import time
import yaml
import math
import copy
import json
import numpy as np
from sentence_transformers import util
from relevance_filter.snippet_relevance import snippet_relevance
from relevance_filter.wikipedia2vec_relevance import get_wikipedia_embeddings, cosine_similarity

CONFIDENCE_THRESHOLD = 0.125 # original score should be at least this 
MAX_ORDER_DIFF = 1.0 # max order of magnitude difference to be tolerated 

def oom_dist(n_i, n_j):
	if min(n_i, n_j) == 0 and max(n_i, n_j) == 0:
		order_diff = 0
	elif min(n_i, n_j) == 0:
		order_diff = np.log(max(n_i, n_j))
	else:
		order_diff = np.log(max(n_i, n_j)/min(max(1, n_i), max(1, n_j)))/np.log(10)
	return min(MAX_ORDER_DIFF, order_diff)


class Node(object):
	"""docstring for Node"""
	def __init__(self):
		super(Node, self).__init__()
		self.id = None
		self.cardinal = None
		self.confidence = 0.0
		self.score = 0.0
		self.consistency = 0.0
		self.answertype = None
		self.sent_rel = None
		self.relevance = None
		self.snippet_id = None
		self.cardinal_id = None
		self.keyphrases = None
		self.keyphrase_vectors = None
		self.text = ""
		self.sentence = ""
		self.snippet = ""
		self.name = ""
		self.set_catgeory = None
		self.centrality = 0.0	
		self.ndist = []


class CountGraph(object):
	"""docstring for CountGraph"""
	def __init__(self, sbert, nlp, config, count_aggregator, sentence_filter):
		super(CountGraph, self).__init__()
		self.sbert = sbert
		self.nlp = nlp
		if count_aggregator == "consistent":
			self.alpha = config["alpha"]
			self.beta = config["beta"]
			self.k_nearest = config["knn"]
			assert self.alpha >= 0 and self.alpha <= 1
			assert self.beta >= 0 and self.beta <= 1
		elif count_aggregator == "central":
			self.alpha = config["alpha"]
			self.node_weight = config["node_weight"]
			assert self.alpha >= 0 and self.alpha <= 1
			assert self.node_weight in ["snippet", "sentence", "none"]
		self.min_confidence_threshold = config["min_confidence_threshold"]
		assert self.min_confidence_threshold >= 0.0 and self.min_confidence_threshold <= 1.0

		self.text_distance = "sentence"
		self.sentence_filter = sentence_filter
		assert self.sentence_filter in [True, False]
		self.count_aggregation = count_aggregator
		assert self.count_aggregation in ["median", "confident", "central", "consistent"]
		self.nodes = {"nodes": [], "snippet_dist": {}, "cardinal_dist": {}}
		# print("CG params: alpha %.2f, beta %.2f, knn: %s"%(self.alpha, self.beta, self.k_nearest))
		

	def node_dist(self, n_i, n_j):
		snippet_dist = self.nodes["snippet_dist"][n_i.cardinal_id][n_j.cardinal_id]
		cardinal_dist = self.nodes["cardinal_dist"][n_i.cardinal_id][n_j.cardinal_id] 
		# print("snippet_dist: ", snippet_dist, " cardinal_dist: ", cardinal_dist)
		dist = self.alpha*snippet_dist + (1-self.alpha)*cardinal_dist
		return dist


	def consistent_node(self):
		N = len(self.nodes["nodes"])
		max_node = {"c": 0, "node": None}
		if N == 0:
			return max_node["node"]
		for i, node_i in enumerate(self.nodes["nodes"]):
			if node_i.confidence < self.min_confidence_threshold:
				continue
			node_i.ndist = []
			for j, node_j in enumerate(self.nodes["nodes"]):
				if i==j or node_j.confidence < self.min_confidence_threshold:
					continue
				node_dist = self.node_dist(node_i, node_j)
				if not np.isnan(node_dist):
					node_i.ndist.append((node_dist, node_j.confidence))
			
			# update consistency:
			node_i.ndist = sorted(node_i.ndist, key=lambda x: x[0])
			k_nn = node_i.ndist[0:min(len(node_i.ndist), self.k_nearest)]
			if len(k_nn) > 0:
				node_i.consistency = (1/len(k_nn))*sum([conf for dist, conf in k_nn])
			else:
				node_i.consistency = 0.0
			node_i.score = self.beta*node_i.consistency + (1-self.beta)*node_i.confidence

			# update consistent node
			if node_i.score > max_node["c"]:
				max_node["c"] = node_i.score
				max_node["node"] = node_i
		return max_node["node"]


	def central_node(self):
		N = len(self.nodes["nodes"])
		max_node = {"c": 0, "node": None}
		if N == 0:
			return max_node["node"]
		for i, node_i in enumerate(self.nodes["nodes"]):
			if node_i.confidence < self.min_confidence_threshold:
				continue
			node_i.ndist = []
			for j, node_j in enumerate(self.nodes["nodes"]):
				if i==j or node_j.confidence < self.min_confidence_threshold:
					continue
				node_dist = self.node_dist(node_i, node_j)
				if not np.isnan(node_dist):
					node_i.ndist.append((node_dist, node_j.confidence))

			#compute centrality
			if len(node_i.ndist) == 0 or sum([dist for dist, _ in node_i.ndist]) == 0:
				node_i.centrality = 0
			else:
				if self.node_weight == "sentence":
					node_weight = node_i.sent_rel
				elif self.node_weight == "snippet":
					node_weight = node_i.relevance
				else:
					node_weight = 1.0
				sum_dist = sum([dist for dist, _ in node_i.ndist])
				node_i.centrality = ((N - 1)*node_weight)/sum_dist
			
			# update central node
			if node_i.centrality > max_node["c"]:
				max_node["c"] = node_i.centrality
				max_node["node"] = node_i
		return max_node["node"]


	def confident_node(self):
		N = len(self.nodes["nodes"])
		max_node = {"c": self.min_confidence_threshold, "node": None}
		if N == 0:
			return max_node["node"]
		for i, node_i in enumerate(self.nodes["nodes"]):
			if node_i.confidence > max_node["c"]:
				max_node["c"] = node_i.confidence
				max_node["node"] = node_i
		return max_node["node"]	


	def median_node(self):
		conf_nodes = [[node.cardinal, node.confidence] for node in self.nodes["nodes"] if node.confidence >= self.min_confidence_threshold]
		N = len(conf_nodes)
		if N == 0:
			return  None, 0.0
		counts, weights = map(np.array, zip(*(conf_nodes)))
		half_weight = 0.5*sum(weights)
		if any(weights > half_weight):
			median = counts[weights == np.max(weights)][0]
			median_score = weights[weights == np.max(weights)][0]
		else:
			cumsum_scores = np.cumsum(weights)
			median_idx = np.where(cumsum_scores <= half_weight)[0][-1]
			if cumsum_scores[median_idx] == half_weight:
				median = np.mean(counts[median_idx:median_idx+2])
				median_score = np.mean(weights[median_idx:median_idx+2])
			else:
				median = counts[median_idx+1]
				median_score = weights[median_idx+1]
		return median, median_score


	def create_nodes(self, snippets, qtuple):
		self.nodes = {"nodes": [], "snippet_dist": {}, "cardinal_dist": {}}
		for snippet in snippets:
			s_key = "sentences" if "sentences" in snippet else "stuples"
			if s_key in snippet:
				if not any(["sentence" in stuple for stuple in snippet[s_key]]):
					sents = [sent.text for sent in self.nlp(snippet["context"]).sents]
					sents_relevance = snippet_relevance(snippet["question"], sents, self.sbert)
					for idx, stuple in enumerate(snippet[s_key]):
						if "sentence" not in stuple:
							stuple["sentence"] = sents[idx]
						if "relevance" not in stuple:
							stuple["relevance"] = sents_relevance[idx]
				# extract best node per snippet
				best_node = None
				for sent_idx, stuple in enumerate(snippet[s_key]):
					if self.sentence_filter and "filter_sent" in stuple and stuple["filter_sent"]:
						continue
					sent = stuple["sentence"]
					stuple_rel = stuple["relevance"]
					if "keyphrases" in stuple:
						keyphrases = stuple["keyphrases"]
					else:
						keyphrases = {}
					for idx, cc in enumerate(stuple["cardinal_contexts"]):
						# if (cc["cardinal"] is not None) and (cc["set_category"] != "unrelated"):
						if cc["cardinal"] is not None:
							n = Node()
							n.id = snippet["rank"]
							n.cardinal = cc["cardinal"]
							n.answertype = cc["answer_type"]
							n.text = cc["text"]
							n.sentence = sent
							n.keyphrases = keyphrases
							n.cardinal_id = str(snippet["rank"])+"_"+str(idx)
							n.snippet_id = snippet["rank"]
							n.relevance = snippet["relevance"]
							n.sent_rel = stuple_rel
							n.snippet = snippet["context"]
							n.name = snippet["name"]
							if cc["answer_type"] is not None:
								n.answertype_rel = snippet_relevance(qtuple["type"], cc["answer_type"], self.sbert)[0]
							else:
								n.answertype_rel = snippet_relevance(qtuple["type"], "", self.sbert)[0]
							n.confidence = n.relevance*n.sent_rel*n.answertype_rel
							if best_node is None or n.confidence > best_node.confidence:
								best_node = copy.deepcopy(n)
				if best_node is not None:
					self.nodes["nodes"].append(best_node)


	def snippet_dist_sentence(self):
		if len(self.nodes["nodes"]) == 0:
			return
		context_dist = {}
		if self.text_distance == "snippet":
			contexts = [n.snippet for n in self.nodes["nodes"] if len(n.snippet)>0]
		else:
			contexts = [n.sentence for n in self.nodes["nodes"] if len(n.sentence)>0]
		# print(contexts)
		context_enc = self.sbert.encode(contexts, convert_to_tensor=True)
		context_sim = util.cos_sim(context_enc, context_enc)
		i=0
		for n_i in self.nodes["nodes"]:
			context_dist[n_i.cardinal_id] = {}
			n_i_length_context = len(n_i.snippet) if self.text_distance == "snippet" else len(n_i.sentence)
			if n_i_length_context > 0:
				j=0
				for n_j in self.nodes["nodes"]:
					n_j_length_context = len(n_j.snippet) if self.text_distance == "snippet" else len(n_j.sentence)
					if n_i.cardinal_id == n_j.cardinal_id:
						context_dist[n_i.cardinal_id][n_j.cardinal_id] = 0
					elif n_j_length_context > 0:
						if abs(context_sim[i][j].item()) > 1:
							context_sim_i_j = 1.0
						else:
							context_sim_i_j = context_sim[i][j].item()
						# From https://developers.google.com/machine-learning/clustering/similarity/measuring-similarity
						context_dist[n_i.cardinal_id][n_j.cardinal_id] = 0.5*math.sqrt(2.0*(1.0 - context_sim_i_j))
						j += 1
					else:
						context_dist[n_i.cardinal_id][n_j.cardinal_id] = 1
				i += 1
			else:
				for n_j in self.nodes["nodes"]:
					if n_i.cardinal_id == n_j.cardinal_id:
						context_dist[n_i.cardinal_id][n_j.cardinal_id] = 0
					else:
						context_dist[n_i.cardinal_id][n_j.cardinal_id] = 1
		self.nodes["snippet_dist"] = context_dist


	def snippet_dist_keyphrase(self):
		if len(self.nodes["nodes"]) == 0:
			return
		context_dist = {}
		context_embeddings = []
		for n in self.nodes["nodes"]:
			num_keyphrases = sum([len(v) for v in n.keyphrases.values()])
			if num_keyphrases == 0:
				continue
			embeddings = np.zeros(500,)
			for k, values in n.keyphrases.items():
				if len(values) == 0:
					continue
				if k == "named_entities":
					get_type = "entity"
				else:
					get_type = "word"
				for v in values:
					vec = get_wikipedia_embeddings(v, "entity")
					if vec is not None:
						embeddings += vec
			if (embeddings == np.zeros(500,)).all():
				context_embeddings.append(None)
			else:
				context_embeddings.append(embeddings)
		i=0
		for n_i in self.nodes["nodes"]:
			context_dist[n_i.cardinal_id] = {}
			n_i_num_keyphrases = sum([len(v) for v in n_i.keyphrases.values()])
			if n_i_num_keyphrases > 0:
				j=0
				for n_j in self.nodes["nodes"]:
					n_j_num_keyphrases = sum([len(v) for v in n_j.keyphrases.values()])
					if n_i.cardinal_id == n_j.cardinal_id:
						context_dist[n_i.cardinal_id][n_j.cardinal_id] = 0
					elif n_j_num_keyphrases > 0:
						context_dist[n_i.cardinal_id][n_j.cardinal_id] = 0.5*math.sqrt(2.0*(1.0 - cosine_similarity(context_embeddings[i], context_embeddings[j])))
						j += 1
					else:
						context_dist[n_i.cardinal_id][n_j.cardinal_id] = 1
				i += 1
			else:
				for n_j in self.nodes["nodes"]:
					if n_i.cardinal_id == n_j.cardinal_id:
						context_dist[n_i.cardinal_id][n_j.cardinal_id] = 0
					else:
						context_dist[n_i.cardinal_id][n_j.cardinal_id] = 1
		self.nodes["snippet_dist"] = context_dist


	def snippet_dist(self):
		if self.text_distance in ["snippet", "sentence"]:
			self.snippet_dist_sentence()
		elif self.text_distance == "keyphrase":
			self.snippet_dist_keyphrase()


	def count_dist(self):
		if len(self.nodes["nodes"]) == 0:
			return
		cardinal_dist = {}
		for n_i in self.nodes["nodes"]:
			cardinal_dist[n_i.cardinal_id] = {}
			for n_j in self.nodes["nodes"]:
				cardinal_dist[n_i.cardinal_id][n_j.cardinal_id] = oom_dist(n_i.cardinal, n_j.cardinal)
		self.nodes["cardinal_dist"] = cardinal_dist


	def nodes_to_json(self):
		nodes = []
		for n in self.nodes["nodes"]:
			nodes.append({
				"id": n.id,
				"cardinal": n.cardinal,
				"text": n.text,
				"atype": n.answertype,
				"atype_sim": n.answertype_rel,
				"sentence": n.sentence,
				"sentence_sim": n.sent_rel,
				"snippet_name": n.name,
				"snippet_text": n.snippet,
				"snippet_sim": n.relevance,
				"centrality": n.centrality,
				"score": n.score,
				"consistency": n.consistency,
				"confidence": n.confidence
			})
		nodes = sorted(nodes, reverse=True, key = lambda x: x["centrality"])
		return {"nodes": nodes}


	def apply_aggregation(self, snippets, qtuple):
		if self.count_aggregation == "median":
			return self.apply_median(snippets, qtuple)
		elif self.count_aggregation == "confident":
			return self.apply_confidence(snippets, qtuple)
		elif self.count_aggregation == "central":
			return self.apply_centrality(snippets, qtuple)
		elif self.count_aggregation == "consistent":
			return self.apply_consistency(snippets, qtuple)
		else:
			raise ValueError("%s did not match any predefined values for count_aggregation. Must be [median, confident, central, consistent]."%self.count_aggregation)


	def apply_median(self, snippets, qtuple):
		tic = time.perf_counter()
		self.create_nodes(snippets, qtuple)
		cardinal, weight = self.median_node()
		print('Aggregation took %.4f secs'%(time.perf_counter() - tic))	
		prediction = {"final": {"numeric": cardinal, "score": weight}}
		return prediction, self.nodes_to_json()


	def apply_confidence(self, snippets, qtuple):
		tic = time.perf_counter()
		self.create_nodes(snippets, qtuple)
		pred_node = self.confident_node()
		print('Aggregation took %.4f secs'%(time.perf_counter() - tic))	
		if pred_node is not None:
			prediction = {"final": {"numeric": pred_node.cardinal, "score": pred_node.confidence}}
		else:
			prediction = {"final": {"numeric": None, "score": 0.0}}
		return prediction, self.nodes_to_json()


	def apply_centrality(self, snippets, qtuple):
		tic = time.perf_counter()
		self.create_nodes(snippets, qtuple)
		self.snippet_dist()
		self.count_dist()
		pred_node = self.central_node()
		print('Aggregation took %.4f secs'%(time.perf_counter() - tic))
		if pred_node is not None:
			prediction = {"final": {"numeric": pred_node.cardinal, "score": pred_node.centrality}}
		else:
			prediction = {"final": {"numeric": None, "score": 0.0}}
		return prediction, self.nodes_to_json()


	def apply_consistency(self, snippets, qtuple):
		tic = time.perf_counter()
		self.create_nodes(snippets, qtuple)
		self.snippet_dist()
		self.count_dist()
		pred_node = self.consistent_node()
		print('Aggregation took %.4f secs'%(time.perf_counter() - tic))
		if pred_node is not None:
			prediction = {"final": {"numeric": pred_node.cardinal, "score": pred_node.score}}
		else:
			prediction = {"final": {"numeric": None, "score": 0.0}}
		return prediction, self.nodes_to_json()


	def calibrate(self, orig_result, reformulations, top_p=3):
		tic = time.perf_counter()
		final_pred = {}
		no_reformulations = reformulations is None or len(reformulations) == 0
		confident_prediction = orig_result["prediction"]["final"]["score"] > CONFIDENCE_THRESHOLD
		if no_reformulations or confident_prediction:
			return {
				"prediction_numeric": orig_result["prediction"]["final"]["numeric"], 
				"top_peers": [], 
				"score": orig_result["prediction"]["final"]["score"]
				}

		top_p = min(len(reformulations), top_p)

		# rank reformulations by question relevance
		q_rel = snippet_relevance(orig_result['question'], [res["question"] for res, _ in reformulations], self.sbert)
		print("q_rel: ", q_rel)
		# print("reform top: ", reformulations[0].keys())
		top_peers = []
		for rel, peer in sorted(zip(q_rel, reformulations), key=lambda x: x[0], reverse=True):
			new_peer = copy.deepcopy(peer[0])
			new_peer["relevance"] = rel
			new_peer["score"] = new_peer["prediction"]["final"]["score"] * rel
			top_peers.append(new_peer)

		# remove less confident peers
		top_peers = [peer for peer in top_peers if peer["prediction"]["final"]["numeric"] is not None and peer["prediction"]["final"]["score"] >= orig_result["prediction"]["final"]["score"]]

		if len(top_peers) == 0:
			return {
				"prediction_numeric": orig_result["prediction"]["final"]["numeric"], 
				"top_peers": [], 
				"score": orig_result["prediction"]["final"]["score"]
				}

		# top_peers = sorted(top_peers, key=lambda x: x["score"], reverse=True)
		top_peers = sorted(top_peers, key=lambda x: (x["prediction"]["final"]["numeric"], x["score"]))
		
		counts, weights = map(np.array, zip(*([[peer["prediction"]["final"]["numeric"], peer["score"]] for peer in top_peers])))
		half_weight = 0.5*sum(weights)

		if any(weights > half_weight):
			median = counts[weights == np.max(weights)][0]
			median_score = np.max(weights)
		else:
			cumsum_scores = np.cumsum(weights)
			median_idx = np.where(cumsum_scores <= half_weight)[0][-1]
			if cumsum_scores[median_idx] == half_weight:
				median = np.mean(counts[median_idx:median_idx+2])
				median_score = np.mean(weights[median_idx:median_idx+2])
			else:
				median = counts[median_idx+1]
				median_score = weights[median_idx+1]
		final_pred["prediction_numeric"] = median
		final_pred["top_peers"] = top_peers
		final_pred["score"] = median_score
		print('Calibration took %.4f secs'%(time.perf_counter() - tic))


		# final_pred["prediction_numeric"] = top_peers[0]["prediction"]["final"]["numeric"]
		# final_pred["top_peers"] = top_peers
		# final_pred["score"] = top_peers[0]["score"]
		return final_pred



# if __name__ == "__main__":
# 	config_file = "config.yml"
# 	config = yaml.safe_load(open(config_file))
# 	nlp = spacy.load(config["spacy"]["model"])
# 	sbert = SentenceTransformer(config["count"]["sbert"], device="cpu")
# 	snippets = []
# 	cg = CountGraph(sbert=sbert, nlp=nlp, config=config)
# 	cg.apply_centrality(snippets)