import os
import time
import json
import argparse
from collections import defaultdict

## set cache directories before loading the predictor module
os.environ["TRANSFORMERS_CACHE"] = "/GW/count_information/static00/.cache/huggingface/transformers/"

import spacy
from transformers import pipeline as tf_pipe
from sentence_transformers import SentenceTransformer

from utils.load import load_config
from utils.utils import unicode_decoder 
from utils.cache import SnippetsCache, CountsCache, SFCache
from question_modelling.question_modelling import QuestionModel
from question_reformulation.question_reformulation import QuestionReformulation
from retrieval.bing_search_v2 import Retriever
from snippet_modelling.snippet_modelling import SnippetModel
from relevance_filter.snippet_relevance import snippet_relevance
from relevance_filter.llm_sentence_filter import SentenceFilter
from relevance_filter.wikipedia2vec_relevance import get_wikipedia_embeddings
# from count_prediction.apply_aggregator import apply_aggregator
from count_prediction.centrality import CountGraph
from enumeration_prediction.enumeration_prediction import predict_enumerations


class Pipeline(object):
	"""docstring for Pipeline"""
	def __init__(self, config, use_cache, use_llm_counts_cache, use_llm_sf_cache, count_aggregator, device):
		super(Pipeline, self).__init__()
		self.config = load_config(config)
		self.load_models(device)
		self.count_aggregator = count_aggregator
		self.count_threshold = float(self.config["count"]["span_prediction"]["threshold"])
		self.sr = Retriever(self.config)
		if use_cache:
			self.scache = SnippetsCache(self.config, use_cache)
			self.scache.load_contexts()
		else:
			self.scache = None
		if use_llm_counts_cache:
			self.counts_cache = CountsCache(use_llm_counts_cache)
			self.counts_cache.load_counts()
		else:
			self.counts_cache = None
		if use_llm_sf_cache:
			self.sf_cache = SFCache(use_llm_sf_cache)
			self.sf_cache.load_sfs()
		else:
			self.sf_cache = None


	def load_models(self, device):
		# nlp for query tuples and count context
		self.nlp = spacy.load(self.config["spacy"]["model"])
		
		# # count span prediction
		# model_path = self.config["count"]["span_prediction"]["model_path"]
		# self.qa_count = tf_pipe("question-answering", model_path)

		# count contextualization
		self.sbert = SentenceTransformer(self.config["count"]["sbert"], device=device) 

		print("All models loaded!!")


	def question_modelling(self, question):
		print('Modelling count question', flush=True)
		tic = time.perf_counter()
		qm = QuestionModel(self.nlp, question)
		qm.parse()
		print("Completed question modelling in %.4f secs."%(time.perf_counter() - tic))
		return qm


	def snippet_retrieval(self, question, **kwargs):
		tic = time.perf_counter()
		## snippets -> list(dict(rank, url, name, context, dateLastCrawled))
		if 'contexts' in kwargs:
			print('Retrieving relevant documents from context args', flush=True)
			snippets = kwargs['contexts']
		else:
			print("Retrieving relevant documents for the original question", flush=True)
			snippets = self.sr.call_bing_api(question)
		return snippets 


	def snippet_retrieval_for_reformulations(self, reformulations, snippets, **kwargs):
		print("Retrieving relevant documents for reformulated question", flush=True)
		reformulated_snippets = []
		for idx, q in enumerate(reformulations):
			q_snippets = self.sr.call_bing_api(q)
			for s in q_snippets:
				dup = False
				for orig_s in snippets+reformulated_snippets:
					url_match = int(orig_s["url"] == s["url"])
					name_match = int(orig_s["name"] == s["name"])
					content_match = int(orig_s["context"] == s["context"])
					if url_match+name_match+content_match>=2:
						dup = True
						break
				if not dup:
					s["rank"] = "_".join(["reform", str(idx), str(s["rank"])])
					reformulated_snippets.append(s)
		print("Number of snippets from reformulated questions: ", len(reformulated_snippets))
		print("Completed retrieval in %.4f secs."%(time.perf_counter() - tic), flush=True)
		return reformulated_snippets


	def get_snippet_relevance(self, question, snippets, topk):
		print("Snippet Relevance", flush=True)
		tic = time.perf_counter()
		contexts = []
		rel_contexts = [item['name']+'. '+item['context'] for item in snippets if len(item['name']+item['context'])>0]
		if len(rel_contexts) == 0:
			raise ValueError("Question: ", question, "Snippets: ", snippets, "==========\n EMPTY context" )
		s_relevance = snippet_relevance(question, rel_contexts, self.sbert)
		_idx = 0
		for snip, rel in zip(snippets, s_relevance):
			snip["relevance"] = rel
		snippets = sorted(snippets, key=lambda x: x["relevance"], reverse=True)
		if len(snippets) > topk:
			snippets = snippets[:topk]
		print("Completed snip relevance in %.4f secs."%(time.perf_counter() - tic))
		return snippets


	def count_candidate_extraction(self, snippets, topk=None):
		print("Count candidate extraction", flush=True)
		tic = time.perf_counter()
		if topk is None:
			topk = len(snippets)
		print("Extracting counts for the top %d snippets."%topk, flush=True)
		for idx, item in enumerate(snippets[:topk]):
			print("Modelling %d of %d snippets.."%(idx, len(snippets)))
			sm = SnippetModel(self.nlp, item["context"], self.config)
			sm.parse()
			item["sentences"] = sm.to_json()
		print("Completed count extraction in %.4f secs."%(time.perf_counter() - tic))
		return snippets


	def llm_count_candidate_extraction(self, question, snippets, topk=None):
		print("LLM Count candidate extraction", flush=True)
		tic = time.perf_counter()
		if topk is None:
			topk = len(snippets)
		if self.counts_cache.has(question):
			count_contexts = self.counts_cache.get(question)
		else:
			count_contexts = {}
		
		print("Extracting counts for the top %d snippets."%topk, flush=True)
		snippet_anns = self.nlp.pipe([item["context"] for item in snippets[:topk]])
		for snippet, ann in zip(snippets[:topk], snippet_anns):
			snippet["sentences"] = []
			for idx, sent in enumerate(ann.sents):
				sent_parse = {"sentence": sent.text, "cardinal_contexts": []}
				if snippet["rank"] in count_contexts and idx in count_contexts[snippet["rank"]]:
					sent_parse["cardinal_contexts"] += count_contexts[snippet["rank"]][idx]
				snippet["sentences"].append(sent_parse)

		print("Completed count extraction in %.4f secs."%(time.perf_counter() - tic))
		return snippets


	def sentence_candidate_filter(self, question, snippets):
		print("Filter out sentences..", flush=True)
		tic = time.perf_counter()
		sf = SentenceFilter(self.config)
		cost = 0.0
		for idx, item in enumerate(snippets):
			# print("Modelling %d of %d snippets.."%(idx, len(snippets)), flush=True)
			sentences = [s["sentence"] for s in item["sentences"] if len(s["sentence"]) > 0]
			s_relevance = snippet_relevance(question, sentences, self.sbert)
			i=0
			for s in item["sentences"]:
				if len(s["sentence"]) > 0:
					s["relevance"] = s_relevance[i]
					i+=1
				else:
					s["relevance"] = 0.0

			# print("SBert relevance computed for %d sentences."%len(sentences), flush=True)
			if self.sf_cache is None:
				for s in item["sentences"]:
					s["filter_sent"] = False
					s["filter_cost"] = 0.0
					s["filter_expl"] = "no filter"
			else:
				sent_ids = []
				sentences = []
				
				for s_id, s in enumerate(item["sentences"]):
					sent_has_cardinal = any([cc["cardinal"] is not None for cc in s["cardinal_contexts"]])
					if len(s["sentence"]) > 0 and sent_has_cardinal:
						sent_ids.append(s_id)
						sentences.append(s["sentence"])
					elif len(s["sentence"]) == 0:
						s["filter_sent"] = True
						s["filter_cost"] = 0.0
						s["filter_expl"] = "empty sentence"
					else:
						s["filter_sent"] = True
						s["filter_cost"] = 0.0
						s["filter_expl"] = "no cardinal found"
				
				# print("==================")
				s_llm_filter = self.sf_cache.get_sent_filters(question, item["rank"], sent_ids)
				# print("sents: ", '\n'.join([str(idx)+":"+i["sentence"] for idx, i in enumerate(item["sentences"])]))
				# print("Rank:", item["rank"], "Sent IDS: ", sent_ids)
				# print("SFS: ", s_llm_filter)
				# print("==================")

				# if len(sent_ids) > 0 and len(s_llm_filter) == 0:
				# 	# sf not in cache:
				# 	s_llm_filter = sf.filter(question, item["name"], sentences)

				for idx, sent_id in enumerate(sent_ids):
					item["sentences"][sent_id]["filter_sent"] = s_llm_filter[idx][0]
					item["sentences"][sent_id]["filter_cost"] = s_llm_filter[idx][1]
					item["sentences"][sent_id]["filter_expl"] = s_llm_filter[idx][2]
					cost += s_llm_filter[idx][1]
		print("Completed filtering in %.4f secs."%(time.perf_counter() - tic))
		return snippets, cost


	def count_context_prediction(self, snippets, qtuple):
		print("Count context prediction", flush=True)
		tic = time.perf_counter()
		cg = CountGraph(
				sbert=self.sbert, 
				nlp=self.nlp, 
				config=self.config[self.count_aggregator], 
				count_aggregator=self.count_aggregator, 
				sentence_filter=self.sf_cache is not None)
		prediction, count_data = cg.apply_aggregation(snippets=snippets, qtuple=qtuple)
		print("Completed prediction in %.4f secs."%(time.perf_counter() - tic))
		return prediction, count_data


	def run(self, question, **kwargs):
		tic_init = time.perf_counter()
		result = {}

		### 1.a. question modeling: namedtuple QTuples('type', 'entity', 'relation' 'context')
		qm = self.question_modelling(question)
		qtuple = qm.qtuple()
		result['qtuple'] = {
			'type': qtuple.answer_type, 
			'entity': ';'.join(qtuple.named_entities), 
			'relation': qtuple.relation, 
			'context': ';'.join(qtuple.context),
			'kb_entities': qm.kb_entities
		}

		# ### 2. Document retrieval (Bing/Wikipedia): JSON 
		snippets = self.snippet_retrieval(question, **kwargs)

		### 3. Snippet Relevance
		snippets = self.get_snippet_relevance(question, snippets, topk=50)
		
		## 3. Count Candidate Extraction
		if self.counts_cache is not None:
			snippets = self.llm_count_candidate_extraction(question, snippets)
		else:
			snippets = self.count_candidate_extraction(snippets)

		## 4. Filter candidates
		snippets, sentence_filter_cost = self.sentence_candidate_filter(question, snippets)

		### 5. Count Context Prediction
		prediction, count_data = self.count_context_prediction(snippets, result['qtuple'])

		### Assemble results
		result["question"] = question
		result["prediction"] = prediction
		result["count_data"] = count_data
		result["annotations"] = snippets

		elapsed_time = time.perf_counter()-tic_init
		print('Total time lapsed %.4f secs'%(elapsed_time), flush=True)
		return result, elapsed_time


	def reformulate(self, question):
		tic = time.perf_counter()
		print("Reformulating the count question", flush=True)
		qm = self.question_modelling(question)
		qtuple = qm.qtuple()
		qr = QuestionReformulation(
			question, 
			self.config,
			qtuple.answer_type, 
			qtuple.named_entities, 
			qtuple.relation, 
			qtuple.context)
		qr.reformulate()
		reformulated_questions = list(set([q for _type in qr.reform_q for q in qr.reform_q[_type]]))
		result = {
			"original": question,
			"unique": reformulated_questions, 
			"cost": qr.reform_cost
		}
		print("Completed question reformulation in %.4f secs."%(time.perf_counter() - tic))
		return result


	def run_reformulations(self, question, **kwargs):
		tic = time.perf_counter()
		result = {}

		reformulations = None
		if "reformulations" in kwargs:
			reformulations = kwargs["reformulations"]
		else:
			reformulations = self.reformulate(question)
		reformulations["results"] = []

		for q in reformulations["unique"]:
			if self.scache.has(q):
				contexts = self.scache.get(q)
				res = self.run(q, contexts=contexts)
			####### uncomment for live run
			# else:
			# 	res = self.run(q)
			# 	self.scache.update(q, res[0]["annotations"])
			reformulations["results"].append(res)
		elapsed_time = time.perf_counter() - tic
		print("Completed reformulation inference in %.4f secs."%(elapsed_time), flush=True)
		return reformulations, elapsed_time


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument("-c", "--config", required=True, type=str, help="path to config file")
	parser.add_argument("-q", "--question", type=str, help="user input count question")
	parser.add_argument("-ctx", "--context", type=str, help="additional context")
	parser.add_argument("-aggregator", "--agg", required=True, type=str, choices=["median", "confident", "central", "consistent"], help="which prediction strategy to apply")
	parser.add_argument("-d", "--dev", default='cpu', type=str, choices=["cpu", "cuda"], help="choose between cpu/gpu for sbert")
	parser.add_argument("-o", "--output_file", required=True, help="file to store query output")
	parser.add_argument("-w", "--write_append", required=True, type=str, choices=["a", "w"], default="a", help="to create (w) or append to (a) output file")
	parser.add_argument("-uc", "--use_cache", type=str, default=None, help="load contexts from this file located in cache dir")
	parser.add_argument("-ullmc", "--use_llm_cache", type=str, default=None, help="filename to load pre-computed count contexts from")
	parser.add_argument("-usfc", "--use_sf_cache", type=str, default=None, help="filename to load pre-computed sentence filters from")
	parser.add_argument("-urfc", "--use_rf_cache", nargs="*", type=str, default=None, help="filename to load pre-computed reformulations from")

	args = parser.parse_args()

	pl = Pipeline(args.config, args.use_cache, args.use_llm_cache, args.use_sf_cache, args.agg, args.device)
