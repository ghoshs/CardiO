import sys
import json
import time
import argparse
from utils.utils import write_to_output, unicode_decoder
from utils.cache import SnippetsCache, RFCache
from utils.load import load_query_data, load_config
from pipeline import Pipeline
from count_prediction.centrality import CountGraph
from sentence_transformers import SentenceTransformer


class Cardinality_Web(object):
	"""docstring for Cardinality_Web"""
	def __init__(self, input_file, config_file, output_file, restart, use_cache, use_rf_cache):
		super(Cardinality_Web, self).__init__()
		self.config_file = config_file
		self.config = load_config(config_file)
		self.output_file = output_file
		self.query_data = load_query_data(input_file)
		self.restart = restart
		if self.restart > 0:
			print("restarting question from idx: %d"%self.restart, flush=True)
			self.query_data = self.query_data[self.restart:]
			self.fmode = "a"
		else:
			self.fmode = "w"
		if use_cache:
			self.scache = SnippetsCache(self.config, use_cache)
			self.scache.load_contexts()
		if use_rf_cache:
			self.rfcache = RFCache(use_rf_cache)
			self.rfcache.load_rfs()


	def run_queries(self, args):
		coqex_pp = Pipeline(
			self.config, 
			args.use_cache, 
			args.use_llm_cache, 
			args.use_sf_cache, 
			args.aggregator, 
			args.device)
		for idx, record in enumerate(self.query_data):
			if args.use_cache and self.scache.has(record["question"]):
				contexts = self.scache.get(record["question"])
				result = coqex_pp.run(record["question"], contexts=contexts)
			elif "contexts" in record:
				result = coqex_pp.run(record["question"], contexts=record["contexts"])
			else:
				raise ValueError("Context cache not provided!!!")
				# #### activate in live run
				# result = coqex_pp.run(record["question"])
			record["result"] = result

			if args.run_reformulation:
				print("Running reformulation inference results...", flush=True)
				if coqex_pp.scache is None:
					raise ValueError("Did not find a snippets cache!!")
				if args.use_rf_cache:
					rfs = {"unique": self.rfcache.get(record["question"])}
					record["reformulations"] = coqex_pp.run_reformulations(record["question"], reformulations=rfs)
				else:
					raise ValueError("run_reformulation is True but not RF cache provided.")
					# ### Activate on Live run
					# record["reformulations"] = coqex_pp.run_reformulations(record["question"])

				cg = CountGraph(
					sbert=coqex_pp.sbert, 
					nlp=coqex_pp.nlp,
					config=coqex_pp.config[coqex_pp.count_aggregator],
					count_aggregator=coqex_pp.count_aggregator, 
					sentence_filter=coqex_pp.sf_cache is not None)

				record["prediction"] = cg.calibrate(
					orig_result=record["result"][0], 
					reformulations=record["reformulations"][0]["results"])
			else:
				record["prediction"] = {
					"prediction_numeric": record["result"][0]["prediction"]["final"]["numeric"], 
					"top_peers": [], 
					"score": record["result"][0]["prediction"]["final"]["score"]
				}

			write_to_output(record, self.output_file, self.fmode)
			if self.fmode == "w":
				self.fmode = "a"
			print("%3d %s: Prediction: %s, Score: %s"%(idx, record["question"], str(record["prediction"]["prediction_numeric"]), str(record["prediction"]["score"])), flush=True)



	def update(self, args):
		# update extraction result without re-running pipeline. 
		# if not args.update_inference:
		print("Initializing CoQEx_pp...", flush=True)
		
		coqex_pp = Pipeline(
			self.config, 
			args.use_cache, 
			args.use_llm_cache, 
			args.use_sf_cache, 
			args.aggregator,
			args.device)

		if not args.update.endswith(".jsonl"):
			updated_res_file = self.output_file.split(".jsonl")[0] + "_" + args.update + '.jsonl'
		elif args.update == self.output_file:
			raise ValueError("Output file and Update file is same (%s)!! Provide another path or a name extension for this output file."%self.output_file)
		else:
			updated_res_file = args.update 
		
		with open(self.output_file, "r") as fp:
			for idx, line in enumerate(fp):
				if idx < self.restart:
					continue
				record = json.loads(line)
				
				cg = CountGraph(
					sbert=coqex_pp.sbert, 
					nlp=coqex_pp.nlp,
					config=coqex_pp.config[coqex_pp.count_aggregator], 
					count_aggregator=coqex_pp.count_aggregator,
					sentence_filter=coqex_pp.sf_cache is not None)

				
				if args.update_inference:
					print("Updating only inference results...", flush=True)
					result = record["result"]
					result[0]["prediction"], result[0]["count_data"] = cg.apply_aggregation(snippets=result[0]["annotations"], qtuple=result[0]["qtuple"])
					
					if "reformulations" in record:
						for result in record["reformulations"][0]["results"]:
							result[0]["prediction"], result[0]["count_data"] = cg.apply_aggregation(snippets=result[0]["annotations"], qtuple=result[0]["qtuple"])
				
				else:
					print("Updating extraction and inference results...", flush=True)
					contexts = []
					for ann in record["result"][0]["annotations"]:
						if isinstance(ann["rank"], int) or ann["rank"].isdigit():
							contexts.append({
								"rank": ann["rank"], 
								"url": ann["url"], 
								"name": ann["name"], 
								"context": unicode_decoder(ann["context"]),
								"dateLastCrawled": ann["dateLastCrawled"], 
								"question": ann["question"]})

					record["result"] = coqex_pp.run(record["question"], contexts=contexts)
				
				if args.run_reformulation:	
					print("Updating reformulation inference results...", flush=True)
					if coqex_pp.scache is None:
						raise ValueError("Did not find a snippets cache!!")
					if args.use_rf_cache:
						rfs = {"unique": self.rfcache.get(record["question"])}
						record["reformulations"] = coqex_pp.run_reformulations(record["question"], reformulations=rfs)
					else:
						raise ValueError("run_reformulation is True but not RF cache provided.")
						# ### Activate on Live run
						# record["reformulations"] = coqex_pp.run_reformulations(record["question"])
					record["prediction"] = cg.calibrate(
						orig_result=record["result"][0], 
						reformulations=record["reformulations"][0]["results"])
					
				elif args.update_reformulation:
					
					record["prediction"] = cg.calibrate(
						orig_result=record["result"][0], 
						reformulations=record["reformulations"][0]["results"])
				else:
					
					record["prediction"] = {
						"prediction_numeric": record["result"][0]["prediction"]["final"]["numeric"], 
						"top_peers": [], 
						"score": record["result"][0]["prediction"]["final"]["score"]
					}


				write_to_output(record, updated_res_file, self.fmode)
				if self.fmode == "w":
					self.fmode = "a"
				print("%3d %s: Prediction: %s, Score: %s"%(idx, record["question"], str(record["prediction"]["prediction_numeric"]), str(record["prediction"]["score"])), flush=True)
				

if __name__ == "__main__":
	parser = argparse.ArgumentParser("Answer count questions on the web")
	parser.add_argument("-i", "--input", required=True, type=str, help="Dataset file in JSON format.")
	parser.add_argument("-c", "--config", required=True, type=str, default="./config.yml", help="Config file with API keys, default paths and other configurations")
	parser.add_argument("-o", "--output", required=True, type=str, help="Predictions file (JSONL)")
	parser.add_argument("-r", "--restart", default=0, type=int, help="question index to restart")
	parser.add_argument("-d", "--device", default='cpu', type=str, choices=["cpu", "cuda"], help="choose between cpu/gpu for sbert")
	parser.add_argument("-rf", "--run_reformulation", action="store_const", const=True, help="run reformulation predictions.")
	
	parser.add_argument("-u", "--update", type=str, help="updated prediction file name")
	parser.add_argument("-uc", "--use_cache", type=str, default=None, help="load contexts from this file located in cache dir")
	parser.add_argument("-ui", "--update_inference", action="store_const", const=True, help="update inference results only, if not mentioned, then update both extraction and inference")
	parser.add_argument("-agg", "--aggregator", required=True, type=str, choices=["median", "confident", "central", "consistent"], help="which prediction strategy to apply")
	parser.add_argument("-urf", "--update_reformulation", action="store_const", const=True, help="update with reformulation predictions.")
	parser.add_argument("-usfc", "--use_sf_cache", nargs="*", type=str, default=None, help="filename to load pre-computed sentence filters from")
	parser.add_argument("-ullmc", "--use_llm_cache", nargs="*", type=str, default=None, help="filename to load pre-computed count contexts from")
	parser.add_argument("-urfc", "--use_rf_cache", nargs="*", type=str, default=None, help="filename to load pre-computed reformulations from")

	# # experiment-specific arguments
	parser.add_argument("-calpha", "--config_agg_alpha", type=float, help="Value in [0,1] for node distance weights")
	parser.add_argument("-mct", "--config_minimum_confidence_threshold", type=float, help="Min confidence between [0,1] for count candidates")
	# # consistency
	parser.add_argument("-cbeta", "--config_agg_beta", type=float, help="Value in [0,1] for consistency/confidence weights")
	parser.add_argument("-cknn", "--config_agg_knn", type=int, choices=[1, 2, 3, 4, 5], help="Number of neighbours for consistency scoring")
	# # centrality
	# parser.add_argument("-nw", "--config_node_weight", type=str, choices=['snippet', 'sentence', 'none'], help="Values in ['snippet', 'sentence', 'none'] for node weights")

	args = parser.parse_args()

	if args.update_reformulation or args.update_inference:
		assert args.update

	start = time.time()
	cw = Cardinality_Web(
		args.input, 
		args.config, 
		args.output, 
		args.restart, 
		args.use_cache,
		args.use_rf_cache)

	print("Arguments: ", args, flush=True)
	if args.update is None:
		print("Running fresh pipeline...", flush=True)
		if args.config_minimum_confidence_threshold is None:
			raise ValueError("Set min config_minimum_confidence_threshold!!")
		else:
			cw.config[args.aggregator]["min_confidence_threshold"] = args.config_minimum_confidence_threshold
		cw.run_queries(args)
	else:

		# print("Updating config aggregation: ")
		# ###Consistency
		# cw.config[args.aggregator]["alpha"] = float(args.config_agg_alpha)
		# cw.config[args.aggregator]["beta"] = float(args.config_agg_beta)
		# cw.config[args.aggregator]["knn"] = int(args.config_agg_knn)
		# cw.config[args.aggregator]["min_confidence_threshold"] = args.config_minimum_confidence_threshold
		
		# ###Centrality
		# cw.config[args.aggregator]["alpha"] = float(args.config_agg_alpha)
		# cw.config[args.aggregator]["node_weight"] = args.config_node_weight
		if args.config_minimum_confidence_threshold is None:
			raise ValueError("Set min config_minimum_confidence_threshold!!")
		else:
			cw.config[args.aggregator]["min_confidence_threshold"] = args.config_minimum_confidence_threshold

		cw.update(args)
	end = time.time()
	print("Total time elapsed: %.4fs"%(end-start))