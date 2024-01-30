import os
import json

class SnippetsCache(object):
	def __init__(self, config, filename):
		if "benchmark" in filename and filename.endswith(".json"):
			self.datapath = filename
			self.benchmark_to_load = True
		else:
			self.datapath = os.path.join(config["cache_path"], filename)
			self.benchmark_to_load = False
		self.contexts = {}


	def create(self, pred_file=None):
		with open(self.datapath, "w") as outfp:
			pass
		if pred_file is not None:
			with open(pred_file, 'r') as fp:
				for line in fp:
					record = json.loads(line)
					contexts = []
					for ann in record["result"][0]["annotations"]:
						if isinstance(ann["rank"], int):
							contexts.append({
								"rank": ann["rank"], 
								"url": ann["url"], 
								"name": ann["name"], 
								"context": ann["context"],
								"dateLastCrawled": ann["dateLastCrawled"], 
								"question": ann["question"]})
					with open(self.datapath, "a") as outfp:
						buff = {record["question"]: contexts}
						outfp.write(json.dumps(buff, ensure_ascii=False)+"\n")


	def load_contexts(self, questions_to_load=None):
		if self.benchmark_to_load:
			data = json.load(open(self.datapath, "r"))["data"]
			for item in data:
				self.contexts[item["question"]] = item["contexts"]
		else:
			try:
				with open(self.datapath, "r") as fp:
					for line in fp:
						record = json.loads(line)
						for question, contexts in record.items():
							if (questions_to_load is None) or (question in questions_to_load):
								self.contexts[question] = contexts	
			except FileNotFoundError:
				print("%s does not exist!!"%self.datapath)
				print("Creating a new snippets cache at: %s"%self.datapath)
				self.create()


	def has(self, q):
		return q in self.contexts


	def get(self, q):
		if self.has(q):
			return self.contexts[q]
		else:
			raise ValueError("%s not found in cache: %s."%(q, self.datapath))


	def update(self, q, res):
		curr_contexts = []
		for ann in res:
			curr_contexts.append({
				"rank": ann["rank"], 
				"url": ann["url"], 
				"name": ann["name"], 
				"context": ann["context"],
				"dateLastCrawled": ann["dateLastCrawled"], 
				"question": ann["question"]})
		self.contexts[q] = curr_contexts
		with open(self.datapath, "a") as fp:
			buff = {q: curr_contexts}
			fp.write(json.dumps(buff, ensure_ascii=False)+"\n")


class CountsCache(object):
	def __init__(self, filenames):
		super(CountsCache, self).__init__()
		self.filenames = filenames
		self.counts = {}

	@staticmethod
	def parse_count(quantity):
		if isinstance(quantity, list):
			quantity = [q for q in quantity if q is not None]
			convert = []
			for q in quantity:
				if q is not None:
					try:
						c = float(q)
					except ValueError:
						c = None
					except Exception as e:
						print(e, "counld not parse %s", str(q))
						c = None
					finally:
						if c is not None:
							convert.append(c)
			if len(convert) == 0:
				return None, 0.0
			elif len(convert) == 1:
				if convert[0] is not None:
					return float(convert[0]), 0.0
				else:
					return None, 0.0
			else:
				lower = float(convert[0]) 
				upper = float(convert[1]) if convert[1] is not None else lower
				mid = (lower+upper)/2.0
				uncert = mid - lower
				return mid, uncert
		elif isinstance(quantity, (int, float)):
			return float(quantity), 0.0
		else:
			return None, 0.0

	@staticmethod
	def parse_math_rel(span, limit):
		if limit is None:
			if any([x in span.lower() for x in ["apprx", "approx", "around", "almost", "about", "nearly", "close to", "roughly", "~"]]):
				return "~"
			elif any([x in span.lower() for x in ["more than", "at least", "exceeds", "over", ">"]]):
				return ">"
			elif any([x in span.lower() for x in ["less than", "at most", "<"]]):
				return "<"
			else:
				return "="
		elif not isinstance(limit, str):
			return ""
		elif limit == "exact":
			return "="
		elif limit == "range":
			return "<>"
		elif limit == "lower":
			return ">"
		elif limit == "upper":
			return "<"
		elif "approx" in limit:
			return "~"
		elif any([x in span.lower() for x in ["apprx", "approx", "around", "almost", "about", "nearly", "close to", "roughly", "~"]]):
			return "~"
		elif any([x in span.lower() for x in ["more than", "at least", "exceeds", "over", ">"]]):
			return ">"
		elif any([x in span.lower() for x in ["less than", "at most", "<"]]):
			return "<"
		else:
			return "="
			

	def load_counts(self):
		for filename in self.filenames:
			with open(filename, 'r') as fp:
				for line in fp:
					data = json.loads(line)
					if data["counts"] is not None:
						for cdata in data["counts"]:
							if not isinstance(cdata, dict):
								continue
							count_data = {}
							if "span" in cdata:
								count_data["text"] = cdata["span"]
							else:
								count_data["span"] = ""
							if "type" in cdata:
								count_data["answer_type"] = cdata["type"]
							else:
								count_data["answer_type"] = ""
							if "quantity" in cdata:
								count_data["cardinal"], count_data["uncertainty"] = self.parse_count(cdata["quantity"])
							else:
								count_data["cardinal"], count_data["uncertainty"] = None, 0.0
							if "limit" in cdata:
								count_data["math_rel"] = self.parse_math_rel(cdata["span"], cdata["limit"])
							else:
								count_data["math_rel"] = ""
							count_data["cost"] = data["cost"]/len(data["counts"])
							if not self.has(data["question"]):
								self.counts[data["question"]] = {}
							if data["rank"] not in self.counts[data["question"]]:
								self.counts[data["question"]][data["rank"]] = {}
							if data["sent_id"] not in self.counts[data["question"]][data["rank"]]:
								self.counts[data["question"]][data["rank"]][data["sent_id"]] = []
							self.counts[data["question"]][data["rank"]][data["sent_id"]].append(count_data)


	def has(self, q):
		return q in self.counts


	def get(self, q):
		if self.has(q):
			return self.counts[q]
		else:
			print("%s not found in cache: %s."%(q, str(self.filenames)))
			return []


class SFCache(object):
	def __init__(self, filenames):
		super(SFCache, self).__init__()
		self.filenames = filenames
		self.sfs = {}


	def load_sfs(self):
		for filename in self.filenames:
			try:
				with open(filename, "r") as fp:
					for line in fp:
						record = json.loads(line)
						if record["question"] not in self.sfs:
							self.sfs[record["question"]] = {}
						if record["rank"] not in self.sfs[record["question"]]:
							self.sfs[record["question"]][record["rank"]] = {}
						self.sfs[record["question"]][record["rank"]][record["sent_id"]] = {
							"filter_sent": record["filter_sent"],
							"filter_cost": record["cost"],
							"filter_expl": record["filter_expl"],
							"sent": record["sent"]
						}
			except FileNotFoundError:
				raise Error("%s does not exist!!"%filename)
			


	def has(self, q, rank=None):
		if rank is None:
			return q in self.sfs
		return q in self.sfs and rank in self.sfs[q]


	def get(self, q, rank):
		if self.has(q, rank):
			return self.sfs[q][rank]
		else:
			if not self.has(q):
				raise KeyError("%s not found in cache: %s."%(q, str(self.filenames)))
			else:
				print(rank, type(rank), self.sfs[q].keys(), type(list(self.sfs[q].keys())[0]))
				raise KeyError("Snippet id:%d for q:%s, not found in cache: %s."%(rank, q, str(self.filenames)))
			return []


	def get_sent_filters(self, q, rank, sent_ids):
		if len(sent_ids) == 0:
			return []
		print("in SF: rank: ", rank, " question: ", q)
		sfs = self.get(q, rank)
		s_llm_filter = []
		# print("All sfs: ", sfs)
		for sent_id in sent_ids:
			if sent_id in sfs:
				s_llm_filter.append([sfs[sent_id]["filter_sent"], sfs[sent_id]["filter_cost"], sfs[sent_id]["filter_expl"]])
			else:
				s_llm_filter.append([True, 0.0, "no output from SF"])
		return s_llm_filter


class RFCache(object):
	def __init__(self, filenames):
		super(RFCache, self).__init__()
		self.filenames = filenames
		self.rfs = {}


	def load_rfs(self):
		for filename in self.filenames:
			try:
				with open(filename, "r") as fp:
					data = json.load(fp)
					for q in data:
						if q in self.rfs:
							self.rfs += data[q]
						else:
							self.rfs[q] = data[q]
			except FileNotFoundError:
				raise Error("%s does not exist!!"%filename)


	def has(self, q):
		return q in self.rfs


	def get(self, q):
		if self.has(q):
			return self.rfs[q]
		else:
			raise Error("%s not found in cache: %s."%(q, str(self.filename)))
			return []		
