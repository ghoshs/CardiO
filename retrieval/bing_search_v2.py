import yaml
import json
import argparse
import requests
from tqdm import tqdm
from utils.utils import write_to_output


class Retriever(object):
	"""docstring for Retriever"""
	def __init__(self, config):
		super(Retriever, self).__init__()
		self.config = config
		self.custom_config = self.config["retriever"]["custom_config"]
		self.url = self.config["retriever"]["url"]
		self.max_retries = 5
		self.headers = {"Ocp-Apim-Subscription-Key": self.config["retriever"]["subscription_key"]}
		self.max_snippets = self.config["retriever"]["max_snippets"]


	def call_bing_api(self, query, count=None):
		if count is None:
			count = self.max_snippets
		params = {
			"q": query, 
			"customconfig": self.custom_config, 
			"mkt": "en-US", 
			"safesearch": "Moderate", 
			"responseFilter": "webPages", 
			"count": count,
			"offset": 0
		}		
		retry = 0
		offset = 0
		snippets = []

		while len(snippets) < count and retry < self.max_retries:
			response = requests.get(self.url, headers=self.headers, params=params)
			response.raise_for_status()
			results = response.json()
			if 'webPages' in results:
				for rank, item in enumerate(results['webPages']['value']):
					webpage = {}
					webpage['rank'] = rank
					webpage['url'] = item['url'] if 'url' in item else ''
					webpage['name'] = item['name'] if 'name' in item else ''
					webpage['context'] = item['snippet'] if 'snippet' in item else ''
					webpage['dateLastCrawled'] = item['dateLastCrawled'] if 'dateLastCrawled' in item else ''
					webpage['question'] = query
					snippets.append(webpage)
			retry += 1
			params["offset"] += len(snippets)
		return snippets

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input", help="JSON file with questions")
	parser.add_argument("-o", "--output", help="JSONL file with snippets")
	parser.add_argument("-c", "--config", help="config file with retriever parameters")
	parser.add_argument("-f", "--filemode", type=str, choices=["a", "w"], required=True, help="to create a new file or append to existing")
	parser.add_argument("-n", "--num_snippets", type=int, default=50, help="number of snippets to retrieve per question")

	args = parser.parse_args()
	config = yaml.safe_load(open(args.config))
	ret = Retriever(config)

	with open(args.input) as fpi:
		# data = json.load(fpi)["data"]
		data = json.load(fpi)
		reform_queries = [oq for oq in data] + [r for oq in data for r in data[oq]]
		print("%d questions loaded!!"%len(reform_queries))
		
		completed_queries = []
		# create output file (will erase old data!!!!)
		if args.filemode == "w":
			with open(args.output, args.filemode) as fptemp:
				pass
		else:
			with open(args.output, "r") as fptemp:
				for line in fptemp:
					record = json.loads(line)
					for key in record:
						completed_queries.append(key)
		
		# for record in data:
		for reform_q in tqdm(reform_queries):
			# snippets = ret.call_bing_api(record["question"], args.num_snippets)
			if reform_q in completed_queries:
				continue
			snippets = ret.call_bing_api(reform_q, args.num_snippets)
			
			if len(snippets) > 50:
				snippets = snippets[0:50]
			# snippets_cache = {record["question"]: snippets}
			snippets_cache = {reform_q: snippets}
			
			write_to_output(snippets_cache, args.output, "a")
	print("Complete!!")