import os
import json
import yaml
# import nltk
# import requests
# import networkx as nx
# from nltk.corpus import wordnet


"""
Utility functions
"""


def write_to_output(record, output_file, mode):
	if record is None and mode == "w":
		with open(output_file, mode, encoding="utf-8") as fp:
			pass
	else:
		with open(output_file, mode, encoding="utf-8") as fp:
			fp.write(json.dumps(record, ensure_ascii=False)+"\n")


# def is_synonym_hypernym_hyponym(term1, term2):
# 	term1 = term1.replace(" ", "_")
# 	term2 = term2.replace(" ", "_")
# 	hypo = lambda s: s.hyponyms()
# 	hyper = lambda s: s.hypernyms()

# 	synsets1 = wordnet.synsets(term1)
# 	synsets2 = wordnet.synsets(term2)

# 	if len(synsets1) == 0 or len(synsets2) == 0:
# 		return "unidentified"

# 	# context is answer type if any synset matches.
# 	for synset1 in synsets1:
# 		for synset2 in synsets2:
# 			if synset1 == synset2:
# 				return "synonym"

# 	term1_hyper_closure = [i for synset in synsets1 for i in synset.closure(hyper)]
# 	# term1_hypo_closure = [i for synset in synsets1 for i in synset.closure(hypo)]

# 	term2_hyper_closure = [i for synset in synsets2 for i in synset.closure(hyper)]
# 	# term2_hypo_closure = [i for synset in synsets2 for i in synset.closure(hypo)]

# 	# context is a answer type if synset2 in hypernym closure of synset1 
# 	if any([s2 in term1_hyper_closure for s2 in synsets2]):
# 		return "hyponym"

# 	# answer type is a context if synset1 in hypernym closure of synset2
# 	if any([s1 in term2_hyper_closure for s1 in synsets1]):
# 		return "hypernym"

# 	else:
# 		return "unrelated"


# def bipartite_match(q, c, sim_scores):
# 	G = nx.Graph()
# 	G.add_nodes_from(q, bipartite=0)
# 	G.add_nodes_from(c, bipartite=1)
# 	for i, item_q in enumerate(q):
# 		for j, item_c in enumerate(c):
# 			G.add_edge(item_q, item_c, weight=sim_scores[i][j].item())
# 	return G, list(nx.max_weight_matching(G))


# def wikidata_entity_search(term):
# 	# implement cache to avoid unncessary lookups
# 	try:
# 		url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={term}&language=en&format=json"
# 		data = requests.get(url).json()
# 		# Return the first id
# 		# print(term, data)
# 		return data['search'][0]['id']
# 	except Exception as e:
# 		print("Disambiguation for: ", term, " ran into error: ",e)
# 		return ""


def unicode_decoder(s):
    y = s.encode().decode('unicode-escape')
    decoded_str = y if len(y) < len(s) else s
    return decoded_str


def to_text(input_file, output_file, questions):
	with open(output_file, "w") as outf:
		with open(input_file, "r") as inpf:
			for line in inpf:
				record = json.loads(line)
				if record["question"] in questions:
					outf.write(record["question"]+"\n")
					cats = ["exact", "subclass", "superclass","peer","unrelated"]
					ann = {str(ann["rank"]): (ann["name"], ann["context"]) for ann in record["result"][0]["annotations"]}
					for cat in cats:
						outf.write("=====%s=====\n"%cat)
						for cc in record["result"][0]["count_data"][cat]:
							outf.write(", ".join([str(cc[0]), str(round(cc[1],2)), ann[str(cc[2])][0], ann[str(cc[2])][1]]))
							outf.write("\n")
					outf.write("==========\n\n")


# if __name__ == "__main__":
# 	pass
	# questions = [
	# 	# "how many world heritage sites are there in the world",
	# 	# "how many animation studios are there in the world",
	# 	# "how many hotels are there in the world"
	# 	# "how many mma fighters have died in the ring",
	# 	# "how many nuclear power plants are there in the world",
	# 	# "how many species of birds are there in the world",
	# 	# "how many sports are there in the Olympics",
	# 	# "how many Grammy awards has Adele won",
	# 	# "how many nurses are there in the world",
	# 	# "how many lakes are there in Switzerland",
	# 	# "how many lawyers in Brazil"
	# 	"how many artists have recorded cover versions of Bob Dylan songs",
	# 	"how many artists have recorded cover versions of Beatles songs"
	# ]
	# to_text("data/output/cardio/cardio_prediction_no_reformulations.jsonl", "examples.txt", questions)

	# cache = Cache({"cache_path": "/GW/count_information/work/class_cardinality_web/data/snippets/snippets_cache.jsonl"})
	# cache.create("/GW/count_information/work/class_cardinality_web/data/output/cardio/centrality/cardio_prediction_no_reformulations.jsonl")