import json
import argparse
import spacy
from utils.load import load_query_data, load_config
from question_modelling.question_modelling import QuestionModel
from question_reformulation.question_reformulation import QuestionReformulation


def extract_reformulations(prediction_file, output_file):
	reform = {}
	with open(prediction_file, "r") as fp:
		for idx, line in enumerate(fp):
			record = json.loads(line)
			if "reformulations" in record:
				reform[record["question"]] = record["reformulations"][0]["unique"]
	
	with open(output_file, "w", encoding="utf-8") as fout:
		json.dump(reform, fout, ensure_ascii=False)
	print("Reforulations for %d questions written to file:%s"%(len(reform), output_file))


def reformulate(benchmark_file, config_file, output_file, restart):
	reform = {}
	query_data = load_query_data(benchmark_file)
	config = load_config(config_file)
	nlp = spacy.load(config["spacy"]["model"])
	if restart:
		reform = json.load(open(output_file))
		print("Num ques in reform: %d"%len(reform))
	for record in query_data:
		if record["question"] in reform:
			continue
		qm = QuestionModel(nlp, record["question"])
		qm.parse()
		qtuple = qm.qtuple()
		qr = QuestionReformulation(
			record["question"], 
			config,
			qtuple.answer_type, 
			qtuple.named_entities, 
			qtuple.relation, 
			qtuple.context)
		qr.reformulate()
		reformulated_questions = list(set([q for _type in qr.reform_q for q in qr.reform_q[_type]]))
		reform[record["question"]] = reformulated_questions

	with open(output_file, "w", encoding="utf-8") as fout:
		json.dump(reform, fout, ensure_ascii=False)
	print("Reformulations for %d questions written to file:%s"%(len(reform), output_file))


def cache_reform(args):
	if args.prediction_file is not None and args.output_file is not None:
		extract_reformulations(args.prediction_file, args.output_file)
	elif args.benchmark_file is not None and args.config_file is not None and args.output_file is not None:
		reformulate(args.benchmark_file, args.config_file, args.output_file, args.restart)
	else:
		raise ValueError("Specify either prediction or benchmark file along with an output file.")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-p', '--prediction_file', type=str, default=None, help="prediction file (JSONL) to extract reformlations")
	parser.add_argument('-b', '--benchmark_file', type=str, default=None, help="benchmark file (JSON) to create reformulations for")
	parser.add_argument('-c', '--config_file', type=str, default=None, help="config file (YAML) with model specifics")
	parser.add_argument('-o', '--output_file', type=str, default=None, help="output file (JSONL) to save reformulations")
	parser.add_argument('-r', '--restart', action="store_const", const=True, help="restart position")

	args = parser.parse_args()
	cache_reform(args)