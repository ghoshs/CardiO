import csv
import json
import yaml

def load_query_data(input_file):
	query_data = []
	with open(input_file, "r") as fp:
		query_data = json.load(fp)
	query_data = query_data["data"]
	print("Read %d queries"%len(query_data))
	return query_data


def load_config(config):
	if isinstance(config, dict):
		return config
	else:
		try:
			with open(config) as fp:
				return yaml.safe_load(fp)
		except FileNotFoundError:
			print("Cannot load config: ", config, ". Have you passed a correct config dict or yml file?")
	return {}


def load_evaluation(input_file):
	evaluations = []
	with open(file, "r") as fp:
		reader = csv.DictReader(fp)
		for row in reader:
			evaluations.append(row)
	return evaluations


def load_predictions(input_file):
	predicitons = []
	with open(input_file, "r") as fp:
		for line in fp:
			record = json.loads(line)
			predicitons.append(record)
	return predicitons