import re
import csv
import ast
import json
import argparse

def clean_answer(ans):
	clean = {"filter_sent": False, "filter_expl": None}
	response = None
	try:
		response = ast.literal_eval(ans)
	except ValueError:
		new_ans = ans.replace("null", "None")
		if new_ans != ans:
			return clean_answer(new_ans)
		else:
			print("ValueError: Could not parse answer:", ans)
	except Exception as e:
		print(e, "Could not parse answer:", ans)
	else:
		print("clean parse of answer:", response)
		if response is not None:
			if "short" in response and response["short"].lower() == "no":
				clean["filter_sent"] = True
			if "long" in response and len(response["long"]) > 0:
				clean["filter_expl"] = response["long"]
	return clean["filter_sent"], clean["filter_expl"]


def compute_cost(model, usage):
	cost_dir = {
		"gpt-3.5-turbo": {
			"input": 0.0010, 
			"output": 0.0020
		}, 
		"gpt-4": {
			"input": 0.03, 
			"output": 0.06
		}
	}
	model_price = [cost_dir[m] for m in cost_dir if m in model][0]
	cost = (model_price["input"]*usage["prompt_tokens"])/1000 + (model_price["output"]*usage["completion_tokens"])/1000
	return cost


def clean_sentence_filter_response(input_file):
	data = []
	with open(input_file, "r") as fp:
		for line in fp:
			response = json.loads(line)
			# read the metadata
			curr_data = response[2]
			curr_data["filter_sent"], curr_data["filter_expl"] = clean_answer(response[1]["choices"][0]["message"]["content"])
			curr_data["cost"] = compute_cost(response[0]["model"], response[1]["usage"])
			data.append(curr_data)
	return data


def write_response_to_csv(data, output_file):
	fieldnames = list(data[0].keys())
	with open(output_file, "w") as fp:
		writer = csv.DictWriter(fp, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(data)
	print("%d rows written to %s."%(len(data), output_file))


def write_response_to_jsonl(data, output_file):
	with open(output_file, "w") as fp:
		for row in data:
			fp.write(json.dumps(row, ensure_ascii=True)+"\n")
	print("%d rows written to %s."%(len(data), output_file))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input", type=str, required=True, help="JSONL file with parallel responses from OpenAI")
	parser.add_argument("-o", "--output", type=str, required=True, help="JSONL or csv output filename. When no extension is provided both formats are generated.")

	args = parser.parse_args()

	data = clean_sentence_filter_response(args.input)
	if args.output.endswith(".csv"):
		write_response_to_csv(data, args.output)
	elif args.output.endswith(".jsonl"):
		write_response_to_jsonl(data, args.output)
	else:
		write_response_to_csv(data, args.output+".csv")
		write_response_to_jsonl(data, args.output+".jsonl")