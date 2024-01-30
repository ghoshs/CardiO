import re
import csv
import ast
import json
import argparse
from utils.cache import CountsCache

parse_count = CountsCache.parse_count

def flotify(mobj):
	num = mobj.group(1)
	print("num:", num)
	comma_idx = num.rfind(",")
	dot_idx = num.rfind(".")
	if comma_idx > -1 and dot_idx > -1:
		if comma_idx < dot_idx:
			num = num.replace(",","")
		else:
			num = num.replace(".","")
			num = num.replace(",",".")
	elif comma_idx > -1:
		if len(num)-comma_idx-1 == 2:
			num = num[:comma_idx] + "." + num[comma_idx+1:]
		num = num.replace(",","")
	elif dot_idx > -1:
		if len(num)-dot_idx-1 > 2:
			num = num.replace(".","")
	try:
		num = ": "+str(float(num)) + ","
	except ValueError:
		num = ': "",'
	return num


def clean_answer(ans):
	clean = []
	try:
		clean = ast.literal_eval(ans)
		if isinstance(clean, dict):
			clean = [clean]
	except ValueError:
		new_ans = ans.replace("null", "None")
		if new_ans != ans:
			clean = clean_answer(new_ans)
		else:
			print("ValueError: Could not parse answer:", ans)
			clean = []
	except SyntaxError:
		new_ans = re.sub(r':\s*(\d+(?:[,\.]\d+)*),', flotify, ans)
		if new_ans != ans:
			clean = clean_answer(new_ans)
		else:
			print("SyntaxError: Could not parse answer:", ans)
			clean = []
	except Exception as e:
		print(e, "Could not parse answer:", ans)
	else:
		print("clean parse of answer:", ans)

	
	try:
		json.dumps(clean)
	except TypeError:
		for qty in clean:
			if "span" in qty and (not isinstance(qty["span"], str)):
				qty["span"] = str(qty["span"])
			if "type" in qty and (not isinstance(qty["type"], str)):
				qty["type"] = str(qty["type"])
			if "modifier" in qty and (not isinstance(qty["modifier"], str)):
				qty["modifier"] = str(qty["modifier"])
			if "quantity" in qty:
				c, unc = parse_count(qty["quantity"])
				if c is None:
					qty["quantity"] = None
				elif unc == 0.0:
					qty["quantity"] = c
				else:
					qty["quantity"] = [c-unc, c+unc]
	return clean


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


def clean_count_response(input_file):
	data = []
	with open(input_file, "r") as fp:
		for line in fp:
			response = json.loads(line)
			# read the metadata
			curr_data = response[2]
			curr_data["counts"] = clean_answer(response[1]["choices"][0]["message"]["content"])
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
			try:
				fp.write(json.dumps(row, ensure_ascii=True)+"\n")
			except Exception as e:
				print(e)
				print("row: ", row)
				break
	print("%d rows written to %s."%(len(data), output_file))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input", type=str, required=True, help="JSONL file with parallel responses from OpenAI")
	parser.add_argument("-o", "--output", type=str, required=True, help="JSONL or csv output filename. When no extension is provided both formats are generated.")

	args = parser.parse_args()

	data = clean_count_response(args.input)
	if args.output.endswith(".csv"):
		write_response_to_csv(data, args.output)
	elif args.output.endswith(".jsonl"):
		write_response_to_jsonl(data, args.output)
	else:
		write_response_to_csv(data, args.output+".csv")
		write_response_to_jsonl(data, args.output+".jsonl")