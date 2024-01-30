import re
import os
import time
import math
import random
from tqdm import tqdm
from quantulum3 import parser
from .run_subprocess import run_subprocess


quant_pattern = re.compile(r"(\[[^\]]*\])\s*(\[[^\]]*\]):(\(\d+,\s\d+\))")
ntuple_pattern = re.compile(r"\[(\S*)\s+(\d+(?:\.\d+)?)([Ee][+-]?\d+)?\s+(.*[^\s]?)\s*\]")


def structured_count(text, cogcomp_result, quant_pattern, ntuple_pattern):
	empty_extraction = []

	if len(cogcomp_result) == 0:
		# print("No extraction from: "+text)
		return empty_extraction

	all_patterns = re.findall(quant_pattern, cogcomp_result) ### [1] [2]:(3)

	if len(all_patterns) == 0:		
		# print("No patterns fromm: "+text)
		return empty_extraction
	
	# for pattern in all_patterns:
	all_ntuples = [re.findall(ntuple_pattern, pattern[1]) for pattern in all_patterns] ### [1 2 3 4] 
	all_ntuples = [ntuple[0] if len(ntuple)>0 else () for idx, ntuple in enumerate(all_ntuples)]
	spans = [pattern[2] for pattern in all_patterns]
	if len(all_ntuples) == 0:
		# print("No triples from: "+text+" : "+cogcomp_result)
		return empty_extraction
	else:
		extraction = list(zip(all_ntuples, spans))
		# print("Extracted: "+ str(extraction))
		return extraction


# return list of tuples
def get_cogcomp_ntuples(snippet, config):
	cc_path = config["count"]["cogcomp"]["path"]
	cc_quantifier = config["count"]["cogcomp"]["quantifier"]
	normalize = config["count"]["cogcomp"]["normalize"]
	tmp_dir = config["default"]["tmp_dir"]

	empty_extraction = []
	tempfile = tmp_dir + "tmp" + "_" + str(random.random())[2:] + ".txt"
	with open(tempfile, "w", encoding="utf-8") as fp:
		fp.write(snippet)
	process_args = ["java", "-cp", cc_path, cc_quantifier, tempfile, normalize]
	cogcomp_result = run_subprocess(process_args)
	os.remove(tempfile)
	structured_result = structured_count(snippet, cogcomp_result, quant_pattern, ntuple_pattern)
	if len(structured_result) == 0:
		return empty_extraction
	else:
		return structured_result


def get_quantulum_ntuples(text):
	try:
		counts_in_snippet = parser.parse(text)
	except Exception as e:
		print("Quantulum could not parse: ", text)
		counts_in_snippet = []
	return counts_in_snippet


def get_count_spans(text, config):
	# counts_in_snippet = get_cogcomp_ntuples(text, config)
	# count_spans = []
	# for ntuple, span in counts_in_snippet:
	# 	if len(ntuple) == 0:
	# 		continue
	# 	relation, quantity, exponent, metric = ntuple
	# 	quantity += exponent
	# 	indices = [index.strip() for index in span[1:-1].split(",")]
	# 	count_spans.append({"relation": relation, "quantity": str(float(quantity)), "start": indices[0], "end": indices[1]})

	counts_in_snippet = get_quantulum_ntuples(text)
	count_spans = []
	for span in counts_in_snippet:
		if span.unit.entity.name not in ["dimensionless", "unknown"]:
			continue
		qty = span.value
		if qty < 0:
			print("cardinality is negative in %s.. Ignoring"%span)
			continue
		elif qty == 0:
			qty = 0
		elif math.modf(qty)[0] > 0 and span.uncertainty is None:
			print("Fraction in %s.. Ignoring"%span)
			continue
		else:
			qty = int(qty) 
		if span.uncertainty is not None:
			count_spans.append({
				"relation": "<>", 
				"quantity": str(qty), 
				"start": span.span[0], 
				"end": span.span[1], 
				"uncertainty": span.uncertainty})
		else:
			approx = any([x in text.lower() for x in ["apprx", "approx", "around", "almost", "about", "nearly", "close to", "roughly"]])
			lower = any([x in text.lower() for x in ["more than", "at least", "exceeds", "over"]])
			upper = any([x in text.lower() for x in ["less than", "at most"]])
			print(text, span.__dict__, span.surface, approx, lower, upper)
			if approx:
				rel = "~"
			elif lower:
				rel = ">"
			elif upper:
				rel = "<"
			else:
				rel = "="
			count_spans.append({
				"relation": rel, 
				"quantity": str(qty), 
				"start": span.span[0], 
				"end": span.span[1], 
				"uncertainty": 0})

	return count_spans


def smallest_cardinality_phrase(nc):
	if any([t.ent_type_.lower() == 'cardinal' for t in nc]):
		return nc
	start = nc.start
	end = None
	for c in nc.subtree:
		if c.ent_type_.lower() == "cardinal":
			end = c.right_edge.i+1
			break
	if end is not None:
		return nc.doc[start:end]
	else:
		return None


def smallest_nc_count(ann, config):
	count_candidates = []

	# remove overlapping subtree noun chunks, where orig nc has no cardinal
	cardinal_ncs = [nc for nc in ann.noun_chunks if any([t.ent_type_.lower() == 'cardinal' for t in nc.root.subtree])]
	print("cardinal ncs: ", cardinal_ncs)
	
	descendants = [nc.root for nc in cardinal_ncs]
	print("descendants: ", descendants)
	
	filtered_cardinal_ncs = []
	for i, nc in enumerate(cardinal_ncs):
		nc_has_ancestor = []
		for j, d in enumerate(descendants):
			if i!=j:
				nc_has_ancestor.append(nc.root.is_ancestor(d))
		# keep nc if nc itself has a cardinal
		# TODO: discard nc if cardinal in nc is from any descendants 
		if (any([t.ent_type_.lower()=="cardinal" for t in nc])) or not any(nc_has_ancestor):
			filtered_cardinal_ncs.append(nc)
	print("ncs: ", filtered_cardinal_ncs)
	
	for nc in filtered_cardinal_ncs:
		np = smallest_cardinality_phrase(nc)
		if np is None:
			continue
		count_span = get_count_spans(np.text, config)
		if len(count_span) > 0:
			cardinal = float(count_span[0]["quantity"])
			relation = count_span[0]["relation"]
			uncertainty = float(count_span[0]["uncertainty"])
		else:
			cardinal = None
			relation = None
			uncertainty = None
		count_candidate = {
			"text": np.text, 
			"cardinal": cardinal, 
			"relation": relation, 
			"uncertainty": uncertainty, 
			"np_ann": np}
		if count_candidate not in count_candidates:
			count_candidates.append(count_candidate)
	print("candidates: ", count_candidates)
	return count_candidates


def count_candidates(ann, config):
	"""
		Returns the following variables
		prediction -> int
		sorted_data -> list(tuple(cardinal, score, id, text, context_class))
		annotated_contexts -> list(dict(
							rank,
							url,
							name,
							context,
							dateLastCrawled,
							cardinal,
							count_span: dict(selected, text, score, context_class)))
	"""
	tic = time.perf_counter()
	
	count_candidates = smallest_nc_count(ann, config)

	time_elapsed_extraction = time.perf_counter() - tic

	print('Extraction took %.4f secs'%(time_elapsed_extraction))
	return count_candidates


if __name__ == '__main__':
	print(get_count_spans("almost 100,000 lakes", None))