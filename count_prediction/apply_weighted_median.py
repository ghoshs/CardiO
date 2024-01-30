import time
import random
import numpy as np
from collections import Counter

PERCENTILE_LEVEL = 50

def prepare_data(contexts, threshold):
	cardinals, scores, ids, texts, sets = [], [], [], [], []
	reduced_threshold = threshold
	MINIMUM_CARDINALS = 5
	# while len(cardinals) < MINIMUM_CARDINALS and reduced_threshold >= 0.0:
	for context in contexts:
		if "stuples" in context:
			for stuple in context["stuples"]:
				for cc in stuple["cardinal_contexts"]:
					if cc["cardinal"] is not None:
						cardinals.append(cc["cardinal"])
						scores.append(context["relevance"])
						ids.append(context["rank"])
						texts.append(cc["text"])
						sets.append(cc["set_category"])
	reduced_threshold = max(0, reduced_threshold)

	data = list(zip(np.array(cardinals), np.array(scores), np.array(ids), np.array(texts, dtype=object), np.array(sets)))
	return data, contexts, round(reduced_threshold, 2)


def get_weighted_prediction(data):
	if len(data) == 0:
		return 	None, data

	# cardinals, scores = np.array(cardinals), np.array(scores)
	sorted_cardinals, sorted_scores, sorted_ids, sorted_texts, sorted_sets = map(np.array, zip(*sorted(data)))
	half_score = (PERCENTILE_LEVEL/100.0) * sum(sorted_scores)
	## in case of zero weights or single data
	if any(sorted_scores > half_score):
		median = (sorted_cardinals[sorted_scores == np.max(sorted_scores)])[0]
	else:
		cumsum_scores = np.cumsum(sorted_scores)
		mid_idx = np.where(cumsum_scores <= half_score)[0][-1]
		if cumsum_scores[mid_idx] == half_score:
			median = np.mean(sorted_cardinals[mid_idx:mid_idx+2])
		else:
			median = sorted_cardinals[mid_idx+1]
	return int(median), list(zip(sorted_cardinals.tolist(), 
		sorted_scores.tolist(), sorted_ids.tolist(), sorted_texts.tolist(), sorted_sets.tolist()))


def apply_aggregator(contexts, threshold):
	tic = time.perf_counter()
	data, annotated_contexts, reduced_threshold = prepare_data(contexts, threshold)
	prediction, sorted_data = {}, {}
	prediction["exact"], sorted_data["exact"] = get_weighted_prediction(data)
	time_elapsed_aggregation = time.perf_counter() - tic
	print('Aggregation took %.4f secs'%(time_elapsed_aggregation))
	return prediction, sorted_data, annotated_contexts