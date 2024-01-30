import time
import random
import numpy as np
from collections import Counter, defaultdict
from operator import itemgetter

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
	return data, round(reduced_threshold, 2)


def get_weighted_prediction(data):
	sorted_cardinals, sorted_scores, sorted_ids, sorted_texts, sorted_sets = map(np.array, zip(*sorted(data)))
	half_score = (PERCENTILE_LEVEL/100.0) * sum(sorted_scores)
	median, conf = None, 0.0
	## in case of zero weights or single data
	if any(sorted_scores > half_score):
		median = (sorted_cardinals[sorted_scores == np.max(sorted_scores)])[0]
		conf = (sorted_scores[sorted_scores == np.max(sorted_scores)])[0]
	else:
		cumsum_scores = np.cumsum(sorted_scores)
		mid_idx = np.where(cumsum_scores <= half_score)[0][-1]
		if cumsum_scores[mid_idx] == half_score:
			median = np.mean(sorted_cardinals[mid_idx:mid_idx+2])
			conf = np.mean(sorted_scores[mid_idx:mid_idx+2])
		else:
			median = sorted_cardinals[mid_idx+1]
			conf = sorted_scores[mid_idx+1]
	return int(median), conf, list(zip(sorted_cardinals.tolist(), 
		sorted_scores.tolist(), sorted_ids.tolist(), sorted_texts.tolist(), sorted_sets.tolist()))


def get_confident_prediction(data):
	sorted_cardinals, sorted_scores, sorted_ids, sorted_texts, sorted_sets = map(np.array, zip(*sorted(data)))
	cardinal_dict = defaultdict(list)
	for c, s in zip(sorted_cardinals, sorted_scores):
		cardinal_dict[c].append(s)
	most_confident = [(k, v, sum(v)/len(v), len(v)) for k, v in cardinal_dict.items()]
	most_confident = sorted(most_confident, key=itemgetter(2,3), reverse=True)
	return most_confident[0][0], most_confident[0][2], list(zip(sorted_cardinals.tolist(),
		sorted_scores.tolist(), sorted_ids.tolist(), sorted_texts.tolist(), sorted_sets.tolist()))


def get_exact_prediction(org_data):
	# filter exact set counts
	data = [item for item in org_data if item[-1] == "exact"]
	if len(data) == 0:
		return 	None, None, data
	# return get_weighted_prediction(data)
	return get_confident_prediction(data)


def get_lower_bound(org_data):
	# filter subclass counts
	data = [item for item in org_data if item[-1] == "subclass"]
	if len(data) == 0:
		return None, None, data
	# sorted_cardinals, sorted_scores, sorted_ids, sorted_texts, sorted_sets = map(np.array, zip(*sorted(data)))
	# max_lower_bound = sorted_cardinals[-1]
	# return int(max_lower_bound), list(zip(sorted_cardinals.tolist(), 
	# 	sorted_scores.tolist(), sorted_ids.tolist(), sorted_texts.tolist(), sorted_sets.tolist()))
	# return get_weighted_prediction(data)
	return get_confident_prediction(data)


def get_upper_bound(org_data):
	# filter superclass counts
	data = [item for item in org_data if item[-1] == "superclass"]
	if len(data) == 0:
		return None, None, data
	# sorted_cardinals, sorted_scores, sorted_ids, sorted_texts, sorted_sets = map(np.array, zip(*sorted(data, key=lambda d: (d[0], -d[1]))))
	# min_upper_bound = sorted_cardinals[0]
	# return int(min_upper_bound), list(zip(sorted_cardinals.tolist(), 
	# 	sorted_scores.tolist(), sorted_ids.tolist(), sorted_texts.tolist(), sorted_sets.tolist()))
	# return get_weighted_prediction(data)
	return get_confident_prediction(data)


def get_peer_range(org_data):
	# filter peer counts
	data = [item for item in org_data if item[-1] == "peer"]
	if len(data) == 0:
		return None, None, data
	# return get_weighted_prediction(data)
	return get_confident_prediction(data)


def get_unrelated_count(org_data):
	# filter unrelated couts
	data = [item for item in org_data if item[-1] == "unrelated"]
	if len(data) == 0:
		return None, None, data
	else:
		sorted_cardinals, sorted_scores, sorted_ids, sorted_texts, sorted_sets = map(np.array, zip(*sorted(data)))
		return None, None, list(zip(sorted_cardinals.tolist(), 
		sorted_scores.tolist(), sorted_ids.tolist(), sorted_texts.tolist(), sorted_sets.tolist()))


'''
	return the final prediction based on inference from different sets
'''
def final_prediction(prediction, confidence):
	sets = ["exact", "subclass", "superclass", "peer"]
	final = [{"numeric": prediction[item], "confidence": confidence[item], "category": item} for item in sets if prediction[item] is not None]
	if len(final) > 0:
		final = sorted(final, key=itemgetter("confidence"), reverse=True)
	else:
		final = [{"numeric": None, "category": None, "confidence": None}]
	# if prediction["exact"] is not None:
	# 	final["numeric"] = prediction["exact"]
	# 	final["category"] = "exact"
	# elif prediction["subclass"] is not None:
	# 	if prediction["superclass"] is not None and prediction["superclass"] > prediction["subclass"]:
	# 		final["numeric"] = (prediction["subclass"]+prediction["superclass"])/2.0
	# 		final["category"] = "medianrange"
	# 	else:
	# 		final["numeric"] = prediction["subclass"]
	# 		final["category"] = "subclass"
	# elif prediction["superclass"] is not None:
	# 	final["numeric"] = prediction["superclass"]
	# 	final["category"] = "superclass"
	# elif prediction["peer"] is not None:
	# 	final["numeric"] = prediction["peer"]
	# 	final["category"] = "peer"
	# else:
	# 	final["numeric"] = None
	# 	final["category"] = None
	return final[0]


def apply_aggregator(contexts, threshold):
	tic = time.perf_counter()
	data, reduced_threshold = prepare_data(contexts, threshold)
	prediction, confidence, sorted_data = {}, {}, {}
	prediction["exact"], confidence["exact"], sorted_data["exact"] = get_exact_prediction(data)
	prediction["subclass"], confidence["subclass"], sorted_data["subclass"] = get_lower_bound(data)
	prediction["superclass"], confidence["superclass"], sorted_data["superclass"] = get_upper_bound(data)
	prediction["peer"], confidence["peer"], sorted_data["peer"] = get_peer_range(data)
	prediction["unrelated"], confidence["unrelated"], sorted_data["unrelated"] = get_unrelated_count(data)
	prediction["final"] = final_prediction(prediction, confidence)
	time_elapsed_aggregation = time.perf_counter() - tic
	print('Aggregation took %.4f secs'%(time_elapsed_aggregation))
	return prediction, sorted_data