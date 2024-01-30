'''
	return the final prediction based on inference from different sets
'''

def final_prediction(prediction):
	final = {}
	if prediction["exact"] is not None:
		final["numeric"] = prediction["exact"]
		final["category"] = "exact"
	elif prediction["subclass"] is not None:
		if prediction["superclass"] is not None and prediction["superclass"] > prediction["subclass"]:
			final["numeric"] = (prediction["subclass"]+prediction["superclass"])/2.0
			final["category"] = "medianrange"
		else:
			final["numeric"] = prediction["subclass"]
			final["category"] = "subclass"
	elif prediction["superclass"] is not None:
		final["numeric"] = prediction["superclass"]
		final["category"] = "superclass"
	elif prediction["peer"] is not None:
		final["numeric"] = prediction["peer"]
		final["category"] = "peer"
	else:
		final["numeric"] = None
		final["category"] = None
	return final