from sentence_transformers import util

def snippet_relevance(question, contexts, sbert):
	q_embeddings = sbert.encode(question, convert_to_tensor=True)
	c_embeddings = sbert.encode(contexts, convert_to_tensor=True)
	relevance = util.cos_sim(q_embeddings, c_embeddings)
	return relevance[0].tolist()