import requests
import numpy as np

def get_wikipedia_embeddings(text, vector_type):
	embeddings_vec = None
	api_url = 'http://localhost:5000/embeddings'
	headers = {'Content-Type': 'application/json'}
	data = {"text": text, "type": vector_type}
	response = requests.post(api_url, json=data, headers=headers)
	if response.status_code == 200:
		embeddings_json = response.json()
		embeddings_list = embeddings_json["embeddings"]
		embeddings_vec = np.array(embeddings_list)
	return embeddings_vec


def cosine_similarity(vec1, vec2):
	if vec1 is None or vec2 is None:
		return 0
	norm_vec1 = np.linalg.norm(vec1)
	norm_vec2 = np.linalg.norm(vec2)
	cos_sim = np.dot(vec1, vec2)/(norm_vec1*norm_vec2)
	return cos_sim


def get_similarity(s1, s2):
	vec1 = get_wikipedia_embeddings(s1)
	vec2 = get_wikipedia_embeddings(s2)
	cos_sim = cosine_similarity(vec1, vec2)
	return cos_sim