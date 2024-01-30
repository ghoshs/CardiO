import sys
sys.path.append("/GW/count_information/work/class_cardinality_web")
import math
import json
import time
import copy
import argparse
import requests
from tqdm import tqdm
from inflection import singularize
from collections import namedtuple
import spacy
from spacy.tokens import Span, Token
from utils.contextspan import ContextSpan
from utils.contexttoken import ContextToken

QTuples = namedtuple('QTuples', 'answer_type relation named_entities context')


class QuestionModel(object):
	def __init__(self, nlp, question):
		self.question = question
		self.question_annotated = nlp(question)
		# self.named_entity_classes = ['person', 'norp', 'fac', 'org', 'gpe', 'loc', 'product', 'event', 'work_of_art', 'law', 'language']
		self.named_entity_classes = ['event', 'fac', 'gpe', 'language', 'law', 'loc', 'norp', 'org', 'person', 'product', 'work_of_art'] +\
		['cardinal', 'date', 'money', 'ordinal', 'percent', 'quantity', 'time']
		self.non_contextual_pos = ['adp', 'aux', 'conj', 'det', 'pron', 'punct', 'sconj']
		self.answer_type, self.relation, self.named_entities, self.context = None, None, None, None
		self.kb_entities = {}
		self.query_constraints = None


	# def question_dependency_tree(self):
	# 	dependency_tree(self.question, self.question_annotated)


	def qtuple(self):
		answer_type_text = '' if self.answer_type is None else singularize(self.answer_type.text)
		relation_text = '' if self.relation is None else self.relation.text
		named_entities_text = tuple(entity.span.text for entity in self.named_entities)
		context_text = tuple(c.span.text for c in self.context) 

		return QTuples(
			answer_type=answer_type_text,
			relation=relation_text,
			named_entities=named_entities_text,
			context=context_text
		)


	def get_constraints(self):
		atype_root_text = self.answer_type.root.text
		# if self.query_constraints is None:
		query_constraints = {"context": set()}
		# atype_modifiers = self.answer_type.text.replace(atype_root_text,"").strip()
		# atype_modifiers = self.answer_type.doc[self.answer_type.start:self.answer_type.root.i]
		# if len(atype_modifiers.text.strip()) > 0:
		# 	# query_constraints["type_modifiers"] = {atype_modifiers}
		# 	query_constraints["context"] = query_constraints["context"].union([atype_modifiers])
		# else:
		# 	query_constraints["type_modifiers"] = set()
		if self.relation is not None and len(self.relation)>0 and self.relation.lemma_ not in ["be","have"]:
			query_constraints["context"] = query_constraints["context"].union([self.relation.lemma_.strip()])
		query_constraints["context"] = query_constraints["context"].union([c.span.lemma_.strip() for c in self.context])
		query_constraints["named_entities"] = [ne.span.text for ne in self.named_entities if ne.span.label_.lower() != "date"]
		query_constraints["temporal_marker"] = [ne.span.text for ne in self.named_entities if ne.span.label_.lower() == "date"]
		query_constraints["answer_type"] = [self.answer_type.lemma_.strip()]
			# self.query_constraints = query_constraints
		# else:
		# 	# reset temporal marker
		# 	self.query_constraints["temporal_marker"] = [ne.span for ne in self.named_entities if ne.span.label_.lower() == "date"]
		return query_constraints


	def parse(self):
		print("Getting keywords....")
		self.keywords()
		# print("Linking to Wikidata...")
		# self.kb_linking()
		# print("Computing specificty...")
		# self.specificity()
		# print("Computing popularity...")
		# self.popularity()


	def keywords(self):
		## gathernig named-entities 
		self.named_entities = self.get_named_entities()

		self.answer_type = self.get_answer_type()
		# remove entity mentions which form a substring of the answer_type
		self.named_entities = [entity for entity in self.named_entities if not entity.substrOf(self.answer_type)]

		self.relation = self.get_relation()

		self.context = self.get_context()


	def get_named_entities(self):
		named_entities = set([ContextSpan(entity) for entity in self.question_annotated.ents if entity.label_.lower() in self.named_entity_classes])
	
		# ## from consecutive PROPN POS tags
		# start = end = None
		# for token in self.question_annotated:
		# 	if token.pos_.lower() == 'propn' and start is None:
		# 		start = token.i
		# 	elif token.pos_.lower() != 'propn' and start is not None:
		# 		end = token.i
		# 		new_ne = ContextSpan(self.question_annotated[start:end])
		# 		contained = False
		# 		contains = -1
		# 		# add only if not already present in named entities or not contained in named entities:
		# 		for idx, ne in enumerate(named_entities):
		# 			if ne == new_ne or ne.contains(new_ne):
		# 				contained = True
		# 				break
		# 			elif new_ne.contains(ne):
		# 				contains = idx
		# 		if not contained:
		# 			if contains >= 0:
		# 				named_entities.pop(contains)
		# 			named_entities.add(new_ne)
		# 		start = end = None
		# ## if propn is at the end of sentence
		# if start is not None:
		# 	named_entities.add(ContextSpan(self.question_annotated[start:start+1]))

		return named_entities


	def get_answer_type(self):
		answer_type = None
		
		for token in self.question_annotated:
			if token.pos_.lower() in ['noun', 'propn'] and answer_type == None and token.text not in ['number', 'count']: ## first noun is the answer type ### heuristic
				start_idx = token.i
				end_idx = token.i+1
				for child in token.children:
					if child.pos_.lower() in ['noun', 'adj', 'propn'] and child.text.lower() != 'many' and child.i < start_idx:
						start_idx = child.i

				# include contiguos propn
				if token.pos_.lower() == "propn":
					while end_idx < len(self.question_annotated):
						if self.question_annotated[end_idx].pos_.lower() != "propn":
							break
						else:
							end_idx += 1

				answer_type = self.question_annotated[start_idx: end_idx]

			elif token.pos_.lower() == 'noun' and answer_type is not None:
				# include noun tokens immediately following the answer type in the answer type
				if type(answer_type) == Token:
					start_idx = answer_type.i
					end_idx = start_idx + 1
				else:
					start_idx = answer_type.start
					end_idx = answer_type.end
				if token.i == end_idx:
					answer_type = self.question_annotated[start_idx:end_idx+1]
				
		return answer_type


	def get_relation(self):
		relation = None
		for token in self.question_annotated:
			if token.pos_.lower() == 'verb':
				if token.dep_ == 'ROOT' or relation is None: ## if the root word is a verb then it is the relation 
					relation = self.question_annotated[token.i:token.i+1]
		return relation


	def get_context(self):
		context = set()
		for np in self.question_annotated.noun_chunks:
			q_phrases = ["how many", "number of", "count of"]
			if np.root.pos_.lower() == 'noun' and (not any([x in np.text.lower() for x in q_phrases])):
				if not any([ContextSpan(np).contains(ne) for ne in self.named_entities]):
					context.add(ContextSpan(np))

		# remove answer type and relation tokens from context
		if self.answer_type is not None:
			a_type_context = ContextSpan(self.answer_type)
			context = set([c for c in context if not a_type_context.contains(c)])
		if self.relation is not None:
			context = context - set([ContextSpan(self.relation)])
		return context


	def kb_linking(self):
		atype_terms = self.decompose_span(self.answer_type)
		ne_terms = list(set([entity.span.text for entity in self.named_entities]))
		terms = list(set(atype_terms + ne_terms))
		terms = [singularize(term) for term in terms]
		for term in terms:
			self.kb_entities[term] = self.entity_disambiguate(term)
			time.sleep(1)


	def decompose_span(self, span):
		if span is None:
			return []
		root = span.root.text
		terms = [root]

		children = [child for child in span.root.children]
		children = children[:span.root.n_lefts]

		for child in children:
			if " ".join([t.text for t in child.subtree]).lower() in ["how many", "number of", "count of"]:
				continue
			term = " ".join([t.text for t in child.subtree]) + " " + root
			terms.append(term)
		terms.append(span.text)
		return list(set(terms))


	def entity_disambiguate(self, term):
		try:
			url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={term}&language=en&format=json"
			data = requests.get(url).json()
			# Return the first id
			# print(term, data)
			return data['search'][0]['id']
		except Exception as e:
			print("Disambiguation for: ", term, " ran into error: ",e)
			return ""


	def specificity(self):
		self.answer_type_specificity = {}
		# specificity without entity linking
		if self.answer_type is not None:
			terms = self.decompose_span(self.answer_type)
			kb_ss_ratio = self.kb_subclass_superclass_ratio(terms)
			print("terms: ", terms)
			print("KB ratio: ", kb_ss_ratio)
			for term in terms:
				if kb_ss_ratio[term] is None:
					self.answer_type_specificity[term] = 1.0 - (1.0/len(term.split(" ")))
				else:
					self.answer_type_specificity[term] = kb_ss_ratio[term]
		else:
			self.answer_type_specificity = {"none": 0.0}
		
		if self.named_entities is not None and len(self.named_entities) > 0:
			self.named_entities_specificity = {ne.span.text: 1.0 for ne in self.named_entities}
		else:
			self.named_entities_specificity = {"none": 0.0}

		atype_specificity = sum([self.answer_type_specificity[term] for term in self.answer_type_specificity])/len(self.answer_type_specificity)
		ne_specificity = sum([self.named_entities_specificity[term] for term in self.named_entities_specificity])/len(self.named_entities_specificity)
		self.specificity_score = (atype_specificity + ne_specificity) / 2.0


	def kb_subclass_superclass_ratio(self, terms):
		kb_ratio = {}
		wd_ids = []
		for term in terms:
			singularized_term = singularize(term)
			if singularized_term in self.kb_entities:
				wd_ids.append(self.kb_entities[singularized_term])
		kb_num_classes = self.wd_num_classes(wd_ids)
		print("kb_num_classes: ", kb_num_classes)
		for term in terms:
			singularized_term = singularize(term)
			if singularized_term in self.kb_entities:
				wd_id = self.kb_entities[singularized_term]
				if wd_id in kb_num_classes:
					kb_ratio[term] = 1.0 - min(1, (kb_num_classes[wd_id]["subclasses"]/kb_num_classes[wd_id]["superclasses"]))
				else:
					kb_ratio[term] = None
			else:
				kb_ratio[term] = None
		return kb_ratio


	def wd_num_classes(self, items):
		if len(items) == 0:
			return {}
		wd_ids = ["wd:"+item for item in items]
		wd_ids = " ".join(wd_ids)
		wd_num_classes = {}
		try:
			url = "https://query.wikidata.org/sparql"
			query = '''select ?term (count (distinct ?sup) as ?superclass) (count (distinct ?sub) as ?subclass) where 
{
  values ?term {'''+wd_ids+'''}
  {?term (wdt:P31|wdt:P279)/wdt:P279* ?sup.}
  UNION
  {?term ^wdt:P279 ?sub.}
} group by ?term'''
			data = requests.get(url, params = {'format': 'json', 'query': query}).json()
		except Exception as e:
			print("Specificity for ",wd_ids, " ran in to error: ", e)
			data = {}
		if "results" in data and "bindings" in data["results"] and len(data["results"]["bindings"]) > 0:
			for item in data["results"]["bindings"]:
				wd_id = item["term"]["value"].split("/")[-1]
				wd_num_classes[wd_id] = {
					"superclasses": int(item["superclass"]["value"]),
					"subclasses": int(item["subclass"]["value"])
				}
		return wd_num_classes


	def popularity(self):
		self.answer_type_popularity, self.named_entities_popularity = {}, {}
		if self.answer_type is not None:
			terms = self.decompose_span(self.answer_type)
			for term in terms:
				singularized_term = singularize(term)
				if singularized_term in self.kb_entities:
					statements = self.wd_statements(self.kb_entities[singularized_term])
					time.sleep(1)
				else:
					statements = 0.0
				self.answer_type_popularity[term] = 1.0 - (1.0/max(1, statements))
		else:
			self.answer_type_specificity = {"none": 0.0}
		
		if self.named_entities is not None and len(self.named_entities) > 0:
			for ne in self.named_entities:
				if ne.span.text in self.kb_entities:
					statements = self.wd_statements(self.kb_entities[ne.span.text])
					time.sleep(1)
				else:
					statements = 0.0
				self.named_entities_popularity[ne.span.text] = 1.0 - (1.0/max(1, statements))
		else:
			self.named_entities_popularity = {"none": 0.0}

		atype_popularity = sum([self.answer_type_popularity[term] for term in self.answer_type_popularity])/len(self.answer_type_popularity)
		ne_popularity = sum([self.named_entities_popularity[term] for term in self.named_entities_popularity])/len(self.named_entities_popularity)
		self.popularity_score = (atype_popularity + ne_popularity) / 2.0


	def wd_statements(self, entity):
		try:
			url = "https://query.wikidata.org/sparql"
			query = '''select (count(*) as ?cnt) where {
  values ?s {wd:'''+entity+'''}.
  ?s ?pd ?o.
  ?p wikibase:directClaim ?pd.
} '''
			data = requests.get(url, params = {'format': 'json', 'query': query}).json()
		except:
			data = {}
		if "results" in data and "bindings" in data["results"] and len(data["results"]["bindings"]) > 0:
			statements = int(data["results"]["bindings"][0]["cnt"]["value"])
		else:
			statements = 0
		return statements


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--input", "-i", required=True, type=str, help="question")
	parser.add_argument("--model", "-m", 
			type=str, 
			default="/GW/count_information/static00/.cache/en_core_web_trf/en_core_web_trf-3.6.1/", 
			help="Path to local dump of SpaCy model")

	args = parser.parse_args()

	nlp = spacy.load(args.model)
	results = []
	start = time.time()
	question_model = QuestionModel(nlp, args.input)
	question_model.parse()
	qtuple = question_model.qtuple()

	print("answer type: ", qtuple.answer_type)
	print("named entities: ", qtuple.named_entities)
	print("relation: ", qtuple.relation)
	print("context: ", qtuple.context)
	