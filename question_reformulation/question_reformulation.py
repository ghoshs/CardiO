import sys
sys.path.append("/GW/count_information/work/class_cardinality_web")
import yaml
import json
import argparse

from utils.load import load_config

from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)


class ChatModel(object):
	"""docstring for ChatModel"""
	def __init__(self, config):
		super(ChatModel, self).__init__()
		self.config = config
		self.chat_model = ChatOpenAI(
				openai_api_key=self.config["chat_model"]["api_key"], 
				model_name=self.config["chat_model"]["model"],
				max_retries=int(self.config["chat_model"]["max_retries"]),
				max_tokens=int(self.config["chat_model"]["max_tokens"]),
				temperature=float(self.config["chat_model"]["temperature"])
			)


	def call_complete(self, prompt):
		response, cost = None, None
		with get_openai_callback() as cb:
			response = self.chat_model.predict(prompt)
			cost = cb.total_cost
		return response, cost
		

	def call_batch_chat(self, batch_messages):
		reform_q = []
		cost = 0.0
		if len(batch_messages) == 0:
			return reform_q, cost
		with get_openai_callback() as cb:
			response = self.chat_model.generate(batch_messages)
			cost = cb.total_cost
		print("# Batch messages: ", len(batch_messages))
		print("Cost($)=== ", cost)

		for chats in response.generations:
			turn = []
			for chat in chats:
				if chat.message.content is not None and len(chat.message.content) > 0:
					for line in chat.message.content.split('\n'):
						q = line.strip()
						if not q.lower().startswith("how many"):
							if "how many" in q.lower():
								idx = q.lower().index("how many")
								q = q[idx:]
						turn.append(q)
					# turn += [q.strip() for q in chat.message.content.split('\n')]
			reform_q.append(turn)

		print("Num reforms: ", sum([len(rf) for rf in reform_q]))
		# return question
		return reform_q, cost


class QuestionReformulation(object):
	"""Takes a count question and its components and returns reformulations"""
	def __init__(self, question, config, answer_type, named_entities, relation, context):
		super(QuestionReformulation, self).__init__()
		self.question = question
		self.answer_type = answer_type
		self.named_entities = named_entities
		self.relation = relation
		self.context = context
		self.chat_model = ChatModel(config)
		self.reform_q = {
			"remove":[], 
			"subclasses": [], 
			"superclasses": [], 
			"synonyms": [], 
			"peers_by_size": [], 
			"peers_by_answer_type": []
		}
		self.reform_cost = 0.0


	def reformulate(self):
		batch_messages = []
		batch_order = []
		# # relaxed context:
		# for word in self.context:
		# 	batch_messages.append(self.create_message(word, "remove"))
		# 	batch_order.append("remove")

		nes = [ne for ne in self.named_entities if ne not in self.answer_type]
		if len(nes) >= 1:
			for ne in self.named_entities:
				# batch_messages.append(self.create_message(ne, "remove"))
				# batch_order.append("remove")
				batch_messages.append(self.create_message(ne, "replace", self.answer_type))
				batch_order.append("peers_by_answer_type")
		atype_nes = [ne for ne in self.named_entities if ne in self.answer_type]
		if len(atype_nes) > 0:
			batch_messages.append(self.create_message(self.answer_type, "replace"))
			batch_order.append("peers_by_size")
		# elif len(self.answer_type) > 0:
		# 		batch_messages.append(self.create_message(self.answer_type, "replace"))
		# 		batch_order.append("peers_by_size")
		
		# if len(self.answer_type) > 0:
		# 	batch_messages.append(self.create_message(self.answer_type, "replace", "superclasses"))
		# 	batch_order.append("superclasses")
		# 	batch_messages.append(self.create_message(self.answer_type, "replace", "subclasses"))
		# 	batch_order.append("subclasses")
		# 	batch_messages.append(self.create_message(self.answer_type, "replace", "synonyms"))
		# 	batch_order.append("synonyms")

		reformulations, self.reform_cost = self.chat_model.call_batch_chat(batch_messages)
		for rq, bo in zip(reformulations, batch_order):
			print("Reformulation by %s: %s"%(bo, rq))
			rq = [q for q in rq if (q != self.question) and (bo not in q) and (len(q) > 0)]
			print("Dedup Reformulation by %s: %s"%(bo, rq))
			self.reform_q[bo] += rq


	def create_message(self, term, mod, replace_with=None):
		if mod == "replace" and replace_with is None:
			peering_group = "size."
		elif mod == "replace":
			peering_group = "number of " + replace_with +"."
		else:
			peering_group = ""
		task = '''Reformulate the following question by replacing "'''+ term + '''"'''\
			+ ''' with entities comparable in terms of ''' + peering_group \
			+ ''' Return each question on a separate line. Start each question with "how many".'''
		system = SystemMessage(content=task)

		# if mod == "remove":
		# 	task = """Reformulate by removing the constraint "%s"."""%(term)
		# elif mod == "replace":
		# 	if replace_with == "peers":
		# 		task = """Reformulate by replacing "%s" with other related entities."""%(term)
		# 	else:
		# 		task = """Reformulate by replacing "%s" with its "%s"."""%(term, replace_with)
		human = HumanMessage(content="Question: "+ self.question + "\nReformulations:")
		return [system, human]
		

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", "-c", default="/GW/count_information/work/class_cardinality_web/config.yml", type=str, help="YAML config file with chat model configurations")
	parser.add_argument("--question", "-q", required=True, type=str)
	parser.add_argument("--atype", "-at", default="", type=str)
	parser.add_argument("--named_entities", "-ne", default=[], type=str, nargs="*")
	parser.add_argument("--relation", "-r", default="", type=str)
	parser.add_argument("--context","-ct", default=[], type=str, nargs="*")

	args = parser.parse_args()
	config = yaml.safe_load(open(args.config))
	qr = QuestionReformulation(args.question, config, args.atype, args.named_entities, args.relation, args.context)
	qr.reformulate()
	print("Reformulations: ", qr.reform_q)
	print("Cost: ", qr.reform_cost)
	