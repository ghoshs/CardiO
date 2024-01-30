import yaml
import json
import time
import argparse

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

class SentenceFilter(object):
    '''docstring for SentenceFilter'''
    def __init__(self, config):
        super(SentenceFilter, self).__init__()
        self.config = config
        self.model_name = self.config["sentence_filter"]["model"]
        self.model_type = "chat"
        self.model = ChatOpenAI(
            openai_api_key=self.config["sentence_filter"]["api_key"], 
            model_name=self.config["sentence_filter"]["model"],
            max_retries=int(self.config["sentence_filter"]["max_retries"]),
            max_tokens=int(self.config["sentence_filter"]["max_tokens"]),
            temperature=float(self.config["sentence_filter"]["temperature"]),
            request_timeout=self.config["sentence_filter"]["request_timeout"]
        )


    def call_chat(self, messages):
        assert isinstance(messages, list) and all([isinstance(m, (HumanMessage,SystemMessage)) for m in messages])
        print("Prompt:\n", messages)
        response, cost = "", 0.0
        has_count_res = {}
        with get_openai_callback() as cb:
            response = self.model(messages)
            cost = cb.total_cost
        if isinstance(response, AIMessage):
            response = response.content.strip()
        print("Response: \n", response)
        try:
            has_count_res = json.loads(response)
        except json.decoder.JSONDecodeError:
            print("Could not parse: \n====\n%s\n==="%response)

        print("Count response: ", has_count_res, "\nCost($): ", cost)
        return has_count_res, cost


    def call_batch(self, batch_messages):
        assert isinstance(batch_messages, list) and all([isinstance(m, (HumanMessage,SystemMessage)) for batch in batch_messages for m in batch])

        llm_generations, total_cost = [], 0.0
        responses, avg_cost = [], []

        with get_openai_callback() as cb:
            llm_generations = self.model.generate(batch_messages)
            total_cost = cb.total_cost

        for gen in llm_generations.generations:
            response = gen[0].message.content.strip()
            print("Response: ", response)
            try:
                response = json.loads(response)
                responses.append(response)
                print("Extracted: ", response)
            except json.decoder.JSONDecodeError:
                print("Could not parse: \n====\n%s\n==="%response)
                responses.append({})
        avg_cost = [total_cost/len(llm_generations.generations)]*len(llm_generations.generations)
        return responses, avg_cost



    def filter(self, question, snippet_name, sentences):
        desc = '''Given a question, a sentence from a text and the title of the text, your job is to determine if the sentence contains the answer related to the question. Answer in the provided format, where "short" takes only yes or no and "long" takes a one-line explanation.'''
        ques = '''Question:  ''' + question 
        title = '''Title: ''' + snippet_name 
        aformat = '''Answer: {"short": str, "long": str}''' 
        answer = "Answer:"
        system = SystemMessage(content=desc)

        batch_messages = []
        filter_sentences = []
        for idx, sent in enumerate(sentences):
            context = '''Sentence: ''' + sent
            human = HumanMessage(content='\n'.join([ques, context, title, aformat, answer]))
            batch_messages.append([system, human])
            has_count_res, cost = self.call_chat([system, human])
        # responses, total_cost = self.call_batch(batch_messages)
        # for has_count_res, cost in zip(responses, total_cost):
            filter_sent = False
            expl = None
            if "short" in has_count_res and has_count_res["short"] == "no":
                filter_sent = True
            if "long" in has_count_res and len(has_count_res["long"]) > 0:
                expl = has_count_res["long"]
            filter_sentences.append((filter_sent, cost, expl))
            if idx < len(sentences) - 1:
                time.sleep(1)
        print("Num filter_sentences: ", len(filter_sentences), flush=True)
        return filter_sentences