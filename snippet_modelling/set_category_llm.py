import yaml
import json
import argparse

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

class SetCategoryLLM(object):
    '''docstring for SetCategoryLLM'''
    def __init__(self, config):
        super(SetCategoryLLM, self).__init__()
        self.config = config
        self.model_name = self.config["llm_set_category"]["model"]
        self.model_type = "chat"
        self.model = ChatOpenAI(
                openai_api_key=self.config["llm_set_category"]["api_key"], 
                model_name=self.config["llm_set_category"]["model"],
                max_retries=int(self.config["llm_set_category"]["max_retries"]),
                max_tokens=int(self.config["llm_set_category"]["max_tokens"]),
                temperature=float(self.config["llm_set_category"]["temperature"])
            )
        

    def call_complete(self, prompt):
        print("Prompt:\n", prompt)
        response, cost = "", 0.0
        count_res = {}
        with get_openai_callback() as cb:
            response = self.model(prompt)
            cost = cb.total_cost
        response = response.strip()
        print("Response:\n", response)
        try:
            count_res = json.loads(response)
        except json.decoder.JSONDecodeError:
            print("Could not parse: \n====\n%s\n==="%response)

        print("Count response: ", count_res, "\nCost($): ", cost)
        return count_res, cost


    def call_chat(self, messages):
        assert isinstance(messages, list) and all([isinstance(m, (HumanMessage,SystemMessage)) for m in messages])
        print("Prompt:\n", messages)
        response, cost = "", 0.0
        count_res = {}
        with get_openai_callback() as cb:
            response = self.model(messages)
            cost = cb.total_cost
        if isinstance(response, AIMessage):
            response = response.content.strip()
        print("Response: \n", response)
        try:
            count_res = json.loads(response)
        except json.decoder.JSONDecodeError:
            print("Could not parse: \n====\n%s\n==="%response)

        print("Count response: ", count_res, "\nCost($): ", cost)
        return count_res, cost


    def get_set_category(self, qm, cc):
        desc = '''Given the following question and context, and an incomplete answer, identify whether the "type of entities counted" in the answer matches the entities in the question exactly, its subset, its super set, its related entities or unrelated. Return the answer in the provided JSON format. The key "category" can take the following values ["exact", "subset", "super set", "related", "unrelated"] depending on how the answer matched the question. The key "keywords" is a list of keywords from the context explaning the chosen category.'''
        ques = '''Question:  "''' + qm.question + '''"'''
        context = '''Context: "''' + cc.ann.doc.text + '''"'''
        cc_answer_type = cc.answer_type.lemma_ if cc.answer_type is not None else ''
        ansformat = '''Answer: {"count": ''' + str(cc.cardinal) + ''', "type of entities counted": "'''+cc_answer_type+'''", "category": str, "keywords": list(str)}'''
        answer = "Answer:"
        human = HumanMessage(content='\n'.join([desc, ques, context, ansformat, answer]))
        count_res, cost = self.call_chat([human])
        count_same = ('count' in count_res) and isinstance(count_res["count"], (int, float)) and (int(count_res["count"]) == int(cc.cardinal))
        type_same = ('type of entities counted' in count_res) and (count_res["type of entities counted"] == cc_answer_type)
        category = "unrelated"
        keywords = {"modifiers": []}
        if count_same:
            if 'category' in count_res and count_res["category"] in ["exact", "subset", "super set", "related", "unrelated"]:
                category = count_res["category"]
            if "keywords" in count_res:
                keywords["modifiers"] = count_res["keywords"]
            if category in ["exact", "unrelated"]:
                pass
            elif category == "subset":
                category = "subclass"
            elif category == "super set":
                category = "superclass"
            elif category == "related":
                category = "peer"
            else:
                category = "unrelated"
        else:
            category = "unrelated"
        print('Count response::::', count_res)
        print('count_same: ', count_same, "  type same: ", type_same)
        print("category: ", category, "\nkeywords: ", keywords, "\nCost: ", cost)
        return category, keywords, cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="/GW/count_information/work/class_cardinality_web/config.yml", type=str, help="YAML config file with chat model configurations")

    args = parser.parse_args()
    config = yaml.safe_load(open(args.config))
    cm = SetCategoryLLM(config)

    snippets = [
        "The number of software developers in the world is growing each day, and currently, there are roughly 4.3 million developers in the US, while Europe has over 6 million developers. Furthermore, the US Bureau of Labor Statistics has projected that by 2029, demand for developers will grow by 22%.",
        "According to a study by EDC, as of 2018, there are 23 million software developers worldwide. This population is expected to grow to 27.7 million by 2023. Image source: Evans Data Corp."
    ]
    question = "how many software developers are there in the world"
    count_ies = []
    costs = []

    for snippet in snippets:
        snippet_ann = nlp(snippet)
        count_ie, cost = None, 0.0 
        if any([ent.label_.lower() == "cardinal" for ent in snippet_ann.ents]):
            cardinals = [ent for ent in snippet_ann.ents if ent.label_.lower() == "cardinal"]
            print("Sent has cardinals:", cardinals)
            system = SystemMessage(content='Complete the JSON objects in the answer. Each JSON object has a "number" from the context. Extract the "type of entities counted" by this number and a list of "modifier" of the count (adjective, temporal, prepositional) from the context.')
            template = '''Context: {snippet}'''.format(snippet=snippet)
            # template += '''\nFormat: [{{"number": str, "type of entities counted": str, "modifier": list(str)}}]'''
            answer_template = []
            for cardinal in cardinals:
                answer_template.append('''{{"text": "{cardinal}", "type of entities counted": str, "modifier": list(str)}}'''.format(cardinal=cardinal.text))
            if len(answer_template) == 0:
                continue
            template += '''\nAnswer:['''+','.join(answer_template)+''']'''
            template += '''\nAnswer:'''
            print("Prompt message: ", template)
            human = HumanMessage(content=template)
            count_ie, cost = cm.call_chat([system, human])
        count_ies.append(count_ie)
        costs.append(cost)

    for snippet, count_ie, cost in zip(snippets, count_ies, costs):
        print("=======")
        print("Snippet: %s"%snippet)
        print("Count Extraction: %s"%json.dumps(count_ie))
        print("Cost ($): %f"%cost)
        print("=======")