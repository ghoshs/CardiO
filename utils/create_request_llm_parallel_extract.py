import json
import time
import argparse
import spacy
from tqdm import tqdm
from utils.load import load_config, load_query_data
from utils.cache import SnippetsCache


class RequestCreator(object):
    def __init__(self, args):
        super(RequestCreator, self).__init__()
        self.input = args.input
        self.config = load_config(args.config)
        self.output_file = args.output
        self.num_jobs = 0
        self.nlp = spacy.load(self.config["spacy"]["model"])
        if args.reformulations is not None:
            data = json.load(open(args.reformulations, "r"))
            self.reform_questions = [rq for oq in data for rq in data[oq]]
            self.all_questions = [oq for oq in data] + self.reform_questions
            print("Loaded %d reform questions!!"%len(self.reform_questions))
        else:
            self.reform_questions = None
        self.complete_questions = []
        self.incomplete = False
        if args.incomplete and len(args.complete_questions) > 0:
            self.incomplete = True
            for filename in args.complete_questions:
                with open(filename, "r") as fp:
                    for line in fp:
                        data = json.loads(line)
                        if data["metadata"]["question"] not in self.complete_questions:
                            self.complete_questions.append(data["metadata"]["question"])
            print("%d questions completed.. "%len(self.complete_questions))


    @staticmethod
    def count_extraction_messages(sent):
        system = '''Extract all cardinals and the types being counted in the provided snippet. Return a list of JSON objects in the provided format. Each value is explained below.
SPAN: snippet substring that containing the cardinal. 
TYPE: a string that represents the nouns being counted.
QUANTITY: the integer value in SPAN. If SPAN is a range, return a list like this: [lower bound, upper bound].
LIMIT: a value from ["upper", "lower", "exact", "approximate", "range"] denoting the uncertainty of the in the SPAN.
Return an empty list if no cardinals are found.
'''
        answer_template = '''Format: {"span": SPAN, "type": TYPE, "quantity": QUANTITY, "limit": LIMIT}'''
        answer_start = '''Answer:'''
        template = '''Snippet: {snippet}'''.format(snippet=sent)
        user = '\n'.join([template, answer_template, answer_start])
        message = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return message


    @staticmethod
    def count_ie_mesages(sent, name, context):
        system = '''Extract all cardinals and the types being counted in the sentence belonging to a main context. Return a list of JSON objects in the provided format. Each value is explained below.
SPAN: snippet substring that containing the cardinal. 
TYPE: a string that represents the nouns being counted.
QUANTITY: the integer value in SPAN. If SPAN is a range, return a list like this: [lower bound, upper bound].
MODIFIER: a value from ["more than", "less than", "exact", "approximate", "range"] closest to the quantitative modifier in the SPAN.
Return an empty list if no cardinals are found.

Format: {"span": SPAN, "type": TYPE, "quantity": QUANTITY, "modifier": MODIFIER}'''
        template = '''Context: {name} | {context}

Sentence: {sent}

Answer:'''.format(name=name, context=context, sent=sent)
        message = [{"role": "system", "content": system}, {"role": "user", "content": template}]
        return message


    @staticmethod
    def count_sent_filter_messages(question, name, sent):
        system = '''Given a question, a sentence from a text and the title of the text, your job is to determine if the sentence contains the answer related to the question. Answer in the provided format, where "short" takes only yes or no and "long" takes a one-line explanation.'''
        template = '''Question: {question}

Sentence: {sent}

Title: {title}'''.format(question=question, sent=sent, title=name)
        user = template + '''\n\nFormat: {"short": str, "long": str}

Answer:'''
        message = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return message


    def create_jobs(self, request_type):
        cache = SnippetsCache(self.config, self.input)
        cache.load_contexts()
        print("Job creation started ===>", flush=True)
        for q, contexts in tqdm(cache.contexts.items()):
            if self.reform_questions is not None and q not in self.reform_questions:
                continue
            if self.incomplete and q in self.complete_questions:
                continue
            pipe_contexts = [context["context"] for context in contexts]
            anns = list(self.nlp.pipe(pipe_contexts))
            for ann, context in zip(anns, contexts):
                for sent_id, sent in enumerate(ann.sents):
                    if not any([ent.label_.lower() == 'cardinal' for ent in sent.ents]):
                        continue
                    if request_type == "countie_sent":
                        messages = self.count_extraction_messages(sent.text)
                        model_config = "count_extraction"
                    elif request_type == "countie_sent_ctxt":
                        messages = self.count_ie_mesages(sent.text, context["name"], context["context"])
                        model_config = "count_extraction"
                    elif request_type == "count_sent_filter":
                        messages = self.count_sent_filter_messages(q, context["name"], sent.text)
                        model_config = "sentence_filter"
                    else:
                        continue

                    job = {
                        "model": self.config[model_config]["model"],
                        "messages": messages,
                        "temperature": self.config[model_config]["temperature"],
                        "metadata": {
                            "question": q,
                            "rank": context["rank"],
                            "sent_id": sent_id,
                            "sent": sent.text}
                    }
                    self.num_jobs += 1
                    with open(self.output_file, "a") as f:
                        json_string = json.dumps(job, ensure_ascii=False)
                        f.write(json_string + "\n")


    def write_jobs(self, request_type):
        tic = time.perf_counter()
        if self.num_jobs == 0:
            with open(self.output_file, "w") as f:
                pass
            self.create_jobs(request_type)
        print("%d jobs written to file: %s"%(self.num_jobs, self.output_file))
        elapsed_time = time.perf_counter() - tic
        print('Total time lapsed %.4f secs'%(elapsed_time), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="snippets cache file name or path to benchmark with contexts")
    parser.add_argument("-rf", "--reformulations", default=None, type=str, help="reformulation queries")
    parser.add_argument("-c", "--config", type=str, required=True, help="config of cache path and model")
    parser.add_argument("-o", "--output", type=str, required=True, help="output file for OpenAI api requests")
    parser.add_argument("-t", "--request_type", type=str, choices=["countie_sent", "countie_sent_ctxt", "count_sent_filter"], required=True, help="type of job to create")
    parser.add_argument('-ic', '--incomplete', action="store_const", const=True, help="resume missed questions")
    parser.add_argument('-cq', '--complete_questions', nargs="*", type=str, default=None, help="file(s) to read completed requests. required if incomplete is True")

    args = parser.parse_args()
    rc = RequestCreator(args)
    rc.write_jobs(args.request_type)