import os
import time
import nltk
import yaml
import json
import torch
import datetime
import argparse
from collections import namedtuple
import spacy
from spacy.tokens import Span, Token
from sentence_transformers import util
from .count_extraction import count_candidates
from .set_category_llm import SetCategoryLLM
from utils.contextspan import ContextSpan
from utils.contexttoken import ContextToken

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

STuples = namedtuple('STuples', 'answer_type named_entities relation context')

EXACT_MATCH_THRESHOLD = 0.9
TYPE_MATCH_THRESHOLD = 0.5
# SNIPPET_RELEVANCE_THRESHOLD = 0.0

"""
 Compares the constraints of question to cardinals.
 Input: cardinal_constraints (dict): {"type_modifiers": set(), "context": set(), "named_entities": set(), "temporal_marker": set()}
        question_constraints (dict): {"type_modifiers": set(), "context": set(), "named_entities": set(), "temporal_marker": set()}
 Output: relation (str): takes values "equal", "subset", "superset", "congruent", "unrelated"
"""

# def cardinal_question_relation(cardinal_constraints, question_constraints, sbert, constraint_order=['named_entities', 'temporal_marker', 'context']):
#     # print("question_constraints: ", question_constraints)
#     # print("cardinal_constraints: ", cardinal_constraints)
#     relation = { }
#     question_flat, cardinal_flat = [], []

#     for o in constraint_order:
#         if o in question_constraints:
#             for constrnt in question_constraints[o]:
#                 if not isinstance(constrnt, str):
#                     question_flat.append(constrnt.text)
#                 else:
#                     question_flat.append(constrnt)
#         if o in cardinal_constraints:
#             for constrnt in cardinal_constraints[o]:
#                 if not isinstance(constrnt, str):
#                     cardinal_flat.append(constrnt.text)
#                 else:
#                     cardinal_flat.append(constrnt)
    
#     if len(question_flat) == 0 or len(cardinal_flat) == 0:
#         if len(question_flat) + len(cardinal_flat) == 0:
#             return 'equal'
#         if len(question_flat) == 0:
#             return 'subset'
#         if len(cardinal_flat) == 0:
#             return 'superset'

#     for co in constraint_order:
#         rel = 'equal'
#         question_encode = []
#         cardinal_encode = []
#         for c in question_constraints[co]:
#             if isinstance(c, str) and len(c.strip()) > 0:
#                 question_encode.append(c.strip())
#                 continue
#             if len(c.text.strip()) > 0:
#                 question_encode.append(c.lemma_.strip())
#         for c in cardinal_constraints[co]:
#             if isinstance(c, str) and len(c.strip()) > 0:
#                 cardinal_encode.append(c.strip())
#                 continue
#             if len(c.lemma_.strip()) > 0:
#                 cardinal_encode.append(c.lemma_.strip())
#         # print('#cardinal encode: %d #question encode: %d' % (len(cardinal_encode), len(question_encode)))
#         if len(cardinal_encode) + len(question_encode) == 0:
#             continue
#         if len(cardinal_encode) > 0 and len(question_encode) == 0:
#             rel = 'subset'
#         elif len(cardinal_encode) == 0 and len(question_encode) > 0:
#             rel = 'superset'
#         elif len(cardinal_encode) == 1 and len(question_encode) == 1:
#             semantic_sim = util.cos_sim(sbert.encode(question_encode), sbert.encode(cardinal_encode))[0][0].item()
#             if semantic_sim >= EXACT_MATCH_THRESHOLD:
#                 rel = 'equal'
#             elif semantic_sim >= TYPE_MATCH_THRESHOLD:
#                 rel = 'congruent'
#             elif co == 'temporal_marker':
#                 rel = 'congruent'
#             else:
#                 rel = 'unrelated'
#         else:
#             semantic_sim = util.cos_sim(sbert.encode(question_encode), sbert.encode(cardinal_encode))
#             G, bp_matching = bipartite_match(question_encode, cardinal_encode, semantic_sim)
#             # print("G edge weights: ", [G.edges[edge]['weight'] for edge in bp_matching])
#             if len(bp_matching) == len(question_encode) and len(question_encode) == len(cardinal_encode):
#                 if all([G.edges[edge]['weight'] >= EXACT_MATCH_THRESHOLD for edge in bp_matching]):
#                     rel = 'equal'
#                 elif all([G.edges[edge]['weight'] >= TYPE_MATCH_THRESHOLD for edge in bp_matching]):
#                     rel = 'congruent'
#                 elif co == 'temporal_marker':
#                     rel = 'congruent'
#                 else:
#                     rel = 'unrelated'
#             elif len(bp_matching) < len(question_encode):
#                 if all([G.edges[edge]['weight'] >= EXACT_MATCH_THRESHOLD for edge in bp_matching]):
#                     rel = 'superset'
#                 elif all([G.edges[edge]['weight'] >= TYPE_MATCH_THRESHOLD for edge in bp_matching]):
#                     rel = 'congruent'
#                 else:
#                     rel = 'unrelated'
#             else:
#                 if all([G.edges[edge]['weight'] >= EXACT_MATCH_THRESHOLD for edge in bp_matching]):
#                     rel = 'subset'
#                 elif all([G.edges[edge]['weight'] >= TYPE_MATCH_THRESHOLD for edge in bp_matching]):
#                     rel = 'congruent'
#                 else:
#                     rel = 'unrelated'
#         relation[co] = rel
#         # print('%s: relation: %s' % (co, relation[co]))
        
#     if all([x == 'equal' for x in relation.items()]):
#         return 'equal'
#     if all([x == 'congruent' for x in relation.items()]):
#         return 'congruent'
#     relation_overall = 'equal'
#     for co in constraint_order:
#         if co in relation and relation[co] == 'equal':
#             continue
#         if co in relation and relation[co] != 'equal':
#             relation_overall = relation[co]
#             return relation_overall
#     return relation_overall


class CardinalContext(object):
    ''' Each CardinalContext consists of:
    answer type(str): type of entities counted,
    cardinal(int): integer value accompanying the answer type,
    text(str): substring containing cardinal and answer type
    '''
    
    def __init__(self = None, ann = None):
        super(CardinalContext, self).__init__()
        self.ann = ann
        self.answer_type = None
        self.text = None
        self.cardinal = None
        self.math_relation = None
        self.uncertainty = None
        self.relation = None
        self.named_entities = None
        self.context = None
        self.type_category = None
        self.constraint_match = None
        self.constraints = None
        self.q_constraints = None
        self.set_category = None
        self.atype_kbentities = { }
        self.cost = 0.0
        self.named_entity_classes = [
            'event',
            'fac',
            'gpe',
            'language',
            'law',
            'loc',
            'norp',
            'org',
            'person',
            'product',
            'work_of_art',
            'date']
        self.literal_entity_classes = [
            'cardinal',
            'date',
            'money',
            'ordinal',
            'percent',
            'quantity',
            'time']
        self.atype_pos = [
            'noun',
            'adj',
            'propn']

    
    def get_answer_type(self):
        count_context_ents = [ ent for ent in self.ann.ents if ent.label_.lower() == 'cardinal']
        print('count context cardinals: ', count_context_ents)
        if len(count_context_ents) > 0:
            max_idx = max([ ent.end for ent in count_context_ents])
            print('count context ann: ', self.ann, '; start idx: ', self.ann.start, '; end idx: ', self.ann.end)
            print('max id: ', max_idx)
            if max_idx <= self.ann.end:
                self.answer_type = self.ann.doc[max_idx:self.ann.end]
            print('answer type: ', self.answer_type)

    
    def get_named_entities(self):
        sent = self.ann.sent
        ne_in_sent = [ContextSpan(entity) for entity in sent.ents if entity.label_.lower() in self.named_entity_classes ]
        return ne_in_sent

    
    def get_relation(self):
        relation = None
        ann_head = self.ann.doc[self.ann.root.i].head
        for token in self.ann.sent:
            if token.pos_.lower() == 'verb':
                # print('Relation: ', self.ann.doc[token.i:token.i + 1])
                if ann_head.is_ancestor(token) and token.lemma_ not in ('be', 'have'):
                    relation = self.ann.doc[token.i:token.i + 1]
                    # print('Relation in context: ', relation)
                    continue
                if token.is_ancestor(ann_head) and relation is None:
                    relation = self.ann.doc[token.i:token.i + 1]
                    # print('Relation in context: ', relation)
        return relation

    
    def get_context(self):
        context = set()
        for np in self.ann.sent.noun_chunks:
            if np.root.pos_.lower() == 'noun' and np.text not in self.text:
                # np_has_ne = any([ContextSpan(np).contains(ne) for ne in self.named_entities])
                np_has_ne = any(list(np.ents))
                np_has_cardinal = any([ne.label_.lower() == 'cardinal' for ne in np.ents ])
                if not (np_has_ne or np_has_cardinal):
                    context.add(ContextSpan(np))
        if self.answer_type is not None and self.answer_type.text.strip() != '':
            atypes = [ContextSpan(self.answer_type)]
            context = set([c for c in context if not any([ atype.contains(c) for atype in atypes])])
        if self.relation is not None:
            context = context - set([ContextSpan(self.relation)])
        return context


    def get_constraints(self):
        cardinal_constraints = {"context": set()}
        cardinal_constraints["answer_type"] = [cardinal_context.answer_type.lemma_.strip()]
        if cardinal_context.relation is not None and cardinal_context.relation.text.strip() != '':
            cardinal_constraints['context'] = cardinal_constraints['context'].union({cardinal_context.relation.lemma_.strip()})
        cardinal_constraints['context'] = cardinal_constraints['context'].union([ c.span.lemma_.strip() for c in cardinal_context.context])
        cardinal_constraints['named_entities'] = [ ne.span.text for ne in cardinal_context.named_entities if ne.span.label_.lower() != 'date' ]
        cardinal_constraints['temporal_marker'] = [ ne.span.text for ne in cardinal_context.named_entities if ne.span.label_.lower() == 'date' ]


class Sentence(object):
    '''Each sentence has cardinal_contexts, relation, named entities, context'''
    
    def __init__(self = None, ann = None, config = None):
        super(Sentence, self).__init__()
        self.ann = ann
        self.config = config
        self.relation = None
        self.named_entities = None
        self.context = None
        self.cardinal_contexts = []
        self.ne_kbentities = { }
        self.named_entity_classes = [
            'event',
            'fac',
            'gpe',
            'language',
            'law',
            'loc',
            'norp',
            'org',
            'person',
            'product',
            'work_of_art',
            'date']

    
    def count_context(self):
        candidates = count_candidates(self.ann, self.config)
        for candidate in candidates:
            print('Cardinal candidate: ', candidate)
            if len(candidate['text']) > 0 and candidate['cardinal'] is not None:
                c = CardinalContext(candidate['np_ann'])
                c.text = candidate['text']
                c.cardinal = candidate['cardinal']
                c.math_relation = candidate['relation']
                c.uncertainty = candidate['uncertainty']
                c.get_answer_type()
                # c.named_entities = c.get_named_entities()
                # c.relation = c.get_relation()
                # c.context = c.get_context()
                self.cardinal_contexts.append(c)    
                # print("Cardinal inserted!!")


    def get_named_entities(self):
        sent = self.ann
        ne_in_sent = [ContextSpan(entity) for entity in sent.ents if entity.label_.lower() in self.named_entity_classes ]
        return ne_in_sent


    def get_relation(self):
        relation = None
        ann_head = self.ann.doc[self.ann.root.i].head
        for token in self.ann.sent:
            if token.pos_.lower() == 'verb':
                # print('Relation: ', self.ann.doc[token.i:token.i + 1])
                if ann_head.is_ancestor(token) and token.lemma_ not in ('be', 'have'):
                    relation = self.ann.doc[token.i:token.i + 1]
                    # print('Relation in context: ', relation)
                    continue
                if token.is_ancestor(ann_head) and relation is None:
                    relation = self.ann.doc[token.i:token.i + 1]
                    # print('Relation in context: ', relation)
        return relation

    
    def get_context(self):
        context = set()
        for np in self.ann.noun_chunks:
            if np.root.pos_.lower() == 'noun':
                # np_has_ne = any([ContextSpan(np).contains(ne) for ne in self.named_entities])
                np_has_ne = any(list(np.ents))
                np_has_cardinal = any([ne.label_.lower() == 'cardinal' for ne in np.ents])
                if not (np_has_ne or np_has_cardinal):
                    context.add(ContextSpan(np))
        # # if self.answer_type is not None and self.answer_type.text.strip() != '':
        # #     atypes = [ContextSpan(self.answer_type)]
        # #     context = set([c for c in context if not any([ atype.contains(c) for atype in atypes])])
        # if self.relation is not None:
        #     context = context - set([ContextSpan(self.relation)])
        return context


    def sentence_keyphrases(self):
        self.relation = self.get_relation()
        # self.named_entities = self.get_named_entities()
        self.context = self.get_context()


class SnippetModel(object):
    
    def __init__(self, nlp, snippet, config):
        self.snippet = snippet
        self.num_cardinals = 0
        self.config = config
        self.snippet_annotated = nlp(snippet)
        self.sentences = []

    
    def to_json(self):
        sentences = []
        for sent in self.sentences:
            context = [c.span.lemma_.strip() for c in sent.context] if sent.context is not None else []
            context += [sent.relation.lemma_.strip()] if sent.relation is not None else []
            nes = [ne.span.text for ne in sent.named_entities if ne.span.label_.lower() != "date"] if sent.named_entities is not None else []
            temp = [ne.span.text for ne in sent.named_entities if ne.span.label_.lower() == "date"] if sent.named_entities is not None else []
            s = {
                'cardinal_contexts': [],
                'sentence': sent.ann.text,
                'keyphrases': {
                    "context": context,
                    "named_entities": nes,
                    "temporal_marker": temp
                }
            }
            for cc in sent.cardinal_contexts:
                temp = {
                    "text": cc.text,
                    "cardinal": cc.cardinal,
                    "math_relation": cc.math_relation,
                    "uncertainty": cc.uncertainty,
                    "answer_type": cc.answer_type.lemma_.strip() if cc.answer_type is not None else "",
                    # "type_category": cc.type_category,
                    # "constraint_match": cc.constraint_match,
                    # "set_category": cc.set_category,
                    "cost": cc.cost
                }
                s['cardinal_contexts'].append(temp)
            sentences.append(s)
        assert json.dumps(sentences)
        return sentences

    
    def parse(self):
        for sent in self.snippet_annotated.sents:
            ann_sent = Sentence(sent, self.config)
            ann_sent.count_context()
            ann_sent.sentence_keyphrases()
            # if len(ann_sent.cardinal_contexts) > 0:
            #     for cc in ann_sent.cardinal_contexts:
            #         if cc.cardinal is not None:
            #             self.num_cardinals += 1
            self.sentences.append(ann_sent)


    # def parse_w_llm(self, qm):
    #     sc_llm = SetCategoryLLM(self.config)
    #     for sent in self.snippet_annotated.sents:
    #         ann_sent = Sentence(sent, self.config)
    #         ann_sent.count_context()
    #         # if len(ann_sent.cardinal_contexts) > 0:
    #         #     for cc in ann_sent.cardinal_contexts:
    #         #         if cc.cardinal is not None:
    #         #             rank_context_llm(sc_llm, qm, cc)
    #         #             self.num_cardinals += 1
    #         #         else:
    #         #             cc.set_category = "pass"
    #         self.sentences.append(ann_sent)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', str, '/GW/count_information/static00/.cache/en_core_web_trf/en_core_web_trf-3.6.1/', 'Path to local dump of SpaCy model', **('type', 'default', 'help'))
    args = parser.parse_args()
    nlp = spacy.load(args.model)
    config = yaml.safe_load(open('/GW/count_information/work/class_cardinality_web/config.yml'))
    snippet = "As of 31 December 2022, there are 6,718 operational satellites in the Earth's orbit, of which 4,529 belong to the United States (3,996 commercial), 590 belong to China, 174 belong to Russia, and 1,425 belong to other nations."
    sm = SnippetModel(nlp, snippet, config)
    sm.parse()
    print(sm.to_json())
