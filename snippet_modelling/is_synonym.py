import nltk
from nltk.corpus import wordnet

def is_synonym(term1, term2):
    synsets1 = wordnet.synsets(term1)
    synsets2 = wordnet.synsets(term2)

    for synset1 in synsets1:
        for synset2 in synsets2:
            if synset1 == synset2:
                return f"{term1} and {term2} are synonyms."
    return None