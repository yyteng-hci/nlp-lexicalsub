#!/usr/bin/env python
import sys
from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np

import gensim
import transformers

import string


def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())
    return s.split() 

def get_candidates(lemma, pos) -> list[str]:

    synonyms = []
    for i in wn.synsets(lemma, pos = pos):
        for j in i.lemmas():
            synonyms.append(j.name())

    for i in range(len(synonyms)):
        if '_' in synonyms[i]:
            synonyms[i] = synonyms[i].replace('_', ' ')

    synonyms = list(set(synonyms))

    for i in synonyms:
        if i == lemma:
            synonyms.remove(i)

    return synonyms

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context : Context) -> str:

    lemma = context.lemma
    pos = context.pos
    dict = {}

    for i in wn.synsets(lemma, pos):
        for j in i.lemmas():
            if j.name() != lemma:
                if j.name() in dict:
                    dict[j.name()] += j.count()
                else:
                    dict[j.name()] = j.count()

    result = max(dict, key = dict.get)
    if '_' in result:
        result = result.replace('_', ' ')

    return result # replace for part 2



def wn_simple_lesk_predictor(context : Context) -> str:

    lemma = context.lemma
    pos = context.pos
    stop_words = stopwords.words('english')

    final_count = 0
    final_synset = None

    overall_count = 0
    overall_synset = None


    for synset in wn.synsets(lemma, pos):
        #
        deft = tokenize(synset.definition())
        result = set(deft) - set(stop_words)
        result.update(set(synset.lemma_names()))

        #All examples for the synset.
        for x in synset.examples():
            x = tokenize(x)
            result2 = set(x) - set(stop_words)
            result.update(result2)

        #The definition and all examples for all hypernyms of the synset.
        for x2 in synset.hypernyms():
            deft2 = tokenize(x2.definition())
            result3 = set(deft2) - set(stop_words)
            result.update(result3)
            result.update(set(x2.lemma_names()))

        context_set = set(context.left_context + context.right_context)
        same_words = result & context_set

        if len(same_words) >= final_count:
            final_count = len(same_words)
            final_synset = synset

        synset_count = 0
        for i in synset.lemmas():
            synset_count += i.count()
        if synset_count >= overall_count and len(synset.lemma_names()) > 1:
            overall_count = synset_count
            overall_synset = synset

    result_return = None
    for i in final_synset.lemmas():
        if i.count() >= 0 and i.name() != lemma:
            result_return = i.name()

    overall_result = None
    for j in overall_synset.lemmas():
        if j.count() >= 0 and j.name() != lemma:
            overall_result = j.name()

    if final_count == 0 or result_return == None:
        return overall_result

    if '_' in result_return:
        result_return = result_return.replace('_', ' ')

    return result_return
   

class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self, context: Context) -> str:
        lemma = context.lemma
        pos = context.pos
        synonyms = get_candidates(lemma, pos)

        count = -2
        result = None

        for x in synonyms:
            try:
                if self.model.similarity(x, lemma) >= count:
                    count = self.model.similarity(x, lemma)
                    result = x
            except:
                pass

        return result


class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        lemma = context.lemma
        pos = context.pos
        synonyms = get_candidates(lemma, pos)
        result = None

        # MASK
        left_context_mask = ''
        for i in context.left_context:
            if i.isalpha() == True:
                left_context_mask = left_context_mask + ' ' + i
            else:
                left_context_mask += i

        context_mask = left_context_mask + ' ' + '[MASK]'

        for j in context.right_context:
            if j.isalpha() == True:
                context_mask = context_mask + ' ' + j
            else:
                context_mask += j


        input_toks = self.tokenizer.encode(context_mask)
        mask_id = self.tokenizer.convert_ids_to_tokens(input_toks).index('[MASK]')


        input_mat = np.array(input_toks).reshape((1, -1))  # get a 1 x len(input_toks) matrix
        outputs = self.model.predict(input_mat, verbose=0)
        predictions = outputs[0]

        best_words = np.argsort(predictions[0][mask_id])[::-1]  # Sort in increasing order
        best_tokens = self.tokenizer.convert_ids_to_tokens(best_words)

        for word in best_tokens:
            if word.replace('_', ' ') in synonyms:
                result = word.replace('_', ' ')
                break

        return result

    

if __name__=="__main__":

    # W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    # predictor = Word2VecSubst(W2VMODEL_FILENAME)

    predictor = BertPredictor()

    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging
        prediction = predictor.predict(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))

    # print(get_candidates('slow', 'a'))


