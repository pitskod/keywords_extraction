#!/usr/bin/env python
# coding: utf-8
import os
import re
from nltk.tokenize import RegexpTokenizer, sent_tokenize
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
import rake_nltk
import multi_rake
import pke
from models import rake
from models import my_rake
#####
from pke import base
base.ISO_to_language['de'] = 'german'
#####

STOPWORDS_DIR = "stopwords"

"""
def get_stopwords(language, dir=STOPWORDS_DIR):
    with open(os.path.join(dir, language)) as f:
        return list(f.read().split('\n'))

stopwords_list = get_stopwords('german')"""

## perprocessing
stopwords_list = stopwords.words('german')


def remove_tags(text):
    tags_re = re.compile(r'<.*?>')
    return tags_re.sub(' ', text)


def clear_data(text):
    text_removed_tags = remove_tags(text)
    sentences = sent_tokenize(text_removed_tags)
    parsed_sentences = []
    for sentence in sentences:
        sentence_lower = sentence.lower()
        tokens = RegexpTokenizer(r'\w+-\w+|\w+').tokenize(sentence_lower)
        tokens = [token for token in tokens if token not in stopwords_list]
        parsed_sentences.append(tokens)
    return parsed_sentences


# ## algorithm RAKE 0 (modified RAKE 1 by me)
def get_modified_rake_keywords(text, n_grams_length, metric=my_rake.Metric.WORD_FREQUENCY):
    rake = my_rake.Rake(language='german', max_length=n_grams_length, punctuations=[], stopwords=[],
                        ranking_metric=metric)
    sentences = clear_data(text)
    rake.extract_keywords_from_sentences(sentences)
    #if not keywords_number:
    #    keywords_number = len(sentences)//3
    return rake.get_ranked_phrases()


# ## algorithm RAKE 1
def get_rake_keywords(text, max_length=100000, metric=None):
    rake = rake_nltk.Rake(language='german', max_length=max_length, stopwords=stopwords_list,
                          ranking_metric=rake_nltk.Metric.DEGREE_TO_FREQUENCY_RATIO)
    rake.extract_keywords_from_text(text)
    #if not keywords_number:
    #    keywords_number = len(sentences)//3
    return rake.get_ranked_phrases()


# ## algorithm RAKE 2
def get_multi_rake_keywords(text, max_words=1):
    m_rake = multi_rake.Rake(min_chars=3, max_words=max_words, min_freq=1, language_code='de', stopwords=stopwords_list)
    keywords = m_rake.apply(text)
    return [k for k, _ in keywords]


# ## algorithm RAKE 3
def get_rake3_keywords(text, max_words=1):
    rake_obj = rake.Rake(os.path.join(STOPWORDS_DIR, 'german'), min_char_length=3, max_words_length=max_words)
    keywords = rake_obj.run(text)
    return [k for k, _ in keywords]


def get_all_rake_keywords(text, keywords_phrase_length):
    # ## RAKE 0
    #res1_0 = get_modified_rake_keywords(text, n_grams_length=keywords_phrase_length, metric=my_rake.Metric.WORD_DEGREE)
    res1_1 = get_modified_rake_keywords(text, n_grams_length=keywords_phrase_length,
                                        metric=my_rake.Metric.WORD_FREQUENCY)
    # ## RAKE 1
    #res2_0 = get_rake_keywords(text, metric=rake_nltk.Metric.WORD_FREQUENCY)
    res2_1 = get_rake_keywords(text)
    res2_2 = get_rake_keywords(text, keywords_phrase_length)
    #res2_3 = get_rake_keywords(text, keywords_phrase_length, metric=rake_nltk.Metric.WORD_DEGREE)

    # ## RAKE 2
    res3 = get_multi_rake_keywords(text, keywords_phrase_length)

    # ## RAKE 3
    res4 = get_rake3_keywords(text, keywords_phrase_length)
    return [res1_1, res2_1, res2_2, res3, res4]


# ## algorithm TF-IDF from pke - python keyphrase extraction
# ## algorithm TopicRank from pke - python keyphrase extraction
keywords_number_stub = 100
def get_tfidf_keywords(text, keywords_phrase_length, POS=None):
    extractor = pke.unsupervised.TfIdf()
    extractor.load_document(input=text, language='de', normalization=None)
    if POS:
        extractor.candidate_selection(n=keywords_phrase_length, stoplist=stopwords_list, pos={"NOUN", "PROPN", "ADJ"})
    else:
        extractor.candidate_selection(n=keywords_phrase_length, stoplist=stopwords_list)
    extractor.candidate_filtering(stoplist=stopwords_list, minimum_length=2)
    extractor.candidate_weighting()
    keywords = extractor.get_n_best(n=keywords_number_stub)
    return [k for k, _ in keywords]


def get_topicrank_keywords(text, keywords_phrase_length, POS=None): ## no n param in candidate_selection
    extractor = pke.unsupervised.TopicRank()
    extractor.load_document(input=text, language='de', normalization=None)
    if POS:
        extractor.candidate_selection(stoplist=stopwords_list, pos={"NOUN", "PROPN", "ADJ"})
    else:
        extractor.candidate_selection(stoplist=stopwords_list)
    extractor.candidate_filtering(stoplist=stopwords_list, minimum_length=2)
    try:
        extractor.candidate_weighting()
    except ValueError:
      return []
    keywords = extractor.get_n_best(n=keywords_number_stub)
    return [k for k, _ in keywords]


def get_all_tfidf_keywords(text, keywords_phrase_length):
    res1_0 = get_tfidf_keywords(text, keywords_phrase_length)
    #res1_1 = get_tfidf_keywords(text, keywords_phrase_length, True)
    return [res1_0]


def get_all_topicrank_keywords(text, keywords_phrase_length):
    #res1_0 = get_topicrank_keywords(text, keywords_phrase_length)
    res1_1 = get_topicrank_keywords(text, keywords_phrase_length, True)
    return [res1_1]


def get_top_n(result_lists, num):
   return [res[:num] for res in result_lists]


def get_all_keywords(text, keywords_number=20, keywords_phrase_length=2):
    rake_result = get_all_rake_keywords(text, keywords_phrase_length)
    rake_result_top = get_top_n(rake_result, keywords_number)

    tfidf_result = get_all_tfidf_keywords(text, keywords_phrase_length)
    tfidf_result_top = get_top_n(tfidf_result, keywords_number)

    topicrank_result = get_all_topicrank_keywords(text, keywords_phrase_length)
    topicrank_result_top = get_top_n(topicrank_result, keywords_number)
    res = {'RAKE': rake_result_top, 'TF-IDF': tfidf_result_top, 'TopicRank': topicrank_result_top}
    return res
    



