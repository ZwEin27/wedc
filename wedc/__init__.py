# -*- coding: utf-8 -*-
# @Author: ZwEin
# @Date:   2016-09-23 12:58:37
# @Last Modified by:   ZwEin
# @Last Modified time: 2016-09-23 13:21:57

import os
import sys
import csv
import numpy
from random import shuffle
from sklearn.utils import resample
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

##################################################################
# Constant
##################################################################

DC_CATEGORY_DICT = {
    'unknown': -1
    'others': 1,
    'massage': 2,
    'escort': 3,
    'job_ads': 4
}


class WEDC(object):

    ##################################################################
    # Basic Methods
    ##################################################################

    def __init__(self, data_path=None, vectorizer_type='count', classifier_type='knn'):
        self.corpus = []
        self.labels = []
        self.size = 0

        if data_path:
            new_corpus, new_labels = self.load_data(filepath=data_path)
            self.corpus += new_corpus
            self.labels += new_labels
            self.size += len(new_labels)

        self.vectorizer = self.load_vectorizer(handler_type=vectorizer_type, binary=True)
        self.classifier = self.load_classifier(handler_type=classifier_type, weights='distance', n_neighbors=5, metric='jaccard')

    def load_data(self, filepath=None):
        dataset = []
        labels = []
        with open(filepath, 'rb') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            for row in reader:
                label =row[0]
                content = row[1].decode('utf-8', 'ignore').encode('ascii', 'ignore')
                dataset.append(content)
                labels.append(label)
        return dataset, labels

    def load_vectorizer(self, handler_type='count', **kwargs):
        vectorizers = {
            'count': CountVectorizer(binary=kwargs.get('binary', True)),
            'tfidf': TfidfVectorizer(min_df=kwargs.get('min_df', .25))
        } 
        return vectorizers[handler_type]

    def load_classifier(self, handler_type='knn', **kwargs)
        classifiers = {
            'knn': KNeighborsClassifier( \
                        weights=kwargs.get('weights', 'distance'), \
                        n_neighbors=kwargs.get('n_neighbors', 5), \
                        metric=kwargs.get('metric', 'jaccard'))
        }
        return classifiers[handler_type]

    ##################################################################
    # Run Methods
    ##################################################################

    def run(self, train_data_path=None, test_data_path=None, train_test_split=.25):

        pass
        


