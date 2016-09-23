# -*- coding: utf-8 -*-
# @Author: ZwEin
# @Date:   2016-09-23 12:58:37
# @Last Modified by:   ZwEin
# @Last Modified time: 2016-09-23 13:53:53

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

    def __run_specific_train_test_data(train_data_path, test_data_path):
        train_data_corpus, train_data_labels = self.load_data(filepath=train_data_path)
        test_data_corpus, test_data_labels = self.load_data(filepath=test_data_path)

        train_data_size = len(train_data_labels)
        test_data_size = len(test_data_labels)

        train_index = range(train_data_size)
        test_index = range(train_data_size, test_data_size)

        corpus = train_data_corpus + test_data_corpus
        labels = train_data_labels + test_data_labels
        size = train_data_size + test_data_size

        vectors = self.vectorizer.fit_transform(corpus).toarray()

        # return corpus, labels, size, train_index, test_index

    def __run_split_train_test_data(train_test_split=.25, random_state=None, n_iter=1):
        corpus = self.corpus
        labels = self.labels
        size = self.size

        for train_index, test_index in cross_validation.ShuffleSplit(size, n_iter=n_iter, test_size=train_test_split, random_state=random_state):
            pass


    def run(self, train_data_path=None, test_data_path=None, train_test_split=.25, random_state=None, n_iter=1):

        # corpus = []
        # labels = []
        # size = 0
        # train_index = []
        # test_index = []

        if train_data_path and test_data_path:
            self.__run_specific_train_test_data(train_data_path, test_data_path)
        elif not train_data_path and not test_data_path:
            self.__run_split_train_test_data(train_test_split=train_test_split, random_state=random_state, n_iter=n_iter)
        else:
            raise Exception('incorrect format')



        


