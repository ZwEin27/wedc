# -*- coding: utf-8 -*-
# @Author: ZwEin
# @Date:   2016-09-13 14:44:46
# @Last Modified by:   ZwEin
# @Last Modified time: 2016-09-16 15:57:29

import os
import sys
import csv
from random import shuffle
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cross_validation
from sklearn.metrics import classification_report

class WEDC(object):

    def __init__(self, data_path):
        self.data_path = data_path
        self.corpus, self.labels = self.load_data(filepath=data_path)
        self.size = len(self.labels)

        # jaccard and knn
        self.vectorizer = CountVectorizer(binary=True)
        self.classifier = KNeighborsClassifier(weights='distance', n_neighbors=5, metric='jaccard')

        # tfidf and cosine
        # self.vectorizer = # TfidfVectorizer(min_df=.25)
        # self.classifier = KNeighborsClassifier(n_neighbors=5, metric='cosine', algorithm='brute')

        self.rs = cross_validation.ShuffleSplit(self.size, n_iter=1, test_size=.25, random_state=12)

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

    def run(self):
        vectors = self.vectorizer.fit_transform(self.corpus).toarray()
        
        for train_index, test_index in self.rs:
            train_X = [vectors[i] for i in range(self.size) if i in train_index]
            train_y = [self.labels[i] for i in range(self.size) if i in train_index]

            test_origin = [self.corpus[i] for i in range(self.size) if i in test_index]
            test_X = [vectors[i] for i in range(self.size) if i in test_index]
            text_y = [self.labels[i] for i in range(self.size) if i in test_index]

            self.classifier.fit(train_X, train_y)

            pred_y = self.classifier.predict(test_X)

            target_names = ['massage', 'escort', 'job_ads']
            print classification_report(text_y, pred_y, target_names=target_names)

            error_index = [i for i in range(len(text_y)) if text_y[i] != pred_y[i]]

            for idx in error_index:
                print '\n\n'
                print '#'*60
                print '# ', target_names[int(text_y[idx])-2], 'error predicted as', target_names[int(pred_y[idx])-2]
                print '#'*60
                print test_origin[idx]
                
            # for idx in error_index:
            #     print '\n\n'
            #     print '#'*60
            #     print '# ', target_names[int(text_y[idx])-2], 'error predicted as', target_names[int(pred_y[idx])-2]
            #     print '#'*60
            #     print test_origin[idx]

    def run_with_specific_training_data(self):
        vectors = self.vectorizer.fit_transform(self.corpus).toarray()
        train_index = [i for i in range(self.size) if int(self.labels[i]) != -1]
        test_index = [i for i in range(self.size) if int(self.labels[i]) == -1]
        
        train_X = [vectors[i] for i in range(self.size) if i in train_index]
        train_y = [self.labels[i] for i in range(self.size) if i in train_index]

        test_origin = [self.corpus[i] for i in range(self.size) if i in test_index]
        test_X = [vectors[i] for i in range(self.size) if i in test_index]
        # text_y = [self.labels[i] for i in range(self.size) if i in test_index]

        self.classifier.fit(train_X, train_y)

        pred_y = self.classifier.predict(test_X)

        target_names = ['massage', 'escort', 'job_ads']
        # print classification_report(text_y, pred_y, target_names=target_names)

        # error_index = [i for i in range(len(text_y)) if text_y[i] != pred_y[i]]

        # for idx in range(len(pred_y)):
        #     print '\n\n'
        #     print '#'*60
        #     print '# ', 'Predicted as', target_names[int(pred_y[idx])-2]
        #     print '#'*60
        #     print test_origin[idx]
        for idx in range(len(pred_y)):
            print pred_y[idx]+','+'\"'+test_origin[idx]+'\"'

    def run_with_specific_training_testing_data(self, testing_data_path):
        vectors = self.vectorizer.fit_transform(self.corpus).toarray()

        testing_corpus, testing_labels = self.load_data(filepath=testing_data_path)
        testing_vectors = self.vectorizer.fit_transform(testing_corpus).toarray()

        train_X = vectors
        train_y = self.labels

        test_origin = testing_corpus
        test_X = testing_vectors
        test_y = testing_labels

        print len(testing_vectors)

        # # train_index = [i for i in range(self.size) if int(self.labels[i]) != -1]
        # # test_index = [i for i in range(self.size) if int(self.labels[i]) == -1]
        
        # train_X = [vectors[i] for i in range(self.size) if i in train_index]
        # train_y = [self.labels[i] for i in range(self.size) if i in train_index]

        # test_origin = [self.corpus[i] for i in range(self.size) if i in test_index]
        # test_X = [vectors[i] for i in range(self.size) if i in test_index]
        # # test_y = [self.labels[i] for i in range(self.size) if i in test_index]

        self.classifier.fit(train_X, train_y)

        pred_y = self.classifier.predict(test_X)

        target_names = ['massage', 'escort', 'job_ads']
        print classification_report(test_y, pred_y, target_names=target_names)

        error_index = [i for i in range(len(test_y)) if test_y[i] != pred_y[i]]

        for idx in error_index:
                print '\n\n'
                print '#'*60
                print '# ', target_names[int(test_y[idx])-2], 'error predicted as', target_names[int(pred_y[idx])-2]
                print '#'*60
                print test_origin[idx]

    def run_test(self):
        vectors = self.vectorizer.fit_transform(self.corpus).toarray()
        
        # for train_index, test_index in self.rs:
        all_index = range(self.size)

        train_index = all_index[:552]
        test_index = all_index[552:]
        train_X = [vectors[i] for i in range(self.size) if i in train_index]
        train_y = [self.labels[i] for i in range(self.size) if i in train_index]

        test_origin = [self.corpus[i] for i in range(self.size) if i in test_index]
        test_X = [vectors[i] for i in range(self.size) if i in test_index]
        text_y = [self.labels[i] for i in range(self.size) if i in test_index]

        self.classifier.fit(train_X, train_y)

        pred_y = self.classifier.predict(test_X)

        target_names = ['massage', 'escort', 'job_ads']
        print classification_report(text_y, pred_y, target_names=target_names)

        error_index = [i for i in range(len(text_y)) if text_y[i] != pred_y[i]]

        for idx in error_index:
            print '\n\n'
            print '#'*60
            print '# ', target_names[int(text_y[idx])-2], 'error predicted as', target_names[int(pred_y[idx])-2]
            print '#'*60
            print test_origin[idx]

    def run_with_differ_training_percentage(self):
        vectors = self.vectorizer.fit_transform(self.corpus).toarray()
        
        # for train_index, test_index in self.rs:
        all_index = range(self.size)

        original_train_index = all_index[:552]
        test_index = all_index[552:]

        # different portion of training index
        original_train_index_size = len(original_train_index)

        for i in range(1, 10):
            percent = float(i) / 10.

            print '#'*60
            print str(int(percent*100))+'%', 'for training'
            print '#'*60

            train_index_size = int(percent * original_train_index_size)
            train_index = list(original_train_index)
            shuffle(train_index)
            train_index = train_index[:train_index_size]

            train_X = [vectors[i] for i in range(self.size) if i in train_index]
            train_y = [self.labels[i] for i in range(self.size) if i in train_index]

            test_origin = [self.corpus[i] for i in range(self.size) if i in test_index]
            test_X = [vectors[i] for i in range(self.size) if i in test_index]
            text_y = [self.labels[i] for i in range(self.size) if i in test_index]

            self.classifier.fit(train_X, train_y)

            pred_y = self.classifier.predict(test_X)

            target_names = ['massage', 'escort', 'job_ads']
            print classification_report(text_y, pred_y, target_names=target_names)

            error_index = [i for i in range(len(text_y)) if text_y[i] != pred_y[i]]

            # for idx in error_index:
            #     print '\n\n'
            #     print '#'*60
            #     print '# ', target_names[int(text_y[idx])-2], 'error predicted as', target_names[int(pred_y[idx])-2]
            #     print '#'*60
            #     print test_origin[idx]



if __name__ == '__main__':
    data_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'data', 'train500_test50.csv')    # need to change
    wedc = WEDC(data_path)
    wedc.run_with_differ_training_percentage()
    




    # data_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'data', 'all_extractions_july_2016_with_500.csv')
    # data_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'data', 'all_extractions_july_2016_with_500.csv')
    # wedc = WEDC(data_path)
    # wedc.run_with_specific_training_data()

    # data_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'data', 'dataset.csv')
    # data_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'data', '20m3_500.csv')
    # data_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'data', '20m3.csv')
    # wedc = WEDC(data_path)
    # wedc.run()

    # data_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'data', 'dataset.csv')
    # testing_data_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'data', 'testing.csv')
    # wedc = WEDC(data_path)
    # wedc.run_with_specific_training_testing_data(testing_data_path)


    # data_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'data', 'train500_test50.csv')
    # wedc = WEDC(data_path)
    # wedc.run_test()





