# -*- coding: utf-8 -*-
# @Author: ZwEin
# @Date:   2016-09-23 12:58:37
# @Last Modified by:   ZwEin
# @Last Modified time: 2016-09-24 14:21:07

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from sklearn.utils import resample
from sklearn.metrics import precision_recall_curve
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import average_precision_score
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize

##################################################################
# Constant
##################################################################

DC_CATEGORY_DICT = {
    'unknown': -1,
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

    def load_classifier(self, handler_type='knn', **kwargs):
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

    def __run_specific_train_test_data(self, train_data_path, test_data_path):
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
        test_origin = [corpus[i] for i in range(size) if i in test_index]

        train_X = [vectors[i] for i in range(size) if i in train_index]
        train_y = [int(labels[i]) for i in range(size) if i in train_index]
        test_X = [vectors[i] for i in range(size) if i in test_index]
        text_y = [int(labels[i]) for i in range(size) if i in test_index]

        self.classifier.fit(train_X, train_y)
        pred_y = self.classifier.predict(test_X)
        pred_proba_y = self.classifier.predict_proba(test_X)
        
        return text_y, pred_y, pred_proba_y


    def __run_split_train_test_data_full_train(self, vectors, labels, train_index, test_index):
        pass

    def __run_split_train_test_data_part_train(self, vectors, labels, train_index, test_index):
        pass

    def __run_split_train_test_data(self, train_test_split=.25, random_state=None, n_iter=1):
        corpus = self.corpus
        labels = self.labels
        size = self.size

        vectors = self.vectorizer.fit_transform(corpus).toarray()

        for train_index, test_index in cross_validation.ShuffleSplit(size, n_iter=n_iter, test_size=train_test_split, random_state=random_state):

            
            test_origin = [corpus[i] for i in range(size) if i in test_index]

            # train_X = [vectors[i] for i in range(size) if i in train_index]
            # train_y = [int(labels[i]) for i in range(size) if i in train_index]
            test_X = [vectors[i] for i in range(size) if i in test_index]
            text_y = [int(labels[i]) for i in range(size) if i in test_index]

            self.classifier.fit(train_X, train_y)
            pred_y = self.classifier.predict(test_X)
            # pred_proba_y = self.classifier.predict_proba(test_X)
            
            for pi in range(1, 11):
                percent = float(pi) / 10.
                precision = []
                recall = []
                fscore = []
                support = []

                cur_train_index = list(train_index)
                cur_train_index.shuffle()
                train_index_size = int(percent * len(cur_train_index))

                for j in range(10):
                    train_index = list(cur_train_index)
                    shuffle(train_index)
                    train_index = train_index[:train_index_size]

                    train_X = [vectors[i] for i in range(self.size) if i in train_index]
                    train_y = [self.labels[i] for i in range(self.size) if i in train_index]

                    # test_origin = [self.corpus[i] for i in range(self.size) if i in test_index]
                    # test_X = [vectors[i] for i in range(self.size) if i in test_index]
                    # text_y = [self.labels[i] for i in range(self.size) if i in test_index]

                    self.classifier.fit(train_X, train_y)

                    pred_y = self.classifier.predict(test_X)

                    tmp_precision, tmp_recall, tmp_fscore, tmp_support = precision_recall_fscore_support(text_y, pred_y)

                    precision.append(tmp_precision)
                    recall.append(tmp_recall)
                    fscore.append(tmp_fscore)
                    support.append(tmp_support)



        
        return text_y, pred_y, None

    def run(self, train_data_path=None, test_data_path=None, train_test_split=.25, random_state=None, n_iter=1):

        if train_data_path and test_data_path:
            y_test, y_pred, y_pred_proba = self.__run_specific_train_test_data(train_data_path, test_data_path)
        elif not train_data_path and not test_data_path:
            y_test, y_pred, y_pred_proba = self.__run_split_train_test_data(train_test_split=train_test_split, random_state=random_state, n_iter=n_iter)
        else:
            raise Exception('incorrect format')

        # self.plot(y_test, y_pred, y_pred_proba=y_pred_proba)




    ##################################################################
    # Statistic Report Methods
    ##################################################################
    
    def display_avg_std(self, percent, precision, recall, fscore, support):
        precision = numpy.array(precision)
        recall = numpy.array(recall)
        fscore = numpy.array(fscore)
        support = numpy.array(support)

        pmean = numpy.mean(precision, axis=0)
        rmean = numpy.mean(recall, axis=0)
        fmean = numpy.mean(fscore, axis=0)
        smean = numpy.mean(support, axis=0)

        pstd = numpy.std(numpy.array(precision), axis=0)
        rstd = numpy.std(numpy.array(recall), axis=0)
        fstd = numpy.std(numpy.array(fscore), axis=0)

        print '                  '.ljust(10), \
            'precision'. center(20), \
            'recall'.center(20), \
            'f1-score'.center(20), \
            'support'.center(20)

        print '    massage       ', \
            str(round(pmean[0], 2)).rjust(8),'|',str(round(pstd[0], 5)).ljust(9), \
            str(round(rmean[0], 2)).rjust(8),'|',str(round(rstd[0], 5)).ljust(9), \
            str(round(fmean[0], 2)).rjust(8),'|',str(round(fstd[0], 5)).ljust(9), \
            str(int(smean[0])).center(20)

        print '     escort       ', \
            str(round(pmean[1], 2)).rjust(8),'|',str(round(pstd[1], 5)).ljust(9), \
            str(round(rmean[1], 2)).rjust(8),'|',str(round(rstd[1], 5)).ljust(9), \
            str(round(fmean[1], 2)).rjust(8),'|',str(round(fstd[1], 5)).ljust(9), \
            str(int(smean[1])).center(20)

        print '    job_ads       ', \
            str(round(pmean[2], 2)).rjust(8),'|',str(round(pstd[2], 5)).ljust(9), \
            str(round(rmean[2], 2)).rjust(8),'|',str(round(rstd[2], 5)).ljust(9), \
            str(round(fmean[2], 2)).rjust(8),'|',str(round(fstd[2], 5)).ljust(9), \
            str(int(smean[2])).center(20)


    ##################################################################
    # Plot Methods
    ##################################################################

    def plot(self, y_test, y_pred, y_pred_proba=None):
        # print y_test
        # print y_pred
        
        target_labels = [2, 3, 4]

        y_test = np.array(y_test)
        y_pred = np.array(y_pred)
        y_pred_proba = np.array(y_pred_proba)

        y_test = label_binarize(y_test, classes=target_labels)

        precision = dict()
        recall = dict()
        average_precision = dict()

        for i in range(len(target_labels)):
            precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                                y_pred_proba[:, i])
            average_precision[i] = average_precision_score(y_test[:, i], y_pred_proba[:, i])

        # # Plot Precision-Recall curve
        # plt.clf()
        # plt.plot(recall[0], precision[0], label='Precision-Recall curve')
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.ylim([0.0, 1.05])
        # plt.xlim([0.0, 1.0])
        # plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision[0]))
        # plt.legend(loc="lower left")
        # plt.show()

        # Plot Precision-Recall curve for each class
        plt.clf()
        # plt.plot(recall["micro"], precision["micro"],
        #          label='micro-average Precision-recall curve (area = {0:0.2f})'
        #                ''.format(average_precision["micro"]))
        
        for i in range(len(target_labels)):
            plt.plot(recall[i], precision[i],
                     label='Precision-recall curve of class {0} (area = {1:0.2f})'
                           ''.format(i, average_precision[i]))

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Extension of Precision-Recall curve to multi-class')
        plt.legend(loc="lower right")
        plt.show()



        


if __name__ == '__main__':
    

    ## Train and test path provided separately
    # dc = WEDC(vectorizer_type='count', classifier_type='knn')
    # train_path = '/Users/ZwEin/job_works/StudentWork_USC-ISI/projects/dig-groundtruth-data/classification/training-data/dig_memex_eval_datasets.csv'
    # test_path = '/Users/ZwEin/job_works/StudentWork_USC-ISI/projects/dig-groundtruth-data/classification/testing-data/testing_data.csv'
    # dc.run(train_data_path=train_path, test_data_path=test_path)

    ## Only one data path provided
    data_path = '/Users/ZwEin/job_works/StudentWork_USC-ISI/projects/dig-groundtruth-data/classification/testing-data/testing_data.csv'
    dc = WEDC(data_path=data_path, vectorizer_type='count', classifier_type='knn')
    dc.run(train_test_split=.25, random_state=23, n_iter=1)


