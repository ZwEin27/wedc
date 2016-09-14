# -*- coding: utf-8 -*-
# @Author: ZwEin
# @Date:   2016-09-13 14:44:46
# @Last Modified by:   ZwEin
# @Last Modified time: 2016-09-13 19:03:41

import os
import sys
import csv
import w2v
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import cross_validation
from sklearn.metrics import classification_report

class WEDC(object):

    def __init__(self, data_path):
        self.data_path = data_path
        self.w2v_model_path = os.path.join(os.path.dirname(__file__), 'w2v_model.bin')
        self.corpus, self.labels = self.load_data()
        self.size = len(self.labels)
        # self.vectorizer = CountVectorizer(min_df=1)
        # self.classifier = KNeighborsClassifier(n_neighbors=5, metric='jaccard')
        # self.rs = cross_validation.ShuffleSplit(self.size, n_iter=1, test_size=.25, random_state=12)

    def load_data(self):
        dataset = []
        labels = []
        with open(self.data_path, 'rb') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            for row in reader:
                label =row[0]
                content = row[1].decode('utf-8', 'ignore').encode('ascii', 'ignore')
                dataset.append(content)
                labels.append(label)
        return dataset, labels

    def load(self):

        w2v.setup_model(input, 
                        output, 
                        binary=1, 
                        cbow=0, 
                        size=300, 
                        window=10, 
                        negative=5, 
                        hs=0, 
                        threads=12, 
                        iter_=5, 
                        min_count=5, 
                        verbose=False)

    def run(self):
        vectors = self.vectorizer.fit_transform(self.corpus).toarray()
        
        """
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
        """



if __name__ == '__main__':
    data_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'data', 'dataset.csv')
    wedc = WEDC(data_path)
    wedc.run()





