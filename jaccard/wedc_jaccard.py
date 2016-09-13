# -*- coding: utf-8 -*-
# @Author: ZwEin
# @Date:   2016-09-13 14:44:46
# @Last Modified by:   ZwEin
# @Last Modified time: 2016-09-13 15:20:36

import os
import sys
import csv
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer

class WEDC(object):


    def __init__(self, data_path):
        self.data_path = data_path
        self.corpus, self.labels = self.load_data()

        self.vectorizer = CountVectorizer(min_df=1)

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

    def run(self):
        # print self.corpus[0]
        # neigh = NearestNeighbors(n_neighbors=1)
        # neigh.fit(self.corpus) 
        # print neigh.predict()
        # print neigh.kneighbors(self.corpus[0])

        

        self.vectorizer.fit_transform(self.corpus)
        neigh = KNeighborsClassifier(n_neighbors=5)
        neigh.fit(self.corpus, self.labels) 
        # print neigh.predict([self.corpus[0][1].split()])



if __name__ == '__main__':
    data_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'data', 'dataset.csv')
    wedc = WEDC(data_path)
    wedc.run()





