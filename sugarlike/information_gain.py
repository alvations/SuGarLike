#!/usr/bin/env python -*- coding: utf-8 -*-

import codecs, csv, sys, random
import math
import cPickle as pickle
from itertools import chain, islice, izip
from collections import Counter,defaultdict
from functools import partial

import numpy as np
from scipy.sparse import csc_matrix

from seedling import udhr

def word2ngrams(text, n=3):
    """ Convert word into character ngrams. """
    return [text[i:i+n] for i in range(len(text)-n+1)]

def sent2ngrams(text, n=3):
    """ Convert sentence into character ngrams. """
    return list(chain(*[word2ngrams(i,n) for i in text.lower().split()]))

def make_mtxfile(datasource='udhr', outfile=None):
    """ 
    Extracts features from SeedLing corpus and outputs a tab-delimited file,
    where rows represents the languages and columns represents the frequencies
    of the features.
    """
    if os.path.exists(datasource+".mtx"):
        return
    
    matrix = defaultdict(Counter)
    all_features = set()
    all_labels = set()
    # Accessing SeedLing corpus and extracting Ngrams. 
    for lang, sent in globals()[datasource].sents():
        features = sent2ngrams(sent)
        matrix[lang].update(features)
        all_labels.add(lang)
        all_features.update(features)
    
    with open(datasource+'.mtx', 'wb') as fout:
        # Use the first two lines to save the labels and features.
        fout.write("\t".join(all_labels)+"\n")
        fout.write("\t".join(all_features)+"\n")
        # Saves the counts of the features in a tab-delimited file.
        for _label in sorted(all_labels):
            line = "\t".join([str(matrix[_label][_feature]) \
                     for _feature in sorted(all_features)])
            fout.write(line+"\n")

def read_mtxfile(mtxfile):
    all_labels, all_features = [], []
    # Reads the first two lines to get the labels and features.
    with codecs.open(mtxfile,"rb","utf8") as fin:
        firstline,secondline = list(islice(fin,2))
        all_labels = firstline.strip().split('\t')
        all_features = secondline.strip().split('\t')
    
    # Reading data into sparse matrix object.
    with open(mtxfile,"rb") as fin:
        next(fin); next(fin) # Skips first two rows.
        # Reads the .mtx files into list of list of int.
        data = map(partial(map,int),list(csv.reader(fin,delimiter='\t')))
        # Converts list of list to sparse matrix.
        matrix = csc_matrix(data)
        ##print sys.getsizeof(matrix)
    
    return matrix, sorted(all_labels), sorted(all_features)

def sum_of(matrix, option, num):
    """
    [in]: a scipy matrix, 
    [out]: sum of column|row given the column|row number.
    Usage:
    >>> sum_of(matrix, "row", 2)
    >>> sum_of(matrix, "col", 2)
    """
    options = {"col":0, "row":1}
    return int(matrix.sum(axis=options[option])[num])

def try_except(myFunction, *params):
    """ Generic try-except to catch ZeroDivisionError, ValueError """
    try:
        return myFunction(*params)
    except ZeroDivisionError as e:
        return np.NAN
    except ValueError as e:
        return 0

class mutual_information():
    def __init__(self, matrix, all_labels, all_features):
        self.matrix = matrix
        self.all_labels = sorted(all_labels)
        self.all_features = sorted(all_features)
        
        self.sum_matrix = matrix.sum()
        
        self.plabel = {}
        self.pfeature = {}
        self.plabel_feature = {}
        
        self.update_probabilities()
            
    def iterate_row(self):
        """ Iterating by row. """
        for label, row in zip(self.all_labels, self.matrix):
            yield label, row
    
    def iterate_col(self):
        """ Iterating by column. """
        for feat, col in zip(self.all_features, self.matrix.transpose()):
            yield feat, col
            
    def iterate_cells(self):
        """ Iterating through each cell in the matrix. """
        _coo = self.matrix.tocoo()
        for i,j,v in izip(_coo.row, _coo.col, _coo.data):
            yield self.all_labels[i], self.all_features[j], v

    def iterate_nonzeros(selfs):
        """ Iterating through non-zero cells in the matrix. """
        rows,cols = self.matrix.nonzero()
        for row_num,col_num in zip(rows,cols):
            _label = self.labels[row_num-1]
            _feat = self.features[col_num-1]
            yield _label, _feat, self.matrix[row_num,col_num]
    
    def update_probabilities(self):
        """ Updates probabilities. """
        for label, feat, count in self.iterate_cells():
            ##print label, feat, int(count)
            self.plabel_feature.setdefault(label,{})[feat] = \
            try_except(math.log, int(count)/float(self.sum_matrix))
        for label, row in self.iterate_row():
            self.plabel[label] = \
            try_except(math.log, row.sum()/float(self.sum_matrix))
        for feat, col in self.iterate_col(): 
            self.pfeature[feat] = \
            try_except(math.log, col.sum()/float(self.sum_matrix))
    
    def prob_label(self, label):
        return self.plabel[label]
    
    def prob_feature(self, feat):
        return self.pfeature[feat]
    
    def prob_label_feature(self, label, feature):
        try:
            return self.plabel_feature[label][feature]
        except KeyError:
            return 0


# Usage:

try:
    mi = pickle.load(open('udhr.mtx.pk', 'rb'))
except IOError:
    ##make_mtxfile('udhr')
    matrix, labels, features = read_mtxfile('udhr.mtx')
    mi = mutual_information(matrix, labels, features)
    pickle.dump(mi, open('udhr.mtx.pk', 'wb'))


random_label = random.choice(labels)
random_feature = random.choice(features)
random_label = "eng".encode('utf8')
random_feature = "the".encode('utf8')

print("p("+random_label+","+random_feature+")"+" = " + \
      str(mi.prob_label_feature(random_label, random_feature)))

print("p("+random_label+")"+" = " + \
      str(mi.prob_label(random_label)))

print("p("+random_feature+")"+" = " + \
      str(mi.prob_feature(random_feature)))
