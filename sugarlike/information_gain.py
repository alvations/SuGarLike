#!/usr/bin/env python -*- coding: utf-8 -*-

import codecs, csv, sys, random, os
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
    if n == "word":
        return text.split()
    return list(chain(*[word2ngrams(i,n) for i in text.lower().split()]))

def make_mtxfile(datasource='udhr', outfilename=None, n=3):
    """ 
    Extracts features from SeedLing corpus and outputs a tab-delimited file,
    where rows represents the languages and columns represents the frequencies
    of the features.
    """
    outfilename = datasource+"-"+str(n)+'grams.mtx' if None else outfilename
    if os.path.exists(outfilename):
        return
    
    matrix = defaultdict(Counter)
    all_features = set()
    all_labels = set()
    # Accessing SeedLing corpus and extracting Ngrams. 
    for lang, sent in globals()[datasource].sents():
        features = sent2ngrams(sent, n=n)
        matrix[lang].update(features)
        all_labels.add(lang)
        all_features.update(features)
    
    all_features = sorted(features)
    all_labels = sorted(labels)
    
    with open(outfilename, 'wb') as fout:
        # Use the first two lines to save the labels and features.
        fout.write("\t".join(all_labels)+"\n")
        fout.write("\t".join(all_features)+"\n")
        # Saves the counts of the features in a tab-delimited file.
        for _label in all_labels:
            line = "\t".join([str(matrix[_label][_feature]) \
                     for _feature in all_features])
            fout.write(line+"\n")

def read_mtxfile(mtxfile, read_labels_features_only=False):
    all_labels, all_features = [], []
    # Reads the first two lines to get the labels and features.
    with codecs.open(mtxfile,"r","utf8") as fin:
        firstline,secondline = list(islice(fin,2))
        all_labels = firstline.strip().split('\t')
        all_features = secondline.strip().split('\t')
    
    if read_labels_features_only:
        return all_labels, all_features
    
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

def log(prob):
    return try_except(math.log, prob)

def calculate_mi(pi , pj, pij):
    p_i = 1-pi
    p_j = 1-pj
    p_ij = pj - pij
    pi_j = pi - pij
    p_i_j = 1- pi -pj + pij
    
    log_pi = log(pi)
    log_pj = log(pj)
    log_p_i = log(p_i)
    log_p_j = log(p_j)
    
    mi5 =   pij * (log(pij) - log_pi - log_pj) + \
            pi_j * (log(pi_j) - log_pi - log_p_j) + \
            p_ij * (log(p_ij) - log_p_i - log_pj) + \
            p_i_j * (log(p_i_j) - log_p_i - log_p_j)
    
    return mi5

class mutual_information():
    def __init__(self, matrix, all_labels, all_features):
        """ 
        *matrix* any scipy matrix, scipy.sparse.csc_matrix recommended.
        """
        self.matrix = matrix
        self.all_labels = all_labels
        self.all_features = all_features
        
        self.sum_matrix = matrix.sum()
        
        self.plabel = {}
        self.pfeature = {}
        self.plabel_feature = {}
        self.update_probabilities()
        
        self.mutualinfo = {}
        self.update_mutualinfo()
            
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

    def iterate_nonzeros(self):
        """ Iterating through non-zero cells in the matrix. """
        rows,cols = self.matrix.nonzero()
        for row_num,col_num in zip(rows,cols):
            _label = self.all_labels[row_num]
            _feat = self.all_features[col_num]
            yield _label, _feat, self.matrix[row_num,col_num]
    
    def update_probabilities(self):
        """ Updates probabilities. """
        print("Caculating Probabilities ...")
        for label, feat, count in self.iterate_nonzeros():
            ##if label == u"eng": print label, feat, int(count);
            self.plabel_feature.setdefault(label,{})[feat] = \
            int(count)/float(self.sum_matrix)
        for label, row in self.iterate_row():
            self.plabel[label] = row.sum()/float(self.sum_matrix)
        for feat, col in self.iterate_col(): 
            self.pfeature[feat] = col.sum()/float(self.sum_matrix)
    
    def prob_label(self, label):
        """ Returns probability of label, i.e. p(label) """
        return self.plabel[label]
    
    def prob_feature(self, feat):
        """ Returns probability of feature, i.e. p(feat). """
        return self.pfeature[feat]
    
    def prob_label_feature(self, label, feature):
        """ Returns of probability of label,feature, i.e. p(label,feat). """
        try:
            return self.plabel_feature[label][feature]
        except KeyError:
            return 0

    def update_mutualinfo(self):
        """ Updates Mutual Information for each cells. """
        print("Caculating Mutual Informations ...")
        # Iterate using iterate_nonzeros()
        for label, feat, _ in self.iterate_nonzeros():
            pi = self.prob_label(label)
            pj = self.prob_feature(feat)
            pij = self.prob_label_feature(label, feat)
            this_mi = calculate_mi(pi, pj, pij)
            ##print("\t".join(map(str, [label, feat, this_mi])))
            self.mutualinfo.setdefault(label,{})[feat] = this_mi

# Usage:

if not os.path.exists('udhr-3grams-mutalinfo.pk'):
    # Creates matrix, labels and features from seeding.udhr
    make_mtxfile('udhr', outfilename='udhr-3grams.mtx', n=3)
    matrix, labels, features = read_mtxfile('udhr-3grams.mtx')
    # Creates the Mutual information object. 
    mi = mutual_information(matrix, labels, features)
    # Dumps into a pickle.
    with open('udhr-3grams-mutalinfo.pk','wb') as fout:
        pickle.dump(mi, fout)
else:
    with open('udhr-3grams-mutalinfo.pk','rb') as fin:
        mi = pickle.load(fin)



# To check the encoding of the labels and features:
print(type(features[0]), features[:10])
# To check the probabilities:
l = unicode('eng'); f = unicode('the')
print(mi.prob_feature(l))
print(mi.prob_label_feature(l,f))

# To check the mutual information of a certain label+feature
##l = unicode('eng'); f = unicode('the')
##print(sorted(mi.mutualinfo[l].keys())) # Check list of feats for a language.
print(mi.mutualinfo[l][f])


'''
matrix = csc_matrix(np.array[[0,1,2,3,4],[1,2,3,4,5]])
labels = ['one', 'two']
feats = ['a','b','c','d','e']
mi = mutual_information(matrix, labels, feats)
print(mi.mutualinfo['a']['b'])
'''