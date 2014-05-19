#!/usr/bin/env python -*- coding: utf-8 -*-

import codecs, csv, sys, random, os
import math, operator
import zipfile
import cPickle as pickle
from itertools import chain, islice, izip
from collections import Counter, defaultdict
from functools import partial

import numpy as np
import scipy as sp
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


def get_features_crubadan(n, featfreqs, all_langs, all_features):
    """
    Return features (n-grams or words) for the crubadan data.
    Language codes are not converted into ISO.
    Allow feature 'word' and feature '5' (return all 1-5 character grams).
    """
    crubadanfile='crub-131119.zip'
    crubadanfile =  './seedling/data/crubadan/' + crubadanfile
    assert os.path.exists(crubadanfile)

    with zipfile.ZipFile(crubadanfile,'r') as inzipfile:
     for infile in sorted(inzipfile.namelist()):
      path, filename = os.path.split(infile)
      if filename.strip() != '':
       if n == 'word' and 'words' in path or n == 5 and 'chars' in path:
        lang = filename.rpartition('.')[0]
        if '-' in lang:
           lang = lang.partition('-')[0]
        all_langs.add(lang)
        for line in inzipfile.open(infile):
            feature, count = line.strip().split(' ')
            featfreqs[lang][feature] = int(count)
            all_features.add(feature)

    return featfreqs, all_langs, all_features


def get_features(datasource, n):
    """
    Return features (n-grams or words) for the datasource. 
    Also return list of all labels (languages) and all features.
    """
    _matrix = defaultdict(Counter)
    all_features = set()
    all_labels = set()

    if datasource=='crubadan':
        _matrix, all_labels, all_features = \
            get_features_crubadan(n, _matrix, all_labels, all_features)
    else:
        _matrix = defaultdict(Counter)
        all_features = set()
        all_labels = set()
        # Accessing SeedLing corpus and extracting Ngrams. 
        for lang, sent in globals()[datasource].sents():
            features = sent2ngrams(sent, n=n)
            print(features)
            _matrix[lang].update(features)
            all_labels.add(lang)
            all_features.update(features)
    print(datasource + ' data read in.')
    return _matrix, all_labels, all_features


def datasource2matrix(datasource='udhr', n=3, option="csc_matrix"):
    outmatrixfile = datasource+"-"+str(n)+'grams.mtx'
    outlabelfile = datasource+"-"+str(n)+'grams.label'
    outfeatfile = datasource+"-"+str(n)+'grams.feats'
    
    if os.path.exists(outmatrixfile):
        with open(outmatrixfile, 'rb') as fin:
            matrix = pickle.load(fin)
        
        with open(outlabelfile, 'rb') as fin:
            all_labels = pickle.load(fin)

        with open(outfeatfile, 'rb') as fin:
            all_features = pickle.load(fin)
        
        return matrix, all_labels, all_features
    
    _matrix, all_labels, all_features = get_features(datasource, n)
   
    all_features = sorted(all_features)
    all_labels = sorted(all_labels)
    
    if option == "dok_matrix": # it's slower.
        matrix = sp.sparse.dok_matrix((len(all_labels), len(all_features)))
        for i,label in enumerate(all_labels):
            for j,feat in enumerate(all_features):
                matrix[i, j] = _matrix[label][feat]
    elif option == "csc_matrix":
        matrix = csc_matrix(np.array([[_matrix[label][feat] \
                                       for feat in all_features] \
                                      for label in all_labels]))
    
    with open(outlabelfile, 'wb') as fout:
        pickle.dump(all_labels, fout)
    
    with open(outfeatfile, 'wb') as fout:
        pickle.dump(all_features, fout)
    
    with open(outmatrixfile, 'wb') as fout:
        pickle.dump(matrix, fout)
        
    return matrix, all_labels, all_features

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
        *matrix* = one of the following scipy sparse matrices: 
        - sp.sparse.csc_matrix
        - sp.sparse.csr_matrix
        - sp.sparse.dok_matrix
        - sp.sparse.lil_matrix
        
        scipy.sparse.csc_matrix recommended for size.
        
        *all_labels* = rows
        *all_features* = column
        
        
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
    
    def topn_features(self, label, topn=10):
        """ Sort by value and then returns the keys of the top n features. """
        return [i for i,j in sorted(self.mutualinfo[label].iteritems(), \
                                  key=operator.itemgetter(1))][:topn]

def test_mutual_information_class():
    # Testing the mutual_information class.
    x = np.array([[5,5,5,3,4],[0,0,0,9,9]])
    csc = csc_matrix(x)
    csr = sp.sparse.csr_matrix(x)
    dok = sp.sparse.dok_matrix(x)
    lil = sp.sparse.lil_matrix(x)
    
    labels = ['one', 'two']
    feats = ['a','b','c','d','e']
    
    for matrix in [csc, csr, dok, lil]:

        mi = mutual_information(matrix, labels, feats)
        print(mi.mutualinfo)
        print


def test_everything(datasource, n=3):
    if not os.path.exists(datasource+'-'+str(n)+'grams-mutalinfo.pk'):
        print(" ".join(["Creating Mutual Information object for",\
                       datasource,str(n)+'gram ...']))
        
        # Creates matrix, labels and features from seeding.udhr
        print(" ".join(["Loading", datasource, "into scipy matrix ..."]))
        matrix, labels, features = datasource2matrix(datasource,n=n, \
                                                     option="csc_matrix")
        
        # Creates the Mutual information object. 
        mi = mutual_information(matrix, labels, features)
        
        # Dumps into a pickle.
        with open(datasource+'-'+str(n)+'grams-mutalinfo.pk','wb') as fout:
            pickle.dump(mi, fout)
    else:
        print(" ".join(["Loading Mutual Information object for",\
                       datasource,str(n)+'gram ...']))
        matrix, labels, features = datasource2matrix(datasource,n=n, \
                                                     option="csc_matrix")
        with open(datasource+'-'+str(n)+'grams-mutalinfo.pk','rb') as fin:
            mi = pickle.load(fin)
    
    # To check the encoding of the labels and features:
    ##print(type(features[0]), features[:100])
    # To check the probabilities:
    '''l = unicode('fuc'); f = unicode('ɗum')
    #print(mi.prob_label(l)) # p(label)
    #print(mi.prob_feature(f)) # p(feat)
    #print(mi.prob_label_feature(l,f)) # p(label,feat)
    #print(log(mi.prob_feature(f))) # log(p(feat))

    # To check the mutual information of a certain label+feature
    ##l = unicode('eng'); f = unicode('the')
    ##print(sorted(mi.mutualinfo[l].keys())) # Check list of feats for a language.
    print(mi.mutualinfo[l][f]) # MI(label,feat)
    
    # Returns nbest features.
    print(mi.topn_features(l, 100))
    print
    
    l = unicode('deu')
    print(mi.topn_features(l, 100))'''


##test_mutual_information_class()

datasource = 'crubadan'
#for n in [1,2,3,4,5,'word']:
for n in ['word']:
    test_everything(datasource, n)


##


