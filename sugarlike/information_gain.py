#!/usr/bin/env python -*- coding: utf-8 -*-

from __future__ import division, print_function

import os, sys, gzip
import cPickle as pickle
from zipfile import ZipFile
from itertools import chain
from collections import Counter, defaultdict
from math import log

import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix, dok_matrix

from seedling import udhr


def mutual_information(pi, pj, pij):
    """
    Calculates the mutual information of two Bernoulli distributions,
    which have marginal probabilities pi and pj, and joint probability pij
    """
    p_i = 1 - pi
    p_j = 1 - pj
    p_ij = pj - pij
    pi_j = pi - pij
    p_i_j = 1 - pi - pj + pij
    
    log_pi = log(pi)
    log_pj = log(pj)
    log_p_i = log(p_i)
    log_p_j = log(p_j)
    
    mi = pij * (log(pij) - log_pi - log_pj) + \
         pi_j * (log(pi_j) - log_pi - log_p_j) + \
         p_i_j * (log(p_i_j) - log_p_i - log_p_j)
    if p_ij != 0:  # For language groups and features, this is the only probability that could be zero, and lim_x->0[x*log(x)] = 0 
        mi += p_ij * (log(p_ij) - log_p_i - log_pj)
    
    return mi

def pointwise_mi(pi, pj, pij):
    """
    Calculates the pointwise mutual information of two events,
    which have marginal probabilities pi and pj, and joint probability pij
    """
    return log(pij) - log(pi) - log(pj)


class csr_matrix_labelled(csr_matrix):
    """
    Usage: csr_matrix_labelled(matrix, codes, features)
    
    Note: do not rename this class!  The 'csr' prefix is actually required by scipy
    """
    def __init__(self, *args, **kwargs):
        """ Initialises the matrix, then adds labels """
        # *args and **kwargs are used so that scipy doesn't break
        # Force datatype to be float:
        kwargs.setdefault('dtype','float')
        super(csr_matrix_labelled, self).__init__(args[0], **kwargs)
        # __init__ is called twice, for some reason; the second time only with arg[0], so we need this if-clause
        if len(args) > 1:
            assert self.shape == (len(args[1]), len(args[2]))  # Check that the number of labels matches the data 
            self.codes = args[1] #codes
            self.feats = args[2] #features
    
    def iter_nonzero(self):
        """ Iterates through non-zero cells in the matrix, with indices """
        for row_num, col_num in zip(*self.nonzero()):
            yield row_num, col_num, self[row_num, col_num]
    
    def iter_nonzero_label(self):
        """ Iterates through non-zero cells in the matrix, with labels """
        for row, col in zip(*self.nonzero()):
            code = self.codes[row]
            feat = self.feats[col]
            yield code, feat, self[row, col]
    
    def iter_row(self):
        """ Iterates through rows/codes """
        for row in range(len(self.codes)):
            yield code[row], self[row,:]
    
    def iter_col(self):
        """ Iterates through columns/features (slow) """
        for col in range(len(self.feats)):
            yield feat[col], self[:,col]
    
    def from_label(self, code, feature):
        """ Looks up a cell, from labels """
        """ Note: not optimised """
        row = self.codes.index(code)
        col = self.feats.index(feature)
        return self[row, col]
    
    def normalise(self):
        """ Entire matrix will sum to 1 """
        norm = 1/self.sum()
        self *= norm
    
    def normalise_rows(self):
        """ Each row will sum to 1 """
        norm = [1/x for x in self.sum(1)]
        for row, col in zip(*self.nonzero()):
            self[row,col] *= norm[row]
    
    def convert(self, function=pointwise_mi):
        """
        Converts all non-zero entries using the given function
        Suggested functions:
         - mutual_information
         - pointwise_mi
        """
        self.normalise()
        code_prob = self.sum(1)
        feat_prob = self.sum(0)
        for row, col, value in self.iter_nonzero():
            self[row, col] = function(code_prob[row,0], feat_prob[0,col], value)


def word2ngrams(text, n=3):
    """ Convert word into character ngrams. """
    return [text[i:i+n] for i in range(len(text)-n+1)]

def sent2ngrams(text, n=3):
    """ Convert sentence into character ngrams. """
    if n == "word":
        return text.split()
    return list(chain(*[word2ngrams(i,n) for i in text.lower().split()]))


def get_raw_crubadan(n, collapse=False):
    """
    Return features (n-grams or words) for the crubadan data.
    Language codes are not converted into ISO.
    Allow feature 'word' and 1 to 5 (for character grams).
    """
    crubadanfile='crub-131119.zip'
    crubadanfile =  os.getcwd() + '/seedling/data/crubadan/' + crubadanfile
    assert os.path.exists(crubadanfile)
    
    matrix = dok_matrix((sys.maxint,sys.maxint))  # We will resize the matrix at the end
    codes = []
    curr = -1
    feats = []
    feat_dict = {}
    
    if n == 'word':        subdir = 'words'
    elif n in [1,2,3,4,5]: subdir = 'chars'
    else: raise TypeError("expected 1, 2, 3, 4, 5, or 'word'")

    with ZipFile(crubadanfile,'r') as inzipfile:
        for infile in sorted(inzipfile.namelist()):
            path, filename = os.path.split(infile)
            if path != subdir: continue
            if filename == '': continue
            lang = filename.rpartition('.')[0]
            if collapse:
                lang = lang.partition('-')[0]
                if codes == [] or lang != codes[-1]:
                    codes.append(lang)
                    curr += 1
            else:
                codes.append(lang)
                curr += 1
            
            for line in inzipfile.open(infile):
                feature, count = line.strip().split(' ')
                if n == 'word' or (n in [1,2,3,4,5] and len(feature.decode('utf-8')) == n):
                    try:
                        index = feat_dict[feature]
                    except KeyError:
                        index = len(feats)
                        feats.append(feature)
                        feat_dict[feature] = index
                    matrix[curr, index] += float(count)
    
    matrix.resize((len(codes), len(feats)))

    return csr_matrix_labelled(matrix, codes, feats)

def get_raw_seedling(datasource, n):
    """
    Return features (n-grams or words) for seedling data.
    Language codes are not converted into ISO.
    Allow feature 'word' and 1 to 5 (for character grams).
    """
    matrix_dict = defaultdict(Counter)
    all_features = set()
    all_labels = set()
    # Access SeedLing corpus and extract Ngrams. 
    for lang, sent in globals()[datasource].sents():
        features = sent2ngrams(sent, n=n)
        matrix_dict[lang].update(features)
        all_labels.add(lang)
        all_features.update(features)
    all_features = sorted(all_features)
    all_labels = sorted(all_labels)
    # The following line is not optimal, since we create a dense array, but this works for the smaller datasources
    matrix = csr_matrix(np.array([[matrix_dict[label][feat] \
                                       for feat in all_features] \
                                       for label in all_labels]))
    
    return csr_matrix_labelled(matrix, all_labels, all_features)

def get_raw(datasource, n, collapse=False):
    """
    Chooses the appropriate function to call to get data
    """
    if datasource == 'crubadan':
        return get_raw_crubadan(n=n, collapse=collapse)
    else:
        return get_raw_seedling(datasource=datasource, n=n)


def get_matrix(datasource='crubadan', n=3, option="raw", collapse=False):
    """
    Loads matrix (if pickled), or calculates it.
    """
    filename = "{}-{}-{}.pk.gz".format(datasource, n, option)
    
    if os.path.exists(filename):
        print("Loading {} data ({}-gram, {}) ...".format(datasource, n, option))
        with gzip.open(filename,'rb') as fin:
            matrix = pickle.load(fin)
        
    elif option == 'raw':
        print("Transferring {} {}-gram data into a matrix ...".format(datasource, n))
        matrix = get_raw(datasource=datasource, n=n, collapse=collapse)
        print("Pickling ...")
        with gzip.open(filename,'wb') as fout:
            pickle.dump(matrix, fout)
        
    elif option == 'mi':
        matrix = get_matrix(datasource=datasource, n=n, option='raw', collapse=collapse)
        print("Calculating mutual information ...")
        matrix.convert(mutual_information)
        print("Pickling ...")
        with gzip.open(filename,'wb') as fout:
            pickle.dump(matrix, fout)
        
    elif option == 'pmi':
        matrix = get_matrix(datasource=datasource, n=n, option='raw', collapse=collapse)
        print("Calculating pointwise mutual information ...")
        matrix.convert(pointwise_mi)
        print("Pickling ...")
        with gzip.open(filename,'wb') as fout:
            pickle.dump(matrix, fout)
        
    else:
        raise NotImplementedError("Available options: 'raw'")
    
    return matrix


if __name__ == "__main__":
    for n in [1,2,3,4,5,'word']:
        for option in ['mi','pmi']:
            get_matrix(datasource='crubadan', n=n, option=option)