#!/usr/bin/env python -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import cPickle as pickle
from zipfile import ZipFile
from itertools import chain
from collections import Counter, defaultdict
from math import log

import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix

from seedling import udhr


def word2ngrams(text, n=3):
    """ Convert word into character ngrams. """
    return [text[i:i+n] for i in range(len(text)-n+1)]

def sent2ngrams(text, n=3):
    """ Convert sentence into character ngrams. """
    if n == "word":
        return text.split()
    return list(chain(*[word2ngrams(i,n) for i in text.lower().split()]))


def get_features_crubadan(n, featfreqs, all_langs, all_features, verbose=False):
    """
    Return features (n-grams or words) for the crubadan data.
    Language codes are not converted into ISO.
    Allow feature 'word' and 1 to 5 (for character grams).
    """
    crubadanfile='crub-131119.zip'
    crubadanfile =  os.getcwd() + '/seedling/data/crubadan/' + crubadanfile
    assert os.path.exists(crubadanfile)

    with ZipFile(crubadanfile,'r') as inzipfile:
     for infile in sorted(inzipfile.namelist()):
      path, filename = os.path.split(infile)
      if filename.strip() != '':
       if n == 'word' and 'words' in path or n in [1,2,3,4,5] and 'chars' in path:
        lang = filename.rpartition('.')[0]
        if '-' in lang:
           lang = lang.partition('-')[0]
        all_langs.add(lang)
        for line in inzipfile.open(infile):
            feature, count = line.strip().split(' ')
            if n in [1,2,3,4,5] and len(feature.decode('utf-8')) == n or n == 'word':
              #print(feature)
              featfreqs[lang][feature] = int(count)
              all_features.add(feature)
    if verbose:
	    for feature in all_features:
	        print (feature), # .decode('utf-8')
	    print(len(all_features))
    return featfreqs, all_langs, all_features


def get_features(datasource, n, verbose=False):
    """
    Return features (n-grams or words) for the datasource. 
    Also return list of all labels (languages) and all features.
    """
    matrix_dict = defaultdict(Counter)
    all_features = set()
    all_labels = set()

    if datasource=='crubadan':
        matrix_dict, all_labels, all_features = \
            get_features_crubadan(n, matrix_dict, all_labels, all_features)
    else:
        matrix_dict = defaultdict(Counter)
        all_features = set()
        all_labels = set()
        # Accessing SeedLing corpus and extracting Ngrams. 
        for lang, sent in globals()[datasource].sents():
            features = sent2ngrams(sent, n=n)
            if verbose:
                print(features)
            matrix_dict[lang].update(features)
            all_labels.add(lang)
            all_features.update(features)
    if verbose:
        print(datasource + ' data read in.')
    return matrix_dict, all_labels, all_features

def datasource2matrix(datasource='crubadan', n=3, option="csr_matrix", verbose=False):
    """
    Calls get_features, and converts the data to a scipy matrix.
    Names of languages and features are stored as lists.
    """
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
    
    matrix_dict, all_labels, all_features = get_features(datasource, n)
   
    all_features = sorted(all_features)
    all_labels = sorted(all_labels)
    
    if option == "dok_matrix": # it's slower.
        matrix = sp.sparse.dok_matrix((len(all_labels), len(all_features)))
        for i,label in enumerate(all_labels):
            for j,feat in enumerate(all_features):
                matrix[i, j] = matrix_dict[label][feat]
    elif option == "csr_matrix":
        matrix = csr_matrix(np.array([[matrix_dict[label][feat] \
                                       for feat in all_features] \
                                       for label in all_labels]))
    else:
        raise Exception('option not recognised')
    if verbose:
        print("Converted features into scipy matrix")
    
    with open(outlabelfile, 'wb') as fout:
        pickle.dump(all_labels, fout)
    
    with open(outfeatfile, 'wb') as fout:
        pickle.dump(all_features, fout)
    
    with open(outmatrixfile, 'wb') as fout:
        pickle.dump(matrix, fout)
        
    return matrix, all_labels, all_features


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
        """ *args and **kwargs are used so that scipy doens't break """
        kwargs.setdefault('dtype','float')
        super(csr_matrix_labelled, self).__init__(args[0], **kwargs)
        """ __init__ is called twice, for some reason; the second time only with arg[0], so we need this if-clause """
        if len(args) > 1:
            assert self.shape == (len(args[1]), len(args[2]))
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
    
    def get(self, code, feature):
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
        Converts all non-zero entries using to the given function
        Suggested functions:
         - mutual_information
         - pointwise_mi
        """
        self.normalise()
        code_prob = self.sum(1)
        feat_prob = self.sum(0)
        for row, col, value in self.iter_nonzero():
            self[row, col] = function(code_prob[row,0], feat_prob[0,col], value)

def get_matrix(datasource='crubadan', n=3):
    filename = "{}-{}-matrix.pk".format(datasource, n)
    
    if not os.path.exists(filename):
        print("Converting {} {}-gram data into a matrix ...".format(datasource, n))
        matrix = csr_matrix_labelled(*datasource2matrix(datasource,n=n))
        print("Pickling ...")
        with open(filename,'wb') as fout:
            pickle.dump(matrix, fout)
    else:
        print("Loading {} {}-gram data ...".format(datasource, n))
        with open(filename,'rb') as fin:
            matrix = pickle.load(fin)
    return matrix


if __name__ == "__main__":
    crub_mx = get_matrix(datasource='crubadan', n=1)
    print("Calculating mutual information ...")
    crub_mx.convert(mutual_information)
    for code, feat, value in crub_mx.iter_nonzero_label():
        if value > 10**-3:
            print(code, feat.decode('utf8'), value)