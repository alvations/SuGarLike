#!/usr/bin/env python -*- coding: utf-8 -*-

from __future__ import division, print_function

import os, gzip, warnings
import cPickle as pickle
from zipfile import ZipFile
from itertools import chain
from collections import Counter
from operator import itemgetter
from math import log, sqrt

from seedling import udhr, odin, omniglot #called using globals() in get_raw_seedling


def word2ngrams(text, n=3):
    """ Convert word into character ngrams. """
    return [text[i:i+n] for i in range(len(text)-n+1)]

def sent2ngrams(text, n=3):
    """ Convert sentence into character ngrams. """
    if n == "word":
        return text.split()
    return list(chain(*[word2ngrams(i,n) for i in text.lower().split()]))

def sent2feats(text):
    """
    Convert sentence into features, as a list of six Counters:
    [word, 1-, 2-, 3-, 4-, 5-gram]
    """
    result = [Counter(text.split())]
    for n in [1,2,3,4,5]:
        result.append(Counter(sent2ngrams(text, n)))
    return result


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


class matrix_dict(dict):
    """
    A dictionary with some numerical methods
    """
    def normalise(self):
        """ Entire matrix will sum to 1 """
        total = 0
        for feat_set in self.values():
            for value in feat_set.values():
                total += value
        norm = 1/total
        for feat_set in self.values():
            for feat in feat_set:
                feat_set[feat] *= norm
        return self
    
    def normalise_rowsq(self):
        """ Each row's squares will sum to 1 """
        for code, feat_set in self.items():
            try:
                norm = 1/sqrt(sum([x**2 for x in feat_set.values()]))
            except ZeroDivisionError:
                warnings.warn('{} has no features'.format(code), UserWarning)
                del self[code]
            for feat in feat_set:
                feat_set[feat] *= norm
        return self
    
    def convert(self, function=pointwise_mi):
        """
        Converts all non-zero entries using the given function
        Suggested functions:
         - mutual_information
         - pointwise_mi
        """
        self.normalise()
        feat_prob = Counter()
        for feat_set in self.values():
            for feat in feat_set:
                feat_prob[feat] += feat_set[feat]
        
        for feat_set in self.values():
            code_prob = sum(feat_set.values())
            for feat in feat_set:
                feat_set[feat] = function(code_prob, feat_prob[feat], feat_set[feat])
        return self
    
    def top_n(self, n):
        """
        Returns the top N features for each code, as a dictionary
        """
        top = {}
        for code, feat_set in self.items():
            tuples = sorted(feat_set.items(), reverse=True, key=itemgetter(1))
            best = {feat for feat, _ in tuples[:n]}
            top[code] = best
        return top
    
    def top_n_combined(self, n):
        """
        Finds the top N features for each code, and combines them into one set
        """
        top = set()
        for feat_set in self.values():
            tuples = sorted(feat_set.items(), reverse=True, key=itemgetter(1))
            best = {feat for feat, _ in tuples[:n]}
            top |= best
        return top
    
    def filter(self, new_set):
        """
        Removes all elements not in the given feature set
        """
        for old_set in self.values():
            for feat in old_set.keys():
                if feat not in new_set:
                    del old_set[feat]
        return self
    
    def split(self, bins):
        """
        Given a set of meta-codes to refer to groups of codes (in the form of a dictionary),
        returns a collapsed matrix_dict for the meta-codes, and a set of smaller matrix_dicts for the groups
        """
        raise NotImplementedError
        collapsed = matrix_dict()
        sub_dicts = set()
        for meta in bins:
            collapsed[meta] = Counter()
            small = matrix_dict()
            sub_dicts.add(small)
        return collapsed, sub_dicts


class classifier():
    """
    Our baseline classifier
    """
    def __init__(self, data):
        """ data as a list of six dictionaries of Counters """
        assert len(data) == 6
        for i in range(5):
            assert data[i].keys() == data[5].keys()
        self.weights = data
    
    def __getitem__(self, key):
        """
        Allows dictionary/list-like access. Overloaded:
         * if int or 'word', returns all data for that feature type
         * if a language code, returns all data for that code
        """
        if key == 'word' or key == 'w':
            key = 0
        if type(key) == int:
            return self.weights[key]
        elif key in self.keys():
            return [self.weights[i][key] for i in range(6)]
        else:
            raise KeyError(key)
    
    def keys(self):
        return self[0].keys()
    
    def __len__(self):
        return len(self.keys())
    
    def id_feat(self, features):
        scores = {code:0 for code in self.keys()}
        for i in range(6):
            for code in self[i]:
                for feat in features[i]:
                    scores[code] += self[i][code][feat] * features[i][feat]
        return sorted(scores.items(), reverse=True, key=itemgetter(1))
    
    def identify(self, sample_text):
        return self.id_feat(sent2feats(sample_text))

class two_stage():
    """
    Our two-stage classifier
    """
    pass


def get_raw_crubadan(n, collapse=False):
    """
    Return features (n-grams or words) for the crubadan data.
    Language codes are not converted into ISO.
    Allow feature 'word' and 1 to 5 (for character grams).
    """
    crubadanfile='crub-131119.zip'
    crubadanfile =  os.getcwd() + '/seedling/data/crubadan/' + crubadanfile
    assert os.path.exists(crubadanfile)
    
    matrix = matrix_dict()
    
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
            
            matrix.setdefault(lang, Counter())
            for line in inzipfile.open(infile):
                feature, count = line.strip().split(' ')
                if n == 'word' or (n in [1,2,3,4,5] and len(feature.decode('utf-8')) == n):
                    matrix[lang][feature] += int(count)
    
    return matrix

def get_raw_seedling(datasource, n):
    """
    Return features (n-grams or words) for seedling data.
    Language codes are not converted into ISO.
    Allow feature 'word' and 1 to 5 (for character grams).
    """
    matrix = matrix_dict()
    
    for lang, sent in globals()[datasource].sents():
        features = sent2ngrams(sent, n=n)
        matrix.setdefault(lang, Counter()).update(features)
    
    return matrix

def get_raw(datasource, n, collapse=False):
    """
    Chooses the appropriate function to call to get data
    """
    if datasource == 'crubadan':
        return get_raw_crubadan(n=n, collapse=collapse)
    else:
        return get_raw_seedling(datasource=datasource, n=n)


def get_matrix(datasource='crubadan', n=3, option="raw", collapse=False, verbose=True):
    """
    Loads matrix (if pickled), or calculates it.
    """
    subdir = "matrices"
    filename = "{}/{}-{}-{}.pk.gz".format(subdir, datasource, n, option)
    if not os.path.exists(subdir):
        os.mkdir(subdir)
    
    if os.path.exists(filename):
        if verbose: print("Loading {} data ({}-gram, {}) ...".format(datasource, n, option))
        with gzip.open(filename,'rb') as fin:
            matrix = pickle.load(fin)
        
    elif option == 'raw':
        if verbose: print("Transferring {} {}-gram data into a matrix ...".format(datasource, n))
        matrix = get_raw(datasource=datasource, n=n, collapse=collapse)
        if verbose: print("Pickling ...")
        with gzip.open(filename,'wb') as fout:
            pickle.dump(matrix, fout)
        
    elif option == 'mi':
        matrix = get_matrix(datasource=datasource, n=n, option='raw', collapse=collapse)
        if verbose: print("Calculating mutual information ...")
        matrix.convert(mutual_information)
        if verbose: print("Pickling ...")
        with gzip.open(filename,'wb') as fout:
            pickle.dump(matrix, fout)
        
    elif option == 'pmi':
        matrix = get_matrix(datasource=datasource, n=n, option='raw', collapse=collapse)
        if verbose: print("Calculating pointwise mutual information ...")
        matrix.convert(pointwise_mi)
        if verbose: print("Pickling ...")
        with gzip.open(filename,'wb') as fout:
            pickle.dump(matrix, fout)
        
    else:
        raise NotImplementedError("Available options: 'raw'")
    
    return matrix


def setup_crubadan():
    for n in [1,2,3,4,5,'word']:
        for option in ['mi','pmi']:
            get_matrix(datasource='crubadan', n=n, option=option)


if __name__ == "__main__":
    data = []
    for i in ['word',1,2,3,4,5]:
        data.append(get_matrix(n=i).normalise_rowsq())
    c = classifier(data)
    for code, value in c.identify('guten tag')[:20]:
        print(code, value)
    '''
    feats = get_matrix(n=1, option='mi').top_n_combined(5)
    for x in sorted(feats):
        print(x)
    print(len(feats))
    '''
    """
    m = get_matrix(n=1, option='raw').filter(feats).normalise_rowsq()
    for code, feats in sorted(m.items()):
        print(code, feats)"""