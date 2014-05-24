#!/usr/bin/env python -*- coding: utf-8 -*-

from __future__ import division, print_function

import os, gzip, warnings
import cPickle as pickle
from zipfile import ZipFile
from itertools import chain
from collections import Counter
from operator import itemgetter
from math import log, sqrt

from seedling import udhr, odin, omniglot
SEEDLING = {'udhr':udhr, 'odin':odin, 'omniglot':omniglot}


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


class MatrixDict(dict):
    """
    A dictionary with some numerical methods
    """
    def iter_all(self):
        for code, feat_set in self.iteritems():
            for feat, value in feat_set.iteritems():
                yield code, feat, value
    
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
        for feat_set in self.itervalues():
            for feat in feat_set:
                feat_prob[feat] += feat_set[feat]
        
        for feat_set in self.itervalues():
            code_prob = sum(feat_set.values())
            for feat in feat_set:
                feat_set[feat] = function(code_prob, feat_prob[feat], feat_set[feat])
        return self
    
    def top_n(self, n):
        """
        Returns the top N features for each code, as a dictionary
        """
        top = {}
        for code, feat_set in self.iteritems():
            tuples = sorted(feat_set.items(), reverse=True, key=itemgetter(1))
            best = {feat for feat, _ in tuples[:n]}
            top[code] = best
        return top
    
    def top_n_combined(self, n):
        """
        Finds the top N features for each code, and combines them into one set
        """
        top = set()
        for feat_set in self.itervalues():
            tuples = sorted(feat_set.items(), reverse=True, key=itemgetter(1))
            best = {feat for feat, _ in tuples[:n]}
            top |= best
        return top
    
    def top_thresh(self, threshold):
        """
        Returns all features above a given threshold, for each code, as a dictionary
        """
        top = {}
        for code, feat, value in self.iter_all():
            if value > threshold:
                top.setdefault(code, set()).add(feat)
        return top
     
    def top_thresh_combined(self, threshold):
        """
        Returns all features above a given threshold
        """
        top = set()
        for _, feat, value in self.iter_all():
            if value > threshold:
                top.add(feat)
        return top
    
    def filter(self, new_set):
        """
        Removes all elements not in the given feature set
        """
        for old_set in self.itervalues():
            for feat in old_set.iterkeys():
                if feat not in new_set:
                    del old_set[feat]
        return self
    
    def split(self, bins):
        """
        Given meta-codes to refer to sets of codes (as a dictionary),
        returns a collapsed MatrixDict for the meta-codes, and
        a dictionary from meta-codes to within-group MatrixDicts
        """
        collapsed = MatrixDict()
        sub_dicts = dict()
        for meta, group in bins.iteritems():
            collapsed[meta] = Counter()
            small = MatrixDict()
            sub_dicts[meta] = small
            for code in group:
                collapsed[meta].update(self[code])
                small[code] = self[code]
        return collapsed, sub_dicts

class MultiDict(tuple):
    """ 6 matrix_dicts """
    def __init__(self, iterable):
        super(MultiDict, self).__init__(iterable)
        assert len(self) == 6
        for i in range(6):
            assert isinstance(self[i], MatrixDict)
        for i in range(5):
            assert self[i].keys() == self[5].keys()
    
    def split(self, bins):
        part_split = [self[i].split(bins) for i in range(6)]
        collapsed = MultiDict(part_split[i][0] for i in range(6))
        sub_dicts = dict()
        for code in bins:
            sub_dicts[code] = MultiDict(part_split[i][1][code] for i in range(6))
        return collapsed, sub_dicts


class Classifier():
    """
    Our one-stage classifier
    """
    def __init__(self, data):
        """ data as a MultiDict or equivalent """
        assert len(data) == 6
        for i in range(5):
            assert data[i].keys() == data[5].keys()
        if isinstance(data, tuple):
            self.weights = data
        else:
            self.weights = tuple(data)
    
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
        features = sent2feats(sample_text)
        return self.id_feat(features)
    
    def identify_top(self, sample_text):
        return max(self.identify(sample_text), key=itemgetter(1))[0]

class TwoStage():
    """
    Our two-stage classifier
    """
    def __init__(self, superdata, subdata=None):
        """
        Data as a MultiDict (or equivalent),
        and a dictionary from group labels to multi_dicts
        """
        if not subdata:  # Allow input as a 2-tuple, instead of separate arguments 
            subdata = superdata[1]
            superdata = superdata[0]
        assert superdata[0].keys() == subdata.keys()
        self.first = Classifier(superdata)
        self.second = dict()
        for code in subdata:
            self.second[code] = Classifier(subdata[code])
    
    def id_feat(self, features):
        group  = self.first.id_feat(features)
        best = max(group, key=itemgetter(1))[0]
        variety = self.second[best].id_feat(features)
        return group, best, variety
    
    def identify(self, sample_text):
        features = sent2feats(sample_text)
        return self.id_feat(features)
    
    def identify_top(self, sample_text):
        _, best, variety = self.identify(sample_text)
        return best, max(variety, key=itemgetter(1))[0]


def get_raw_crubadan(n, collapse=False):
    """
    Return features (n-grams or words) for the crubadan data.
    Language codes are not converted into ISO.
    Allow feature 'word' and 1 to 5 (for character grams).
    """
    crubadanfile='crub-131119.zip'
    crubadanfile =  os.getcwd() + '/seedling/data/crubadan/' + crubadanfile
    assert os.path.exists(crubadanfile)
    
    matrix = MatrixDict()
    
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
    matrix = MatrixDict()
    
    for lang, sent in SEEDLING[datasource].sents():
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
    if n == 0: n = 'word'  # To allow easier access to several matrices
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

def get_multi(datasource='crubadan', option="raw", collapse=False, verbose=True):
    """
    Loads all matrices for all feature types
    """
    return MultiDict(get_matrix(n=i,
        datasource=datasource, option=option, collapse=collapse, verbose=verbose)
        for i in range(6))


def setup_crubadan():
    """
    Sets up matrices of raw features, mutual information, and pointwise mutual information,
    for all feature types, using Crubadan
    """
    for n in [1,2,3,4,5,'word']:
        for option in ['mi','pmi']:
            get_matrix(datasource='crubadan', n=n, option=option)


if __name__ == "__main__":
    # Two-stage classifier
    bins = {'germanic':{'en','de'}, 'romance':{'fr','it'}}
    m = get_multi().split(bins)
    for i in range(6):
        m[0][i].normalise_rowsq()
        for meta in m[1]:
            m[1][meta][i].normalise_rowsq()
    c = TwoStage(m)
    print(c.identify_top("hello world"))
    print(c.identify_top("guten tag"))
    print(c.identify_top("bonjour"))
    print(c.identify_top("buonjiorno"))
    """
    # One-stage classifier:
    data = []
    for i in ['word',1,2,3,4,5]:
        data.append(get_matrix(n=i).normalise_rowsq())
    c = Classifier(data)
    for code, value in c.identify('guten tag')[:20]:
        print(code, value)
    """
    """
    # Feature selection with one-stage classifier:
    data = []
    n_feats = [1000,10,100,1000,1000,1000]
    for i in range(6):
        feats = get_matrix(n=i, option='mi').top_n_combined(n_feats[i])
        data.append(get_matrix(n=i).filter(feats).normalise_rowsq())
    c = Classifier(data)
    for code, value in c.identify('guten tag')[:20]:
        print(code, value)
    """