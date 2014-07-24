#!/usr/bin/env python -*- coding: utf-8 -*-

from __future__ import print_function
from collections import defaultdict
from information_gain import get_raw
from operator import itemgetter

unigrams = get_raw('crubadan', 1)

test = [(['t', 'd', 'k', 'A'], 'Latin'),
        (['и', 'ц'], 'Cyrillic'),
        (['ⰰ'], 'Glagolitic'),
        (['ن'], 'Arabic'),
        (['े'], 'Devanagari'),
        (['人'], 'Hanzi'),
        (['α'], 'Greek'),
        (['ᑌ', 'ᐊ'], 'Canadian'),
        (['ა'], 'Georgian'),
        (['ⴀ'], 'Khutsuri'),
        (['ա'], 'Armenian'),
        (['า'], 'Thai'),
        (['າ'], 'Lao'),
        (['ት'], "Ge'ez"),
        (['ⲁ'], 'Coptic'),
        (['ന'], 'Malayalam'),
        (['ಲ'], 'Kannada'),
        (['া'], 'Bengali'),
        (['న'], 'Telugu'),
        (['ର'], 'Oriya'),
        (['ન'], 'Gujarati'),
        (['ਰ'], 'Gurmukhi'),
        (['᜔'], 'Baybayin'),
        (['꣄'], 'Saurashtra'),
        (['த'], 'Tamil'),
        (['න'], 'Sinhala'),
        (['ⵉ'], 'Tifinagh'),
        (['Ꮝ'], 'Cherokee'),
        (['ר'], 'Hebrew'),
        (['ས'], 'Tibetan'),
        (['ᤠ'], 'Limbu'),
        (['މ'], 'Tana'),
        (['ܢ'], 'Syriac'),
        (['ꃅ'], 'Yi'),
        (['ߘ'], "N'Ko"),
        (['ꤢ'], 'Kayah Li'),
        (['ꓭ'], 'Lisu'),  # almost Latin, but different Unicode
        #(['Ŧ'], 'Straits Salish'),  # uppercase Latin with diacritics
        (['တ'], 'Burmese'),
        #(['ၼ'], 'Shan'),  # mostly Burmese
        (['ᠠ'], 'Mongolian'),
        (['하'], 'Korean'),
        (['ន'], 'Khmer'),
        (['ꕉ'], 'Vai')]

ignore = ['my', 'xcl-Latn', 'swn-x-foqaha', 'yue-Hans', 'auj', 'trf', 'en-Shaw']
# 'auj' 2
# 'lif' 7
# 'cay' 10
# 'he-Latn' 11
# 'ksf' 13
# 'bft-Latn' 16
# 'oma' 18
# 'bfq' 18
# 'trf' noise
# 'en-Shaw' wtf

dirty = {'lzh':'Hanzi', 'npi':'Devanagari'}

# Record which script each language uses
scripts = {label:set() for _, label in test}

for code, charset in unigrams.items():
    # Ignore languages if there is too little data
    if code in ignore:
        continue
    """for char, count in charset.items():
        if count > 12:
            break
    else:
        print(code, "too little data")
        # 0: xcl-Latn, my, swn-x-foqaha, yue-Hans
        # small: auj, cay
        continue"""
    # Some languages are just too dirty
    if code in dirty:
        scripts[dirty[code]].add(code)
        continue
    # Keep track of which scripts are used
    n = set()
    for items, label in test:
        for i in items:
            if charset[i] > 8:  # some languages have dirty data
                scripts[label].add(code)
                n.add(label)
    # Some latin-based languages have very little data
    if len(n) == 0:
        if charset['a']:
            scripts['Latin'].add(code)
            n.add('Latin')
    # Catch languages that weren't assigned a unique script
    if len(n) != 1:
        top10 = sorted(unigrams[code].items(), key=itemgetter(1), reverse=True)[0:10]
        top10str = u' '.join([x[0].decode('utf8') for x in top10])
        print(code, top10str)
        print(n)
        if 'Latin' in n:
            scripts['Latin'].remove(code)

"""
# Print orthographic groups
sorted_scripts = sorted(scripts.items(), key=lambda x:len(x[1]), reverse=True)
for label, langs in sorted_scripts:
    print("{} ({}): {}".format(label, len(langs), ' '.join(langs)))
"""

# Metadata from Crubadan
datafile = 'seedling/data/crubadan/crubadancode.txt'

# 0  Code
# 1  Name (English)
# 2  Name (Native)
# 3  ISO 639-3
# 4  Country
# 5  Docs
# 6  Words
# 7  Characters
# 8  SIL
# 9  WT
# 10 WP
# 11 UDHR
# 12 Close to
# 13 Polluters
# 14 FLOSS SplChk
# 15 Contacts
# 16 Updated
# 17 Alternate names
# 18 Classification


# Consider a subset of languages to split geneologically
subset = scripts['Latin']

groups = defaultdict(set)
names = dict()

with open(datafile, 'r') as f:
    for rawline in f:
        line = rawline.split('\t')
        code = line[0]
        # Only consider languages in the subset we want
        if not code in subset:
            continue
        # Find the ISO 639-3 code
        code_main, _, code_extn = code.partition('-')
        if len(code_main) == 2:
            code_main = line[3]
        # Find the language family
        family = line[18].strip().rstrip('.').split(', ')
        # Find a name
        names[code] = line[1]
        """
        if len(family) > 1:
            groups[tuple(family[0:5])].add(code)
        else:
            groups[family[0], ].add(code)
        """
        #groups[family[0], ].add(code)
        
        if family[0] == "Oto-Manguean":
            groups[tuple(family[0:2])].add(code)
        else:
            continue
        
        """
        if family[0] != "Austronesian":
            continue
        elif family[1] != "Malayo-Polynesian":
            groups[tuple(family[0:2])].add(code)
        elif family[2] != "Central-Eastern":
            groups[tuple(family[0:3])].add(code)
        elif family[3] != "Eastern Malayo-Polynesian":
            groups[tuple(family[0:4])].add(code)
        elif family[4] != "Oceanic":
            groups[tuple(family[0:5])].add(code)
        else:
            groups[tuple(family[0:5])].add(code)
        """
        """
        if family[0] != "Niger-Congo":
            continue
        elif family[1] != "Atlantic-Congo":
            groups[tuple(family[0:2])].add(code)
        elif family[2] != "Volta-Congo":
            groups[tuple(family[0:3])].add(code)
        elif family[3] != "Benue-Congo":
            groups[tuple(family[0:4])].add(code)
        elif family[4] != "Bantoid":
            groups[tuple(family[0:5])].add(code)
        elif family[5] != "Southern":
            groups[tuple(family[0:6])].add(code)
        elif family[6] != "Narrow Bantu":
            groups[tuple(family[0:7])].add(code)
        else:
            groups[tuple(family[0:7])].add(code)
        """

# Consider '-' as isolates
#groups['Language Isolate'] |= groups['-']
#del groups['-']

# Consider singleton languages as isolates
#for g, langs in groups.items():
#    if len(langs) == 1:
#        groups['Language Isolate'] |= langs
#        del groups[g]


# Print groups
sorted_groups = sorted(groups.items(), key=lambda x:len(x[1]), reverse=True)
for label, langs in sorted_groups:
    lang_names = [names[code] for code in langs]
    print("{} ({}): {}".format(', '.join(label), len(langs), ', '.join(langs)))