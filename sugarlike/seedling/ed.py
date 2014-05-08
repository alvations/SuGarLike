
import cPickle as pickle
import codecs
from collections import defaultdict

pkfile = 'data/ethnologue/livinglanguages_with_info.pk'
pseudo_ethnologue = pickle.load(codecs.open(pkfile, 'rb'))

def retrieve_endangerment_level(livinglanguages):
  endangerment_languages = defaultdict(list)
  for i in livinglanguages:
    if i in ["nob","nno"]: i = "nor"
    try:
      endangerment_level = pseudo_ethnologue[i][0][-1].split()[0]
    except:
      continue
    endangerment_languages[endangerment_level].append(i)
  return endangerment_languages

from ed2 import wikipedia, udhr, combined, odin, omniglot

for i in ['odin', 'omniglot', 'wikipedia', 'udhr', 'combined']:
  endanger = retrieve_endangerment_level(globals()[i])
  print i
  for level in sorted(endanger):
    print level, len(endanger[level])
  print
  