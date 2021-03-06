#!/usr/bin/python

import nltk, re, fileinput

punkt = nltk.data.load('tokenizers/punkt/english.pickle')
word_tokenizer = nltk.tokenize.WordPunctTokenizer()
stopwords = set(nltk.corpus.stopwords.words('english'))
real_word = re.compile('[A-Za-z][a-z]+')

header = ['num_sentences','num_unique_words','num_words'] 
result = ['NA','NA','NA']

def uni(input):
  r = ''
  try:
    r = unicode(input).encode('ascii', 'ignore')
  except Exception, e:
    raise e
  return r

print "\t".join(map(uni, header))

for line in fileinput.input():
  sentences = punkt.tokenize(line)
  # Number of sentences
  result[0] = len(sentences)

  words = [ word for word in word_tokenizer.tokenize(line) if word not in stopwords and real_word.match(word) ]
  unique_words = set(words)
  # Number of unique words
  result[1] = len(unique_words)

  # Number of total words
  result[2] = len(words)

  print "\t".join(map(uni, result))

