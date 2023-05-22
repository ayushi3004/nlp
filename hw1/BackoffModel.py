import math, collections

class BackoffModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.bigramCounts = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    self.unigramCounts = collections.defaultdict(lambda: 0)
    self.total = 0
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """
    for sentence in corpus.corpus:
      previous_word = '<s>'
      self.unigramCounts[previous_word] += 1
      self.total += 1

      for datum in sentence.data:
        word = datum.word

        self.bigramCounts[previous_word][word] += 1
        self.unigramCounts[word] += 1

        previous_word = word
        self.total += 1

      word = '</s>'
      self.total += 1
      self.bigramCounts[previous_word][word] += 1
      self.unigramCounts[word] += 1

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    score = 0.0
    previous_word = None

    for word in sentence:
      if previous_word is not None:
        bigram_count = self.bigramCounts[previous_word][word]
        bigram_uni_count = self.unigramCounts[previous_word]
        unigram_count = self.unigramCounts[word]

        if bigram_count > 0:
          score += math.log(bigram_count)
          score -= math.log(bigram_uni_count)

        else:
          score += math.log(0.5)
          # add lambda
          score += math.log(unigram_count + 0.001)
          score -= math.log(self.total)
      previous_word = word

    word = '</s>'
    bigram_count = self.bigramCounts[previous_word][word]
    bigram_uni_count = self.unigramCounts[previous_word]
    unigram_count = self.unigramCounts[word]
    if bigram_count > 0:
      score += math.log(bigram_count)
      score -= math.log(bigram_uni_count)
    else:
      score += math.log(0.5) + math.log(unigram_count + + 0.001)
      score -= math.log(self.total + (len(self.unigramCounts)))

    return score