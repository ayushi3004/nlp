import math, collections

class SmoothBigramModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.bigramCounts = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    self.unigramCounts = collections.defaultdict(lambda: 0)
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model.
        Compute any counts or other corpus statistics in this function.
    """
    for sentence in corpus.corpus:
      previous_word = '<s>'
      self.unigramCounts[previous_word] += 1

      for datum in sentence.data:
        word = datum.word

        # if previous_word is not None:
        self.bigramCounts[previous_word][word] += 1
        self.unigramCounts[word] += 1

        previous_word = word

      word = '</s>'
      self.bigramCounts[previous_word][word] += 1
      self.unigramCounts[word] += 1

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the
        sentence using your language model. Use whatever data you computed in train() here.
    """
    score = 0.0
    previous_word = '<s>'
    alpha = 0.0095
    for word in sentence:
        # add lambda
      count1 = self.bigramCounts[previous_word][word]
      # mu = float(count1)/float(self.unigramCounts[previous_word]+1)
      count2 = self.unigramCounts[previous_word] + alpha*len(self.unigramCounts)
      if count1 > 0:
        score += math.log(count1)
      else:
        score += math.log(count1 + alpha)
      score -= math.log(count2)
      previous_word = word

    word = '</s>'
    count1 = self.bigramCounts[previous_word][word]
    count2 = self.unigramCounts[previous_word] + alpha*len(self.unigramCounts)
    if count1 > 0:
      score += math.log(count1)
    else:
      score += math.log(count1 + alpha)
    score -= math.log(count2)
    return score
