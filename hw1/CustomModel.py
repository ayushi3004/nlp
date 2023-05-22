import math, collections
class CustomModel:

  def __init__(self, corpus):
    """Initial custom language model and structures needed by this mode"""
    self.trigramCounts = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: 0)))
    self.bigramCounts = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    self.unigramCounts = collections.defaultdict(lambda: 0)
    self.total = 0
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model.
        Compute any counts or other corpus statistics in this function.
    """
    for sentence in corpus.corpus:
      previous_word_2 = None
      previous_word = '<s>'
      word = None

      # For the start char
      self.unigramCounts[previous_word] = self.unigramCounts[previous_word] + 1
      self.total += 1

      for datum in sentence.data:
        word = datum.word

        if previous_word_2 is not None:
          self.trigramCounts[previous_word_2][previous_word][word] += 1
        self.bigramCounts[previous_word][word] += 1
        self.unigramCounts[word] += 1

        previous_word_2 = previous_word
        previous_word = word
        self.total += 1

      # For the end char
      previous_word_2 = previous_word
      previous_word = word
      word = '</s>'
      self.trigramCounts[previous_word_2][previous_word][word] += 1
      self.bigramCounts[previous_word][word] += 1
      self.unigramCounts[word] += 1
      self.total += 1


  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the
        sentence using your language model. Use whatever data you computed in train() here.
    """
    score = 0.0
    previous_word = '<s>'
    previous_word_2 = None
    alpha = 0.598
    lam = 0.001

    for word in sentence:
      trigram_count = self.trigramCounts[previous_word_2][previous_word][word]
      trigram_bi_count = self.bigramCounts[previous_word_2][previous_word]

      bigram_count = self.bigramCounts[previous_word][word]
      bigram_uni_count = self.unigramCounts[previous_word]

      unigram_count = self.unigramCounts[word]

      if trigram_count > 0:
        score += math.log(trigram_count)
        score -= math.log(trigram_bi_count)

      elif bigram_count > 0:
        score += math.log(alpha)
        score += math.log(bigram_count)
        score -= math.log(bigram_uni_count)

      else:
        score += math.log(alpha)
        score += math.log(alpha)
        score += math.log(unigram_count + lam)
        score -= math.log(self.total + lam*(len(self.unigramCounts)))

      previous_word_2 = previous_word
      previous_word = word

    # sentence close char
    previous_word_2 = previous_word
    previous_word = word
    word = '</s>'

    trigram_count = self.trigramCounts[previous_word_2][previous_word][word]
    trigram_bi_count = self.bigramCounts[previous_word_2][previous_word]

    bigram_count = self.bigramCounts[previous_word][word]
    bigram_uni_count = self.unigramCounts[previous_word]

    unigram_count = self.unigramCounts[word]

    if trigram_count > 0:
      score += math.log(trigram_count)
      score -= math.log(trigram_bi_count)

    elif bigram_count > 0:
      score += math.log(alpha)
      score += math.log(bigram_count)
      score -= math.log(bigram_uni_count)

    else:
      score += math.log(alpha)
      score += math.log(alpha)
      score += math.log(unigram_count + lam)
      score -= math.log(self.total + lam*(len(self.unigramCounts)))

    return score