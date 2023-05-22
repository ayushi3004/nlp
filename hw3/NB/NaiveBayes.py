import sys
import getopt
import os
import math
from collections import defaultdict
import collections
import operator
import re

class NaiveBayes:
    class TrainSplit:
        """Represents a set of training/testing data. self.train is a list of Examples, as is self.test.
        """
        def __init__(self):
          self.train = []
          self.test = []

    class Example:
        """Represents a document with a label. klass is 'pos' or 'neg' by convention.
           words is a list of strings.
        """
        def __init__(self):
            self.klass = ''
            self.words = []


    def __init__(self):
        """NaiveBayes initialization"""
        self.FILTER_STOP_WORDS = False
        self.BOOLEAN_NB = False
        self.BEST_MODEL = False
        self.stopList = set(self.readFile('data/english.stop'))
        self.numFolds = 10

        # the frequency of the classifier across docs
        self.classifier_freq = {}
        # the frequency of words
        self.word_freq = {}
        # frequency of words per classifier
        self.word_freq_per_klass = {}
        # the number of words per classifier
        self.word_cnt_per_klass = {}
        # doc count per word per class
        self.doc_cnt_w_k = {}

        self.doc_unique_words = {}

        self.p_sentiment = {}
        self.p_words_sentiment = {}
        self.p_sentiment_words = {}

    #     for best model
        self.training_tokens = []
        self.bi_word_cnt_per_klass = {}
        self.bigramCounts = defaultdict(lambda: defaultdict(int))
        self.unigramCounts = defaultdict(int)
        self.bi_classifier_freq = defaultdict(int)
        self.bi_words_cnt_per_klass = defaultdict(int)
        self.bi_doc_unique_words = {}
        self.vocab = set()
        self.UNK = "UNK"

    #############################################################################
    # TODO TODO TODO TODO TODO
    # Implement the Multinomial Naive Bayes classifier and the Naive Bayes Classifier with
    # Boolean (Binarized) features.
    # If the BOOLEAN_NB flag is true, your methods must implement Boolean (Binarized)
    # Naive Bayes (that relies on feature presence/absence) instead of the usual algorithm
    # that relies on feature counts.
    #
    # If the BEST_MODEL flag is true, include your new features and/or heuristics that
    # you believe would be best performing on train and test sets.
    #
    # If any one of the FILTER_STOP_WORDS, BOOLEAN_NB and BEST_MODEL flags is on, the
    # other two are meant to be off. That said, if you want to include stopword removal
    # or binarization in your best model, write the code accordingl

    def laplace_smoothing(self, word, klass):
        num = self.word_freq_per_klass[klass][word] + 1

        vocab_size = sum([len(s) for s in self.doc_unique_words.values()])
        den = self.word_cnt_per_klass[klass] + vocab_size
        return math.log(float(num) / den)

    def classify_gen(self, words):
        class_scores = {}
        data_size = sum(self.word_freq.values())
        for word in words:
            for klass in ["pos", "neg"]:
                prob_word_given_klass = self.laplace_smoothing(word, klass)
                if klass not in class_scores:
                    class_scores[klass] = float(self.classifier_freq[klass]) / data_size
                class_scores[klass] += prob_word_given_klass

        return max(class_scores, key=class_scores.get)

    def preprocess(self, words):
        def to_lower(w):
            return [x.lower() for x in w]

        def take_an(w):
            # clean = re.sub("[^a-z\s]+", " ", w)
            # return re.sub("(\s+)", " ", clean)
            return [re.sub("[^a-z0-9\s]+", " ", x) for x in w]

        def filter_out_empty(w):
            filt = []
            for e in w:
                if not e.isspace():
                    filt.append(e)
            return filt


        w1 = to_lower(words)
        w2 = take_an(w1)
        return filter_out_empty(w2)

    def count_unigrams(self, words):
        s = '<s>'
        if s not in self.unigramCounts:
            self.unigramCounts[s] = 0
        self.unigramCounts[s] += 1
        self.training_tokens.append(s)

        for word in words:
            self.training_tokens.append(word)
            if word not in self.unigramCounts:
                self.unigramCounts[word] = 0
            self.unigramCounts[word] += 1

        s = '</s>'
        if s not in self.unigramCounts:
            self.unigramCounts[s] = 0
        self.unigramCounts[s] += 1
        self.training_tokens.append(s)

    def replace_unk_train_words(self):
        for i in range(len(self.training_tokens)):
            w = self.training_tokens[i]
            if self.unigramCounts[w] == 1:
                self.training_tokens[i] = self.UNK

    def replace_unk_test_words(self, words):
        for i in range(len(words)):
            word = words[i]
            if not self.training_tokens.__contains__(word):
                words[i] = self.UNK
        return words

    def recount_unigrams(self):
        self.unigramCounts = {}
        for word in self.training_tokens:
            if word not in self.unigramCounts:
                self.unigramCounts[word] = 0
            self.unigramCounts[word] += 1

    def count_bigrams(self, klass):
        for i in range(len(self.training_tokens) - 1):
            word_pair = self.training_tokens[i], self.training_tokens[i + 1]
            if word_pair != ('</s>', '<s>'):
                if klass not in self.bigramCounts:
                    self.bigramCounts[klass] = defaultdict(int)

                if word_pair in self.bigramCounts[klass]:
                    self.bigramCounts[klass][word_pair] += 1
                else:
                    self.bigramCounts[klass][word_pair] = 1

                if word_pair not in self.vocab:
                    self.vocab.add(word_pair)

                self.bi_words_cnt_per_klass[klass] += 1
        print("Bigram counted")

    def classify_best(self, words):
        words = self.filterStopWords(words)
        words = self.preprocess(words)
        # words = self.replace_unk_test_words(words)
        class_scores = {}
        data_size = len(self.vocab)

        for i in range(len(words) - 1):
            w1 = words[i]
            w2 = words[i + 1]
            for klass in ["pos", "neg"]:
                num = self.bigramCounts[klass][(w1, w2)] + 1

                vocab_size = len(self.vocab)
                den = self.bi_words_cnt_per_klass[klass] + vocab_size
                prob_words_given_klass = math.log(float(num) / den)

                if klass not in class_scores:
                    class_scores[klass] = float(self.bi_classifier_freq[klass]) / vocab_size
                class_scores[klass] += prob_words_given_klass

                print("prob_words_given_klass", prob_words_given_klass)

        res = max(class_scores, key=class_scores.get)
        print("res:", res)
        return res

    def classify(self, words):
        """ TODO
            'words' is a list of words to classify. Return 'pos' or 'neg' classification.
        """
        if self.FILTER_STOP_WORDS:
            words = self.filterStopWords(words)
            self.classify_gen(words)
        if self.BOOLEAN_NB:
            self.classify_binary(words)
        if self.BEST_MODEL:
            self.classify_best(words)


    def classify_binary(self, words):
        """ TODO
            'words' is a list of words to classify. Return 'pos' or 'neg' classification.
        """
        if self.FILTER_STOP_WORDS:
            words = self.filterStopWords(words)

        class_scores = {}

        doc_tot = defaultdict(int)
        for word in words:
            if (word, "pos") in self.doc_cnt_w_k:
                doc_tot["pos"] += self.doc_cnt_w_k[(word, "pos")]
            if (word, "neg") in self.doc_cnt_w_k:
                doc_tot["neg"] += self.doc_cnt_w_k[(word, "neg")]

        for word in words:
            for klass in ["pos", "neg"]:
                if (word, klass) in self.doc_cnt_w_k:
                    num = self.doc_cnt_w_k[(word, klass)]
                    den = doc_tot[klass]
                    p_word = float(num) / den
                else:
                    p_word = 0

                if klass not in class_scores:
                    class_scores[klass] = 0
                class_scores[klass] += p_word

        return max(class_scores, key=class_scores.get)

    def addExamplebest(self, klass, words):
        # Cleaning
        words = self.filterStopWords(words)
        words = self.preprocess(words)

        self.bi_classifier_freq[klass] += 1

        # Building unigram dict, removing UNK
        self.count_unigrams(words)
        # self.replace_unk_train_words()
        # self.recount_unigrams()

        # Build bigram dict
        self.count_bigrams(klass)
        return

    def addExample(self, klass, words):
        """
         * TODO
         * Train your model on an example document with label klass ('pos' or 'neg') and
         * words, a list of strings.
         * You should store whatever data structures you use for your classifier
         * in the NaiveBayes class.
         * Returns nothing
        """
        if self.FILTER_STOP_WORDS:
            words = self.filterStopWords(words)
        elif self.BEST_MODEL:
            print("Training bigram")
            self.addExamplebest(klass, words)
            return

        # the frequency of the classifier across docs
        if klass not in self.classifier_freq:
            self.classifier_freq[klass] = 0
        self.classifier_freq[klass] += 1

        l = len(self.doc_unique_words)
        self.doc_unique_words[l] = set()

        counted = {}
        for word in words:
            # the frequency of words
            if word not in self.word_freq:
                self.word_freq[word] = 0
            self.word_freq[word] += 1

            # frequency of words per classifier
            if klass not in self.word_freq_per_klass:
                self.word_freq_per_klass[klass] = defaultdict(int)
            self.word_freq_per_klass[klass][word] += 1

            # the number of words per classifier
            if klass not in self.word_cnt_per_klass:
                self.word_cnt_per_klass[klass] = 0
            self.word_cnt_per_klass[klass] += 1

            self.doc_unique_words[l].add(word)

            # Write code here


    # END TODO (Modify code beyond here with caution)
    #############################################################################


    def readFile(self, fileName):
        """
         * Code for reading a file.  you probably don't want to modify anything here,
         * unless you don't like the way we segment files.
        """
        contents = []
        f = open(fileName)
        for line in f:
            contents.append(line)
        f.close()
        result = self.segmentWords('\n'.join(contents))
        return result

  
    def segmentWords(self, s):
        """
         * Splits lines on whitespace for file reading
        """
        return s.split()

  
    def trainSplit(self, trainDir):
        """Takes in a trainDir, returns one TrainSplit with train set."""
        split = self.TrainSplit()
        posTrainFileNames = os.listdir('%s/pos/' % trainDir)
        negTrainFileNames = os.listdir('%s/neg/' % trainDir)
        for fileName in posTrainFileNames:
            example = self.Example()
            example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
            example.klass = 'pos'
            split.train.append(example)
        for fileName in negTrainFileNames:
            example = self.Example()
            example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
            example.klass = 'neg'
            split.train.append(example)
        return split

    def train(self, split):
        for example in split.train:
            words = example.words
            if self.FILTER_STOP_WORDS:
                words =  self.filterStopWords(words)
            self.addExample(example.klass, words)


    def crossValidationSplits(self, trainDir):
        """Returns a lsit of TrainSplits corresponding to the cross validation splits."""
        splits = []
        posTrainFileNames = os.listdir('%s/pos/' % trainDir)
        negTrainFileNames = os.listdir('%s/neg/' % trainDir)
        #for fileName in trainFileNames:
        for fold in range(0, self.numFolds):
            split = self.TrainSplit()
            for fileName in posTrainFileNames:
              example = self.Example()
              example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
              example.klass = 'pos'
              if fileName[2] == str(fold):
                  split.test.append(example)
              else:
                  split.train.append(example)
            for fileName in negTrainFileNames:
                example = self.Example()
                example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
                example.klass = 'neg'
                if fileName[2] == str(fold):
                    split.test.append(example)
                else:
                    split.train.append(example)
            yield split


    def test(self, split):
        """Returns a list of labels for split.test."""
        labels = []
        for example in split.test:
            words = example.words
            if self.FILTER_STOP_WORDS:
                words =  self.filterStopWords(words)
            guess = self.classify(words)
            labels.append(guess)
        return labels
  
    def buildSplits(self, args):
        """Builds the splits for training/testing"""
        trainData = []
        testData = []
        splits = []
        trainDir = args[0]
        if len(args) == 1:
            print('[INFO]\tPerforming %d-fold cross-validation on data set:\t%s' % (self.numFolds, trainDir))

            posTrainFileNames = os.listdir('%s/pos/' % trainDir)
            negTrainFileNames = os.listdir('%s/neg/' % trainDir)
            for fold in range(0, self.numFolds):
                split = self.TrainSplit()
                for fileName in posTrainFileNames:
                    example = self.Example()
                    example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
                    example.klass = 'pos'
                    if fileName[2] == str(fold):
                        split.test.append(example)
                    else:
                        split.train.append(example)
                for fileName in negTrainFileNames:
                    example = self.Example()
                    example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
                    example.klass = 'neg'
                    if fileName[2] == str(fold):
                        split.test.append(example)
                    else:
                        split.train.append(example)
                splits.append(split)
        elif len(args) == 2:
            split = self.TrainSplit()
            testDir = args[1]
            print('[INFO]\tTraining on data set:\t%s testing on data set:\t%s' % (trainDir, testDir))
            posTrainFileNames = os.listdir('%s/pos/' % trainDir)
            negTrainFileNames = os.listdir('%s/neg/' % trainDir)
            for fileName in posTrainFileNames:
                example = self.Example()
                example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
                example.klass = 'pos'
                split.train.append(example)
            for fileName in negTrainFileNames:
                example = self.Example()
                example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
                example.klass = 'neg'
                split.train.append(example)

            posTestFileNames = os.listdir('%s/pos/' % testDir)
            negTestFileNames = os.listdir('%s/neg/' % testDir)
            for fileName in posTestFileNames:
                example = self.Example()
                example.words = self.readFile('%s/pos/%s' % (testDir, fileName))
                example.klass = 'pos'
                split.test.append(example)
            for fileName in negTestFileNames:
                example = self.Example()
                example.words = self.readFile('%s/neg/%s' % (testDir, fileName))
                example.klass = 'neg'
                split.test.append(example)
            splits.append(split)
        return splits
  
    def filterStopWords(self, words):
        """Filters stop words."""
        filtered = []
        for word in words:
            if not word in self.stopList and word.strip() != '':
                filtered.append(word)
        return filtered

def test10Fold(args, FILTER_STOP_WORDS, BOOLEAN_NB, BEST_MODEL):
    nb = NaiveBayes()
    splits = nb.buildSplits(args)
    avgAccuracy = 0.0
    fold = 0
    classifier = None
    for split in splits:
        classifier = NaiveBayes()
        classifier.FILTER_STOP_WORDS = FILTER_STOP_WORDS
        classifier.BOOLEAN_NB = BOOLEAN_NB
        classifier.BEST_MODEL = BEST_MODEL
        accuracy = 0.0
        for example in split.train:
            words = example.words
            classifier.addExample(example.klass, words)

        for example in split.test:
            words = example.words
            guess = classifier.classify(words)
            if example.klass == guess:
                accuracy += 1.0
        accuracy = accuracy / len(split.test)
        avgAccuracy += accuracy
        print('[INFO]\tFold %d Accuracy: %f' % (fold, accuracy))
        fold += 1
    avgAccuracy = avgAccuracy / fold
    print('[INFO]\tAccuracy: %f' % avgAccuracy)

    # interpret the decision rule of the model of the last fold
    pos_signal_words, neg_signal_words = analyze_model(classifier)
    print('[INFO]\tWords for pos class: %s' % ','.join(pos_signal_words))
    print('[INFO]\tWords for neg class: %s' % ','.join(neg_signal_words))

    
def classifyFile(FILTER_STOP_WORDS, BOOLEAN_NB, BEST_MODEL, trainDir, testFilePath):
    classifier = NaiveBayes()
    classifier.FILTER_STOP_WORDS = FILTER_STOP_WORDS
    classifier.BOOLEAN_NB = BOOLEAN_NB
    classifier.BEST_MODEL = BEST_MODEL
    trainSplit = classifier.trainSplit(trainDir)
    classifier.train(trainSplit)
    testFile = classifier.readFile(testFilePath)
    print(classifier.classify(testFile))
    
def main():
    FILTER_STOP_WORDS = False
    BOOLEAN_NB = False
    BEST_MODEL = False
    (options, args) = getopt.getopt(sys.argv[1:], 'fbm')
    if ('-f','') in options:
        FILTER_STOP_WORDS = True
    elif ('-b','') in options:
        BOOLEAN_NB = True
    elif ('-m','') in options:
        BEST_MODEL = True

    if len(args) == 2 and os.path.isfile(args[1]):
        classifyFile(FILTER_STOP_WORDS, BOOLEAN_NB, BEST_MODEL, args[0], args[1])
    else:
        test10Fold(args, FILTER_STOP_WORDS, BOOLEAN_NB, BEST_MODEL)


def analyze_model(nb_classifier):
    # TODO: This function takes a <nb_classifier> as input, and outputs two word list <pos_signal_words> and
    #  <neg_signal_words>. <pos_signal_words> is a list of 10 words signaling the positive klass, and <neg_signal_words>
    #  is a list of 10 words signaling the negative klass.
    pos_signal_words = dict(sorted(nb_classifier.word_freq_per_klass["pos"].items(), key=lambda item: item[1], reverse=True))
    neg_signal_words = dict(sorted(nb_classifier.word_freq_per_klass["neg"].items(), key=lambda item: item[1], reverse=True))

    i = 0
    pos = {}
    neg = {}
    for (k, v) in pos_signal_words.items():
        if i > 10:
            break
        pos[k] = v
        i += 1
    i = 0
    for (k, v) in neg_signal_words.items():
        if i > 10:
            break
        neg[k] = v
        i += 1
    return pos, neg


if __name__ == "__main__":
    main()
