#!/usr/bin/python

"""
Implement a trigrm HMM and viterbi here.
You model should output the final tags similar to `viterbi.pl`.

Usage:  python train_trigram_hmm.py tags text > tags

"""
import sys, re
import math
from collections import defaultdict

trans_seq = []
emit_seq = []

TAG_FILE = sys.argv[1]  # tag file
TOKEN_FILE = sys.argv[2]  # token file
INPUT_FILE = "data/ptb.23.txt"  # input file
OOV_WORD = "OOV"
INIT_STATE = "init"
FINAL_STATE = "final"


def deleted_interpolation(trans_tri, trans_total_tri, trans_bi, trans_tot_bi, emissions_tot, tag_tot):
    lambda1 = 0
    lambda2 = 0
    lambda3 = 0
    for prev_prev_tag in trans_tri:
        for prev_tag in trans_tri[prev_prev_tag]:
            for tag in trans_tri[prev_prev_tag][prev_tag]:
                # C(t1,t2,t3)
                trigram_count = trans_tri[prev_prev_tag][prev_tag][tag]
                # C(t1,t2)
                trigram_tot_count = trans_total_tri[prev_prev_tag][prev_tag]

                # C(t2,t3)
                bigram_count = trans_bi[prev_tag][tag]
                # C(t2)
                bigram_tot_count = trans_tot_bi[prev_tag]

                # C(t3)
                unigram_count = emissions_tot[tag]
                # N
                unigram_tot_count = tag_tot

                val1 = 0
                val2 = 0
                val3 = 0
                if trigram_tot_count > 1:
                    val1 = float(trigram_count - 1) / (trigram_tot_count - 1)
                if bigram_tot_count > 1:
                    val2 = float(bigram_count - 1) / (bigram_tot_count - 1)
                if unigram_tot_count > 1:
                    val3 = float(unigram_count - 1) / (unigram_tot_count - 1)

                if val1 > val2 and val1 > val3:
                    lambda1 += trigram_count
                elif val2 > val1 and val2 > val3:
                    lambda2 += trigram_count
                elif val3 > val1 and val3 > val2:
                    lambda3 += trigram_count

    l_sum = lambda1 + lambda2 + lambda3
    lambda1 = float(lambda1) / l_sum
    lambda2 = float(lambda2) / l_sum
    lambda3 = float(lambda3) / l_sum
    return (lambda1, lambda2, lambda3)


def train_trigram_hmm():
    global OOV_WORD
    global INIT_STATE
    global FINAL_STATE
    voc = {}
    tag_tot = 0

    emissions = {}
    emissions_tot = defaultdict(int)

    # C([prev_prev_tag][prev_tag][tag])
    trans_tri = {}
    # count of each tri-tag
    trans_total_tri = defaultdict(lambda: defaultdict(int))

    # C([prev_tag][tag])
    trans_bi = {}
    # count of each bi-tag
    trans_tot_bi = defaultdict(int)

    with open(TAG_FILE) as tagFile, open(TOKEN_FILE) as tokenFile:
        for tagString, tokenString in zip(tagFile, tokenFile):
            tags = re.split("\s+", tagString.rstrip())
            tokens = re.split("\s+", tokenString.rstrip())
            pairs = zip(tags, tokens)

            prev_tag = INIT_STATE
            prev_prev_tag = INIT_STATE
            for (tag, token) in pairs:
                if token not in voc:
                    voc[token] = 1
                    token = OOV_WORD
                if tag not in emissions:
                    emissions[tag] = defaultdict(int)
                if prev_tag not in trans_bi:
                    trans_bi[prev_tag] = defaultdict(int)
                if prev_prev_tag not in trans_tri:
                    trans_tri[prev_prev_tag] = defaultdict(lambda: defaultdict(int))
                if prev_tag not in trans_tri[prev_prev_tag]:
                    trans_tri[prev_prev_tag][prev_tag] = defaultdict(int)

                emissions[tag][token] += 1
                emissions_tot[tag] += 1

                # trigram transitions
                # c(s|u,v)
                trans_tri[prev_prev_tag][prev_tag][tag] += 1
                trans_total_tri[prev_prev_tag][prev_tag] += 1

                # bigram transitions
                # c(u|v)
                trans_bi[prev_tag][tag] += 1
                trans_tot_bi[prev_tag] += 1

                if prev_tag != INIT_STATE:
                    prev_prev_tag = prev_tag
                prev_tag = tag
                tag_tot += 1

            # stop probability for each sentence
            if prev_prev_tag not in trans_tri:
                trans_tri[prev_prev_tag] = defaultdict(lambda: defaultdict(int))
            if prev_tag not in trans_tri[prev_prev_tag]:
                trans_tri[prev_prev_tag][prev_tag] = defaultdict(int)
            trans_tri[prev_prev_tag][prev_tag][FINAL_STATE] += 1
            trans_total_tri[prev_prev_tag][prev_tag] += 1

            if prev_tag not in trans_bi:
                trans_bi[prev_tag] = defaultdict(int)
            trans_bi[prev_tag][FINAL_STATE] += 1
            trans_tot_bi[prev_tag] += 1

    # applying interpolation
    lambda1, lambda2, lambda3 = deleted_interpolation(trans_tri, trans_total_tri, trans_bi, trans_tot_bi, emissions_tot,
                                                      tag_tot)

    for prev_prev_tag in trans_tri:
        for prev_tag in trans_tri[prev_prev_tag]:
            for tag in trans_tri[prev_prev_tag][prev_tag]:
                trigram_count = trans_tri[prev_prev_tag][prev_tag][tag]
                trigram_tot_count = trans_total_tri[prev_prev_tag][prev_tag]
                p_trigram = float(trigram_count) / trigram_tot_count

                bigram_count = trans_bi[prev_tag][tag]
                bigram_tot_count = trans_tot_bi[prev_tag]
                p_bigram = float(bigram_count) / bigram_tot_count

                unigram_count = emissions_tot[tag]
                unigram_tot_count = tag_tot
                p_unigram = float(unigram_count) / unigram_tot_count

                tri_prob = lambda1 * p_trigram + lambda2 * p_bigram + lambda3 * p_unigram
                bi_prob = lambda2 * p_bigram + lambda3 * p_unigram
                trans_seq.append((prev_prev_tag, prev_tag, tag, tri_prob, bi_prob))

    for tag in emissions:
        for token in emissions[tag]:
            emit_seq.append((tag, token, float(emissions[tag][token]) / (emissions_tot[tag])))


def viterbi_trigram():
    global OOV_WORD
    global INIT_STATE
    global FINAL_STATE
    global trans_seq
    global emit_seq

    states = {}
    A_tri = {}
    A_bi = {}
    B = {}
    voc = {}
    for tup in trans_seq:
        prev_prev_tag, prev_tag, tag, tri_prob, bi_prob = tup
        if prev_prev_tag not in A_tri:
            A_tri[prev_prev_tag] = {}
        if prev_tag not in A_tri[prev_prev_tag]:
            A_tri[prev_prev_tag][prev_tag] = {}
        A_tri[prev_prev_tag][prev_tag][tag] = math.log(float(tri_prob))

        if prev_tag not in A_bi:
            A_bi[prev_tag] = {}
        A_bi[prev_tag][tag] = math.log(float(bi_prob))

        states[prev_prev_tag] = 1
        states[prev_tag] = 1
        states[tag] = 1

    for tup in emit_seq:
        tag, token, em_prob = tup
        if tag not in B:
            B[tag] = {}
        B[tag][token] = math.log(float(em_prob))
        voc[token] = 1
        states[tag] = 1

    with open(INPUT_FILE) as input_fp:
        lines = input_fp.readlines()
        for line in lines:
            line = line.split(' ')
            n = len(line)
            line.insert(0, "")
            vit_arr = {}
            backtrace = {}
            vit_arr[(0, INIT_STATE, INIT_STATE)] = 0.0
            for i in range(1, n+1):
                token = line[i]
                if token not in voc:
                    token = OOV_WORD

                for prev_prev_st in states:
                    for prev_st in states:
                        for current_st in states:
                            if current_st in B and token in B[current_st] and (i - 1, prev_prev_st, prev_st) in vit_arr:
                                if prev_prev_st in A_tri and prev_st in A_tri[prev_prev_st] and current_st in A_tri[prev_prev_st][prev_st]:
                                    v = vit_arr[(i - 1, prev_prev_st, prev_st)] + A_tri[prev_prev_st][prev_st][current_st] + B[current_st][token]
                                    if (i, prev_st, current_st) not in vit_arr or v > vit_arr[(i, prev_st, current_st)]:
                                        vit_arr[(i, prev_st, current_st)] = v
                                        backtrace[(i, prev_st, current_st)] = prev_prev_st

                                elif prev_st in A_bi and current_st in A_bi[prev_st]:
                                    v = vit_arr[(i - 1, prev_prev_st, prev_st)] + A_bi[prev_st][current_st] + B[current_st][token]
                                    if (i, prev_st, current_st) not in vit_arr or v > vit_arr[(i, prev_st, current_st)]:
                                        vit_arr[(i, prev_st, current_st)] = v
                                        backtrace[(i, prev_st, current_st)] = prev_prev_st

            goal = float('-inf')
            current_st = INIT_STATE
            prev_st = INIT_STATE
            found_goal = False
            for x in states:
                for y in states:
                    v = float('-inf')
                    if (n - 1, x, y) in vit_arr:
                        if x in A_tri and y in A_tri[x] and FINAL_STATE in A_tri[x][y]:
                            v = vit_arr[(n - 1, x, y)] + A_tri[x][y][FINAL_STATE]
                        elif y in A_bi and FINAL_STATE in A_bi[y]:
                            v = vit_arr[(n - 1, x, y)] + A_bi[y][FINAL_STATE]

                        if not found_goal or v > goal:
                            goal = v
                            current_st = y
                            prev_st = x
                            found_goal = True

            if found_goal:
                t = [current_st, prev_st]

                for k in range(n - 2, 1, -1):
                    bt = backtrace[(k + 1, prev_st, current_st)]
                    t.append(bt)
                    current_st = prev_st
                    prev_st = bt
                t.reverse()
                print(' '.join(t))
            else:
                print()


if __name__ == "__main__":
    train_trigram_hmm()
    viterbi_trigram()
