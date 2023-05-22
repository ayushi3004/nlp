#!/usr/bin/python

"""
Implement the Viterbi algorithm in Python (no tricks other than logmath!), given an
HMM, on sentences, and outputs the best state path.
Please check `viterbi.pl` for reference.

Usage:  python viterbi.py hmm-file < text > tags

special keywords:
 $init_state   (an HMM state) is the single, silent start state
 $final_state  (an HMM state) is the single, silent stop state
 $OOV_symbol   (an HMM symbol) is the out-of-vocabulary word
"""
import math
import re
import sys
from collections import defaultdict

init_state = "init"
final_state = "final"
OOV_symbol = "OOV"
verbose = 0
A = {}
B = {}
states = defaultdict(int)
voc = defaultdict(int)
hmm_file = sys.argv[1]
data_file = sys.stdin
start_st = ""

def hmm_viterbi():
    global start_st
    with open(hmm_file) as hmm_file_fp:
        lines = hmm_file_fp.readlines()
        for prob_seq in lines:
            tag_type, prev_tag, tag, prob = re.split("\s+", prob_seq.rstrip())
            if tag_type == "trans":
                if prev_tag not in A:
                    A[prev_tag] = defaultdict(float)
                A[prev_tag][tag] = math.log(float(prob))
                states[prev_tag] = 1
                states[tag] = 1

            else:
                if prev_tag not in B:
                    B[prev_tag] = defaultdict(float)
                B[prev_tag][tag] = math.log(float(prob))
                states[prev_tag] = 1
                voc[tag] = 1

        start_st = prev_tag

def data_viterbi():
    goal = 0
    lines = data_file.readlines()
    for line in lines:
        w = line.split(' ')
        n = len(w)
        w.insert(0, "")
        vit_arr = {}
        backtrace = {}
        vit_arr[0] = {}
        vit_arr[0][init_state] = 0.0
        for i in range(1, n+1):
            if w[i] not in voc:
                w[i] = OOV_symbol

            for current_st in states:
                for prev_st in states:
                    if(prev_st in A and current_st in A[prev_st]
                    and current_st in B and w[i] in B[current_st]
                    and (i-1) in vit_arr and prev_st in vit_arr[i-1]):
                        v = vit_arr[i - 1][prev_st] + A[prev_st][current_st] + B[current_st][w[i]]

                        if (i not in vit_arr or current_st not in vit_arr[i]) or (v > vit_arr[i][current_st]):
                            if i not in vit_arr:
                                vit_arr[i] = {}
                            vit_arr[i][current_st] = v

                            if i not in backtrace:
                                backtrace[i] = {}
                            backtrace[i][current_st] = prev_st

        found_goal = False
        current_st = start_st
        for prev_st in states:
            if(prev_st in A and final_state in A[prev_st]
            and n in vit_arr and prev_st in vit_arr[n]):
                v = vit_arr[n][prev_st] + A[prev_st][final_state]
                if (not found_goal) or (v > goal):
                    goal = v
                    found_goal = True
                    current_st = prev_st

        if found_goal:
            t = []
            for i in range(n, 0, -1):
                t.insert(0, current_st)
                current_st = backtrace[i][current_st]
            print(" ".join(t))
        else:
            print()


hmm_viterbi()
data_viterbi()