"""
COMS 4705 Natural Language Processing Spring 2022 Zhou Yu
Adapted from COMS 4705 Natural Language Processing Spring 2021 Kathy McKeown
Homework 3: Sentiment Classification with Naive Bayes - Models File
<YOUR NAME HERE>
<YOUR UNI HERE>
"""

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn


class RecurrentNetwork(nn.Module):
    def __init__(self):
        super(RecurrentNetwork, self).__init__()

        ########## YOUR CODE HERE ##########
        # TODO: Here, create any layers and attributes your network needs.
        raise NotImplementedError

    # x is a PaddedSequence for an RNN
    def forward(self, x):
        ########## YOUR CODE HERE ##########
        # TODO: Fill in the forward pass of your neural network.
        # TODO: (The backward pass will be performed by PyTorch magic for you!)
        # TODO: Your architecture should...
        # TODO: 1) Put the words through an Embedding layer (which was initialized with the pretrained embeddings);
        # TODO: 2) Feed the sequence of embeddings through a 2-layer RNN; and
        # TODO: 3) Feed the last output state into a dense layer to become a 4-vector of values, one for each class
        raise NotImplementedError
