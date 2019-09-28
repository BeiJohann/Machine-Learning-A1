import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch import Tensor, LongTensor, zeros


def open_data(self):
    train_sent = []

    data = [line.rstrip("\n") for line in open('./myTrains.txt')]
    train_sent = [list(line.split('\t')[0]) for line in data]
    train_label = [list(line.split('\t')[1]) for line in data]
    # print(train_sent[4]+'--'+train_label[4])*




def get_numeric_representations_sents(sents):
    '''
        We first convert strings to vectors. To do that, I am using ord() here - can replace
        it by something else later.
        Then we convert the vectors to tensors.
    '''
    # using ord() to convert words to integers/numeric representation and creating numeric representations through that
    # vectors = []
    # for sent in sents:
    #     vec = [ord(ch) for ch in sent]
    #     vectors.append(vec)

    # using vocabulary to get word-to-integer mapping and creating numeric representations through that
    # sent_char_lists = [list(sent) for sent in sents]
    vocabulary = list(set(sum(sents, [])))
    char_to_int_mapping = {char: i + 1 for i, char in enumerate(vocabulary)}
    # print(char_to_int_mapping)
    sent_vectors = [[char_to_int_mapping[char]
                     for char in list(sent)] for sent in sents]

    sent_vectors_tensors = [Tensor(vec) for vec in sent_vectors]
    return sent_vectors_tensors, vocabulary