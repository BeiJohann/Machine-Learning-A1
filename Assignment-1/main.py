import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch import Tensor, LongTensor, zeros

from datenloader import open_data, get_numeric_representations_sents

import daten_loader as a



tensor, vocabulary = get_numeric_representations_sents(train_sent)

type(train_sent[1])

print(tensor[0][0])


def char_to_index(char: object) -> object:
    return vocabulary.index(char)


print(char_to_index('n'))


def char_to_tensor(char: object) -> object:
    ret = torch.zeros(1, len(vocabulary))
    ret[0][char_to_index(char)] = 1
    return ret


def sentence_to_tensor(sentence: object) -> object:
    ret = torch.zeros(100, 1, len(vocabulary))
    for i, char in range(100):
        ret[i][0][char_to_index(char)] = 1
    return ret


def create_one_hot_vectors(sequences, vocabulary):
    # one_hot_vectors = torch.Tensor((len(sequences), len(vocabulary)))
    one_hot_vectors = []
    for seq in sequences:
        # one_hot_vec = zeros(len(vocabulary))
        one_hot_vec = [-1.0] * len(vocabulary)
        for char_int in seq:
            char_int = int(char_int.item())
            if char_int == -1:
                break
            one_hot_vec[char_int - 1] = 1.0
        one_hot_vectors.append(one_hot_vec)

    # one_hot_vectors = LongTensor(one_hot_vectors)
    return one_hot_vectors


hotvec = create_one_hot_vectors(tensor, voc)
print(len(hotvec))

# %%

len(voc)
