import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import torch.optim as optim

from torch import Tensor, LongTensor, zeros
from torch.autograd import Variable

from datenloader import open_data

'''
# The Real RNN, but it doesnt work
class GRUNet(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, num_layers, output_size, dropout=0.01):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, input_size)
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, dropout=dropout)
        self.lin1 = nn.Linear(hidden_size, output_size)

        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, sequence):
        hidden_layer = self.init_hidden(len(sequence[0]))
        #print(sequence.size(),'1')
        output = self.embed(sequence)
        #print(output.size(),'2')
        output, _ = self.gru(output, hidden_layer)
        #print(output.size(),'3')
        #take the take the final row of the middle dim. reduce it with squeez
        #output = output.contiguous().view(-1, self.hidden_size * len(sequence[0]))
        output = output[:,len(sequence[0])-1:,:]
        output = output.squeeze(dim=1)
        #print(output.size(),'4')
        output = self.lin1(output)
        return output

    def set_dev(self, dev):
        self.dev = dev

    def init_hidden(self, seq_len):
        return torch.zeros(self.num_layers, seq_len, self.hidden_size).float().to(self.dev)
'''

# from datenloader import get_numeric_representations_sents


train_x = []

data = [line.rstrip("\n") for line in open('./myTrains.txt')]
train_x = [list(line.split('\t')[0]) for line in data]
train_y = [line.split('\t')[1] for line in data]

quantity_of_lines = len(train_x)
print(quantity_of_lines)

print(train_x[1])
# print(train_sent[4]+'--'+train_label[4])

listOfLang = [line.split()[0] for line in open('./data/testLang.txt')]
print(listOfLang)


def get_vocabulary(sents):
    #sents = [[x for x in sent] for sent in sents]
    #print(sents)
    vocabulary = list(set(sum(sents, [])))
    #print(vocabulary)
    char_to_int_mapping = {char: i + 1 for i, char in enumerate(vocabulary)}
    return char_to_int_mapping, vocabulary


char_to_int_mapping, vocabulary = get_vocabulary(train_x)



def char_to_index(char: object) -> object :
    return vocabulary.index(char)


# print(char_to_index('n'))


def char_to_tensor(char: object) -> object :
    ret = torch.zeros(1, len(vocabulary))
    ret[0][char_to_index(char)] = 1
    return ret


def sentence_to_tensor(sentence: object) -> object :
    ret = torch.zeros(100, 1, len(vocabulary))
    for i, char in enumerate(sentence) :
        ret[i][0][char_to_index(char)] = 1
    return ret


print(sentence_to_tensor(train_x[0]).size())


class Netz(nn.Module) :
    def __init__(self, input, hiddens, output) :
        super(Netz, self).__init__()
        self.hiddens = hiddens
        self.hid = nn.Linear(input + hiddens, hiddens)
        self.out = nn.Linear(input + hiddens, output)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden) :
        x = torch.cat((x, hidden), 1)
        new_hidden = self.hid(x)
        output = self.logsoftmax(self.out(x))
        return output, new_hidden

    def initHidden(self) :
        return Variable(torch.zeros(1, self.hiddens))



model = Netz(len(vocabulary), 128, 10)


def lang_from_output(out) :
    _, i = out.data.topk(1)
    return listOfLang[i[0][0]]


def get_all_index_of_lang(lang) :
    return [i for i, e in enumerate(train_y) if e == lang]


# print(get_all_index_of_lang('zea'))
# print(train_label[100:])

def get_train_data(i) :
    lang = train_y[i]
    sentence = train_x[i]
    sentence_tensor = Variable(sentence_to_tensor(sentence))
    lang_tensor = Variable(torch.LongTensor([listOfLang.index(lang)]))
    return lang, sentence, lang_tensor, sentence_tensor


criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters())


def train(lang_tensor, sentence_tensor) :
    hidden = model.initHidden()
    model.zero_grad()
    #print(sentence_tensor.size()[0])
    for i in range(sentence_tensor.size()[0]) :
        output, hidden = model(sentence_tensor[i], hidden)
    loss = criterion(output, lang_tensor)
    loss.backward()
    optimizer.step()
    #for i in model.parameters() :
    #    i.data.add_(-0.05, i.grad.data)

    return output, loss


lang, sentence, lang_tensor, sentence_tensor = get_train_data(1)
#print(lang, lang_tensor.size())

avg = []
sum = 0
for i in range(1, quantity_of_lines) :
    lang, sentence, lang_tensor, sentence_tensor = get_train_data(i)
    output, loss = train(lang_tensor, sentence_tensor)
    sum = sum + loss.data

    if i % 100 == 0 :
        avg.append(sum / 100)
        sum = 0
        print(i / 500, "% done")

plt.figure()
plt.plot(avg)
plt.show()
