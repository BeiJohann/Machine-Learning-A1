# Imports
import torch.nn as nn
import torch
import argparse
import os
import joblib

from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

from data_loader import open_data, get_vocabulary, convert_into_num_tensor, convert_into_clipped

# Parameters
BATCH_SIZE = 300
SEQUENZ_LENGTH = 100
DEVICE = 'cuda:1'

# The NN
class GRUNet(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, num_layers, output_size, dropout=0.01):
        super().__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, input_size)
        self.gru = nn.GRU(input_size, hidden_size, num_layers=self.num_layers, dropout=dropout)
        self.lin1 = nn.Linear(hidden_size * input_size, output_size)

    def forward(self, sequence):
        # print(sequence.size())
        output = self.embed(sequence)
        # print(output.size())
        hidden_layer = self.init_hidden(len(sequence[0]))
        output, _ = self.gru(output, hidden_layer)
        output = output.contiguous().view(-1, self.hidden_size * len(sequence[0]))
        # print(output.size())
        output = self.lin1(output)
        return output

    def set_dev(self, dev):
        self.dev = dev

    def init_hidden(self, seq_len):
        return torch.zeros(self.num_layers, seq_len, self.hidden_size).float().to(self.dev)


def test(model, mapping, test_x, test_y):
    all_pred = 0
    correct_pred_increment = 0
    intotal_correct = 0

    for x, y in zip(test_x, test_y):
        # get 100 clipped and converted test_x for x
        x, _ = convert_into_clipped([x],[y])

        test_x_tensor = convert_into_num_tensor(x, mapping)
        # get 100 padded sequences
        padded_sequences = pad_sequence(
            test_x_tensor, batch_first=True, padding_value=0.0)
        #print(padded_sequences[:102])

        #collecting data for evaluation
        num_until_correct = -1
        correct_pred_per_sentence = 0
        sentence_iterator = 0

        for seq in padded_sequences:
            all_pred += 1
            sentence_iterator += 1
            output = model(torch.stack([seq]).long().to(DEVICE))
            _, prediction = torch.max(output.data, dim=1)
            #print('for: ', seq, 'pred: ',prediction,'aim: ', y)
            if prediction == y:
                correct_pred_per_sentence += 1
                if num_until_correct == -1:
                    num_until_correct = sentence_iterator
        if num_until_correct == -1 :
            correct_pred_increment += 100
        else:
            correct_pred_increment += num_until_correct
        intotal_correct += correct_pred_per_sentence

    print('Average incremets after prediction: ', correct_pred_increment / len(test_y))
    print('Average propability for predicting right: ', intotal_correct / all_pred)


if __name__ == '__main__':
    # commandline arguments
    parser = argparse.ArgumentParser( description="Train a recurrent network for language identification")
    parser.add_argument("-L", "--load_model", dest="model_path", type=str, help="Specify the model path")
    parser.add_argument("-X", "--test_sent", dest="x_file", type=str, help="Specify the file from where the sentences are loaded")
    parser.add_argument("-Y", "--test_label", dest="y_file", type=str, help="Specify the file from where the Labels are loaded")
    args = parser.parse_args()

    # open the data
    test_x, test_y, list_of_lang = open_data('my_data_test')

    # open the vocabs and mapping
    mapping = joblib.load('./data/my_data_mapping.sav')
    vocabulary = joblib.load('./data/my_data_vocabulary.sav')

    # put the labels into indexes
    test_y =[list_of_lang.index(label) for label in test_y ]

    # Initializing the Network
    vocab_size = len(vocabulary) + 1
    print('number of character: ',vocab_size)
    output_size = len(list_of_lang)

    # Loading the network
    print('Loading the Net')
    model = torch.load('savedNet.pt')
    # is Cuda available
    if not(torch.cuda.is_available()):
        DEVICE = 'cpu'
    # put everything to the choosen device
    dev = torch.device(DEVICE)
    model = model.to(dev)
    model.set_dev(dev)

    # test the model
    print('testing')
    test(model, mapping, test_x, test_y)