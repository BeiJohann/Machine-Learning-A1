
import torch.nn as nn
import torch.nn.functional as F
import torch

from data_loader import open_data, get_vocabulary, convert_into_num_tensor


if __name__ == '__main__':
    #open the data
    train_x, train_y, list_of_lang = open_data()

    #generate the vocabs
    mapping, vocabulary = get_vocabulary(train_x)

    numeric, train_y = convert_into_num_tensor(train_x, train_y, mapping)
    print(numeric[:100])


class GRUNet(nn.Module):
    def __init__(self, vocab_size, seq_len, input_size, hidden_size, num_layers, output_size, dropout=0.01):
        super().__init__()
        self.num_layers = 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(vocab_size, input_size)
        self.gru = nn.GRU(input_size, hidden_size, num_layers=self.num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size * seq_len, output_size)

    def forward(self, sequence):
        output = self.emb(sequence)
        hidden_layer = self.init_hidden(len(sequence[0]))
        output, _ = self.gru(output, hidden_layer)
        output = output.contiguous().view(-1, self.hidden_size * len(sequence[0]))
        output = self.fc(output)
        return output

    def init_hidden(self, seq_len):
        return torch.zeros(self.num_layers, seq_len, self.hidden_size).float()