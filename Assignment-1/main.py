# Imports
import torch.nn as nn
import torch

from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

from data_loader import open_data, get_vocabulary, convert_into_num_tensor

# Parameters
LEARNING_RATE = 0.01
HIDDEN_SIZE = 100
NUM_LAYERS = 2
INPUT_SIZE = 200
EPOCH = 1
BATCH_SIZE = 300
SEQUENZ_LENGTH = 100

# The NN
class GRUNet(nn.Module):
    def __init__(self, vocab_size, seq_len, input_size, hidden_size, num_layers, output_size, dropout=0.01):
        super().__init__()
        self.num_layers = 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, input_size)
        self.gru = nn.GRU(input_size, hidden_size, num_layers=self.num_layers, dropout=dropout)
        self.lin1 = nn.Linear(hidden_size * seq_len, output_size)

    def forward(self, sequence):
        #print(sequence.size())
        output = self.embed(sequence)
        #print(output.size())
        hidden_layer = self.init_hidden(len(sequence[0]))
        output, _ = self.gru(output, hidden_layer)
        output = output.contiguous().view(-1, self.hidden_size * len(sequence[0]))
        #print(output.size())
        output = self.lin1(output)
        return output

    def init_hidden(self, seq_len):
        return torch.zeros(self.num_layers, seq_len, self.hidden_size).float()


def padded_batching(train_x, train_y, batch_size):
    #REMIND SHUFFELING
    for i in range(0,len(train_x),batch_size):
        x_batch = train_x[i: i + batch_size]
        #print(x_batch[:100])
        x_batch = pad_sequence(x_batch, batch_first=True, padding_value=0.0)
        x_batch = x_batch.long()
        y_batch = train_y[i: i + batch_size]
        y_batch = torch.LongTensor(y_batch)
        #remind the last cutoff
        yield x_batch, y_batch


def train(model, train_x, train_y, criterion, optimizer, batch_size=BATCH_SIZE, epoch=EPOCH):

    if torch.cuda.is_available():
        dev = torch.device('cuda:1')
        model = model.to(dev)
        model.set_dev(dev)

    for epoch in range(epoch):
        print("Epoch: %d" % (epoch + 1))
        epoch_loss = 0.0

        print(len(train_x), len(train_y))

        for x_batch, y_batch in padded_batching(train_x, train_y, batch_size):

            if torch.cuda.is_available():
                x_batch = torch.LongTensor(x_batch).to(dev)
                y_batch = torch.LongTensor(y_batch).to(dev)
            else:
                x_batch = torch.LongTensor(x_batch)
                y_batch = torch.LongTensor(y_batch)

            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print("Loss at epoch %d: %.7f" % (epoch + 1, epoch_loss))
    return model


def test(model, mapping, test_x, test_y):
    all_pred = 0
    correct_pred = 0

    for x, y in zip(test_x, test_y):
        # get 100 clipped and converted test_x for x
        test_x_tensor, _ = convert_into_num_tensor(x, y, mapping)

        # get 100 padded sequences
        padded_sequences = pad_sequence(
            test_x_tensor, batch_first=True, padding_value=0.0)

        for seq in padded_sequences:
            all_pred += 1
            output = model(torch.stack([seq]).long())
            _, prediction = torch.max(output.data, dim=1)
            if prediction == y:
                correct_pred += 1
                break

    print("Accuracy of model: {}".format(
        (correct_pred / all_pred) * 100))


if __name__ == '__main__':
    # open the data
    x, y, list_of_lang = open_data()

    # put the labels into indexes
    y =[list_of_lang.index(label) for label in y ]

    # creating train and test data
    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.2, shuffle=True)

    # generate the vocabs
    mapping, vocabulary = get_vocabulary(train_x)

    # creating a num tensor and a extend both label und sentences to 100 length
    train_x_tensor, train_y_num_longer = convert_into_num_tensor(train_x, train_y, mapping)
    print(train_x_tensor[:15])

    print(train_y_num_longer[:15])

    # Initializing the Network
    vocab_size = len(vocabulary) + 1
    print(vocab_size)
    output_size = len(list_of_lang)

    model = GRUNet(vocab_size, SEQUENZ_LENGTH, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, output_size)

    # Initializing criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # train the model
    model = train(model, train_x_tensor, train_y_num_longer, criterion, optimizer)

    # test the model
    test(model, mapping, test_x, test_y)