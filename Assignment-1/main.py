# Imports
import torch.nn as nn
import torch

from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

from data_loader import open_data, get_vocabulary, convert_into_num_tensor, convert_into_clipped

# Parameters
LEARNING_RATE = 0.01
HIDDEN_SIZE = 200
NUM_LAYERS = 2
INPUT_SIZE = 200
EPOCH = 5
BATCH_SIZE = 300
SEQUENZ_LENGTH = 100
DEVICE = 'cuda:1'

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
    
    def set_dev(self, dev):
        self.dev = dev

    def init_hidden(self, seq_len):
        return torch.zeros(self.num_layers, seq_len, self.hidden_size).float().to(self.dev)


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


def train(model, train_x, train_y, criterion, optimizer, batch_size=BATCH_SIZE, epoch=EPOCH, device=DEVICE):

    for epoch in range(epoch):
        print("Epoch: %d" % (epoch + 1))
        epoch_loss = 0.0

        for x_batch, y_batch in padded_batching(train_x, train_y, batch_size):

            # sending to Cuda
            x_batch = torch.LongTensor(x_batch).to(dev)
            y_batch = torch.LongTensor(y_batch).to(dev)

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
        correct_pred_increment += num_until_correct
        intotal_correct += correct_pred_per_sentence

    print('Average incremets after prediction: ', correct_pred_increment / len(test_y))
    print('Average propability for predicting right: ', intotal_correct / all_pred)


if __name__ == '__main__':
    # open the data
    x, y, list_of_lang = open_data()
    
    # generate the vocabs
    mapping, vocabulary = get_vocabulary(x)

    # put the labels into indexes
    y =[list_of_lang.index(label) for label in y ]

    # creating train and test data
    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.2, shuffle=True)
    
    #extend both label und sentences to 100 length
    train_x, train_y = convert_into_clipped(train_x, train_y)

    # creating a num tensor
    train_x_tensor = convert_into_num_tensor(train_x, mapping)
    #print(train_x_tensor[:15])


    # Initializing the Network
    vocab_size = len(vocabulary) + 1
    print('Anzahl an Zeichen: ',vocab_size)
    output_size = len(list_of_lang)

    model = GRUNet(vocab_size, SEQUENZ_LENGTH, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, output_size)
    model.set_dev(torch.device('cpu'))

    # Initializing criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    
    if not(torch.cuda.is_available()):
        DEVICE = 'cpu'
        
    dev = torch.device(DEVICE)
    model = model.to(dev)
    model.set_dev(dev)

    # train the model
    model = train(model, train_x_tensor, train_y, criterion, optimizer)

    # test the model
    test(model, mapping, test_x, test_y)