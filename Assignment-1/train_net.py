# Imports
import torch.nn as nn
import torch
import argparse
import joblib

from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence

from data_loader import open_data, get_vocabulary, convert_into_num_tensor, convert_into_clipped

# Parameters
LEARNING_RATE = 0.001
HIDDEN_SIZE = 300
NUM_LAYERS = 2
INPUT_SIZE = 100
EPOCH = 5
BATCH_SIZE = 800
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


def padded_batching(train_x, train_y, batch_size):
    # REMIND SHUFFELING
    for i in range(0, len(train_x), batch_size):
        x_batch = train_x[i: i + batch_size]
        # print(x_batch[:100])
        x_batch = pad_sequence(x_batch, batch_first=True, padding_value=0.0)
        x_batch = x_batch.long()
        y_batch = train_y[i: i + batch_size]
        y_batch = torch.LongTensor(y_batch)
        # remind the last cutoff
        yield x_batch, y_batch


def train(model, train_x, train_y, criterion, optimizer, batch_size, epoch, loss_func=1, device=DEVICE):
    model.train()
    for epoch in range(epoch):
        print("Epoch: %d" % (epoch + 1))
        epoch_loss = 0.0
        #epoch_steps = 0
        # print(train_x[:100],train_y[:100])

        for x_batch, y_batch in padded_batching(train_x, train_y, batch_size):
            #epoch_steps += 1
            # sending to Cuda
            x_batch = torch.LongTensor(x_batch).to(dev)
            y_batch = torch.LongTensor(y_batch).to(dev)

            optimizer.zero_grad()
            output = model(x_batch)
            # print(output.shape, y_batch.shape)

            loss = criterion(output, y_batch)
            #insert the 3 loss methodes
            if loss_type != 1:
                # get the number of characters in each sequence in the batch
                char_lengths = []
                for t in local_batch:
                    non_zero_indices = torch.nonzero(t)
                    char_lengths.append(non_zero_indices.size(0) / 100.0)
                char_lengths = torch.Tensor(char_lengths)
                char_lengths = char_lengths.to(dev)

                if loss_type == 2:
                    # multiply the losses of the batch with the character lengths
                    loss *= char_lengths
                elif loss_type == 3:
                    # add the losses of the batch with the character lengths
                    loss += char_lengths
            # take mean of the loss
            loss = loss.mean()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print("Loss at epoch %d: %.7f" % (epoch + 1, epoch_loss))
    return model


if __name__ == '__main__':
    # commandline arguments
    parser = argparse.ArgumentParser(description="Train a recurrent network for language identification")
    parser.add_argument("-E", "--epochs", dest="num_epochs", type=int, help="Specify the number of epochs for training the model")
    parser.add_argument("-LF", "--loss", dest="loss_func", type=int, help="Specify the loss function to be used. Choose between 1,2,3")
    parser.add_argument("-S", "--save", dest="save", type=bool, help="Specify if the model should be saved")
    #only one Argument for the Data, becaus X,Y are in the same file und shouldn't be seperated
    parser.add_argument("-D", "--train_data", dest="data_path", type=str, help="Specify the file from where the data is loaded")
    args = parser.parse_args()
    EPOCH = args.num_epochs

    # open the data 
    train_x, train_y, list_of_lang = open_data(args.data_path)

    # open the vocabs and mapping
    mapping = joblib.load('./data/my_data_mapping.sav')
    vocabulary = joblib.load('./data/my_data_vocabulary.sav')

    # put the labels into indexes
    train_y = [list_of_lang.index(label) for label in train_y]

    # extend both label und sentences to 100 length
    train_x, train_y = convert_into_clipped(train_x, train_y)

    # creating a num tensor
    train_x_tensor = convert_into_num_tensor(train_x, mapping)

    vocab_size = len(vocabulary) + 1
    print('number of character: ', vocab_size)
    output_size = len(list_of_lang)
    # is cuda available
    if not (torch.cuda.is_available()):
        DEVICE = 'cpu'

    # Initializing the Network
    model = GRUNet(vocab_size, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, output_size)
    dev = torch.device(DEVICE)
    model = model.to(dev)
    model.set_dev(dev)

    # Initializing criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # train the model
    print('Training the model with %d epochs' % EPOCH)
    model = train(model, train_x_tensor, train_y, criterion, optimizer, BATCH_SIZE, EPOCH, args.loss_func)

    # saving if it was specified in the commandline
    if args.save:
        print('saving the Net')
        torch.save(model, 'savedNet.pt')
