# Imports
import torch.nn as nn
import torch
import argparse
import joblib

from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence

from data_loader import open_data, convert_into_num_tensor, convert_into_clipped
from GRUNN import GRUNN

# Parameters
LEARNING_RATE = 0.001
HIDDEN_SIZE = 300
NUM_LAYERS = 2
INPUT_SIZE = 100
BATCH_SIZE = 800
SEQUENZ_LENGTH = 100
DEVICE = 'cuda:1'


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


def train(model, train_x, train_y, criterion, optimizer, batch_size, epochs, loss_func):
    model.train()
    for epochs in range(epochs):
        print("Epoch: %d" % (epochs + 1))
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

            # measure number of chars in prefix
            for prefix in y_batch:
                char_len = torch.nonzero(prefix)
                prefix_len.append(char_len.size(0))
                # vocab_len.append(len(X_batch[0]))
            prefix_len = torch.FloatTensor(prefix_len)
            prefix_len = prefix_len.to(dev)

            #insert the 3 loss methodes
            if args.loss_type == 1:
                loss = criterion(output, y_batch)
            # loss including character prefix length
            if args.loss_type == 2:
                loss = criterion(output, y_batch)
                #loss *= (prefix_len/vocab_len)
                loss *= prefix_len
                loss = loss.mean()
            # additive loss including character prefix
            if args.loss_type == 3:
                loss = criterion(output, y_batch)
                loss += prefix_len
                loss = loss.mean()
            # take mean of the loss
            loss = loss.mean()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print("Loss at epoch %d: %.7f" % (epochs + 1, epoch_loss))
    return model


if __name__ == '__main__':
    # commandline arguments
    parser = argparse.ArgumentParser(description="Train a recurrent network for language identification")
    parser.add_argument("-E", "--epochs", dest="num_epochs", type=int, default=10, help="Specify the number of epochs for training the model")
    parser.add_argument("-LF", "--loss", dest="loss_func", type=int, default=1, help="Specify the loss function to be used. Choose between 1,2,3")
    parser.add_argument("-S", "--save", dest="save", action='store_true',  help="Specify if the model should be saved")
    #only one Argument for the Data, becaus X,Y are in the same file und shouldn't be seperated
    parser.add_argument("-D", "--train_data", dest="data_path", type=str, default='my_data_train', help="Specify the file from where the data is loaded")
    args = parser.parse_args()

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
    model = GRUNN(vocab_size, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, output_size)
    dev = torch.device(DEVICE)
    model = model.to(dev)
    model.set_dev(dev)

    # Initializing criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # train the model
    print('Training the model with %d epochs' % args.num_epochs)
    model = train(model, train_x_tensor, train_y, criterion, optimizer, BATCH_SIZE, args.num_epochs, args.loss_func)

    # saving if it was specified in the commandline
    if args.save:
        print('saving the Net')
        torch.save(model, 'savedNet.pt')
