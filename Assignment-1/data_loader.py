from torch import Tensor, LongTensor, zeros
from torch.nn.utils.rnn import pad_sequence




def open_data():
    data = [line.rstrip("\n") for line in open('./myTrains.txt')]
    train_x = [list(line.split('\t')[0]) for line in data]
    train_y = [line.split('\t')[1] for line in data]

    listOfLang = [line.split()[0] for line in open('./data/testLang.txt')]
    #print(listOfLang)
    return train_x, train_y, listOfLang

def get_vocabulary(sents):
    vocabulary = list(set(sum(sents, [])))
    # print(vocabulary)
    char_to_int_mapping = {char: i + 1 for i, char in enumerate(vocabulary)}
    return char_to_int_mapping, vocabulary

def convert_into_num_tensor(train_x, train_y, mapping):
    # this function returns 100 sentences for each sentence
    train_x_increment = []
    proportioned_labels = []
    for sent, label in zip(train_x, train_y):
        # this will add 100 sentences for each sent
        for index in range(len(sent)):
            train_x_increment.append([x for x in sent[:index + 1]])
            proportioned_labels.append(label)

    x_train_numeric = [[ mapping[char] for char in sentences] for sentences in train_x_increment]
    x_train_tensors = [Tensor(sentence) for sentence in x_train_numeric]

    numeric_tensor_paddded = pad_sequence(
        x_train_tensors, batch_first=True, padding_value=0.0)
    return numeric_tensor_paddded, train_y

def char_to_index(char, vocabulary):
    return vocabulary.index(char)
