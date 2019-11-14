from torch import Tensor


def open_data(file):
    data = [line.rstrip("\n") for line in open('./data/'+file+'.txt')]
    train_x = [list(line.split('\t')[0]) for line in data]
    train_y = [line.split('\t')[1] for line in data]

    listOfLang = [line.split()[0] for line in open('./data/test_lang.txt')]
    #print(listOfLang)
    return train_x, train_y, listOfLang

def convert_into_clipped(train_x, train_y):
    # this function returns 100 sentences for each sentence
    train_x_increment = []
    y_extended = []
    for sent, label in zip(train_x, train_y):
        # this will add 100 sentences for each sent
        for index in range(len(sent)):
            train_x_increment.append([x for x in sent[:index + 1]])
            y_extended.append(label)

    return train_x_increment, y_extended

def convert_into_num_tensor(train_x, mapping):
    x_train_numeric = [[ mapping[char] for char in sentences] for sentences in train_x]
    x_train_tensor = [Tensor(sentence) for sentence in x_train_numeric]
    return x_train_tensor


def char_to_index(char, vocabulary):
    return vocabulary.index(char)
