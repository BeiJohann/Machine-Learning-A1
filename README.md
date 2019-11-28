# Assignment #1: language identification with as few characters as possible 
## Part 0: choosing "your" GPU
I use Cuda:1  if Cuda is available. Otherwise cpu as device.

## Part 1: data preparation

### Languages
* Modern Greek
* Central Kurdish
* Afrikaans
* Zeeuws
* Maltese
* Swedish
* Hebrew
* Portuguese
* Neapolitan
* Russian

I chose a lot if different languages, because it is harder to identify similar languages and it is a bit easier for beginners.  

### Generating train, test and vocab(mapping) files

Use the data_prep.py file to generate train, test and vocab(mapping) files. All the non python files are in the `data` folder.
I used a separated script for this, because the vocabulary and mapping must be done on all characters. Otherwise there could be some character missing afterwards.
I output two files with the train and test data. Every file got the labels in it. The vocabulary and mapping store all the characters used in the file and an numerical representation.   

* `-X` Path of the X data. Must be placed in \data folder.
* `-Y` Path of the X data. Must be placed in \data folder.
* `-S` The path were the data is saved. Going to be saved in \data dir. One name for the whole set.
* `-L` The path of the file from which the script is loading the selected language.  Must be placed in \data dir.

The generated files I used for training and testing are:

* my_data_test.txt
* my_data_train.txt
* my_data_vocabulary.sav
* my_data_mapping.sav

You can only give the whole set of data one prefix In the save argument for the command line. `my_data` is the path I used to save the files. 


## Part 2: model and training 

### Network model

The GRU network model is in the file `GRUNN.py`. The model has an input, embedding, hidden, linear and an output layer.
 It contains the criterion for CrossEntropyLoss, which combines LogSoftmax and NLLLoss in one single class, with and without reduction. 
 The data is fed into the embedding layer. The hidden layer is initialized with the number of layers, length of the sentence and the layer's hidden size. 
 The loss is calculated and printed for every trained epoch.

###Training

The network model is trained by running the script `train_model.py` using the parameters:

* `-E`: Number of Epochs
* `-L`: The loss function. 1:CrossEntropyLoss, 2:CrossEntropyLoss without reduction multiplied with character prefix length, 3: CrossEntropyLoss without reduction with additive character prefix length
* `-S`: Path where the model should be saved.
* `-D`: The path of the train data.

The data is loaded from the selected files into training, labels and vocabulary data. The `data_loader.py` contains the functions for converting the data into number representation and tensors.
the vocabulary is used to find the right representation. The encoded data is put into generator function. Then it feeds the data in the selected number of batches into the model, updated loss after each batch.
When the model has finished training, the model is saved.


## Part 3: evaluation

### Testing

The network model is tested by running the script `test_model.py` using the parameters:

* `-L`: Path of the model the script should load.
* `-D`:  The path of the test data.

It loads the test data with labels and the vocabulary. For each sentence, the  prefixes and padding are done using the same functions as in training.
Every padded sentence is going thru the testing even if the model predict it right. It saves how often a language was correctly predicted and if a whole sentence was never correctly predicted. It also saves the number of character until it hits first.

## Part 4: reporting 

The tables show how many sentences that were correctly classified, at what percentage, the average number of characters until hit score at which is was correctly predicted and how many sentences that were never predicted correctly.

|                                        | Func1   | Func2   | Func3    |
|----------------------------------------|---------|---------|----------|
| Total accuracy                         | 83.9%   | 85.1%   | 83.6%    |
| increments after first right prediction| 9.051   | 8.732   | 9.515    |

#### Loss Function 1

| Language           | Never hit the whole sentence|
|--------------------|-----------------------------|
| Modern Greek       | 0 times                     |
| Central Kurdish    | 0 times                     |
| Afrikaans          | 9 times                     |
| Zeeuws             | 17 times                    |
| Maltese            | 1 times                     |
| Swedish            | 8 times                     |
| Hebrew             | 0 times                     |
| Portuguese         | 4 times                     |
| Neapolitan         | 5 times                     |
| Russian            | 1 times                     |

#### Loss Function 2

| Language           | Never hit the whole sentence|
|--------------------|-----------------------------|
| Modern Greek       | 0 times                     |
| Central Kurdish    | 0 times                     |
| Afrikaans          | 11 times                    |
| Zeeuws             | 10 times                    |
| Maltese            | 3 times                     |
| Swedish            | 7 times                     |
| Hebrew             | 0 times                     |
| Portuguese         | 2 times                     |
| Neapolitan         | 3 times                     |
| Russian            | 0 times                     |

#### Loss Function 3

| Language           | Never hit the whole sentence|
|--------------------|-----------------------------|
| Modern Greek       | 0 times                     |
| Central Kurdish    | 0 times                     |
| Afrikaans          | 12 times                    |
| Zeeuws             | 15 times                    |
| Maltese            | 1 times                     |
| Swedish            | 6 times                     |
| Hebrew             | 0 times                     |
| Portuguese         | 3 times                     |
| Neapolitan         | 10 times                    |
| Russian            | 1 times                     |

