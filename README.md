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

I chose a lot if different languages, because it is harder to indetify similar languages and it is a bit more easier in the beginning.  

### Generating train, test and vocab(mapping) files

Use the data_prep.py file to generate train, test and vocab(mapping) files. All the non python files are in the `data` folder.

* `-X` The name of the file from which the script is loading the X data. Musst be placed in \data dir. Default is 'x_train.txt'
* `-Y`The name of the file from which the script is loading the Y data. Musst be placed in \data dir. Default is 'y_train.txt'
* `-S`The name of the file from which the script is saving the data. Going to be saved in \data dir. Default is 'my_data.txt'. You don't need to write .txt at the end
* `-L`The name of the file from which the script is taking the selected language.  Must be placed in \data dir. Default is 'test_lang.txt'. You don't need to write .txt at the end

The generated files in the for training and test files in the repo:

* x_train.txt
* y_train.txt
* x_test.txt
* y_test.txt

## Part 2: model and training 

### Network model

The GRU network model is in the file `GRUNN.py`. The model has an input, embedding, hidden, linear and an output layer.
 It contains the criteron for  CrossEntropyLoss, which combines LogSoftmax and NLLLoss in one single class, with and without reduction. 
 The data is fed into the embedding layer which creates randomly-initialized vectors corresponding to the indices in the sentences. 
 The hidden layer is initialized with the number of layers, length of the sentence and the layer's hidden size. The loss is calculated and printed for every trained batch.

###Training

The network model is trained by running the script `train_model.py` using the parameters:

* `-E`: Number of Epochs
* `-L`: The loss function. 1:CrossEntropyLoss, 2:CrossEntropyLoss without reduction multiplied with character prefix length (prefix lenth/sentence length), 3: CrossEntropyLoss without reduction with additive character prefix length
* `-S`: Path where the model should be saved.
* `-D`: The path of the train data.


### Testing

The network model is tested by running the script `test_model.py` using the parameters:

* `-L`: Path of the model the script should load.
* `-D`:  The path of the test data.

The data is loaded from the selected files into training data, labels and vocabulary. The `utils.py` contains the functions for creating the  data fed into the network model.
First, the sentences are made into prefixes, made up of one list for every following character up to the lenth of the sentence.
The list of prefixes is encoded by swiching each character to the integer representation in the vocabulary, and are then padded up to the length of the longest sentence with zeroes.

The script will check if cuda is available, and then select `cuda:1`, if unavailable it will select `cpu`.

The encoded data is put into a Dataset from `dataset.py` and put into a dataloader. The dataloader pins the dataloader to memory if `cuda:0` is selected. Then it feeds the data in the selected number of batches into the model, printing the updated loss after each batch.

When the model has finished training, the model is saved to disk togeather with the vocabulary to be used in the evaluation.

## Part 3: evaluation

### Testing

The script will check if cuda is available, and then select `cuda:0`, if unavailable it will select `cpu`.
It loads the test data and labels and the vocabulary.

For each sentence, the  prefixes and padding are done using the same functions as in training. For each sentence, an instance of n (length of sentence) prefixes are created and used for testing. The n prefixes are matched with the language label and fed into a dataloader (the reason for this was because it kept having the wrong shape otherwise) and it does not shuffle the data. Then, it tries to predict each prefix. If the prefix is correctly classified, the loop breaks and the testing continues with the next sentence. This turned out to make the testing considerably shorter, as some sentences were correctly predicted at the first character.

For every language, it saves all the predictions and the correct results. If a language was correctly predicted, the prefix number at which it was correct is also saved, otherwise the instance will be `None`. At a correct classification, the result is printed in the terminal.

The results for every language are saved as:

* The predicted language
* The language that was correct
* Percentages of the correctly predicted languages
* The mean prefix number at which each language was correctly predicted
* The number of sentences for a language that were never predicted

## Part 4: reporting