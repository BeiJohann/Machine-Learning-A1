import argparse
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from data_loader import open_data, get_vocabulary


# to select a List of Languages, a File or 10 Arguments
def select_language(selected_lang):
    selected_lang = [line.split()[0] for line in open('./data/' + selected_lang+ ".txt")]

    if not selected_lang:
        selected_languages = ['als', 'ckb', 'gle', 'zea', 'vls', 'swe', 'sco', 'por', 'nap', 'lit']
        print("selected the default List")
    else:
        selected_languages = selected_lang

    print(selected_languages)
    return selected_languages


# Save the selected languages in a File
def prep_data(selected_lang, saved_file):
    y = [line.split()[0] for line in open('./data/y_train.txt')]
    x = [line.rstrip("\n") for line in open('./data/x_train.txt')]

    print('generating vocabulary and mapping')
    mapping, vocabulary = get_vocabulary(x)
    print('saving both')
    joblib.dump(mapping, saved_file+'_mapping.sav')
    joblib.dump(vocabulary, saved_file+'vocabulary.sav')

    print('splitting data')
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, shuffle=True)
    # print(y_train[400:800])

    print('select language')
    selected_languages = select_language(selected_lang)
    print('saving train and test data')
    output_train = open('./data/' + saved_file + "_train.txt", "w")
    output_test = open('./data/' + saved_file + "_test.txt", "w")

    for x, y in zip(x_train, y_train):
        if y in selected_languages and len(x) >= 100:
            output_train.write(x[:100] + "\t" + y + "\n")

    for x, y in zip(x_test, y_test):
        if y in selected_languages and len(x) >= 100:
            output_test.write(x[:100] + "\t" + y + "\n")


# return all availabe languages
def getListOfLang():
    AllLines = [line.split()[0] for line in open('./data/labels.csv')]
    AllWords = [x.split(';') for x in AllLines]
    return AllWords

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Prepare the Data for training and testing. But everything for the selected language in one file")
    #parser.add_argument("-G", "--getfile", dest="get_file", type=str,
    #                    help="The name of the file from which the script is loading the data. Musst be placed in \data dir. Default is 'x_train.txt'")
    parser.add_argument("-S", "--savefile", dest="save_file", type=str, default='my_data',
                        help="The name of the file from which the script is saving the data. Going to be saved in \data dir. Default is 'my_data.txt'. You don't need to write .txt at the end")
    parser.add_argument("-L", "--language", dest="language", type=str, default='test_lang',
                        help="The name of the file from which the script is taking the selected language.  Musst be placed in \data dir. Default is 'test_lang.txt'. You don't need to write .txt at the end")
    args = parser.parse_args()

    prep_data(args.language, args.save_file)
    print('Language selected and saved in', args.save_file)