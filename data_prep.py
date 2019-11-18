import argparse
from sklearn.model_selection import train_test_split
import joblib
#from data_loader import open_data


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

def get_vocabulary(sents):
    sents = [[x for x in sent] for sent in sents]
    vocabulary = list(set(sum(sents, [])))
    # print(vocabulary)
    char_to_int_mapping = {char: i + 1 for i, char in enumerate(vocabulary)}
    return char_to_int_mapping, vocabulary

# Save the selected languages in a File
def prep_data(x_path, y_path, selected_lang, saved_file):
    print('open files')
    y = [line.split()[0] for line in open('./data/'+y_path)]
    x = [line.rstrip("\n") for line in open('./data/'+x_path)]

    print('select language')
    selected_languages = select_language(selected_lang)

    x_selec = []
    y_selec = []
    for x, y in zip(x,y):
        if y in selected_languages and len(x) >= 100:
            x_selec.append(x)
            y_selec.append(y)

    print('generating vocabulary and mapping')
    mapping, vocabulary = get_vocabulary(x_selec)
    print('saving both')
    joblib.dump(mapping,'./data/' +saved_file+'_mapping.sav')
    joblib.dump(vocabulary,'./data/' +saved_file+'_vocabulary.sav')

    print('splitting data')
    x_train, x_test, y_train, y_test = train_test_split(
        x_selec, y_selec, test_size=0.2, shuffle=True)
    # print(y_train[400:800])


    print('saving train and test data')
    output_train = open('./data/' + saved_file + "_train.txt", "w")
    output_test = open('./data/' + saved_file + "_test.txt", "w")

    for x, y in zip(x_train, y_train):
        output_train.write(x[:100] + "\t" + y + "\n")

    for x, y in zip(x_test, y_test):
        output_test.write(x[:100] + "\t" + y + "\n")

# return all availabe languages
def getListOfLang():
    AllLines = [line.split()[0] for line in open('./data/labels.csv')]
    AllWords = [x.split(';') for x in AllLines]
    return AllWords

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Prepare the Data for training and testing. But everything for the selected language in one file")
    parser.add_argument("-X", "--getXfile", dest="getX_file", type=str, default='x_train.txt', help="The name of the file from which the script is loading the X data. Musst be placed in \data dir. Default is 'x_train.txt'")
    parser.add_argument("-Y", "--getYfile", dest="getY_file", type=str, default='y_train.txt', help="The name of the file from which the script is loading the Y data. Musst be placed in \data dir. Default is 'y_train.txt'")
    parser.add_argument("-S", "--savefile", dest="save_file", type=str, default='my_data', help="The name of the file from which the script is saving the data. Going to be saved in \data dir. Default is 'my_data.txt'. You don't need to write .txt at the end")
    parser.add_argument("-L", "--language", dest="language", type=str, default='test_lang', help="The name of the file from which the script is taking the selected language.  Musst be placed in \data dir. Default is 'test_lang.txt'. You don't need to write .txt at the end")
    args = parser.parse_args()

    prep_data(args.getX_file, args.getY_file, args.language, args.save_file)
    print('Language selected and saved in', args.save_file)