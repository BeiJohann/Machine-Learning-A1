import argparse


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
    y_train = [line.split()[0] for line in open('./data/y_train.txt')]
    x_train = [line.rstrip("\n") for line in open('./data/x_train.txt')]

    # print(y_train[400:800])

    selected_languages = select_language(selected_lang)

    output = open('./data/' + saved_file + ".txt", "w")

    for x, y in zip(x_train, y_train):
        if y in selected_languages and len(x) >= 100:
            output.write(x[:100] + "\t" + y + "\n")


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