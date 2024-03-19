# This coding exercise is done by 
#   - Jiecheng Mei
#   - Khiem Do
import json
import sys
import pandas as pd
import os
import time


def split_words(sentence):
    chars_to_remove = [',', '.', '-', '!', '\"', ':', ')', '(', '\\']
    for char in chars_to_remove:
        sentence = sentence.replace(char, '')
    sentence = sentence.split()
    split_sentence = [element.lower() for element in sentence]
    return split_sentence


# delete some character and split the sentence to words
def split_tag_words(sentence, tag):
    _massive_words = []
    chars_to_remove = [',', '.', '-', '!', '\"', ':', ')', '(']
    for char in chars_to_remove:
        sentence = sentence.replace(char, '')
    sentence = sentence.split()
    sentence = [element.lower() for element in sentence]
    for each in sentence:
        _massive_words.append((str(each), int(tag)))
    return _massive_words


# count the tag from massive dataset to store dataset
def count_words(dataset):
    store_data = {}
    for x in dataset:
        word, tag = x[0], x[1]
        if not isinstance(word, str):
            word = str(word)
        if word not in store_data:
            store_data[word] = {"total": 0, "one": 0, "two": 0, "three": 0, "four": 0, "five": 0}
        # total count
        store_data[word]["total"] += 1
        if tag == 1:
            store_data[word]["one"] += 1
        elif tag == 2:
            store_data[word]["two"] += 1
        elif tag == 3:
            store_data[word]["three"] += 1
        elif tag == 4:
            store_data[word]["four"] += 1
        elif tag == 5:
            store_data[word]["five"] += 1
    return store_data


def write_to_local(dataset, file_name, dir_results):
    full_path = os.path.join(dir_results, file_name)
    with open(full_path, 'w') as _file:
        json.dump(dataset, _file, indent=2)


def load_from_local(file_name, dir_results):
    full_path = os.path.join(dir_results, file_name)
    with open(full_path, 'r') as _file:
        data_dict = json.load(_file)
    return data_dict


# this part is the algorithm for training classifier
def pre_process_train_data(t_size: int):
    dir_path = './dataset/train'
    df = pd.read_csv('dataset/Reviews.csv')
    t_size_f = t_size / 100
    data_train = df[:int(len(df) * t_size_f)]
    _train_start = time.time()
    train_data_simplify = data_train[["Score", "Summary", "Text"]]
    train_dataset = pd.DataFrame(train_data_simplify)
    massive_train_dataset = []
    for i in range(1, len(train_dataset)):
        row = train_dataset.iloc[i]
        tag = row.iloc[0]
        summary = row.iloc[1]
        text = row.iloc[2]
        sentence = f'{summary} {text}'
        massive_train_dataset.extend(split_tag_words(sentence, tag))
    train_dataset = count_words(massive_train_dataset)
    write_to_local(train_dataset, f'{t_size}.json', dir_path)
    _train_enclape = time.time() - _train_start
    print(f"    TRAIN dataset build, time: {_train_enclape:2f} seconds")
    return train_dataset


def pre_process_test_data():
    dir_path = './dataset/test'
    _Reviews_dataset = pd.read_csv('dataset/Reviews.csv')
    _Reviews_dataset = _Reviews_dataset[['Score', 'Summary', 'Text']]
    test_data = _Reviews_dataset[int(len(_Reviews_dataset)*0.8):]
    _test_start = time.time()
    data_for_csv = []
    for i, r in test_data.iterrows():
        tag = r['Score']
        sentence = f'{r["Summary"]} {r["Text"]}'
        words = split_words(sentence)
        data_for_csv.append([words, tag])
    test_data = pd.DataFrame(data_for_csv, columns=["Words", "Score"])
    test_data.to_csv(os.path.join(dir_path, "test.csv"), index=False)
    _test_enclaps = time.time() - _test_start
    print(f'    TEST dataset build, time: {_test_enclaps:2f} seconds')
    return test_data


if __name__ == '__main__':
    # Global Variables
    tagged_word = {}
    train_size = 80
    _file_name = ''

    # This part handling parameter pass by command
    if len(sys.argv) == 2:
        try:
            arg_val = int(sys.argv[1])
            if 20 <= arg_val <= 80:
                train_size = arg_val
            else:
                raise ValueError
        except ValueError:
            pass
    print(f'Mei Jiecheng A20439795, Khiem Do A20483713 solution:\n'
          f'Training set size = {train_size}%')

    # Detect if the ##% train data on local, if not create it, else use it
    print("Training Classifier...")
    json_files_train = []
    _dir_path = './dataset/train'
    for filename in os.listdir(_dir_path):
        if filename.endswith('.json'):
            if filename not in json_files_train:
                json_files_train.append(filename)

    if f"{train_size}.json" not in json_files_train:
        train_data = pre_process_train_data(train_size)
    else:
        print('     Existing TRAIN dataset detected...')
        train_data = load_from_local(f'{train_size}.json', _dir_path)
    # IMPORTANT:
    # this part is pre-process for the V size and P(label) part
    V = len(train_data)
    num_label_1 = sum(x['one'] for x in train_data.values())
    num_label_2 = sum(x['two'] for x in train_data.values())
    num_label_3 = sum(x['three'] for x in train_data.values())
    num_label_4 = sum(x['four'] for x in train_data.values())
    num_label_5 = sum(x['five'] for x in train_data.values())
    num_total = sum(x['total'] for x in train_data.values())
    P_label_1 = num_label_1 / num_total
    P_label_2 = num_label_2 / num_total
    P_label_3 = num_label_3 / num_total
    P_label_4 = num_label_4 / num_total
    P_label_5 = num_label_5 / num_total

    print("Testing classifier...")
    _dir_path = './dataset/test'
    csv_test_file = []
    for filename in os.listdir(_dir_path):
        if filename.endswith('.csv'):
            csv_test_file.append(filename)
    if filename != "test.csv":
        test_dataset = pre_process_test_data()
    else:
        print('     Existing TEST dataset detected...')
        test_dataset = pd.read_csv(os.path.join(_dir_path, "test.csv"))




    # This part placehold for confusion matrix
    # This part placehold for Sentence with naive bayes classifier
    # P(label|S) = P(label)*P(word1|label)*P(word2|label)...
