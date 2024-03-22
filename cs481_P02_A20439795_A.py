# This coding exercise is done by 
#   - Jiecheng Mei
#   - Khiem Do
import json
import sys
import pandas as pd
import os
import time


def CM_deduction(_CM, _label):
    _TP = _TN = _FP = _FN = 0
    for i in range(len(_CM)):
        for j in range(len(_CM[i])):
            if i == j and i == _label:
                _TP += _CM[i][j]
            elif i == _label and j != _label:
                _FN += _CM[i][j]
            elif i != _label and j == _label:
                _FP += _CM[i][j]
            elif i != _label and j != _label:
                _TN += _CM[i][j]
    return [_TP, _FN, _FP, _TN]


def s_cm_eval(_TP, _FN, _FP, _TN):
    _sensitive = round(_TP / (_TP + _FN), 5) if _TP + _FN != 0 else 0
    _specificity = round(_TN / (_TN + _FP), 5) if _TN + _FP != 0 else 0
    _precision = round(_TP / (_TP + _FP), 5) if _TP + _FP != 0 else 0
    _n_predictive = round(_TN / (_TN + _FN), 5) if _TN + _FN != 0 else 0
    _accurancy = round((_TP + _TN) / (_TP + _TN + _FP + _FN), 5) if _TP + _TN + _FP + _FN != 0 else 0
    _f_score = round(2 * ((_precision * _sensitive) / (_precision + _sensitive)), 5) if _precision+_sensitive != 0 else 0
    return [_sensitive, _specificity, _precision, _n_predictive, _accurancy, _f_score]


def analyze_test(dataset, list_of_words, p_label, n_label, V_size):
    # predict variable will initial all 5 of the P(label)
    predict = p_label.copy()
    # results will store predict and actual label for future adding to Big_CM
    _results = []
    # get the row data
    for index, row in list_of_words.iterrows():
        # type of data: Array, Int
        _Test_words, _Act_label = eval(row['Words']), row['Score']
        # loop all five labels
        for _word in _Test_words:
            for _lbl in range(1, 6):
                _curr_label = n_to_str.get(_lbl)
                words_occur = dataset.get(_word, {"total": 0, "one": 0, "two": 0, "three": 0, "four": 0, "five": 0})
                words_occur_with_label = words_occur.get(_curr_label)
                predict[_lbl - 1] *= (words_occur_with_label + 1) / (n_label[_lbl - 1] + V_size)

        # will get the max number location
        p_max_idx = predict.index(max(predict))
        # concat two label [actual, predict] for future adding to CM
        _results.append([p_max_idx, int(_Act_label - 1)])
        predict = p_label.copy()
    # once concat all the result, returns to main
    return _results


def split_words(sentence):
    chars_to_remove = [',', '.', '-', '!', '\"', ':', ')', '(', '\\']
    for char in chars_to_remove:
        sentence = sentence.replace(char, '')
    sentence = sentence.split()
    split_sentence = [element for element in sentence]
    return split_sentence


# delete some character and split the sentence to words
def split_tag_words(sentence, tag):
    _massive_words = []
    chars_to_remove = [',', '.', '-', '!', '\"', ':', ')', '(']
    for char in chars_to_remove:
        sentence = sentence.replace(char, '')
    sentence = sentence.split()
    sentence = [element for element in sentence]
    for each in sentence:
        _massive_words.append((str(each), int(tag)))
    return _massive_words


# count the tag from massive dataset to store dataset
def count_words(dataset):
    store_data = {}
    for x in dataset:
        _word, tag = x[0], x[1]
        if _word not in store_data:
            store_data[_word] = {"total": 0, "one": 0, "two": 0, "three": 0, "four": 0, "five": 0}
        # total count
        store_data[_word]["total"] += 1
        if tag == 1:
            store_data[_word]["one"] += 1
        elif tag == 2:
            store_data[_word]["two"] += 1
        elif tag == 3:
            store_data[_word]["three"] += 1
        elif tag == 4:
            store_data[_word]["four"] += 1
        elif tag == 5:
            store_data[_word]["five"] += 1
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
    train_data_simplify = data_train[["Score", "Summary", "Text"]]
    _train_dataset = pd.DataFrame(train_data_simplify)
    massive_train_dataset = []
    for i in range(1, len(_train_dataset)):
        _row = _train_dataset.iloc[i]
        tag = _row.iloc[0]
        summary = _row.iloc[1]
        text = _row.iloc[2]
        sentence = f'{summary} {text}'
        massive_train_dataset.extend(split_tag_words(sentence, tag))
    _train_dataset = count_words(massive_train_dataset)
    write_to_local(_train_dataset, f'{t_size}.json', dir_path)
    return _train_dataset


def pre_process_test_data():
    dir_path = './dataset/test'
    _Reviews_dataset = pd.read_csv('dataset/Reviews.csv')
    _Reviews_dataset = _Reviews_dataset[['Score', 'Summary', 'Text']]
    test_data = _Reviews_dataset[int(len(_Reviews_dataset) * 0.8):]
    data_for_csv = []
    for i, r in test_data.iterrows():
        tag = r['Score']
        sentence = f'{r["Summary"]} {r["Text"]}'
        words_set = split_words(sentence)
        data_for_csv.append((words_set, tag))
    test_data = pd.DataFrame(data_for_csv, columns=["Words", "Score"])
    test_data.to_csv(os.path.join(dir_path, "test.csv"), index=False)
    return test_data


if __name__ == '__main__':
    # Global Variables
    tagged_word = {}
    train_size = 80
    _file_name = ''
    n_to_str = {
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five"
    }

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
        print("     Train dataset work finished...")
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
    num_label = [num_label_1, num_label_2, num_label_3, num_label_4, num_label_5]
    P_label_1 = num_label_1 / num_total
    P_label_2 = num_label_2 / num_total
    P_label_3 = num_label_3 / num_total
    P_label_4 = num_label_4 / num_total
    P_label_5 = num_label_5 / num_total
    P_label = [P_label_1, P_label_2, P_label_3, P_label_4, P_label_5]
    Big_CM = [[0 for _ in range(5)] for _ in range(5)]

    print("Testing classifier...")
    # load test data if existing
    _dir_path = './dataset/test'
    csv_test_file = []
    for filename in os.listdir(_dir_path):
        if filename.endswith('.csv'):
            csv_test_file.append(filename)
    if filename != "test.csv":
        test_dataset = pre_process_test_data()
        print("     Test dataset work finished...")
    else:
        print('     Existing TEST dataset detected...')
        test_dataset = pd.read_csv(os.path.join(_dir_path, "test.csv"))

    # This part will do the label part
    results = analyze_test(train_data, test_dataset, P_label, num_label, V)
    # this is plug all result into the big_cm
    for point in results:
        _predict = int(point[0])
        _actual = int(point[1])
        Big_CM[_actual][_predict] += 1
    # this is the small Confusion matrix list for store all the data from the Small CM

    print("DEBUG Big CM")
    for x in Big_CM:
        print(x)
    print("DEBUG Big CM")

    Small_CM = []
    for cur_label in range(5):
        Small_CM.append(CM_deduction(Big_CM, cur_label))
    # this part will be handling calculation
    small_CM_eval = []
    for cm in Small_CM:
        TP, FN, FP, TN = cm[0], cm[1], cm[2], cm[3]
        small_CM_eval.append(s_cm_eval(TP, FN, FP, TN))

    for info in range(5):
        TP, FN, FP, TN = Small_CM[info][0], Small_CM[info][1], Small_CM[info][2], Small_CM[info][3]
        sensitive = small_CM_eval[info][0]
        specificity = small_CM_eval[info][1]
        precision = small_CM_eval[info][2]
        n_predictive = small_CM_eval[info][3]
        accurancy = small_CM_eval[info][4]
        f_score = small_CM_eval[info][5]
        label = info + 1
        # printing information
        print()
        print(f"================================== Label: {label} ==================================\n"
              f"# of TP: {TP}\n"
              f"# of FN: {FN}\n"
              f"# of FP: {FP}\n"
              f"# of TN: {TN}\n"
              f"Sensitive: {sensitive}\n"
              f"Specificity: {specificity}\n"
              f"Precision: {precision}\n"
              f"Negative Predictive: {n_predictive}\n"
              f"Accurancy: {accurancy}\n"
              f"F-Score: {f_score}\n"
              f"================================== Label: {label} ==================================\n")

    # This part placehold for Sentence with naive bayes classifier
    # P(label|S) = P(label)*P(word1|label)*P(word2|label)...
    while True:
        print()
        userSentence = input("Enter your sentence:\n"
                             "     Sentence S:\n")
        bow = split_words(userSentence)
        user_P_label = P_label
        for label in range(1, 6):
            current_label = n_to_str.get(label)
            for word in bow:
                word_occur = train_data.get(word, {"total": 0, "one": 0, "two": 0, "three": 0, "four": 0, "five": 0})
                word_occur_with_label = word_occur.get(current_label)
                user_P_label[label - 1] *= (word_occur_with_label + 1) / (num_label[label - 1] + V)
        user_p_max_idx = user_P_label.index(max(user_P_label))
        user_predict_label = user_p_max_idx + 1
        print(f'was classifier as {user_predict_label}\n'
              f'P(label 1 | S) = {user_P_label[0]}\n'
              f'P(label 2 | S) = {user_P_label[1]}\n'
              f'P(label 3 | S) = {user_P_label[2]}\n'
              f'P(label 4 | S) = {user_P_label[3]}\n'
              f'P(label 5 | S) = {user_P_label[4]}\n')
        userSelection = input("Do you want to enter another sentence [Y/N]? (invalid input leads another run)\n")
        if userSelection.lower() == 'n':
            break
