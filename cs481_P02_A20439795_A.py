# This coding exercise is done by 
#   - Jiecheng Mei
#   - Khiem Do
import json
import sys
import pandas as pd
import os
import time
import math

# Modify the split_count_words function to return just words for predict_class
def split_count_words(sentence, tag=None):
    massive_words = []
    chars_to_remove = [',', '.', '-', '!', '\"', ':', ')', '(']
    for char in chars_to_remove:
        sentence = sentence.replace(char, '')
    words = sentence.split()
    words = [element.lower() for element in words]
    if tag is not None:
        words = [(str(word), int(tag)) for word in words]
    return words


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
    with open(full_path, 'w') as file:
        json.dump(dataset, file, indent=2)


def load_from_local(file_name, dir_results):
    full_path = os.path.join(dir_results, file_name)
    with open(full_path, 'r') as file:
        data_dict = json.load(file)
    return data_dict


# 这个部分读去数据并返回逐行数据本身，并且只获取score，summary，text
def pre_process_train_data(t_size: int):
    dir_path = './dataset/train'
    df = pd.read_csv('dataset/Reviews.csv')
    t_size_f = t_size / 100
    # print(f'[=TEST=] train_size in method = {t_size_f}')
    data_train = df[:int(len(df) * t_size_f)]
    print(f'Length of the train dataset size: {len(data_train)}')
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
        massive_train_dataset.extend(split_count_words(sentence, tag))
    print('[TRAIN PART] split words finished... Beginning count words')

    train_dataset = count_words(massive_train_dataset)
    write_to_local(train_dataset, f'{t_size}.json', dir_path)
    _train_enclape = time.time() - _train_start
    print(f"total train time take: {_train_enclape}")
    return train_dataset


def pre_process_test_data():
    dir_path = './dataset/test'
    Reviews_dataset = pd.read_csv('dataset/Reviews.csv')
    # remove useless column
    Reviews_dataset = Reviews_dataset[['Score', 'Summary', 'Text']]
    # get last 20% of data
    test_dataset = Reviews_dataset[int(len(Reviews_dataset)*0.8):]
    print(f'Length of the test dataset size: {len(test_dataset)}')
    _test_start = time.time()
    test_dataset = pd.DataFrame(test_dataset)
    massive_test_dataset = []
    for i in range(len(test_dataset)):
        row = test_dataset.iloc[i]
        tag = row.iloc[0]
        summary = row.iloc[1]
        text = row.iloc[2]
        sentence = f"{summary} {text}"
        massive_test_dataset.extend(split_count_words(sentence, tag))
    print("[TEST PART] split words finished... Beginning count words")

    test_dataset = count_words(massive_test_dataset)
    write_to_local(test_dataset, f'test.json', dir_path)
    _test_enclaps = time.time() - _test_start
    print(f'Total test time take: {_test_enclaps}')
    return test_dataset


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

    # Detect if the last 20% data on local, if not create one, else use it
    json_files_test = []
    _dir_path = './dataset/test'
    for filename in os.listdir(_dir_path):
        if filename.endswith('.json'):
            if filename not in json_files_test:
                json_files_test.append(filename)
    if "test.json" not in json_files_test:
        test_data = pre_process_test_data()
    else:
        print('Existing TEST dataset detected...')
        test_data = load_from_local('test.json', _dir_path)

    # Detect if the ##% train data on local, if not create it, else use it
    json_files_train = []
    _dir_path = './dataset/train'
    for filename in os.listdir(_dir_path):
        if filename.endswith('.json'):
            if filename not in json_files_train:
                json_files_train.append(filename)

    if f"{train_size}.json" not in json_files_train:
        train_data = pre_process_train_data(train_size)
    else:
        print('Existing TRAIN dataset detected...')
        train_data = load_from_local(f'{train_size}.json', _dir_path)

    print("[=TEST=] See this message means counter is working properly")

    # This part placehold for confusion matrix

    # This part placehold for Sentence with naive bayes classifier
    # P(label|S) = P(label)*P(word1|label)*P(word2|label)...


###### Test classifiers ######
def calculate_class_probabilities(train_data):
    class_counts = {k: 0 for k in train_data[next(iter(train_data))].keys() if k != 'total'}
    total_counts = 0

    for word, tags in train_data.items():
        for class_label in class_counts:
            class_counts[class_label] += tags[class_label]
            total_counts += tags[class_label]

    class_probabilities = {class_label: (class_counts[class_label] / total_counts) for class_label in class_counts}
    return class_probabilities

def predict_class(words, classifier, class_probabilities, vocab_size):
    class_scores = {class_label: math.log(class_probability, 10) for class_label, class_probability in class_probabilities.items()}

    for word in words:
        if word in classifier:  # Only consider words that are in the classifier
            for class_label in class_scores.keys():
                word_freq = classifier[word].get(class_label, 0) + 1  # Laplace smoothing
                class_scores[class_label] += math.log(word_freq / (class_probabilities[class_label] + vocab_size), 10)

    predicted_class = max(class_scores, key=class_scores.get)
    return predicted_class

def test_classifier(test_data, classifier):
    class_probabilities = calculate_class_probabilities(classifier)
    vocab_size = len(classifier)

    # Initialize counters for metrics
    true_positives = true_negatives = false_positives = false_negatives = 0

    for sentence_data in test_data:
        sentence, actual_class = sentence_data
        predicted_class = predict_class(sentence, classifier, class_probabilities, vocab_size)

        if actual_class == predicted_class:
            if actual_class == 'five':  # Assuming 'five' is the positive class
                true_positives += 1
            else:
                true_negatives += 1
        else:
            if actual_class == 'five':
                false_negatives += 1
            else:
                false_positives += 1

    # Calculate metrics
    sensitivity = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0
    specificity = true_negatives / (true_negatives + false_positives) if true_negatives + false_positives else 0
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
    f_score = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0

    # Print out metrics
    print(f"True Positives: {true_positives}")
    print(f"True Negatives: {true_negatives}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F-Score: {f_score:.4f}")


test_data = load_from_local('test.json', './dataset/test')
classifier = load_from_local('80.json', './dataset/train')

# Create test_data_formatted as a list of tuples (words, actual_class)
test_data_formatted = []
for actual_class, sentences in test_data.items():
    for sentence in sentences:
        words = split_count_words(sentence)
        test_data_formatted.append((words, actual_class))

# Ensure the class_probabilities calculation is correct and matches your JSON structure
class_probabilities = calculate_class_probabilities(classifier)

# Run the classifier test and print metrics
test_classifier(test_data_formatted, classifier)
