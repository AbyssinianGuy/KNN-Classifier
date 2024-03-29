import sys
import string
from scipy.spatial import distance

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer, sent_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD

from matplotlib import pyplot as plt
import numpy as np
import time

# nltk.download('stopwords')

start = time.process_time()


def read_file(file_name, encoding="utf8"):
    """
    read train and test file
    :param file_name: the name of the file
    :param encoding: encoding of the text
    :return: list of the lines read.
    """
    lines = []
    with open(file_name, "r", encoding=encoding) as file:
        reader = file.readline()
        while reader:
            lines.append(reader.lower())
            reader = file.readline()
    return lines


def train_and_validation(train_dt):
    """
    splits the train dt into train and validate sets after randomizing.
    :param train_dt: the training dt.
    :return: tuple (train_dt, validate_dt)
    """
    train_dt = np.array(train_dt)
    np.random.shuffle(train_dt)  # randomize the dataset
    # 75% train - 25% validation
    size = int(len(train_dt) * .9)
    return train_dt[:size], train_dt[size:]


stop_words = set(stopwords.words('english'))
punct = set(string.punctuation)  # punctuations used in english
tokenizer = RegexpTokenizer(r"\w+")


def parse_train_data(train_data):
    """
    splits the label from the review.
    :param train_data: list of the train_data
    :return: the filtered text.
    """
    class_1 = []  # positive reviews
    class_2 = []  # negative reviews
    reviews = []

    ref = {}  # reference for cross validation.
    for i in range(len(train_data)):
        if train_data[i][:2] == "+1":
            ref[i] = '+1'
        else:
            ref[i] = '-1'
        reviews.append(train_data[i][2:])
    # list.append(" ".join(map(str, review)))
    # remove stop-words form corpus

    filtered_corpus = []  # positive reviews
    sentence = ""
    for i in range(len(reviews)):
        tkn = tokenizer.tokenize(reviews[i])
        for t in tkn:
            if len(t) > 2 and t not in stop_words and t not in punct and not t.isdigit():
                sentence += t + " "
        filtered_corpus.append(sentence[:-1])
        sentence = ""
    return filtered_corpus, ref


def parse_test_data(data):
    test_class = []
    sentence = ""
    for i in range(len(data)):
        tkn = tokenizer.tokenize(data[i])
        for t in tkn:
            if len(t) > 2 and t not in stop_words and t not in punct and not t.isdigit():
                sentence += t + " "
        test_class.append(sentence[:-1])
        sentence = ""

    return test_class


# [(1,2),(3,4),(5,6)]  ---> train
# [(6,7)]


def euclidean_dis(train, test):
    return np.sqrt(np.sum(np.square(train - test)))



print('reading files...')
train_data = read_file("train_data.txt")
test_data = read_file("test_data.txt")

print("training and validation splitting...")
train_split, validation_split = train_and_validation(train_data)

print('parsing data...')
training_data, train_ref = parse_train_data(train_split)
validation_data, valid_ref = parse_train_data(validation_split)
testing_data = parse_test_data(test_data)
print('vectorize document...')
tfid_vectorizer = TfidfVectorizer(norm='l2', smooth_idf=False, use_idf=True, max_features=7000)
training_vector = tfid_vectorizer.fit_transform(training_data)
validation_vector = tfid_vectorizer.fit_transform(validation_data)
testing_vector = tfid_vectorizer.fit_transform(testing_data)
print(training_vector.shape)
print(validation_vector.shape)
print(testing_vector.shape)
print('calculating svd...')

svd = TruncatedSVD(n_components=100)
train_matrix = svd.fit_transform(training_vector)
validation_matrix = svd.fit_transform(validation_vector)
test_matrix = svd.fit_transform(testing_vector)
print(train_matrix.shape)
print(validation_matrix.shape)
print(test_matrix.shape)
# print("#"*100)
# print(test_matrix)
# fig, (ax1, ax2) = plt.subplots(2)
# fig.suptitle('csr matrix and svd matrix')
# [(1,2),(3,4)]
# x, y = train_matrix[:][0], train_matrix[:][1]
# x2, y2 = test_matrix[:][0], test_matrix[:][1]

# a, b = training_vector.toarray()[:][0], training_vector.toarray()[:][1]
# a2, b2 = testing_vector.toarray()[:][0], testing_vector.toarray()[:][1]
# ax1.plot(x, y, 'o', color='black')
# ax1.plot(x2, y2, 'o', color='red')
# ax2.plot(a, b, 'o', color='black')
# ax2.plot(a2, b2, 'o', color='red')
# plt.savefig("training-vs-test")
# input("Press enter to continue to go through the test data....")

k = 7
counter = 0
correct_guess = 0
accuracy = .00
for row in validation_matrix:
    counter += 1
    dist = np.array(euclidean_dis(np.array(train_matrix), row))  # get the distances
    nearest_neighbors = dist.sort()[:k]  # splice only k amount
    neighbors_score = [train_ref[index] for index in nearest_neighbors]
    pos_rev = neighbors_score.count("+1")
    neg_rev = neighbors_score.count("-1")
    if pos_rev > neg_rev:
        if valid_ref[counter-1] == "+1":
            correct_guess += 1
            accuracy = correct_guess / counter
    else:
        if valid_ref[counter-1] == "-1":
            correct_guess += 1
            accuracy = correct_guess / counter
        # print(valid_ref[counter], "\t", "-1")
    sys.stdout.write("\rTraining time: {:.2f} seconds\t training accuracy: {:.2f}%\t data points processed: {}/1500".
                     format(time.process_time() - start, accuracy * 100.00, counter))
    sys.stdout.flush()
sys.stdout.write("\n")
sys.stdout.flush()
# print("positive reviews: ", pos_rev)
# print("negative reviews", neg_rev)
# input("Press enter to continue to go through the test data....")
# Prediction starts here...
accuracy = .00
counter = 0
guess = []
dist = []
for row in test_matrix:
    counter += 1
    dist = np.array(euclidean_dis(np.array(train_matrix), row))
    dist = np.array(dist)
    nearest_neighbors = dist.sort()[:k]  # splice only k amount
    neighbors_score = [train_ref[index] for index in nearest_neighbors]
    pos_rev = neighbors_score.count("+1")
    neg_rev = neighbors_score.count("-1")
    if pos_rev > neg_rev:
        guess.append("+1")
    else:
        guess.append("-1")
    sys.stdout.write("\rTesting Time: {:.2f} seconds\t data points processed: {}/15000".
                     format(time.process_time() - start, counter))
    sys.stdout.flush()

with open("Solution.txt", "w", encoding='utf8') as file:
    for score in guess:
        file.write(score + "\n")
