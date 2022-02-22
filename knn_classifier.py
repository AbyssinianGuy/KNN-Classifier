import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer, sent_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import numpy as np
# nltk.download('stopwords')


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


def parse_train_data(train_data):
    """
    splits the label from the review.
    :param train_data: list of the train_data
    :return: the filtered text.
    """

    class_1 = []  # positive reviews
    class_2 = []  # negative reviews
    tokenizer = RegexpTokenizer(r"\w+")  # to avoid any non-alphabetical characters
    for i in range(len(train_data)):
        if train_data[i][:2] == "+1":
            class_1.append(train_data[i][2:])
        else:
            class_2.append(train_data[i][2:])

    # remove stop-words form corpus
    stop_words = set(stopwords.words('english'))
    punct = set(string.punctuation)  # punctuations used in english
    filtered_class_1 = []  # positive reviews
    filtered_class_2 = []  # negative reviews
    sent_tkn = ()
    sent_tkn2 = ()
    for i in range(len(class_1)):
        tkn = tokenizer.tokenize(class_1[i])
        for t in tkn:
            if t not in stop_words and t not in punct:
                filtered_class_1.append(t)
        tkn = tokenizer.tokenize(class_2[i])
        for t in tkn:
            if t not in stop_words and t not in punct:
                filtered_class_2.append(t)

    return filtered_class_1, filtered_class_2


train_data = read_file("train_data.txt")
test_data = read_file("test_data.txt")

# print("train_data length = ",len(train_data))
# print("test_data length = ",len(test_data))
train_label, train_text = parse_train_data(train_data)
# print(train_label[0].__str__())
# print(train_text)
concatenated_sentence = []
concatenated_sentence2 = []
for word, word2 in zip(train_label, train_text):
    if word != "br":
        if word.isalpha():
            concatenated_sentence.append(word)
    if word2 != "br":
        if word2.isalpha():
            concatenated_sentence2.append(word2)

concatenated_sentence = concatenated_sentence[:-1]
concatenated_sentence2 = concatenated_sentence2[:-1]
# print(concatenated_sentence)

def freq(corpus1, corpus2):
    freq1 = nltk.FreqDist(corpus1)
    freq2 = nltk.FreqDist(corpus2)
    dic1 = dict((w, fr) for w, fr in freq1.items())
    dic2 = dict((w, fr) for w, fr in freq2.items())
    return dic1, dic2


d1, d2 = freq(concatenated_sentence, concatenated_sentence2)
# print(d1)
label_arry = np.array(d1.keys()).astype(str)
vector_arry = np.array(d1.values())


print(label_arry)


# print(d2)
# cv = CountVectorizer()
# with open("Test.txt", 'w+')as file:
#     file.write(concatenated_sentence)
# file = open("Test.txt")
# word_count_vector = cv.fit_transform(file)
# print(word_count_vector.shape)
# print(word_count_vector)
# tfidf_transformer = TfidfVectorizer()
# train_data = tfidf_transformer.fit_transform(concatenated_sentence)
# print(type(train_data))
# print(train_data)
# print(train_data)
u, s, v = np.linalg.svd([label_arry, vector_arry], compute_uv=True, full_matrices = False)

