from nltk import tokenize
import numpy as np


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
            lines.append(reader)
            reader = file.readline()
    return lines


def parse_train_data(train_data):
    """
    splits the label from the review.
    :param train_data: list of the train_data
    :return: tuple of data and the review text.
    """
    label = []
    data = []
    for i in range(len(train_data)):
        label.append(train_data[i][:2])
        data.append(train_data[i][2:])
    return label, data


def calculate_dist(dim, dt_pts):
    """
    calculates the median distance from the origin to the nearest neighbor.
    :param dim: the dimension of the sphere.
    :param dt_pts: the number of data points.
    :return: the median distance
    """
    return (1 - 0.5 ** (1 / dt_pts)) ** (1 / dim)






train_data = read_file("train_data.txt")
test_data = read_file("test_data.txt")

# print("train_data length = ",len(train_data))
# print("test_data length = ",len(test_data))
train_label, train_text = parse_train_data(train_data)
