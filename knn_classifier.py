import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer, sent_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD

from matplotlib import pyplot as plt


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
    for i in range(len(train_data)):
        if train_data[i][:2] == "+1":
            class_1.append(train_data[i][2:])
        else:
            class_2.append(train_data[i][2:])
    # list.append(" ".join(map(str, review)))
    # remove stop-words form corpus

    filtered_class_1 = []  # positive reviews
    filtered_class_2 = []  # negative reviews
    sentence = ""
    for i in range(len(class_1)):
        tkn = tokenizer.tokenize(class_1[i])
        for t in tkn:
            if len(t) > 2 and t not in stop_words and t not in punct and not t.isdigit():
                sentence += t + " "
        filtered_class_1.append(sentence[:-1])
        sentence = ""
        tkn = tokenizer.tokenize(class_2[i])
        for t in tkn:
            if len(t) > 2 and t not in stop_words and t not in punct and not t.isdigit():
                sentence += t + " "
        filtered_class_2.append(sentence[:-1])

    return filtered_class_1 + filtered_class_2


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


print('reading files...')
train_data = read_file("train_data.txt")
test_data = read_file("test_data.txt")
print('parsing data...')
training_data = parse_train_data(train_data)
testing_data = parse_test_data(test_data)
print('vectorize document...')
tfid_vectorizer = TfidfVectorizer()
training_vector = tfid_vectorizer.fit_transform(training_data)
testing_vector = tfid_vectorizer.fit_transform(testing_data)
print('calculating svd...')

svd = TruncatedSVD(n_components=100)
train_matrix = svd.fit_transform(training_vector)
test_matrix = svd.fit_transform(testing_vector)
# print(train_matrix.shape)
# print("#"*100)
# print(test_matrix)

x, y = train_matrix[:][0], train_matrix[:][1]
x2, y2 = test_matrix[:][0], test_matrix[:][1]
plt.plot(x, y, 'o', color='black')
plt.plot(x2, y2, 'o', color='red')
plt.show()
