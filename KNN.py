import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import string
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


# nltk.download('stopwords')

stop_words = set(stopwords.words('english')).union(set(string.punctuation))
tokenizer = RegexpTokenizer(r"\w+")

class KNN:
    def __init__(self, training_file, test_file, k=3):
        self.k = k
        self.training_file = self.read_file(training_file)
        self.testing_file = self.read_file(test_file)
        self.train_set = None
        self.validation_set = None
        self.test_set = None
        self.tokenizer = RegexpTokenizer(r"\w+")  # to avoid any non-alphabetical characters
        self.stop_words = set(stopwords.words('english'))
        self.punct = set(string.punctuation)  # punctuations used in english
        self.num_comp = 100  # number of components to be used for svd

    def train(self):
        """
        uses the given dataset to train and validate using KNN algorithm
        :return: None.
        """
        self.train_set, self.validation_set = self.__train_and_validation_split__(self.training_file)
        self.test_set = self.__parse_test_data__(self.testing_file)
        print("split train, validation and testing sets...")
        training_reviews = self.__parse_train_data__(self.train_set)
        validation_reviews = self.__parse_train_data__(self.validation_set)
        print("parse training and validation sets...")
        truth_values = []  # this will be used to cross-check training accuracy
        for reviews in self.validation_set:
            truth_values.append(reviews[:2])
        print(len(truth_values))
        print("starting svd matrix")
        training_matrix = self.svd(self.__vectorize__(training_reviews))
        validation_matrix = self.svd(self.__vectorize__(validation_reviews))
        print("get svd matrix for training and validation sets...")
        euclidean_dis = []
        for row in range(len(training_matrix)):
            euclidean_dis.append(self.__calculate_distance__(training_matrix[row], validation_matrix[0]))
        print("Euclidean distance for the first row...")
        min_distance = min(euclidean_dis)
        guess = euclidean_dis.index(min_distance)
        print("the min distance is: ", min_distance)
        if guess > 5624:
            print(training_reviews[guess])
            print(validation_reviews[0])
            print("review is negative")
        else:
            print(training_reviews[guess])
            print(validation_reviews[0])
            print("review is positive")
        pass

    def __parse_train_data__(self, train_data):
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

        # remove stop-words form corpus

        filtered_class_1 = []  # positive reviews
        filtered_class_2 = []  # negative reviews
        sentence = ""
        for i in range(len(class_1)):
            tkn = tokenizer.tokenize(class_1[i])
            for t in tkn:
                if len(t) > 2 and t not in stop_words and not t.isdigit():
                    sentence += t + " "
            filtered_class_1.append(sentence[:-1])
            sentence = ""
        for i in range(len(class_2)):
            tkn = tokenizer.tokenize(class_2[i])
            for t in tkn:
                if len(t) > 2 and t not in stop_words and not t.isdigit():
                    sentence += t + " "
            filtered_class_2.append(sentence[:-1])

        return filtered_class_1 + filtered_class_2
    def __parse_test_data__(self, test_data):
        test_class = []
        sentence = ""
        for i in range(len(test_data)):
            tkn = self.tokenizer.tokenize(test_data[i])
            for t in tkn:
                if len(t) > 2 and t not in self.stop_words and not t.isdigit():
                    sentence += t + " "
            test_class.append(sentence[:-1])
            sentence = ""

        return test_class

    def __vectorize__(self, corpus):
        """
        convert the corpus to numeric values.
        :param corpus: the text.
        :return: vector representation of the corpus.
        """
        tfid_vectorizer = TfidfVectorizer()
        return tfid_vectorizer.fit_transform(corpus)

    def svd(self, vector):
        """
        calculates the svd from the sparse matrix.
        :param vector: sparse matrix
        :return: transformed matrix
        """
        svd = TruncatedSVD()
        return svd.fit_transform(vector)

    def __calculate_distance__(self, x, y):
        """
        calculates euclidean distance of two sets.
        :param x: set one
        :param y: set two
        :return: the euclidean distance
        """
        return np.sqrt(np.sum((x - y) ** 2))

    def predict(self, test_dataset):
        """
        Based on previous train dt, predicts the class of the test dt.
        :param test_dataset: np array type, the test dataset
        :return: the classification.
        """
        pass

    def read_file(self, file_name, encoding="utf8"):
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

    def __train_and_validation_split__(self, train_dt):
        """
        splits the train dt into train and validate sets after randomizing.
        :param train_dt: the training dt.
        :return: tuple (train_dt, validate_dt)
        """
        train_dt = np.array(train_dt)
        np.random.shuffle(train_dt)  # randomize the dataset
        # 75% train - 25% validation
        size = int(len(train_dt) * .75)
        return train_dt[:size], train_dt[size:]
