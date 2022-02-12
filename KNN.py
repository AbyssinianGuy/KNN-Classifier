import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import string


class KNN:
    def __init__(self, k=4):
        self.k = k
        self.train_set = None
        self.validation_set = None
        self.test_set = None

    def train(self, train_dataset):
        """
        uses the given dataset to train and validate using KNN algorithm
        :param train_dataset: np array type, the dataset to be trained on
        :return: None.
        """
        self.train_set, self.validation_set = self.__train_and_validation_split__(train_dataset)
        '''
        1. tokenize the corpus
        2. filter high and low frequency words
        3. perform stemming to avoid processing same root words multiple times (optional)
        4. 
        '''
        positive_reviews, negative_reviews = self.__parse_train_data__(self.train_set)
        pos_parag = []
        neg_parag = []
        for words1, words2 in zip(positive_reviews, negative_reviews):
            #  make sure to filter the word "br" leftover from "</br>"
            if words1 != "br":
                pos_parag.append(words1)
            if words2 != "br":
                neg_parag.append(words2)


        pass

    def __parse_train_data__(self, train_data):
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
        for i in range(len(class_1)):
            tkn = tokenizer.tokenize(class_1[i])
            for t in tkn:
                if t not in stop_words and t not in punct:
                    filtered_class_1.append(t)
            print(filtered_class_1)
            tkn = tokenizer.tokenize(class_2[i])
            for t in tkn:
                if t not in stop_words and t not in punct:
                    filtered_class_2.append(t)

        return filtered_class_1, filtered_class_2

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
