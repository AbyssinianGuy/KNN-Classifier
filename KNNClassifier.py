# Importing libraries
from sklearn.model_selection import train_test_split
from scipy.stats import mode
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import TruncatedSVD

from datetime import datetime
import nltk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords

stopwords = nltk.corpus.stopwords.words('english')
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.stem import WordNetLemmatizer


# K Nearest Neighbors Classification
class K_Nearest_Neighbors_Classifier():

    def __init__(self, K):
        self.K = K

    def extractFeatures(self, input_values):
        # lemmatization (stemming and lemmatization?)
        lemmatizer = WordNetLemmatizer()
        xValues = []
        # text preprocessing
        for i in range(0, len(input_values)):
            review = re.sub('[^a-zA-Z]', ' ', input_values[i])
            review = review.lower()
            review = review.split()
            review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords)]
            review = ' '.join(review)
            xValues.append(review)

        ##############################################################
        # tf idf
        tf_idf = TfidfVectorizer()
        # applying tf idf to training data
        X_train_tf = tf_idf.fit_transform(xValues)  # for training data
        svd = TruncatedSVD(n_components=50)
        train_X = svd.fit_transform(X_train_tf)

        return train_X

    def formTrainDataset(self, input_values, output_values):
        # convert list to array
        output_values = np.array(output_values)
        # convert 1D array 2D array
        output_values = np.reshape(output_values, (len(output_values), 1))  # print(output_values.shape)

        tf_idf_vector = self.extractFeatures(input_values)
        # Put back the labels
        input_values = tf_idf_vector  # input_values = tf_idf_vector.toarray()
        # input_values = X_train_tf.toarray()

        # Concatenating operation to add output labels at the end of the sample
        # axis = 1 implies that it is being done column-wise
        dataset = np.concatenate((input_values, output_values), axis=1)  # merge input and output
        return dataset

    def formTestDataset(self, input_values):
        tf_idf_vector = self.extractFeatures(input_values)
        input_values = tf_idf_vector  # input_values = tf_idf_vector.toarray()
        dataset = input_values
        return dataset

    def get_train_dataset(self):
        inputFile = open("train_data.txt", 'r', encoding='utf-8')
        lines = inputFile.readlines()
        inputFile.close()
        output_values = []  # train_y
        input_values = []  # train_x
        for sample in lines:
            output_values.append(int(sample[0:2]))  # extract label (+1 or -1)
            input_values.append(sample[2:])  # extract text

        # converting the text into 2D numbers
        dataset = self.formTrainDataset(input_values, output_values)
        return dataset

    def get_test_dataset(self):
        inputFile = open("test_data.txt", 'r', encoding='utf-8')
        lines = inputFile.readlines()
        inputFile.close()
        input_values = []  # test_x
        for sample in lines:
            input_values.append(sample)  # extract text

        # converting the text into 2D numbers
        dataset = self.formTestDataset(input_values)
        return dataset

    # Function to store training set
    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

        # no_of_training_examples, no_of_features
        self.m, self.n = X_train.shape

    # Function for prediction
    def predict(self, X_test):
        self.X_test = X_test
        # no_of_test_examples, no_of_features
        self.m_test, self.n = X_test.shape

        # initialize Y_predict
        Y_predict = np.zeros(self.m_test)
        for i in range(self.m_test):
            x = self.X_test[i]
            # find the K nearest neighbors from current test example
            neighbors = np.zeros(self.K)
            neighbors = self.find_neighbors(x)

            # most frequent class in K neighbors
            # Y_predict[i] = max(set(neighbors_output_values), key=neighbors_output_values.count)
            Y_predict[i] = mode(neighbors)[0][0]
        return Y_predict

    # Function to find the K nearest neighbors to current test example
    def find_neighbors(self, x):
        # calculate all the euclidean distances between current
        # test example x and training set X_train
        euclidean_distances = np.zeros(self.m)
        for i in range(self.m):
            d = self.euclidean(x, self.X_train[i])
            euclidean_distances[i] = d

        # sort Y_train according to euclidean_distance_array and
        # store into Y_train_sorted
        inds = euclidean_distances.argsort()
        Y_train_sorted = self.Y_train[inds]
        return Y_train_sorted[:self.K]

    # Function to calculate euclidean distance
    def euclidean(self, x, x_train):
        return np.sqrt(np.sum(np.square(x - x_train)))


# main function
def main():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    # Defined KNN Model
    model = K_Nearest_Neighbors_Classifier(K=7)    # K is the number of neighbors

    # Importing dataset
    train_dataset = model.get_train_dataset()
    test_dataset = model.get_test_dataset()
    X = train_dataset[:, 0:len(train_dataset[0]) - 1]  # input
    Y = train_dataset[:, [len(train_dataset[0]) - 1]]  # output: last column of the train_dataset
    Y = Y.ravel()

    # comment out stats here
    # Splitting dataset into train and test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 / 3, random_state=0)
    print("Training dataset shape")
    print(X_train.shape)
    print("Test dataset shape")
    print(X_test.shape)
    model.fit(X_train, Y_train)

    model1 = KNeighborsClassifier(n_neighbors=7)  # model from sklearn.neighbors
    model1.fit(X_train, Y_train)

    # Prediction on test set
    Y_pred = model.predict(X_test)
    Y_pred1 = model1.predict(X_test)

    # measure performance
    correctly_classified = 0
    correctly_classified1 = 0

    # counter
    count = 0
    for count in range(np.size(Y_pred)):
        if Y_test[count] == Y_pred[count]:
            correctly_classified = correctly_classified + 1

        if Y_test[count] == Y_pred1[count]:
            correctly_classified1 = correctly_classified1 + 1

        count = count + 1

    print("Accuracy on test set by our model	 : ", (correctly_classified / count) * 100)
    print("Accuracy on test set by sklearn model : ", (correctly_classified1 / count) * 100)
    # comment out ends here
    """
    print("Training input shape")
    print(X.shape)
    print("Training output shape")
    print(Y.shape)
    print("Test input shape")
    print(test_dataset.shape)
    model.fit(X_train, Y_train)    # training
    predictions = model.predict(test_dataset)   # prediction
    output_file = open("format.txt", "w")
    for i in range(0, len(predictions)):
        if 1 == int(predictions[i]):
            output_file.writelines("+1")
        else:
            output_file.writelines(str(int(predictions[i])))
        output_file.writelines("\n")
    output_file.close()
    """
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)


if __name__ == "__main__":
    main()
