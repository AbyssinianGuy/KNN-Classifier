import KNN

knn_classifier = KNN.KNN("train_data.txt", "test_data.txt", 5)

knn_classifier.train()
