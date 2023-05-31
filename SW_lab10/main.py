import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import cluster
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def convert_descriptors_to_histogram(descriptors, vocab_model):
    predictions = vocab_model.predict(descriptors)
    histogram = np.histogram(predictions, bins=NB_WORDS)
    histogram = np.array([val/histogram[0].sum() for val in histogram[0]]).reshape(1, NB_WORDS)
    return histogram


def apply_feature_transform(images, feature_detector_descriptor, vocab_model):
    histograms = np.empty((0, NB_WORDS), dtype=float)
    for img in images:
        keypoints, descriptors = feature_detector_descriptor.detectAndCompute(img, None)
        histogram = convert_descriptors_to_histogram(descriptors, vocab_model)
        histograms = np.concatenate((histograms, histogram), axis=0)
    return histograms


boats = []
targets = []
boat_dirs = ['example/sailboat', 'example/warship']

# Read data
for boat_dir in boat_dirs:
    for img_name in os.listdir(boat_dir):
        boats.append(cv2.imread(os.path.join(boat_dir, img_name)))
        if boat_dir == 'example/sailboat':
            targets.append(0)
        else:
            targets.append(1)

# Split data into training, validation and test sets
boats_train, boats_test, targets_train, targets_test = train_test_split(boats, targets, test_size=0.2, random_state=42,
                                                                        stratify=targets)
# boats_train, boats_val, targets_train, targets_val = train_test_split(boats_train, targets_train, test_size=0.25,
#                                                                       random_state=42, stratify=targets_train)

# Calculate descriptors of features in training set
akaze = cv2.AKAZE_create()
des_arr = np.empty((0, 61), dtype=int)
for boat in boats_train:
    kpts, des = akaze.detectAndCompute(boat, None)
    des_arr = np.concatenate((des_arr, des), axis=0)

# Clustering model
NB_WORDS = 15
kmeans = cluster.KMeans(n_clusters=NB_WORDS, random_state=42)
kmeans.fit(des_arr)

# Calculate histograms
histograms_train = apply_feature_transform(boats_train, akaze, kmeans)
# histograms_val = apply_feature_transform(boats_val, akaze, kmeans)
histograms_test = apply_feature_transform(boats_test, akaze, kmeans)

# Final classification
parameters = {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': range(1, 10),
        'min_samples_split': range(2, 10),
        'min_samples_leaf': range(1, 5)
    }

clf = GridSearchCV(DecisionTreeClassifier(random_state=0), parameters, cv=3)
clf.fit(histograms_train, targets_train)
print('GridSearch DTC best params: ', clf.best_params_)
# print('GridSearch DTC best score: ', clf.best_score_)
# print('Classification score: ', clf.best_estimator_.score(histograms_val, targets_val))
predicted = clf.best_estimator_.predict(histograms_test)
print('Classification score: ', clf.best_estimator_.score(histograms_test, targets_test))

cm = confusion_matrix(targets_test, predicted)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
