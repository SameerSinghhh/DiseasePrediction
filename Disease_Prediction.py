# -*- coding: utf-8 -*-
"""ENGR_100_Project.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1gdPoxn1qHFGQqARoE99tnmnVB8y4p1Bm

#Disease Classification

Sameer Singh, Joe Marcotte, Ian Nadeau, Sriram Kumaran

1. Import the data from a csv file

 a. How many samples (rows of data) do we have?
"""

import pandas as pd
import numpy as np

df = pd.read_csv("diagnosis_and_symptoms.csv")

print("1a.")
print("Number of samples: ", df.shape[0])

"""2. Separate data into labels and symptoms

 a. Split the data into an array of labels and an array of symptom lists. Do this such that each
index of the label array corresponds to that same index of the symptom lists array. You
should end with two variables, one as an array of labels, and the other as an array of
symptom lists.

 b. Iterate over each sample, and for each sample, split the list of symptoms into individual
strings. The final output should look as follows:
labels[0] = ‘common cold’
symptoms[0] = [‘runny_nose’, ‘cough’, ‘fever’]

"""

lab_and_symp = df.iloc[:, 0]
other_symptoms = df.iloc[:, 1:]

labels = []
symptoms = []

idx = 0

for row in lab_and_symp:
  info = row.split(';')

  labels.append(info[0])
  symptoms.append(info[1:])

for i in range(0, 15):
  rowIdx = 0
  for row in other_symptoms.iloc[:, i]:
    isNaN = pd.isna(row)
    if isNaN == False:
      symptoms[rowIdx].append(row)
    rowIdx = rowIdx + 1

"""3. One-hot encode symptoms into a feature matrix

 a. How many total unique symptoms are in our data?

 b. Generate a one-hot encoded matrix of symptom data. What are the dimensions of this one-hot encoded symptom matrix? Return the total list of unique symptoms and the one-hot encoded matrix for machine learning.
"""

unique_symptoms = []

for row in symptoms:
  for symptom in row:
    if np.isin(symptom, unique_symptoms) == False:
      unique_symptoms.append(symptom)

unique_symptoms = np.array(unique_symptoms)

print("3a.")
print("Unique symptoms: ", unique_symptoms.shape[0])

encoded_df = pd.DataFrame(0, index=labels, columns=unique_symptoms)

for i in range(0, 5362):
  for j in range(0, 131):
    if unique_symptoms[j] in symptoms[i]:
      encoded_df.iloc[i].at[unique_symptoms[j]] = 1

print("3b.")
print("Dimensions of one-hot encoded symptom matrix: ", encoded_df.shape)
print("\nTotal list of unique symptoms:\n", unique_symptoms)
print("\nOne-hot encoded matrix:\n", encoded_df)

"""4. Dimensionality reduction
  
  a. Perform dimensionality reduction with PCA, justify your choice of n_components. How
much of the explained variance ratio do your PCA features describe? Next, you will
compare the classification model predictions using the PCA features vs. the original
feature space in questions 5-6.
"""

from sklearn.decomposition import PCA

X = encoded_df.iloc[:, 0:].values

pca = PCA(n_components=40)
pca_symptoms = pca.fit_transform(X)

print("4a.")
print("n_components = 40 because the total explained variance ratio at this value is close to 1, but the number of components is on the lower end.")
print("The PCA features describe ", sum(pca.explained_variance_ratio_), " of the explained variance ratio")

import matplotlib.pyplot as plt
pca_test = PCA(n_components=131)
pca_symptoms_test = pca_test.fit_transform(X)

plt.bar(range(1, 132), pca_test.explained_variance_ratio_, alpha=0.5, align='center')
plt.step(range(1, 132), np.cumsum(pca_test.explained_variance_ratio_), where='mid')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')

plt.show()

"""5. Implement at least two supervised classification models, where the input features are the
PCA features and the output prediction is a diagnosis

 a. Justify your train/test split ratio and your ML model selection choice.

 b. Use GridSearchCV to optimize the supervised classification model hyperparameters.
Justify your choices of which hyperparameters to optimize.

 c. Report the best_estimator_, best_score_ and best_params_ identified from
GridSearchCV .

 d. What is the classification accuracy on test set for each of the models after hyperparameter
tuning?
"""

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

y = encoded_df.index

X_train, X_test, y_train, y_test = train_test_split(pca_symptoms, y, test_size=0.2, stratify=y)

params = {'criterion': ['gini', 'entropy'], 'n_estimators': [30, 40, 50, 75, 100, 125, 150, 175, 200], 'max_depth': [2, 3, 4, 5, 6]}
tree = GridSearchCV(RandomForestClassifier(), params, n_jobs=-1, verbose=1, cv=6)

tree.fit(X_train, y_train)

y_pred_test = tree.predict(X_test)

print("5a.")
print("The 80%/20% train/test split ratio was chosen because there are over 5000 data points, so giving 20% of these to the test set leaves enough")
print("for the model to be tested well. A random forest model was chosen because it is an ensemble method. A decision tree model was chosen because")
print("it is what makes up random forest models and they are able to deal with problems that involve a lot of possible classifications.")

print("Random Forest")

print("\n5b.")
print("We chose two different possible criterion since they determine splits differently. We chose a wide range of the number of trees to have")
print("in the random forest since the number of trees will impact the accuracy of the model. We chose a range of max depths for the trees since the")
print("depth of each tree also impacts the accuracy of the model.")

print("\n5c.")
print("Best estimator: ", tree.best_estimator_)
print("Best score: ", tree.best_score_)
print("Best params: ", tree.best_params_)

print("\n5d.")
print("Test accuracy: ", accuracy_score(y_pred_test, y_test))

from sklearn.tree import DecisionTreeClassifier

params = {'criterion': ['gini', 'entropy'], 'min_samples_split': [2, 3, 4, 5, 6], 'max_depth': [2, 3, 4, 5, 6]}
tree2 = GridSearchCV(DecisionTreeClassifier(), params, n_jobs=-1, verbose=1, cv=6)

tree2.fit(X_train, y_train)

y_pred_test_2 = tree2.predict(X_test)

print("Decision Tree")

print("\n5b.")
print("We chose two different possible criterion since they determine splits differently. We chose a range of the minimum number of samples required")
print("to split at a node because this impacts the number of splits and the splits impact the accuracy. We chose a range of max depths since the depth")
print("impacts the accuracy of the model.")

print("\n5c.")
print("Best estimator: ", tree2.best_estimator_)
print("Best score: ", tree2.best_score_)
print("Best params: ", tree2.best_params_)

print("\n5d.")
print("Test accuracy: ", accuracy_score(y_pred_test_2, y_test))

"""6. Compare classifier performance trained on the reduced dimensionality dataset (PCA)
against the original dataset without dimensionality reduction

 a. Repeat Question 5 but with the original feature space. Compare and contrast model
performance using the PCA feature space vs. the original feature space.
"""

X, y = encoded_df.iloc[:, :].values, encoded_df.index

X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, stratify=y)

params = {'criterion': ['gini', 'entropy'], 'n_estimators': [30, 40, 50, 75, 100, 125, 150, 175, 200], 'max_depth': [2, 3, 4, 5, 6]}
tree3 = GridSearchCV(RandomForestClassifier(), params, n_jobs=-1, verbose=1, cv=6)

tree3.fit(X_train2, y_train2)

y_pred_test3 = tree3.predict(X_test2)

print("6.")

print("Random Forest")

print("We chose two different possible criterion since they determine splits differently. We chose a wide range of the number of trees to have")
print("in the random forest since the number of trees will impact the accuracy of the model. We chose a range of max depths for the trees since the")
print("depth of each tree also impacts the accuracy of the model.")

print("\nBest estimator: ", tree3.best_estimator_)
print("Best score: ", tree3.best_score_)
print("Best params: ", tree3.best_params_)

print("\nTest accuracy: ", accuracy_score(y_pred_test3, y_test2))

params = {'criterion': ['gini', 'entropy'], 'min_samples_split': [2, 3, 4, 5, 6], 'max_depth': [2, 3, 4, 5, 6]}
tree4 = GridSearchCV(DecisionTreeClassifier(), params, n_jobs=-1, verbose=1, cv=6)

tree4.fit(X_train2, y_train2)

y_pred_test_4 = tree4.predict(X_test2)

print("6.")
print("Decision Tree")

print("We chose two different possible criterion since they determine splits differently. We chose a range of the minimum number of samples required")
print("to split at a node because this impacts the number of splits and the splits impact the accuracy. We chose a range of max depths since the depth")
print("impacts the accuracy of the model.")

print("\nBest estimator: ", tree4.best_estimator_)
print("Best score: ", tree4.best_score_)
print("Best params: ", tree4.best_params_)

print("\nTest accuracy: ", accuracy_score(y_pred_test_4, y_test2))

print("6a.")
print("Both of the models that use the PCA feature space perform better than the models that use the original feature space.")

"""7. Implement additional performance metrics for the best classifier from question 6.

 a. Implement four additional performance metrics. Where TP = true positive, FP = false
positive, TN = true negative, and FN = false negative.

 i. Balanced Accuracy: (sensitivity + specificity) / (2)

 ii. Precision: TP / (TP + FP)

 iii. Recall: TP / (TP + FN)

 iv. F1-Score: (2 * Recall * Precision) / (Recall + Precision)

 b. Measure the performance of your tuned classification models against the above four
metrics. Report your findings.
"""

from sklearn.metrics import recall_score, precision_score, f1_score, balanced_accuracy_score

print("7a. and 7b.")
print("\nPCA Random Forest")
print("Balanced accuracy: ", balanced_accuracy_score(y_test, y_pred_test))
print("Precision: ", precision_score(y_test, y_pred_test, average="macro"))
print("Recall: ", recall_score(y_test, y_pred_test, average="macro"))
print("F1-Score: ", f1_score(y_test, y_pred_test, average="macro"))

print("\nPCA Decision Tree")
print("Balanced accuracy: ", balanced_accuracy_score(y_test, y_pred_test_2))
print("Precision: ", precision_score(y_test, y_pred_test_2, average="macro"))
print("Recall: ", recall_score(y_test, y_pred_test_2, average="macro"))
print("F1-Score: ", f1_score(y_test, y_pred_test_2, average="macro"))

print("\nOriginal Random Forest")
print("Balanced accuracy: ", balanced_accuracy_score(y_test2, y_pred_test3))
print("Precision: ", precision_score(y_test2, y_pred_test3, average="macro", zero_division=0))
print("Recall: ", recall_score(y_test2, y_pred_test3, average="macro"))
print("F1-Score: ", f1_score(y_test2, y_pred_test3, average="macro"))

print("\nOriginal Decision Tree")
print("Balanced accuracy: ", balanced_accuracy_score(y_test2, y_pred_test_4))
print("Precision: ", precision_score(y_test2, y_pred_test_4, average="macro", zero_division=0))
print("Recall: ", recall_score(y_test2, y_pred_test_4, average="macro"))
print("F1-Score: ", f1_score(y_test2, y_pred_test_4, average="macro"))

"""8. Analyze performance and discuss findings
  
  a. Which of your implemented classifiers performed the best? With which feature space?
According to which performance metrics?

  b. Do your findings lead you to believe that your classifiers should be used to predict a
diagnosis? Justify your answer
"""

print("8a.")
print("The random forest model with the PCA feature space and the decision tree model with the PCA feature space performed the best")
print("according to the balanced accuracy, precision, recall, and F1-score metrics. These two models performed similarly, and they both")
print("did considerably better in terms of all the metrics compared to the random forest model and the decision tree model with the original feature space.")

print("\n8b.")
print("The random forest model with the PCA feature space should be used to predict diagnoses according to the")
print("balanced accuracy, precision, recall, and F1-score metrics because they are all high and they are all the best across all the models.")
print("However, this model is not perfect, so its use in the real world should be done with a certain level of skepticism considering what the consequences of giving a wrong diagnosis can be.")

"""9. Generate average feature vector for each disease
  
  a. Using the original one-hot encoded data (before dimensionality reduction), generate an average feature vector for diagnosis. For example, every sample with a diagnosis of “common cold” would be averaged into an averaged “common cold” feature vector of symptoms. Return a dataset with c rows and d columns, where c is the number of unique prognoses and d is the total number of symptoms
"""

unique_diagnoses = []
number_of_diagnoses = []

for label in labels:
    if np.isin(label, unique_diagnoses) == False:
      unique_diagnoses.append(label)
      number_of_diagnoses.append([label, 1])
    else:
      for info in number_of_diagnoses:
        if info[0] == label:
          info[1] = info[1] + 1;

number_of_diagnoses = np.array(number_of_diagnoses)

average_features = pd.DataFrame(0, index=unique_diagnoses, columns=unique_symptoms)

for i in range(0, 5362):
    average_features.loc[encoded_df.index[i]] = average_features.loc[encoded_df.index[i]] + encoded_df.iloc[i]

for i in range(0,41):
  average_features.iloc[i] = average_features.iloc[i] / number_of_diagnoses[i][1].astype('float')

print("Average feature vector shape: ", average_features.shape)

"""10. Cluster diseases and tune for optimal k-clusters
  
  a. Perform k-means clustering on your dataset of averaged feature vectors for each
diagnosis. Optimize for k, and provide justification for your final choice.
  
  b. Using your final choice of k, report the clusters of diseases. For example:
Cluster 1: Common Cold, Influenza
Cluster 2: Gallstones, Peptic Ulcers, Celiac Diseas
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X_k_means, y_k_means = average_features.iloc[:, :].values, average_features.index

inertia_values = []
mean_silhouette = []

for i in range(2, 40):
  kmeans = KMeans(init='k-means++', n_clusters=i, n_init='auto').fit(X_k_means)
  predicted_labels = kmeans.predict(X_k_means)
  inertia_values.append(kmeans.inertia_)
  mean_silhouette.append(silhouette_score(X_k_means, predicted_labels))

figure, axis_one = plt.subplots()
axis_two = axis_one.twinx()

one = axis_one.plot(range(2, 40), inertia_values, marker="o", label="Distortion")
axis_one.set_xlabel("Number of Clusters")
axis_one.set_ylabel("Distortion")

two = axis_two.plot(range(2, 40), mean_silhouette, marker="o", color="orange", label="Mean Silhouette Coeff.")
axis_two.set_ylabel("Mean Silhouette Coeff.")

axis_one.legend(one+two, [one[0].get_label(), two[0].get_label()], loc=0)

plt.title("Distortion and Mean Silhouette Coeff. vs Number of Clusters")
plt.show()

print("10a.")
print("The final choice for k is 8 because the mean silhouette coefficient is relatively high compared to other values of k.")

print("10b.")
cluster_num = 8

clusters = []
cluster_nums = []
idx = 0

predictions = KMeans(init='k-means++', n_clusters=cluster_num, n_init='auto').fit_predict(X_k_means)

for i in predictions:
    if np.isin(i, cluster_nums) == False:
      clusters.append([i, y_k_means[idx]])
      cluster_nums.append(i)
    else:
      for info in clusters:
        if info[0] == i:
          info.append(y_k_means[idx])
    idx = idx + 1

for i in range(0, cluster_num):
  for j in range(0, cluster_num):
    if(clusters[j][0] == i):
      print("Cluster ", clusters[j][0],": ", clusters[j][1:])

"""11. Dimensionality reduction and visualization
  
  a. Perform PCA on the dataset of average feature vectors to reduce dimensionality down to
n_components = 3. Also report how much of the explained variance ratio these three components describe, note the difference with PCA performed in step 4. Use matplotlib to visualize a 3-dimensional scatter plot, where each point is the average features of a diagnosis. Color each point based on its cluster assignment. Report this cluster plot.
"""

pca_k_means = PCA(n_components=3)
pca_average_features = pca_k_means.fit_transform(X_k_means)

component_one = pca_average_features[:, 0]
component_two = pca_average_features[:, 1]
component_three = pca_average_features[:, 2]

figure = plt.figure(figsize=[7,7])
axis = figure.add_subplot(projection='3d')

scatter = axis.scatter(component_one, component_two, component_three, c=predictions, cmap="tab20")
axis.set_xlabel("Component One", labelpad=5)
axis.set_ylabel("Component Two", labelpad=5)
axis.set_zlabel("Component Three", labelpad=5)
axis.legend(*scatter.legend_elements(), title="Clusters", loc='upper left')

axis.set_box_aspect(aspect=None, zoom=0.93)

plt.tight_layout()
plt.title("Disease Clusters")

print("11a.")
print("These three components explain ", sum(pca_k_means.explained_variance_ratio_), " of the explained variance ratio. This is significantly")
print("less than the components generated from the PCA performed in step 4 which explained about .938 of the")
print("explained variance ratio.")

"""12. Additional analysis
  
  a. Create your own question to answer and report on it (feel free to do more than one question). For example, you could try other ML models, predict classification probabilities, or examine the effects of outliers. Ask us if you are unsure about which questions are good to investigate
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = encoded_df.iloc[:, :].values, encoded_df.index

X_train, X_test, y_train, y_test = train_test_split(pca_symptoms, y, test_size=0.2, stratify=y)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, stratify=y)

pca_accuracies = []
accuracies = []

for i in range(1, 31):
  model = KNeighborsClassifier(n_neighbors=i)
  pca_model = KNeighborsClassifier(n_neighbors=i)

  model.fit(X_train2, y_train2)
  pca_model.fit(X_train, y_train)

  y_pred = model.predict(X_test2)
  pca_y_pred = pca_model.predict(X_test)

  accuracy = accuracy_score(y_pred, y_test2)
  pca_accuracy = accuracy_score(pca_y_pred, y_test)

  accuracies.append(accuracy)
  pca_accuracies.append(pca_accuracy)

plt.plot(range(1,31), pca_accuracies, marker="o", label="PCA Accuracy")
plt.plot(range(1,31), accuracies, color="orange", marker="o", label="Non-PCA Accuracy")
plt.xlabel("Number Of Neighbors")
plt.ylabel("Accuracy")
plt.title("Model Accuracy vs Number Of Neighbors")
plt.legend(loc="lower right")
plt.tight_layout()

from sklearn.metrics import recall_score, precision_score, f1_score, balanced_accuracy_score

pca_model = KNeighborsClassifier(n_neighbors=10)
model = KNeighborsClassifier(n_neighbors=10)

model.fit(X_train2, y_train2)
pca_model.fit(X_train, y_train)

y_pred = model.predict(X_test2)
pca_y_pred = pca_model.predict(X_test)

print("PCA Test Accuracy: ", accuracy_score(pca_y_pred, y_test))
print("Regular Test Accuracy: ", accuracy_score(y_pred, y_test2))

print("\nPCA K-Nearest Neighbors Metrics: ")
print("Balanced accuracy: ", balanced_accuracy_score(y_test, pca_y_pred))
print("Precision: ", precision_score(y_test, pca_y_pred, average="macro"))
print("Recall: ", recall_score(y_test, pca_y_pred, average="macro"))
print("F1-Score: ", f1_score(y_test, pca_y_pred, average="macro"))

print("\nOriginal K-Nearest Neighbors Metrics: ")
print("Balanced accuracy: ", balanced_accuracy_score(y_test2, y_pred))
print("Precision: ", precision_score(y_test2, y_pred, average="macro"))
print("Recall: ", recall_score(y_test2, y_pred, average="macro"))
print("F1-Score: ", f1_score(y_test2, y_pred, average="macro"))