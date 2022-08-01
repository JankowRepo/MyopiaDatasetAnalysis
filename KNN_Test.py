import math

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import cross_val_score, cross_validate
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.over_sampling import SMOTE


def Run_KNN(number_of_repeats, X, y, K, test_size):
    m, sc_mean, gmean_mean, acc_mean = 0, 0, 0, 0
    while m != number_of_repeats:
        X, y = SMOTE().fit_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        X_train = scale(X_train)
        X_test = scale(X_test)

        classifier = KNeighborsClassifier(n_neighbors=K, metric='minkowski')

        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)



        conf_matx = confusion_matrix(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        Sensitivity = conf_matx[0][0] / (conf_matx[0][0] + conf_matx[0][1])
        Specificity = conf_matx[1][1] / (conf_matx[1][0] + conf_matx[1][1])

        gmean = math.sqrt(Sensitivity * Specificity)
        sc_mean += sc
        gmean_mean += gmean
        acc_mean += acc
        print('Repeat number: ', m)
        print("Acc:", acc)
        print("F1_score:", sc)
        print("Geometric mean:", gmean)
        print('Confusion matrix: \n', conf_matx)
        Number_of_instances_of_decision_classes(y_train)
        m += 1
    print("Mean accuracy of models: ", acc_mean / number_of_repeats)
    print("Mean f1_score of models: ", sc_mean / number_of_repeats)
    print("Mean G-mean of models: ", gmean_mean / number_of_repeats)


def Show_acc_plots(X, y, K, test_size):
    X, y = SMOTE().fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    classifier = KNeighborsClassifier(n_neighbors=K)

    classifier.fit(X_train, y_train)
    error = []

    for i in range(1, 50):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error.append(np.mean(pred_i == y_test))

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 50), error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('ACC Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean accuracy')
    plt.show()


def Show_test_and_train_plot(X, y, K):
    plt.style.use('ggplot')
    X, y = SMOTE().fit_resample(X, y)
    knn = KNeighborsClassifier(n_neighbors=K)

    size = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    results = {}
    for i in size:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i)
        knn.fit(X_train, y_train)

        test = knn.score(X_test, y_test)
        train = knn.score(X_train, y_train)

        results[i] = train, test

    size_df = pd.DataFrame(results).transpose()
    size_df.columns = ['Train accuracy', 'Test accuracy']

    size_df.plot()
    plt.show()


def Run_KNN_with_CV(df, K, numberOfParts):
    df = df.sample(frac=1).reset_index(drop=True)
    X = df.iloc[:, :-4].values  # statystyki pacjentów
    y1 = df.iloc[:, -4].values  # Czy wystąpiło pogorszenie wzroku: pierwszy rok
    y2 = df.iloc[:, -3].values  # Czy wystąpiło pogorszenie wzroku: drugi rok
    y3 = df.iloc[:, -2].values  # Czy wystąpiło pogorszenie wzroku: trzeci rok
    yF = df.iloc[:, -1].values  # Podsumowanie poprzednich lat
    y = yF
    print("\n\n\nK-Fold Cross validation:\n")
    classifier = KNeighborsClassifier(n_neighbors=K)

    X, y = SMOTE().fit_resample(X, y)

    X = SelectKBest(score_func=chi2, k=5).fit_transform(X, y)
    cv_sc = cross_val_score(classifier, X, y, cv=numberOfParts, scoring='f1')
    cv_acc = cross_val_score(classifier, X, y, cv=numberOfParts, scoring='accuracy')
    Number_of_instances_of_decision_classes(y)
    for i in range(numberOfParts):
        print('Part', i + 1, ' sc:', cv_sc[i])
    print('CV f1_score mean: {}'.format(np.mean(cv_sc)), "\n")

    for i in range(numberOfParts):
        print('Part', i + 1, ' acc:', cv_acc[i])
    print('CV accuracy mean: {}'.format(np.mean(cv_acc)), "\n")

    k_range = range(1, 101)
    k_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, y, cv=numberOfParts, scoring='f1')
        k_scores.append(scores.mean())
    plt.plot(k_range, k_scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()


def Number_of_instances_of_decision_classes(y):
    print("Patients not at risk: " + str(np.count_nonzero(y == 0)))
    print("Patients at risk: " + str(np.count_nonzero(y == 1)), "\n\n")


df = pd.read_excel("4. Normalized_Data3.xlsx")

X = df.iloc[:, :-4].values  # statystyki pacjentów
y1 = df.iloc[:, -4].values  # Czy wystąpiło pogorszenie wzroku: pierwszy rok
y2 = df.iloc[:, -3].values  # Czy wystąpiło pogorszenie wzroku: drugi rok
y3 = df.iloc[:, -2].values  # Czy wystąpiło pogorszenie wzroku: trzeci rok
yF = df.iloc[:, -1].values  # Podsumowanie poprzednich lat
Y = yF

number_of_repeats = 1000
k = 5
test_size = 0.2

# Run_KNN(number_of_repeats, X, Y, k, test_size)

# Show_acc_plots(X, Y, k, test_size)

# Show_test_and_train_plot(X, Y, k)

CV_number_of_parts = 5
Run_KNN_with_CV(df, k, CV_number_of_parts)
