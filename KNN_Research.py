import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, matthews_corrcoef, roc_curve
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.over_sampling import SMOTE
from sklearn import metrics


def ChooseSamples(df, method):
    df = df.sample(frac=1).reset_index(drop=True)
    X = df.iloc[:, :-4].values  # attributes
    y = df.iloc[:, -1].values  # decision classes
    X = SelectKBest(score_func=chi2, k=2).fit_transform(X, y)

    if method == 1:
        X, y = SMOTE().fit_resample(X, y)

    elif method == 2:
        patients_not_at_risk = df.where(df["Degradation of sight final decision"] == 0)
        patients_at_risk = df.loc[:10].where(df["Degradation of sight final decision"] == 1)
        df = pd.concat([patients_not_at_risk, patients_at_risk])
        df = df.reset_index(drop=True).dropna()
        X = df.iloc[:, :-4].values
        y = df.iloc[:, -1].values
        X = SelectKBest(score_func=chi2, k=2).fit_transform(X, y)

    # elif method==3:

    return X, y


def Number_of_instances_of_decision_classes(df, method):
    X, y = ChooseSamples(df, method)
    print("Patients not at risk: " + str(np.count_nonzero(y == 0)))
    print("Patients at risk: " + str(np.count_nonzero(y == 1)), "\n")


def Run_KNN(number_of_repeats, df, K, numberOfParts, method):
    acc_sum, sc_sum, gmean_sum, mcc_sum = 0, 0, 0, 0
    for repeat in range(number_of_repeats):
        X, y = ChooseSamples(df, method)
        acc, sc, gmean, mcc = Run_KNN_Algorithm(X, y, K, numberOfParts, repeat)
        acc_sum += acc
        sc_sum += sc
        gmean_sum += gmean
        mcc_sum += mcc

    print("Mean accuracy of models: ", acc_sum / number_of_repeats)
    print("Mean f1_score of models: ", sc_sum / number_of_repeats)
    print("Mean G-mean of models: ", gmean_sum / number_of_repeats)
    print("Mean MCC of models: ", mcc_sum / number_of_repeats)
    print("\n\n")
    Number_of_instances_of_decision_classes(df, method)


def Run_KNN_Algorithm(X, y, K, numberOfParts, loop):
    classifier = KNeighborsClassifier(n_neighbors=K)

    cv_sc = cross_validate(classifier, X, y, cv=numberOfParts, scoring='f1').get('test_score')

    cv_acc = cross_validate(classifier, X, y, cv=numberOfParts, scoring='accuracy').get('test_score')

    y_pred = cross_val_predict(classifier, X, y, cv=numberOfParts)
    conf_matx = confusion_matrix(y, y_pred)

    Sensitivity = conf_matx[0][0] / (conf_matx[0][0] + conf_matx[0][1])
    Specificity = conf_matx[1][1] / (conf_matx[1][0] + conf_matx[1][1])
    gmean = math.sqrt(Sensitivity * Specificity)

    mcc = matthews_corrcoef(y, y_pred)

    print("Number of loop: ", loop)
    print('Accuracy: ', np.mean(cv_acc))
    print('F1_score: ', np.mean(cv_sc))
    print('G_mean: ', gmean)
    print('MCC', mcc)
    print('Confusion matrix:\n {}'.format(conf_matx))
    print("\n")

    return np.mean(cv_acc), np.mean(cv_sc), gmean, mcc


def Show_KNN_Plot(df, numberOfParts, max_k, repeats, method):
    k_range = range(1, max_k + 1, 2)
    k_scores = []
    k_scores_sum = np.zeros(k_range[-1])
    for i in range(repeats):
        X, y = ChooseSamples(df, method)
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, X, y, cv=numberOfParts, scoring='f1')
            k_scores.append(scores.mean())
        k_scores_sum = [x + y for x, y in zip(k_scores_sum, k_scores)]
        print("Loop ", i)

    plt.plot(k_range, [score / repeats for score in k_scores_sum])
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Cross-Validated score')
    plt.show()


def Show_ROC_Plot(df, numberOfParts, method, k):
    X, y = ChooseSamples(df, method)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train = scale(X_train)
    X_test = scale(X_test)

    KNN = KNeighborsClassifier(n_neighbors=k)

    KNN.fit(X_train, y_train)

    y_scores = KNN.predict_proba(X_test)
    fpr, tpr, threshold = roc_curve(y_test, y_scores[:, 1])
    roc_auc = metrics.auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=('ROC curve', round(roc_auc, 2)))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) for KNN, k=' + str(k))
    plt.legend(loc="lower right")
    plt.show()


df = pd.read_excel("4. Normalized_Data3.xlsx")

method_for_balance_data = 1  # 0=No method, 1=Oversampling, 2=Undersampling
number_of_repeats = 100
k = 13
number_of_parts_for_CV = 5
years = 5
# Run_KNN(number_of_repeats, df, k, number_of_parts_for_CV, method_for_balance_data)

max_k = 100
# Show_KNN_Plot(df, number_of_parts_for_CV, max_k,number_of_repeats, method_for_balance_data)


Show_ROC_Plot(df, number_of_parts_for_CV, method_for_balance_data, k)
