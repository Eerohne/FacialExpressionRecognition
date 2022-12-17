import pickle
from sklearn.metrics import accuracy_score
from classification import train_models, test_models


# Code to train models and test accuracy
# svm_full_hog, svm_full_lbp, svm_sal_hog, svm_sal_lbp = train_models("../assets/FER/train/")
# acc_score_full_hog, acc_score_full_lbp, acc_score_sal_hog, acc_score_sal_lbp = test_models("../assets/FER/test/",
#                                                                                            svm_full_hog,
#                                                                                            svm_full_lbp,
#                                                                                            svm_sal_hog,
#                                                                                            svm_sal_lbp)

# Load values
svm_full_hog = pickle.load(open("../assets/svm/svm_hog_full.pkl", "rb"))
svm_full_lbp = pickle.load(open("../assets/svm/svm_lbp_full.pkl", "rb"))
svm_sal_hog = pickle.load(open("../assets/svm/svm_hog_salient.pkl", "rb"))
svm_sal_lbp = pickle.load(open("../assets/svm/svm_lbp_salient.pkl", "rb"))

# Load training sets
X_full_hog = pickle.load(open("../assets/features/X_full_hog_train.pkl", "rb"))
X_full_lbp = pickle.load(open("../assets/features/X_full_lbp_train.pkl", "rb"))
X_salient_hog = pickle.load(open("../assets/features/X_salient_hog_train.pkl", "rb"))
X_salient_lbp = pickle.load(open("../assets/features/X_salient_lbp_train.pkl", "rb"))

y_full = pickle.load(open("../assets/features/y_full_train.pkl", "rb"))
y_sal = pickle.load(open("../assets/features/y_salient_train.pkl", "rb"))


# Predict values
y_full_hog_pred = svm_full_hog.predict(X_full_hog)
y_full_lbp_pred = svm_full_lbp.predict(X_full_lbp)
y_sal_hog_pred = svm_sal_hog.predict(X_salient_hog)
y_sal_lbp_pred = svm_sal_lbp.predict(X_salient_lbp)

# Compute accuracies
accuracy_score_full_hog = accuracy_score(y_full_hog_pred, y_full)
accuracy_score_full_lbp = accuracy_score(y_full_lbp_pred, y_full)
accuracy_score_sal_lbp = accuracy_score(y_sal_hog_pred, y_sal)
accuracy_score_sal_hog = accuracy_score(y_sal_lbp_pred, y_sal)

print("SVM Full HOG Train Accuracy: " + str(accuracy_score_full_hog))
print("SVM Full LBP Train Accuracy: " + str(accuracy_score_full_lbp))
print("SVM Salient HOG Train Accuracy: " + str(accuracy_score_sal_hog))
print("SVM Salient LBP Train Accuracy: " + str(accuracy_score_sal_lbp))






