from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier # Stochastic gradient descent classifier
from sklearn.multiclass import OneVsOneClassifier # One vs One strategy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold # splits the dataset into stratified folds for cross-validation
from sklearn.base import clone # this is used to clone other model keeping parameters and throwing away the data
from sklearn.metrics import precision_score, recall_score


# P.S. I wrote the code on jupyter, so some lines might not print 
# how do I check distribution in this case ? normal or uniform

mnist = fetch_openml('mnist_784', version=1, as_frame= False)
# This dataset is pretty big so some codes might take lil bit time

print(mnist.keys())
print(mnist.DESCR)

X, y = mnist["data"], mnist["target"]

print(type(X))  # Nice numpy is always nice

print(X.shape)
print(y.shape)

# As stated in the DESCR each image is 28x28 pixels
# We can try to plot a digit now
some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image)
plt.axis("off")
plt.show() # Don't really need this in jupyter

print(some_digit_image)
print(type(y[0]))
# We gotta convert it to int
y = y.astype('int')


# So according to DESCR the dataset it already split, so we don't need to do a train test split ✌️
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# Am gonna use some cross-validation folds in this code, that's just creating tinier train and validation sets
# The dataset is already shuffled toh like achi baat hai

# Training a Binary Classifier
# So like first thing I am gonna train a binary classifier
# Cause why not
# Also training a bigger one in a single attempt seems a bit tough to me

# So this classifier is going to differentiate between fives and non fives
# So our training data is going to be the same
# But like our labels just gonna be true or false
# true for when its a five and false otherwise
y_train_5 = ( y_train == 5 )
print(y_train_5) # Voila
y_test_5 = ( y_test == 5 )


# Its good for online learning
# ASK: It's working and how it varies wrt the normal gradient descent strategy taught in class ?
sgd_classifier = SGDClassifier(random_state = 42) # so results can be resiprocates
# Imported an instance now we train it
sgd_classifier.fit(X_train, y_train_5)

# Let's predict it now
print(sgd_classifier.predict([some_digit]))
print(y[0])
# Yes the classifier correctly indentifies the 5

# Now we trained a binary classifier 
# Now we are going to test it's accuracy

# First let's jus see how many correct predictions it makes


# now first we split our training data into train and validation
# also cross validation cause nai toh poora dataset use nai hoga
# Setting a random_state has no effect since shuffle is False. 
# You should leave random_state to its default (None), or set shuffle=True.
strat_k_folds = StratifiedKFold(n_splits = 5) # try three training phases

for train_indices, test_indices in strat_k_folds.split(X_train, y_train_5):
    # now we clone our gradient descent model, jus the model tho not its fits and estimates
    clone_classifier = clone(sgd_classifier)
    
    # Don't use our actual test case here, jus the train is split 
    X_train_folds, y_train_folds = X_train[train_indices], y_train_5[train_indices]
    X_test_folds, y_test_folds = X_train[test_indices], y_train_5[test_indices]
    
    # training it now
    clone_classifier.fit(X_train_folds, y_train_folds)
    answers = clone_classifier.predict(X_test_folds)
    correct = np.sum(answers == y_test_folds)
    print(correct/len(y_test_folds))
    # Accuracy is pretty good

cross_val_score(sgd_classifier, X_train, y_train_5, cv=5, scoring='accuracy') # this does the same thing 
# It uses 5 fold cross validation by default too 

# The accuracy above isn't that accurate of a measure 
# That's because only like 10% os the images are fives
# We'll get a 90% accuracy even if we predict them all as non-fives
# So like in case of classification we can't predict a model just based on the correct predictions

# CONFUSION MATRIX

# Like cross_val_score returns a score on each cross validation
# cross_val_predict returns the predictions for each fold
def conf_two(actual, predicted, ret_precision = False): # confusion matrix but just for two
    true_positives = (actual&predicted).sum()
    true_negatives = (~actual&~predicted).sum()
    false_negatives = (actual&~predicted).sum()
    false_positives = (~actual&predicted).sum()
    precision = true_positives/(true_positives + false_positives)
    if ret_precision:
        return np.array([[true_negatives, false_positives],[false_negatives, true_positives]]), precision
    return np.array([[true_negatives, false_positives],[false_negatives, true_positives]])


# Now we are goign to use cross validation on our confusion matrix
y_train_pred = cross_val_predict(sgd_classifier, X_train, y_train_5, cv = 5)
# Ok so what cross_val_score is gonna do is its gonna split tis dataset into 5 parts
# then taking each part as the test_set its gonna predict 
# then its gonna group all these tes_sets to give us back the total predictions
print(y_train_pred.shape)

print(conf_two(y_train_5, y_train_pred))

conf_mat_5 = conf_two(y_train_5, y_train_pred)

def ret_precision(conf_matrix):
    TP = conf_matrix[1][1]
    FP = conf_matrix[0][1]
    return TP/(TP + FP)

def ret_recall(conf_matrix):
    TP = conf_matrix[1][1]
    FN = conf_matrix[1][0]
    return TP/(TP + FN)

precision = ret_precision(conf_mat_5)
# We can't really use precision alone tho 
# cause we can jus lower it down by making only those preds we're extremely sure about
recall = ret_recall(conf_mat_5) # low recall good for medical tests

print('Precision :- ',precision)
print('Recall :- ',recall)

# Can also use the sklearn ones
print(precision_score(y_train_5, y_train_pred)) 
print(recall_score(y_train_5, y_train_pred))
# Now the classifier looks a bit nice than it did a while ago

f1_score = 2*precision*recall/(precision + recall)
print(f1_score) # f1_score is high iff both recall and precision are high

# Increasing precision reduces recall and vice versa
# SGD classifier makes it decisions based ona threshold 
# It evaluates a value for each instance and decides wheter it passes or fails the threshold based on that value
# We can increase the precision by increasing the threshold
# An we can increare the recall by lowering the threshold
# the sgd classifier provides us with th decision scores on its own 
y_score = sgd_classifier.decision_function([some_digit])

y_scores = sgd_classifier.decision_function(X_train)

y_scores.max()
y_pred_1000 = y_scores > 1000
y_pred_5000 = y_scores > 5000
y_pred_8000 = y_scores > 8000
conf_1000 = conf_two(y_train_5, y_pred_1000)
conf_5000 = conf_two(y_train_5, y_pred_5000)
conf_8000 = conf_two(y_train_5, y_pred_8000)
print("When threshold is 1000 - ", ret_precision(conf_1000), ret_recall(conf_1000))
print("When threshold is 5000 - ", ret_precision(conf_5000), ret_recall(conf_5000))
print("When threshold is 8000 - ", ret_precision(conf_8000), ret_recall(conf_8000))

y_scores = cross_val_predict(sgd_classifier, X_train, y_train_5, cv=5, method="decision_function")

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

for i in range(10):
    print(precisions[i], recalls[i], thresholds[i])

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds): 
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision") 
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.legend(['Precision', 'Recall'])

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, 'b--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    
plot_precision_vs_recall(precisions, recalls)
plt.show()

# We can first try and aim for more than 90% precision 
# so like uske liye threshold dhoondhna padega
reqd_index = np.argmax(precisions >= 0.90)
reqd_threshold = thresholds[reqd_index]

print(reqd_threshold)

y_train_pred_90 = (y_scores > reqd_threshold)

conf_matrix_90precision = conf_two(y_train_5, y_train_pred_90)
print(ret_precision(conf_matrix_90precision), ret_recall(conf_matrix_90precision))
# the recall is pretty low tbh
# like we don't want our classifier to just skip over alot of the fives

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
f1_scores = 2*precisions*recalls/(precisions + recalls)
plt.plot(thresholds, f1_scores[:-1])
# It reaches max f1_score for a threshold of zero I'm guessing, mightve been offset ? 
f1_max = np.argmax(f1_scores)
print(precisions[f1_max], recalls[f1_max], thresholds[f1_max])

print(recalls.shape,precisions.shape,thresholds.shape)

# ROC curve plots TPR (recall(sensitivity)) against FPR( 1- TNR(specificity)), for different values of threshold
FPR, TPR, thresholds = roc_curve(y_train_5, y_scores)

def plot_roc_curve(FPR, TPR, label = None):
    plt.plot(FPR, TPR, linewidth = 2, label = label)
    plt.plot([0,1],[0,1],'k--') # line that represents a bad classifier

plot_roc_curve(FPR, TPR)
plt.show()

# The area under the ROC curve for a perfect classifier would be 1
# Lets calculate the area for ours
roc_auc_score(y_train_5, y_scores)
# That's pretty good but like we saw in the PR curve its not doing that good

# Now lets try a random forest classifier, tbh i don't exaclty know what it is
forest_classifier = RandomForestClassifier(random_state=42)
y_probs_forest = cross_val_predict(forest_classifier, X_train, y_train_5, cv = 5, method='predict_proba')
# This takes a while to run

print(y_probs_forest)
# So it gives the probabilities as
# Not five, five
y_scores_forest = y_probs_forest[:, 1]

# Now lets plot an ROC curve for this too
FPR_forest, TPR_forest, THRESHOLD_forest = roc_curve(y_train_5, y_scores_forest)
plt.plot(FPR, TPR, 'g-',label= 'SGD')
plt.plot(FPR_forest, TPR_forest, 'b-',label='random forest')
plt.plot([0,1],[0,1],'k--')
plt.legend(loc="lower right")
plt.show()

roc_auc_score(y_train_5, y_scores_forest) # woahhhh, this seems like overfitting


# MUTLI-CLASS CLASSIFIERS
sgd_classifier.fit(X_train, y_train) # One vs All strategy
print(sgd_classifier.predict([X_train[2]]))
print(y_train[2])

print(sgd_classifier.decision_function([X_train[2]])) 
# scores achieved by each of the 10 under the hood binary classifiers

print(sgd_classifier.classes_)

ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42)) 
ovo_clf.fit(X_train, y_train)
ovo_clf.predict([some_digit]) # Hey the OvsA SGD classifier predicted this wrong

# Random forest classifiers can directly sort data into multiple classes
forest_classifier.fit(X_train, y_train)
print(forest_classifier.predict([some_digit])) # It sorts 5 correctly toooo

print(forest_classifier.predict_proba([some_digit]))

# Now les evaluate them 
cross_val_score(sgd_classifier, X_train, y_train, cv = 5, scoring='accuracy') # this takes real long to run 5-10mins
# cause it's building like 50 classifiers
# it's got 88% + accuracy

cross_val_score(forest_classifier, X_train, y_train, cv = 5, scoring = 'accuracy')
# nice

y_train_pred = cross_val_predict(forest_classifier, X_train, y_train, cv = 5)
conf_final = confusion_matrix(y_train, y_train_pred)
conf_final

plt.matshow(conf_final, cmap='gray')
plt.show()

row_sums = conf_final.sum(axis = 1, keepdims = True)
normalised_conf_mx = conf_final/row_sums

plt.matshow(normalised_conf_mx, cmap='gray')

np.fill_diagonal(normalised_conf_mx, 0)
plt.matshow(normalised_conf_mx, cmap='gray')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 8 gets misclassified alot as other digits
# but fir bhi like a lot of digits don't get misclassified as 8

report = classification_report(y_train, y_train_pred)
print(report) # That's pretty good

knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)
y_train_knn_pred = knn_classifier.predict(X_train)
print(classification_report(y_train, y_train_knn_pred))
# better than forest classifier

# now lets try it on the test set
knn_classifier.fit(X_train, y_train)
knn_pred = knn_classifier.predict(X_test)

forest_classifier.fit(X_train, y_train)
forest_pred = forest_classifier.predict(X_test)

print('KNN results - ')
print(classification_report(y_test, knn_pred))
print('Forest predictions')
print(classification_report(y_test, forest_pred))