import sframe
products = sframe.SFrame('amazon_baby.gl/')


def remove_punctuation(text):
    import string
    return text.translate(None, string.punctuation)

products['review_clean'] = products['review'].apply(remove_punctuation)

products = products[products['rating'] != 3]

products['sentiment'] = products['rating'].apply(lambda rating : +1 if rating > 3 else -1)

train_data, test_data = products.random_split(.8, seed=1)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
     # Use this token pattern to keep single-letter words
# First, learn vocabulary from the training data and assign columns to words
# Then convert the training data into a sparse matrix
train_matrix = vectorizer.fit_transform(train_data['review_clean'])

# Second, convert the test data into a sparse matrix, using the same word-column mapping
test_matrix = vectorizer.transform(test_data['review_clean'])

train_data['sentiment'].shape
train_matrix.shape

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model = model.fit(train_matrix, train_data['sentiment'])

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true=test_data['sentiment'].to_numpy(), y_pred=model.predict(test_matrix))
print "Test Accuracy: %s" % accuracy



baseline = len(test_data[test_data['sentiment'] == 1])/float(len(test_data))
print "Baseline accuracy (majority class classifier): %s" % baseline

from sklearn.metrics import confusion_matrix
cmat = confusion_matrix(y_true=test_data['sentiment'].to_numpy(),
                        y_pred=model.predict(test_matrix),
                        labels=model.classes_)    # use the same order of class as the LR model.
cmat
print ' target_label | predicted_label | count '
print '--------------+-----------------+-------'
# Print out the confusion matrix.
# NOTE: Your tool may arrange entries in a different order. Consult appropriate manuals.
for i, target_label in enumerate(model.classes_):
    for j, predicted_label in enumerate(model.classes_):
        print '{0:^13} | {1:^15} | {2:5d}'.format(target_label, predicted_label, cmat[i,j])


fp = 1453
fn = 803
tp = 27292
tn = 3788
fp * 100 + fn * 1

#fdr
fp/float(tp+fp)


#### Precision Recall
from sklearn.metrics import precision_score
precision = precision_score(y_true=test_data['sentiment'].to_numpy(),
                            y_pred=model.predict(test_matrix))
print "Precision on test data: %s" % precision


from sklearn.metrics import recall_score
recall = recall_score(y_true=test_data['sentiment'].to_numpy(),
                      y_pred=model.predict(test_matrix))
print "Recall on test data: %s" % recall


import numpy as np
recall = recall_score(y_true=test_data['sentiment'].to_numpy(),
                      y_pred=np.ones(len(test_data['sentiment'])))
print "Recall on test data: %s" % recall


#### Varying the threshold
def apply_threshold(probabilities, threshold):
    ret = np.zeros(len(probabilities))
    for i in xrange(len(probabilities)):
        if(probabilities[i] >= threshold):
            ret[i] = +1
        else:
            ret[i] = -1
    return ret

print apply_threshold([0.3,0.4,0.5], 0.45)

probabilities = model.predict_proba(test_matrix)[:,1]

prob_threshold_05 = apply_threshold(probabilities, 0.5)
prob_threshold_09 = apply_threshold(probabilities, 0.9)

sum(prob_threshold_05 == 1)
sum(prob_threshold_09 == 1)

#### Exploring the associated precision and recall as the threshold varies

precision = precision_score(y_true=test_data['sentiment'].to_numpy(),
                            y_pred=prob_threshold_05)
print "Precision on test data: %s" % precision

recall = recall_score(y_true=test_data['sentiment'].to_numpy(),
                      y_pred=prob_threshold_05)
print "Recall on test data: %s" % recall

#### ------------------------------------------------- ####

precision = precision_score(y_true=test_data['sentiment'].to_numpy(),
                            y_pred=prob_threshold_09)
print "Precision on test data: %s" % precision

recall = recall_score(y_true=test_data['sentiment'].to_numpy(),
                      y_pred=prob_threshold_09)
print "Recall on test data: %s" % recall


#### Precision-recall curve

threshold_values = np.linspace(0.5, 1, num=100)
print threshold_values


precision_all = np.zeros(len(threshold_values))
recall_all = np.zeros(len(threshold_values))
for i in xrange(len(threshold_values)):

    prob_thres = apply_threshold(probabilities, threshold_values[i])
    precision_all[i] = precision_score(y_true=test_data['sentiment'].to_numpy(),
                                y_pred=prob_thres)
    recall_all[i] = recall_score(y_true=test_data['sentiment'].to_numpy(),
                          y_pred=prob_thres)

def plot_pr_curve(precision, recall, title):
    plt.rcParams['figure.figsize'] = 7, 5
    plt.locator_params(axis = 'x', nbins = 5)
    plt.plot(precision, recall, 'b-', linewidth=4.0, color = '#B0017F')
    plt.title(title)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.rcParams.update({'font.size': 16})

import matplotlib.pyplot as plt
plot_pr_curve(precision_all, recall_all, 'Precision recall curve (all)')
plt.show()



for i in xrange(len(threshold_values)):
    if(precision_all[i]>=0.965):
        print threshold_values[i]
        break


prob_threshold_098 = apply_threshold(probabilities, 0.98)
cmat = confusion_matrix(y_true=test_data['sentiment'].to_numpy(),
                        y_pred=prob_threshold_098,
                        labels=model.classes_)

print ' target_label | predicted_label | count '
print '--------------+-----------------+-------'
# Print out the confusion matrix.
# NOTE: Your tool may arrange entries in a different order. Consult appropriate manuals.
for i, target_label in enumerate(model.classes_):
    for j, predicted_label in enumerate(model.classes_):
        print '{0:^13} | {1:^15} | {2:5d}'.format(target_label, predicted_label, cmat[i,j])
fn = 8220


#### Precision-Recall on all baby related items

baby_reviews = test_data[test_data['name'].apply(lambda x: 'baby' in x.lower())]

baby_matrix = vectorizer.transform(baby_reviews['review_clean'])
probabilities = model.predict_proba(baby_matrix)[:,1]

threshold_values = np.linspace(0.5, 1, num=100)

precision_all = np.zeros(len(threshold_values))
recall_all = np.zeros(len(threshold_values))
for i in xrange(len(threshold_values)):

    prob_thres = apply_threshold(probabilities, threshold_values[i])
    precision_all[i] = precision_score(y_true=baby_reviews['sentiment'].to_numpy(),
                                y_pred=prob_thres)
    recall_all[i] = recall_score(y_true=baby_reviews['sentiment'].to_numpy(),
                          y_pred=prob_thres)


for i in xrange(len(threshold_values)):
    if(precision_all[i]>=0.965):
        print threshold_values[i]
        break




plot_pr_curve(precision_all, recall_all, 'Precision recall curve (all)')
plt.show()

#
