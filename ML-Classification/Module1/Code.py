import sframe
products = sframe.SFrame('amazon_baby.gl/')

#cleanup the reviews (remove punctuation)
def remove_punctuation(text):
    import string
    return text.translate(None, string.punctuation)

products['review_clean'] = products['review'].apply(remove_punctuation)

#fill nas
products = products.fillna('review','')  # fill in N/A's in the review column

#remove rating 3 (middle)
products = products[products['rating'] != 3]
#create sentiment from ratings
products['sentiment'] = products['rating'].apply(lambda rating : +1 if rating > 3 else -1)
#split training and test set
train_data, test_data = products.random_split(.8, seed=1)

#create the sparse matrix (work counts) as features
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
# Use this token pattern to keep single-letter words
# First, learn vocabulary from the training data and assign columns to words
# Then convert the training data into a sparse matrix
train_matrix = vectorizer.fit_transform(train_data['review_clean'])
# Second, convert the test data into a sparse matrix, using the same word-column mapping
test_matrix = vectorizer.transform(test_data['review_clean'])

#train a linear model (logistic regression)
from sklearn import linear_model

logreg = linear_model.LogisticRegression()
logreg.fit(train_matrix, train_data["sentiment"])

logreg.coef_.shape
len(logreg.coef_[0,])
sum(logreg.coef_[0,]>=0)
#Quiz question: How many weights are >= 0? 86211

#look at sampesl 10:13
sample_test_data = test_data[10:13]
print sample_test_data
sample_test_data[0]['review']
sample_test_data[1]['review']

#look at the scores
sample_test_matrix = vectorizer.transform(sample_test_data['review_clean'])
scores = logreg.decision_function(sample_test_matrix)
print scores

#predict the results (based on the scores)
logreg.predict(sample_test_matrix)
test_data["pred"] = logreg.predict(test_matrix)

import numpy as np
def predict_from_score(s):
    if( s<= 0 ):
        return -1
    else:
        return 1

predict_from_score_v = np.vectorize(predict_from_score)
predict_from_score_v(scores)

#calculate the probabilies
logreg.predict_proba(sample_test_matrix)
from scipy.stats import logistic
logistic.cdf(scores)
#Quiz question: Of the three data points in sample_test_data, which one (first, second, or third) has the lowest probability of being classified as a positive review?
#Third 2.95 * 10^-5


test_data["pred_proba"] = logreg.predict_proba(test_matrix)[:,1]
test_data_tmp = test_data.sort("pred_proba", ascending=False)
test_data_tmp.print_rows(21)
test_data_tmp = test_data.sort("pred_proba", ascending=True)
test_data_tmp.print_rows(21)

#Quiz Question: Which of the following products are represented in the 20 most positive reviews?
#Britax
#The First..., Peg-Perego..., Safety....

#Calc Accuracy
test_data["pred"] = logreg.predict(test_matrix)
test_data[1:100]

sum(test_data["pred"] == test_data['sentiment'])/float(len(test_data))

#Quiz Question: What is the accuracy of the sentiment_model on the test_data? Round your answer to 2 decimal places (e.g. 0.76).
#0.93

#fewer words
significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves',
      'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed',
      'work', 'product', 'money', 'would', 'return']

vectorizer_word_subset = CountVectorizer(vocabulary=significant_words) # limit to 20 words
train_matrix_word_subset = vectorizer_word_subset.fit_transform(train_data['review_clean'])
test_matrix_word_subset = vectorizer_word_subset.transform(test_data['review_clean'])

simple_model = linear_model.LogisticRegression()
simple_model.fit(train_matrix_word_subset, train_data["sentiment"])

simple_model_coef_table = sframe.SFrame({'word':significant_words,
                                         'coefficient':simple_model.coef_.flatten()})

simple_model_coef_table = simple_model_coef_table.sort("coefficient", ascending=False)
simple_model_coef_table.print_rows(20)
sum(simple_model_coef_table["coefficient"] > 0)
#Quiz Question: Consider the coefficients of simple_model. How many of the 20 coefficients (corresponding to the 20 significant_words) are positive for the simple_model?
# 10


for i in significant_words:
    print(i)
    print(logreg.coef_.flatten()[vectorizer.vocabulary_[i]])

#Quiz Question: Are the positive words in the simple_model also positive words in the sentiment_model?
#YES

train_data_pred = logreg.predict(train_matrix)
train_data_pred_simple = simple_model.predict(train_matrix_word_subset)

sum(train_data_pred == train_data['sentiment'])/float(len(train_data))
sum(train_data_pred_simple == train_data['sentiment'])/float(len(train_data))

#Quiz Question: Which model (sentiment_model or simple_model) has higher accuracy on the TRAINING set?
# sentiment_model


test_data["pred_simle"] = simple_model.predict(test_matrix_word_subset)

sum(test_data["pred"] == test_data['sentiment'])/float(len(test_data))
sum(test_data["pred_simle"] == test_data['sentiment'])/float(len(test_data))

#Quiz Question: Which model (sentiment_model or simple_model) has higher accuracy on the TEST set?
# sentiment_model


sum(test_data['sentiment']==1)/float(len(test_data))
#Quiz Question: Enter the accuracy of the majority class classifier model on the test_data. Round your answer to two decimal places (e.g. 0.76).
#0,84

#Quiz Question: Is the sentiment_model definitely better than the majority class classifier (the baseline)?
#YES
