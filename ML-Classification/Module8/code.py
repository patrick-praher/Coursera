import sframe
loans = sframe.SFrame('lending-club-data.gl/')

loans.column_names()

# safe_loans =  1 => safe
# safe_loans = -1 => risky
loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.remove_column('bad_loans')

target = 'safe_loans'
features = ['grade',                     # grade of the loan (categorical)
            'sub_grade_num',             # sub-grade of the loan as a number from 0 to 1
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'payment_inc_ratio',         # ratio of the monthly payment to income
            'delinq_2yrs',               # number of delinquincies
             'delinq_2yrs_zero',          # no delinquincies in last 2 years
            'inq_last_6mths',            # number of creditor inquiries in last 6 months
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'open_acc',                  # number of open credit accounts
            'pub_rec',                   # number of derogatory public records
            'pub_rec_zero',              # no derogatory public records
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
            'int_rate',                  # interest rate of the loan
            'total_rec_int',             # interest received to date
            'annual_inc',                # annual income of borrower
            'funded_amnt',               # amount committed to the loan
            'funded_amnt_inv',           # amount committed by investors for the loan
            'installment',               # monthly payment owed by the borrower
           ]

loans, loans_with_na = loans[[target] + features].dropna_split()

# Count the number of rows with missing data
num_rows_with_na = loans_with_na.num_rows()
num_rows = loans.num_rows()
print 'Dropping %s observations; keeping %s ' % (num_rows_with_na, num_rows)

######## Balance Data ########

safe_loans_raw = loans[loans[target] == 1]
risky_loans_raw = loans[loans[target] == -1]

# Undersample the safe loans.
percentage = len(risky_loans_raw)/float(len(safe_loans_raw))
safe_loans = safe_loans_raw.sample(percentage, seed = 1)
risky_loans = risky_loans_raw
loans_data = risky_loans.append(safe_loans)

print "Percentage of safe loans                 :", len(safe_loans) / float(len(loans_data))
print "Percentage of risky loans                :", len(risky_loans) / float(len(loans_data))
print "Total number of loans in our new dataset :", len(loans_data)


######## One-hot encoding ########

loans_data = risky_loans.append(safe_loans)

categorical_variables = []
for feat_name, feat_type in zip(loans_data.column_names(), loans_data.column_types()):
    if feat_type == str:
        categorical_variables.append(feat_name)

for feature in categorical_variables:
    loans_data_one_hot_encoded = loans_data[feature].apply(lambda x: {x: 1})
    loans_data_unpacked = loans_data_one_hot_encoded.unpack(column_name_prefix=feature)

    # Change None's to 0's
    for column in loans_data_unpacked.column_names():
        loans_data_unpacked[column] = loans_data_unpacked[column].fillna(0)

    loans_data.remove_column(feature)
    loans_data.add_columns(loans_data_unpacked)

loans_data.column_names()

######## train / validation split ########
train_data, validation_data = loans_data.random_split(.8, seed=1)


######## Training ########

from sklearn.ensemble import GradientBoostingClassifier
import numpy as np


features_oh = train_data.column_names()
features_oh.remove(target)


model_5 = GradientBoostingClassifier(n_estimators=5, max_depth=6).fit(train_data[features_oh].to_numpy(), train_data[target].to_numpy())

######## Prediction ########
validation_safe_loans = validation_data[validation_data[target] == 1]
validation_risky_loans = validation_data[validation_data[target] == -1]

sample_validation_data_risky = validation_risky_loans[0:2]
sample_validation_data_safe = validation_safe_loans[0:2]

sample_validation_data = sample_validation_data_safe.append(sample_validation_data_risky)
sample_validation_data

#Quiz question: What percentage of the predictions on sample_validation_data did model_5 get correct?
model_5.predict(sample_validation_data[features_oh].to_numpy())
sample_validation_data[target]

#Quiz Question: Which loan has the highest probability of being classified as a safe loan?
model_5.predict_proba(sample_validation_data[features_oh].to_numpy())

model_5.score(validation_data[features_oh].to_numpy(),validation_data[target].to_numpy())

#Quiz question: What is the number of false positives on the validation_data?
#FP
pred_m5 = model_5.predict(validation_data[features_oh].to_numpy())
correct = validation_data[target].to_numpy()
fp = 0
for i in xrange(len(pred_m5)):
    if pred_m5[i] == 1 and correct[i] == -1:
        fp += 1
print fp

#FN
fn = 0
for i in xrange(len(pred_m5)):
    if pred_m5[i] == -1 and correct[i] == 1:
        fn += 1
print fn


######## cost ########
#Quiz Question: Using the same costs of the false positives and false negatives, what is the cost of the mistakes made by the boosted tree model (model_5) as evaluated on the validation_set?

cost = 10000 * fn  + 20000 * fp
cost

######## Most positive & negative loans.

pred_proba_m5 = model_5.predict_proba(validation_data[features_oh].to_numpy())
validation_data["pred_proba"] = pred_proba_m5[:,1]

validation_data_tmp = validation_data.sort("pred_proba", ascending=False)

#Quiz question: What grades are the top 5 loans?
validation_data_tmp[ 'grade.A', 'grade.B', 'grade.C', 'grade.D', 'grade.E', 'grade.F', 'grade.G'].print_rows(5)
#grade.A


#Effects of adding more trees

model_10 = GradientBoostingClassifier(n_estimators=10, max_depth=6).fit(train_data[features_oh].to_numpy(), train_data[target].to_numpy())
model_50 = GradientBoostingClassifier(n_estimators=50, max_depth=6).fit(train_data[features_oh].to_numpy(), train_data[target].to_numpy())
model_100 = GradientBoostingClassifier(n_estimators=100, max_depth=6).fit(train_data[features_oh].to_numpy(), train_data[target].to_numpy())
model_200 = GradientBoostingClassifier(n_estimators=200, max_depth=6).fit(train_data[features_oh].to_numpy(), train_data[target].to_numpy())
model_500 = GradientBoostingClassifier(n_estimators=500, max_depth=6).fit(train_data[features_oh].to_numpy(), train_data[target].to_numpy())

######## Compare accuracy on entire validation set

acc_m10 = model_10.score(validation_data[features_oh].to_numpy(),validation_data[target].to_numpy())
acc_m50 = model_50.score(validation_data[features_oh].to_numpy(),validation_data[target].to_numpy())
acc_m100 = model_100.score(validation_data[features_oh].to_numpy(),validation_data[target].to_numpy())
acc_m200 = model_200.score(validation_data[features_oh].to_numpy(),validation_data[target].to_numpy())
acc_m500 = model_500.score(validation_data[features_oh].to_numpy(),validation_data[target].to_numpy())
print acc_m10
print acc_m50
print acc_m100
print acc_m200
print acc_m500

# Quiz Question: Which model has the best accuracy on the validation_data?
#500

# Quiz Question: Is it always true that the model with the most trees will perform best on test data?
#no

######## Plot the training and validation error vs. number of trees

import matplotlib.pyplot as plt
%matplotlib inline
def make_figure(dim, title, xlabel, ylabel, legend):
    plt.rcParams['figure.figsize'] = dim
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend is not None:
        plt.legend(loc=legend, prop={'size':15})
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()

train_acc_m10 = model_10.score(train_data[features_oh].to_numpy(),train_data[target].to_numpy())
train_acc_m50 = model_50.score(train_data[features_oh].to_numpy(),train_data[target].to_numpy())
train_acc_m100 = model_100.score(train_data[features_oh].to_numpy(),train_data[target].to_numpy())
train_acc_m200 = model_200.score(train_data[features_oh].to_numpy(),train_data[target].to_numpy())
train_acc_m500 = model_500.score(train_data[features_oh].to_numpy(),train_data[target].to_numpy())

train_errors = [1-train_acc_m10, 1-train_acc_m50, 1-train_acc_m100, 1-train_acc_m200, 1-train_acc_m500]

validation_errors = [1-acc_m10, 1-acc_m50, 1-acc_m100, 1-acc_m200, 1-acc_m500]

plt.plot([10, 50, 100, 200, 500], train_errors, linewidth=4.0, label='Training error')
plt.plot([10, 50, 100, 200, 500], validation_errors, linewidth=4.0, label='Validation error')

make_figure(dim=(10,5), title='Error vs number of trees',
            xlabel='Number of trees',
            ylabel='Classification error',
            legend='best')


#Quiz question: Does the training error reduce as the number of trees increases?
#yes

#Quiz question: Is it always true that the validation error will reduce as the number of trees increases?
#no


#
