import sframe
loans = sframe.SFrame('lending-club-data.gl/')

loans.column_names()

# safe_loans =  1 => safe
# safe_loans = -1 => risky
loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.remove_column('bad_loans')

percent_safe = loans[loans['safe_loans']==1].num_rows()/float(loans.num_rows())
print "Risky loans: " + str(1-percent_safe)
print "Safe loans: " + str(percent_safe)

#Extract Features
features = ['grade',                     # grade of the loan
            'sub_grade',                 # sub-grade of the loan
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'term',                      # the term of the loan
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
           ]

target = 'safe_loans'                    # prediction target (y) (+1 means safe, -1 is risky)

# Extract the feature columns and target column
loans = loans[features + [target]]

#Balance Classes
safe_loans_raw = loans[loans[target] == +1]
risky_loans_raw = loans[loans[target] == -1]
print "Number of safe loans  : %s" % len(safe_loans_raw)
print "Number of risky loans : %s" % len(risky_loans_raw)

# Since there are fewer risky loans than safe loans, find the ratio of the sizes
# and use that percentage to undersample the safe loans.
percentage = len(risky_loans_raw)/float(len(safe_loans_raw))

risky_loans = risky_loans_raw
safe_loans = safe_loans_raw.sample(percentage, seed=1)

# Append the risky_loans with the downsampled version of safe_loans
loans_data = risky_loans.append(safe_loans)

#One-hot encoding
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

#Split Train/Validation
train_data, validation_data = loans_data.random_split(.8, seed=1)

#Learning
import sklearn
import sklearn.tree

features_onehot = train_data.column_names()
features_onehot.remove("safe_loans")

decision_tree_model = sklearn.tree.DecisionTreeClassifier(max_depth=6)
decision_tree_model = decision_tree_model.fit(train_data[features_onehot].to_numpy(), train_data[target].to_numpy())

small_model = sklearn.tree.DecisionTreeClassifier( max_depth=2)
small_model = small_model.fit(train_data[features_onehot].to_numpy(), train_data[target].to_numpy())

#Predictions
validation_safe_loans = validation_data[validation_data[target] == 1]
validation_risky_loans = validation_data[validation_data[target] == -1]

sample_validation_data_risky = validation_risky_loans[0:2]
sample_validation_data_safe = validation_safe_loans[0:2]

sample_validation_data = sample_validation_data_safe.append(sample_validation_data_risky)
sample_validation_data

#------------------ decision_tree_model ------------------

#predictions on sample validation
predictions = decision_tree_model.predict(sample_validation_data[features_onehot].to_numpy())
sum(predictions == sample_validation_data[target].to_numpy())

#Quiz Question: What percentage of the predictions on sample_validation_data did decision_tree_model get correct?
#0.5

#prediction probabilty
predictions_proba = decision_tree_model.predict_proba(sample_validation_data[features_onehot].to_numpy())
#Quiz Question: What percentage of the predictions on sample_validation_data did decision_tree_model get correct?
print(predictions_proba)

#accuracy training data
decision_tree_model.score(train_data[features_onehot].to_numpy(), train_data[target].to_numpy())
#accuracy validation data
#Quiz Question: What is the accuracy of decision_tree_model on the validation set, rounded to the nearest .01?
decision_tree_model.score(validation_data[features_onehot].to_numpy(), validation_data[target].to_numpy())
#0.64

#------------------ small_model ------------------

#predictions on sample validation
predictions_small = small_model.predict(sample_validation_data[features_onehot].to_numpy())
sum(predictions_small == sample_validation_data[target].to_numpy())

#prediction probabilty
predictions_proba_small = small_model.predict_proba(sample_validation_data[features_onehot].to_numpy())
#Quiz Question: Notice that the probability preditions are the exact same for the 2nd and 3rd loans. Why would this happen?
predictions_proba_small

#Quiz Question: Based on the visualized tree, what prediction would you make for this data point (according to small_model)?
small_model.predict(sample_validation_data[features_onehot].to_numpy()[1])
#-1

#accuracy
small_model.score(train_data[features_onehot].to_numpy(), train_data[target].to_numpy())
#accuracy validation data
small_model.score(validation_data[features_onehot].to_numpy(), validation_data[target].to_numpy())

#big_model
big_model = sklearn.tree.DecisionTreeClassifier(max_depth=10)
big_model = big_model.fit(train_data[features_onehot].to_numpy(), train_data[target].to_numpy())

#accuracy training data
big_model.score(train_data[features_onehot].to_numpy(), train_data[target].to_numpy())
#accuracy validation data
big_model.score(validation_data[features_onehot].to_numpy(), validation_data[target].to_numpy())

#FP / FN
predictions = decision_tree_model.predict(validation_data[features_onehot].to_numpy())

FP = 0
for i in xrange(0,len(predictions)):
    if(predictions[i] == 1 and validation_data[target].to_numpy()[i] == -1):
        FP +=1
print FP

FN = 0
for i in xrange(0,len(predictions)):
    if(predictions[i] == -1 and validation_data[target].to_numpy()[i] == 1):
        FN +=1
print FN

cost = FP*20000 + FN*10000
cost
#50390000
xrange(0,len(predictions))
