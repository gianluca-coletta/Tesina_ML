import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load weatherAUS dataframe
rain = pd.read_csv("weatherAUS.csv")

# Dimension of dataset
print(rain.shape)

# Summary of dataset
print(rain.info())

# Summary statistics
print(rain.describe().to_string())

# Categorical feature
categorical_features = [column_name for column_name in rain.columns if rain[column_name].dtype == 'O']
print("Number of Categorical Features: {}".format(len(categorical_features)))
print("Categorical Features: ", categorical_features)

# Unique value for categorical features (Check cardinality)
for each_feature in categorical_features:
    unique_values = len(rain[each_feature].unique())
    print("Cardinality(no. of unique values) of {} are: {}".format(each_feature, unique_values))

# Feature engineering of date column
rain['Date'] = pd.to_datetime(rain['Date'])
rain['year'] = rain['Date'].dt.year
rain['month'] = rain['Date'].dt.month
rain['day'] = rain['Date'].dt.day



# Drop column date
rain.drop('Date', axis=1, inplace=True)
print(rain.head().to_string())

# Missing values for categorical features
categorical_features = [column_name for column_name in rain.columns if rain[column_name].dtype == 'O']
print(rain[categorical_features].isnull().sum())

# Drop missing values
rain = rain.dropna(subset=categorical_features)
print(rain[categorical_features].isnull().sum())

# Dimension of dataset after remove missing values of categorical features
print(rain.shape)

# Encoding categorical variables to numeric ones (non la miglior scelta, meglio dummies)
from sklearn.preprocessing import LabelEncoder

for c in rain.columns:
    if rain[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(rain[c].values))
        rain[c] = lbl.transform(rain[c].values)
print(rain.head().to_string())

# Numerical variables

# Missing values for numerical features
numerical_features = [column_name for column_name in rain.columns if rain[column_name].dtype == 'float64']
print(rain[numerical_features].isnull().sum())

# Impute missing values with respective column median
for col in numerical_features:
    col_median = rain[col].median()
    rain[col].fillna(col_median, inplace=True)
print(rain[numerical_features].isnull().sum())

# Summary statistics of numerical feature to find possible outliers
print(round(rain[numerical_features].describe()).to_string(), 2)

# BoxPlot numerical features
plt.figure(1, figsize=[18, 16])
rain.boxplot(column=numerical_features)
plt.xticks(rotation=45)

# BoxPlot numerical features without Pressure
plt.figure(2, figsize=[18, 16])
rain.boxplot(column=['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
                     'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
                     'Humidity3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm'])
plt.xticks(rotation=45)
plt.show()

# BoxPlot rainfall
plt.figure(3, figsize=[18, 16])
rain.boxplot(column='Rainfall')
plt.xticks(rotation=45)
plt.show()

# Remove outlier
rain = rain.drop(rain[rain.Rainfall > 367].index)
numerical_features = [column_name for column_name in rain.columns if rain[column_name].dtype == 'float64']

# Correlation matrix
plt.figure(4, figsize=(18,16))
sns.heatmap(rain.corr(), annot=True, cmap=plt.cm.CMRmap_r)
plt.show()

# Count of RainTomorrow for month
plt.figure(5, figsize=[18, 16])
sns.countplot(x=rain.month, hue=rain.RainTomorrow, data = rain)

# Count of RainToday for month
plt.figure(6, figsize=[18, 16])
sns.countplot(x=rain.month, hue=rain.RainToday, data = rain)
plt.show()

# Remove correlated features
rain.drop(['Temp9am','Temp3pm','Pressure3pm'],inplace= True,axis=1)

print(rain.columns)
print(rain.head().to_string())
print(rain.shape)

# Show count of RainToday and RainTomorrow
print(rain.RainToday.value_counts())
print(rain.RainTomorrow.value_counts())

fig, ax =plt.subplots(1,2)
sns.countplot(data=rain,x='RainToday',ax=ax[0])
sns.countplot(data=rain,x='RainTomorrow',ax=ax[1])
plt.show()

# Splitting dataset and re-scaling with standardization
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

x = rain.drop('RainTomorrow', axis=1).values
t = rain['RainTomorrow']
X_train, X_test, t_train, t_test = train_test_split(x, t, test_size = 0.20, random_state = 0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# grid search logistic regression model on the sonar dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

# GRID SEARCH LOGISTIC REGRESSION

# # define model
# model = LogisticRegression()
# # define search space
# space = dict()
# space['solver'] = ['lbfgs','newton-cg', 'sag', 'saga']
# space['penalty'] = ['none',  'l2']
# space['C'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
# space['max_iter'] = [100, 1000, 2500, 5000]
# # define search
# search = GridSearchCV(model, space, scoring='accuracy', n_jobs=-1, cv=3)
# # execute search
# result = search.fit(X_train, t_train)
# # summarize result
# print('Best Score: %s' % result.best_score_)
# print('Best Hyperparameters: %s' % result.best_params_)

# Best parameter -> c:10 max_iter_:5000 penalty: L2 solver: sag ||  F1-Score = 0.59

from sklearn.metrics import classification_report
model = LogisticRegression(C=10, max_iter=5000, penalty='l2', solver='sag')
model.fit(X_train, t_train)

t_pred = model.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(t_test, t_pred))

# Confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(t_test, t_pred)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])

# Visualize
plt.figure(7, figsize=[18, 16])
cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.show()

# GRID SEARCH NEURAL NETWORKS

from sklearn.neural_network import MLPClassifier
# mlp_gs = MLPClassifier()
# parameter_space = {
#     'hidden_layer_sizes': [(100, 100), (100, 100, 100)],
#     'activation': ['tanh', 'relu'],
#     'solver': ['sgd','adam'],
#     'alpha': [0.0001, 0.001, 0.01, 0.1],
#     'early_stopping': [True, False],
#     'max_iter': [200]
# }
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import classification_report
# clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=3, scoring='f1', verbose=10, error_score="raise")
# best_clf = clf.fit(X_train, t_train) # X is train samples and t is the corresponding labels
#
# # best score achieved during the GridSearchCV
# print('GridSearch CV best score : {:.4f}\n\n'.format(clf.best_score_))
#
# # print parameters that give the best results
# print('Parameters that give the best results :','\n\n', (clf.best_params_))
#
# # print estimator that was chosen by the GridSearch
# print('\n\nEstimator that was chosen by the search :','\n\n', (clf.best_estimator_))


# Best parameter NN -> alpha = 0.001 hidden_layer_sizes = (100,100) early_stopping = true || F1-Score = 0.6377

clf_NN = MLPClassifier(alpha=0.001, hidden_layer_sizes=(100, 100), early_stopping=True).fit(X_train, t_train)
t_pred_NN=clf_NN.predict(X_test)
print(classification_report(t_test, t_pred_NN))

# Confusion matrix

from sklearn.metrics import confusion_matrix

cm1 = confusion_matrix(t_test, t_pred_NN)

print('Confusion matrix\n\n', cm1)

print('\nTrue Positives(TP) = ', cm1[0,0])

print('\nTrue Negatives(TN) = ', cm1[1,1])

print('\nFalse Positives(FP) = ', cm1[0,1])

print('\nFalse Negatives(FN) = ', cm1[1,0])

# Visualize
plt.figure(6, figsize=[18, 16])
cm1_matrix = pd.DataFrame(data=cm1, columns=['Actual Positive:1', 'Actual Negative:0'],
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm1_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.show()