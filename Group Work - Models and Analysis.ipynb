##############################################################################
# Import modules and libraries #

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn import metrics
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
import sklearn
import time

##############################################################################

##############################################################################
# Data Clean-up #

wine_red = pd.read_csv('winequality-red.csv',sep=';')
wine_white = pd.read_csv('winequality-white.csv',sep=';')

# RED WINE DATA

wine_red.drop_duplicates(inplace=True)
#wine_red.dropna(inplace=True)

#No NA's
#print(wine_red.isna().any().any())

# WHITE WINE DATA

wine_white.drop_duplicates(inplace=True)
#wine_white.dropna(inplace=True)

#No NA's
#print(wine_white.isna().any().any())

# Merge White/Red

wines = [wine_red,wine_white]
all_wines = pd.concat(wines, ignore_index=True)

x = []
for element in all_wines.quality:
    if ((element >=0) and (element <5)):
        x.append('poor')
    elif ((element >= 5) and (element<7)):
        x.append('decent')
    else:
        x.append('good')

all_wines['adjusted quality'] = x

###############################################################################

##############################################################################
# Target and Features Arrays #

target_qual = all_wines.quality
target_adj_qual = all_wines['adjusted quality']

y=[]
for element in wine_red.quality:
    if ((element >=0) and (element <5)):
        y.append('poor')
    elif ((element >= 5) and (element<7)):
        y.append('decent')
    else:
        y.append('good')

wine_red['adjusted quality'] = y
target_red = wine_red.quality
target_adj_red = wine_red['adjusted quality']
features_red = wine_red.iloc[:,:-2]


z=[]
for element in wine_white.quality:
    if ((element >=0) and (element <5)):
        z.append('poor')
    elif ((element >= 5) and (element<7)):
        z.append('decent')
    else:
        z.append('good')

wine_white['adjusted quality'] = z    
target_white = wine_white.quality
target_adj_white = wine_white['adjusted quality']
features_white = wine_white.iloc[:,:-2]

# Feature Selection 
# Uses Scipy Module 

features = all_wines.iloc[:, :-2]

features_selected = []
feature_names = [x for x in features.columns]
for name in feature_names:
    coef,p = kendalltau(features[f'{name}'], target_qual)
    if p < .01:
        features_selected.append(name)
       #print(f'NAME:{name}, COEF:{coef}, P-VAL: {p}','\n')

selected_features = all_wines[features_selected]
selected_features_red = features_red[features_selected]
selected_features_white = features_white[features_selected]

features2 = all_wines.iloc[:, :-1]
corr_matrix = features2.round(2).corr()
f, ax = plt.subplots(figsize = (10,10))
cm_plot = sns.heatmap(corr_matrix, annot=True, linewidths=.8,fmt='.2f',ax=ax)

###############################################################################

###############################################################################
# Logistic Regression #

# All Wines



# w/ Feature Selection, 

# 3-9 Scale

start_time = time.time() #starts timer

feature_train,feature_test,target_train,target_test = train_test_split(selected_features,
                                                      target_qual,
                                                      test_size=0.3,
                                                      random_state=1111) 

pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
pipe.fit(feature_train, target_train)

log_score_sel_qual = pipe.score(feature_test,target_test)

cm_plot = sklearn.metrics.plot_confusion_matrix(pipe, feature_test, target_test)

# 'poor','decent','good' Scale

feature_train,feature_test,target_train,target_test = train_test_split(selected_features,
                                                      target_adj_qual,
                                                      test_size=0.3,
                                                      random_state=1111) 

pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
pipe.fit(feature_train, target_train)

log_score_sel_adj_qual = pipe.score(feature_test,target_test)

end_time = time.time() #ends timer
#print("Red Wine Model's Execution Times:")
#print("Logistic Regression :", log_end_time-start_time)
aw_log_time = end_time-start_time

cm_plot_alt = sklearn.metrics.plot_confusion_matrix(pipe, feature_test, target_test)



# Red Wine Only w FS

# 3-9 Scale

feature_train,feature_test,target_train,target_test = train_test_split(selected_features_red,
                                                      target_red,
                                                      test_size=0.3,
                                                      random_state=1111) 

pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
pipe.fit(feature_train, target_train)

log_score_sel_red = pipe.score(feature_test,target_test)

# 'poor','decent','good' Scale

feature_train,feature_test,target_train,target_test = train_test_split(selected_features_red,
                                                      target_adj_red,
                                                      test_size=0.3,
                                                      random_state=1111) 

pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
pipe.fit(feature_train, target_train)

log_score_sel_adj_red = pipe.score(feature_test,target_test)


# White Wine Only w/ FS

# 3-9 Scale

feature_train,feature_test,target_train,target_test = train_test_split(selected_features_white,
                                                      target_white,
                                                      test_size=0.3,
                                                      random_state=1111) 

pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
pipe.fit(feature_train, target_train)

log_score_sel_white = pipe.score(feature_test,target_test)


# 'poor','decent','good' Scale

feature_train,feature_test,target_train,target_test = train_test_split(selected_features_white,
                                                      target_adj_white,
                                                      test_size=0.3,
                                                      random_state=1111) 

pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
pipe.fit(feature_train, target_train)

log_score_sel_adj_white = pipe.score(feature_test,target_test)


###############################################################################

###############################################################################
# Random Forest #

# All Wines

# w/ Feature Selection, 

# 3-9 Scale

feature_train,feature_test,target_train,target_test = train_test_split(selected_features,
                                                      target_qual,
                                                      test_size=0.3,
                                                      random_state=1111) 

clf=RandomForestClassifier(n_estimators=100)

pipe = make_pipeline(StandardScaler(), clf)
pipe.fit(feature_train, target_train)

rf_sel_qual = pipe.score(feature_test,target_test)

# 'poor','decent','good' Scale

start_time = time.time()

feature_train,feature_test,target_train,target_test = train_test_split(selected_features,
                                                      target_adj_qual,
                                                      test_size=0.3,
                                                      random_state=1111) 

pipe = make_pipeline(StandardScaler(), clf)
pipe.fit(feature_train, target_train)

rf_sel_adj_qual = pipe.score(feature_test,target_test)

end_time = time.time() #ends timer
#print("Red Wine Model's Execution Times:")
#print("Logistic Regression :", log_end_time-start_time)
aw_rf_time = end_time-start_time



# Red Wine Only w/ FS

# 'poor','decent','good' Scale

feature_train,feature_test,target_train,target_test = train_test_split(selected_features_red,
                                                      target_adj_red,
                                                      test_size=0.3,
                                                      random_state=1111) 

pipe = make_pipeline(StandardScaler(), clf)
pipe.fit(feature_train, target_train)

rf_sel_adj_qual_red = pipe.score(feature_test,target_test)


# White Wine Only w/ FS

# 'poor','decent','good' Scale

feature_train,feature_test,target_train,target_test = train_test_split(selected_features_white,
                                                      target_adj_white,
                                                      test_size=0.3,
                                                      random_state=1111) 

pipe = make_pipeline(StandardScaler(), clf)
pipe.fit(feature_train, target_train)

rf_sel_adj_qual_white = pipe.score(feature_test,target_test)

###############################################################################

###############################################################################
# K-Nearest Neighbors #

# ALL Wines

# w/ Feature Selection, 

# 3-9 Scale

feature_train,feature_test,target_train,target_test = train_test_split(selected_features,
                                                      target_qual,
                                                      test_size=0.3,
                                                      random_state=1111)

classifier = KNeighborsClassifier(n_jobs=1, n_neighbors=7, weights='distance',p=1, leaf_size=1)
classifier.fit(feature_train,target_train)

target_pred = classifier.predict(feature_test)

KNN_sel_score = accuracy_score(target_test,target_pred)


# 'poor','decent','good' Scale

start_time = time.time()

feature_train,feature_test,target_train,target_test = train_test_split(selected_features,
                                                      target_adj_qual,
                                                      test_size=0.3,
                                                      random_state=1111)

classifier = KNeighborsClassifier(n_jobs=1, n_neighbors=7, weights='distance',p=1, leaf_size=1)
classifier.fit(feature_train,target_train)

target_pred = classifier.predict(feature_test)

KNN_sel_adj_score = accuracy_score(target_test,target_pred)

end_time = time.time() #ends timer
#print("Red Wine Model's Execution Times:")
#print("Logistic Regression :", log_end_time-start_time)
aw_KNN_time = end_time-start_time


# Red Wine Only w/ FS

feature_train,feature_test,target_train,target_test = train_test_split(selected_features_red,
                                                 target_adj_red,
                                                 test_size=0.3,
                                                 random_state=1111)


classifier = KNeighborsClassifier(n_jobs=1, n_neighbors=7, weights='distance',p=1, leaf_size=1)
classifier.fit(feature_train,target_train)

target_pred = classifier.predict(feature_test)

KNN_sel_adj_score_red = accuracy_score(target_test,target_pred)


# White Wine Only w/ FS

feature_train,feature_test,target_train,target_test = train_test_split(selected_features_white,
                                                 target_adj_white,
                                                 test_size=0.3,
                                                 random_state=1111)


classifier = KNeighborsClassifier(n_jobs=1, n_neighbors=7, weights='distance',p=1, leaf_size=1)
classifier.fit(feature_train,target_train)

target_pred = classifier.predict(feature_test)

KNN_sel_adj_score_white = accuracy_score(target_test,target_pred)


###############################################################################

###############################################################################
# Charts and Graphs #

plt.figure()
aw_hist = all_wines.quality.plot(kind='hist', title="Quality Distribution")

adj_aw_hist = all_wines['adjusted quality'].value_counts()
plt.figure()
adj_aw_hist.plot(kind="bar",ylabel='Frequency', title = "Quality Distribution")