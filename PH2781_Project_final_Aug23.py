#!/usr/bin/env python
# coding: utf-8

# In[1]:


# this will help in making the Python code more structured automatically (good coding practice)
get_ipython().run_line_magic('load_ext', 'nb_black')

# Libraries to help with reading and manipulating data
import numpy as np
import pandas as pd
import datetime

# Libraries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# split the data into train and test
from sklearn.model_selection import train_test_split

# to build linear regression_model
from sklearn.linear_model import LinearRegression

# to check model performance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# to build linear regression_model using statsmodels
import statsmodels.api as sm

# to compute VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[2]:


df = pd.read_csv(r"C:\Users\munee\Desktop\covid.csv")
df.head()


# In[3]:


# df = pd.read_csv(r"C:\Users\munee\Desktop\thesisbothtable.csv")
# df
# del df["Unnamed: 0"]
# df
df["row_num"] = np.arange(len(df))
df


# In[4]:


df["TEST_RESULT_Dummy"] = np.where(df["TEST_RESULT"] == "POSITIVE", 1, 0)
df.head()


# In[5]:


df["DIAGNOSIS_CD_Dummy"] = np.where(df["DIAGNOSIS_CD"] == "U071", 1, 0)
df.head()


# In[6]:


# see full table
# pd.set_option("display.max_rows", None)
# pd.set_option("display.max_columns", None)
# pd.set_option("display.width", None)
# pd.set_option("display.max_colwidth", None)


# In[7]:


# calculate a
tp = df.loc[(df["DIAGNOSIS_CD"] == "U071") & (df["TEST_RESULT"] == "POSITIVE")]
tp
len(tp)


# In[8]:


# calculate C
fn = df.loc[(df["DIAGNOSIS_CD"] == "U071") & (df["TEST_RESULT"] == "NEGATIVE")]
fn
len(fn)


# In[9]:


# calculate d
fp = df.loc[(df["DIAGNOSIS_CD"] != "U071") & (df["TEST_RESULT"] == "NEGATIVE")]
fp
len(fp)


# In[10]:


# calculate b
tn = df.loc[(df["DIAGNOSIS_CD"] != "U071") & (df["TEST_RESULT"] == "POSITIVE")]
tn
len(tn)


# In[11]:


df.info()


# In[12]:


import pandas as pd

df5 = pd.DataFrame({"value": ["POSITIVE", "NEGATIVE"]})
df5.value.eq("POSITIVE").astype(int)
df5.value.eq("NEGATIVE").astype(int)
df5.columns = ["TEST_RESULT"]
df5
df.append(df5)
df.head()


# In[13]:


df6 = pd.DataFrame({"value": ["U071", "=!U071"]})
df6.value.eq("U071").astype(int)
df6.value.eq("=!U071").astype(int)
df6.columns = ["DIAGNOSIS_CD"]
df6


# In[14]:


df.append(df5)
df.head()
df.append(df6)
df.head()


# In[15]:


df.info()


# In[50]:


# statology
from sklearn.preprocessing import MultiLabelBinarizer as mlb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from matplotlib import pyplot

# raw confusion matrix
df = pd.DataFrame(df, columns=["DIAGNOSIS_CD_Dummy", "TEST_RESULT_Dummy"])
confusion_matrix = pd.crosstab(
    df["TEST_RESULT_Dummy"],
    df["DIAGNOSIS_CD_Dummy"],
    rownames=["Test Result"],
    colnames=["Diagnosis"],
)
print(confusion_matrix)


# Logistic Regression Confusion Matrix
from sklearn.preprocessing import MultiLabelBinarizer as mlb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
from sklearn import metrics

X = df[["DIAGNOSIS_CD_Dummy"]]
y = df[["TEST_RESULT_Dummy"]]
# X = pd.DataFrame(df.iloc[:, -1])
# y = pd.DataFrame(df.iloc[:, :-1])


# split into training and test using scikit
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y.values.ravel(), test_size=0.3, random_state=1, stratify=y
)
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

# use logistic regression model to make predictions
y_score = log_model.predict_proba(X_test)[:, 1]

y_pred = log_model.predict(X_test)
y_pred = np.round(y_pred)
confusion_matrix = confusion_matrix(y_test, y_pred)
print("\n")
print("Confusion Matrix LR")
print(confusion_matrix)
print("\n")
print(classification_report(y_test, y_pred, zero_division=0))

# calculate precision and recall
precision, recall, thresholds = precision_recall_curve(y_test, y_score)

# create precision recall curve
fig, ax = plt.subplots()
ax.plot(recall, precision, color="purple")

# add axis labels to plot
ax.set_title("Precision-Recall Curve")
ax.set_ylabel("Precision")
ax.set_xlabel("Recall")

# display plot
plt.show()

# precision-recall curve
# generate 2 class dataset
X = df[["DIAGNOSIS_CD_Dummy"]]
y = df[["TEST_RESULT_Dummy"]]

# X = pd.DataFrame(df.iloc[:, :-1])
# y = pd.DataFrame(df.iloc[:, -1])

# split into train/test sets
trainX, testX, trainy, testy = train_test_split(
    X, y.values.ravel(), test_size=0.3, random_state=2
)
# fit a model
model = LogisticRegression(solver="lbfgs")
model.fit(trainX, trainy)

# predict probabilities
lr_probs = model.predict_proba(testX)
# probs_rf = model_rf.predict_proba(testX)[:, 1]

# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]

# predict class values
yhat = model.predict(testX)
lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
lr_f1, lr_auc = f1_score(testy, yhat), auc(lr_recall, lr_precision)

# precision_rf, recall_rf, _ = precision_recall_curve(testy, probs_rf)
# f1_rf, auc_rf = f1_score(testy, yhat), auc(recall_rf, precision_rf)
# auc_rf = auc(recall_rf, precision_rf)

pred = model.predict(X_test)

# calculating precision and reall
precision = precision_score(y_test, pred)
recall = recall_score(y_test, pred)

print("Precision: ", precision)
print("Recall: ", recall)


# summarize scores
print("Logistic: f1=%.3f auc=%.3f" % (lr_f1, lr_auc))

# plot the precision-recall curves
no_skill = len(testy[testy == 1]) / len(testy)
pyplot.plot([0, 1], [no_skill, no_skill], linestyle="--", label="No Skill")
pyplot.plot(lr_recall, lr_precision, marker=".", label="Logistic")

plt.plot(lr_precision, lr_recall, label=f"AUC (Logistic Regression) = {lr_auc:.2f}")

# axis labels
pyplot.xlabel("Recall")
pyplot.ylabel("Precision")
# show the legend
pyplot.legend()
# show the plot
pyplot.show()


## Random Forest ##
#model_rf = RandomForestClassifier()
#model_rf.fit(trainX, trainy)
# model_rf = RandomForestClassifier().fit(trainX, trainy)

# predict probabilities
#lr_probs = model.predict_proba(testX)
#probs_rf = model_rf.predict_proba(testX)

# keep probabilities for the positive outcome only
probs_rf = probs_rf[:, 1]

# predict class values
#yhat = model.predict(testX)
#precision_rf, recall_rf, _ = precision_recall_curve(testy, probs_rf)
#f1_rf, auc_rf = f1_score(testy, yhat), auc(recall_rf, precision_rf)
#auc_rf = auc(recall_rf, precision_rf)

#print("Random Forest: f1=%.3f auc=%.3f" % (f1_rf, auc_rf))

# plot the precision-recall curves
#no_skill = len(testy[testy == 1]) / len(testy)
#pyplot.plot([0, 1], [no_skill, no_skill], linestyle="--", label="No Skill")
#pyplot.plot(lr_recall, lr_precision, marker=".", label="Random Forest")

#plt.plot(recall_rf, precision_rf, label=f"AUC (Random Forests) = {auc_rf:.2f}")

# axis labels
#pyplot.xlabel("Recall")
#pyplot.ylabel("Precision")
# show the legend
#pyplot.legend()
# show the plot
#pyplot.show()


# In[17]:


from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(y, X)
sns.heatmap(conf_mat, square=True, annot=True, cmap="Blues", fmt="d", cbar=False)
print("Test Result")


# In[18]:


df.info()


# In[19]:


df.head()


# In[20]:


pip install -U notebook-as-pdf


# In[21]:


jupyter-nbconvert --to PDFviaHTML df.ipynb


# In[22]:


import os

os.environ["PATH"].split(";")


# In[ ]:


#using Random Forest


# In[76]:


# statology
from numpy import argmax
from sklearn.preprocessing import MultiLabelBinarizer as mlb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from matplotlib import pyplot

# raw confusion matrix
df = pd.DataFrame(df, columns=["DIAGNOSIS_CD_Dummy", "TEST_RESULT_Dummy"])
confusion_matrix = pd.crosstab(
    df["TEST_RESULT_Dummy"],
    df["DIAGNOSIS_CD_Dummy"],
    rownames=["Test Result"],
    colnames=["Diagnosis"],
)
print("\n")
print(confusion_matrix)
print("\n")

# RF Confusion Matrix
from sklearn.preprocessing import MultiLabelBinarizer as mlb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

X = df[["DIAGNOSIS_CD_Dummy"]]
y = df[["TEST_RESULT_Dummy"]]
# X = pd.DataFrame(df.iloc[:, -1])
# y = pd.DataFrame(df.iloc[:, :-1])


# split into training and test using scikit
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y.values.ravel(), test_size=0.5, random_state=2, stratify=y
)


# Random Forest
model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)
# model_rf = RandomForestClassifier().fit(trainX, trainy)

# use RF model to make predictions
y_score = model_rf.predict_proba(X_test)[:, 1]

y_pred_test = model_rf.predict(X_test)
y_pred = np.round(y_pred_test)
confusion_matrix = confusion_matrix(y_test, y_pred)
print("\n")
print("RF Confusion Matrix")
print(confusion_matrix)
print("\n")
print(classification_report(y_test, y_pred, zero_division=0))

# predict probabilities
probs_rf = model_rf.predict_proba(X_test)

# keep probabilities for the positive outcome only
probs_rf = probs_rf[:, 1]

# predict class values
yhat = model_rf.predict(X_test)
precision_rf, recall_rf, _ = precision_recall_curve(y_test, probs_rf)
f1_rf, auc_rf = f1_score(y_test, yhat), auc(recall_rf, precision_rf)
auc_rf = auc(recall_rf, precision_rf)

print("Random Forest: f1=%.3f auc=%.3f" % (f1_rf, auc_rf))

pred = model_rf.predict(X_test)

# calculating precision and reall
precision = precision_score(y_test, pred)
recall = recall_score(y_test, pred)

print("Precision: ", precision)
print("Recall: ", recall)

# plot the precision-recall curves
no_skill = len(y_test[y_test == 1]) / len(y_test)
pyplot.plot([0, 1], [no_skill, no_skill], linestyle="--", label="No Skill")
# pyplot.plot(lr_recall, lr_precision, marker=".", label="LR")
pyplot.plot(precision_rf, recall_rf, marker=".", label="RF")

# plt.plot(lr_precision, lr_recall, label=f"AUC (Logistic Regression) = {lr_auc:.2f}")
plt.plot(recall_rf, precision_rf, label=f"AUC (Random Forests) = {auc_rf:.2f}")

# axis labels
pyplot.xlabel("Recall")
pyplot.ylabel("Precision")
# show the legend
pyplot.legend()
# show the plot
pyplot.show()



# In[90]:


# trial 2: finding best threshold

from numpy import arange
from numpy import argmax
from sklearn.preprocessing import MultiLabelBinarizer as mlb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from matplotlib import pyplot

# raw confusion matrix
df = pd.DataFrame(df, columns=["DIAGNOSIS_CD_Dummy", "TEST_RESULT_Dummy"])
confusion_matrix = pd.crosstab(
    df["TEST_RESULT_Dummy"],
    df["DIAGNOSIS_CD_Dummy"],
    rownames=["Test Result"],
    colnames=["Diagnosis"],
)
print("\n")
print(confusion_matrix)
print("\n")

# RF Confusion Matrix
from sklearn.preprocessing import MultiLabelBinarizer as mlb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

X = df[["DIAGNOSIS_CD_Dummy"]]
y = df[["TEST_RESULT_Dummy"]]
# X = pd.DataFrame(df.iloc[:, -1])
# y = pd.DataFrame(df.iloc[:, :-1])


# split into training and test using scikit
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y.values.ravel(), test_size=0.3, random_state=2, stratify=y
)


# Random Forest
model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)
# model_rf = RandomForestClassifier().fit(trainX, trainy)

# use RF model to make predictions
y_score = model_rf.predict_proba(X_test)[:, 1]

y_pred_test = model_rf.predict(X_test)
y_pred = np.round(y_pred_test)
confusion_matrix = confusion_matrix(y_test, y_pred)
print("\n")
print("RF Confusion Matrix")
print(confusion_matrix)
print("\n")
print(classification_report(y_test, y_pred, zero_division=0))

# predict probabilities
probs_rf = model_rf.predict_proba(X_test)

# keep probabilities for the positive outcome only
probs_rf = probs_rf[:, 1]

# predict class values
yhat = model_rf.predict(X_test)
precision_rf, recall_rf, _ = precision_recall_curve(y_test, probs_rf)
f1_rf, auc_rf = f1_score(y_test, yhat), auc(recall_rf, precision_rf)
auc_rf = auc(recall_rf, precision_rf)

print("Random Forest: f1=%.3f auc=%.3f" % (f1_rf, auc_rf))

pred = model_rf.predict(X_test)

# calculating precision and reall
precision = precision_score(y_test, pred)
recall = recall_score(y_test, pred)

print("Precision: ", precision)
print("Recall: ", recall)

# plot the precision-recall curves
no_skill = len(y_test[y_test == 1]) / len(y_test)
pyplot.plot([0, 1], [no_skill, no_skill], linestyle="--", label="No Skill")
# pyplot.plot(lr_recall, lr_precision, marker=".", label="LR")
pyplot.plot(precision_rf, recall_rf, marker=".", label="RF")

# plt.plot(lr_precision, lr_recall, label=f"AUC (Logistic Regression) = {lr_auc:.2f}")
plt.plot(recall_rf, precision_rf, label=f"AUC (Random Forests) = {auc_rf:.2f}")

# axis labels
pyplot.xlabel("Recall")
pyplot.ylabel("Precision")
# show the legend
pyplot.legend()
# show the plot
pyplot.show()


from numpy import argmax
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot

# predict probabilities
yhat = model_rf.predict_proba(X_test)
probs_rf = model_rf.predict_proba(X_test)

# keep probabilities for the positive outcome only
yhat = yhat[:, 1]
probs_rf = probs_rf[:, 1]

# calculate curves
precision, recall, thresholds_rf = precision_recall_curve(y_test, yhat)
precision_rf, recall_rf, _ = precision_recall_curve(y_test, probs_rf)
# f1_rf, auc_rf = f1_score(y_test, yhat), auc(recall_rf, precision_rf)
# auc_rf = auc(recall_rf, precision_rf)

# find best optimal threshold for PR curve
# convert to f score

fscore = (2 * precision_rf * recall_rf) / (precision_rf + recall_rf)
# locate the index of the largest f score
ix = argmax(fscore)
print("Best Threshold=%f, F-Score=%.3f" % (thresholds[ix], fscore[ix]))
# plot the roc curve for the model
no_skill = len(y_test[y_test == 1]) / len(y_test)
pyplot.plot([0, 1], [no_skill, no_skill], linestyle="--", label="No Skill")
pyplot.plot(recall_rf, precision_rf, marker=".", label="RF")
pyplot.scatter(recall_rf[ix], precision_rf[ix], marker="o", color="black", label="Best")
# axis labels
pyplot.xlabel("Recall")
pyplot.ylabel("Precision")
pyplot.legend()
# show the plot
pyplot.show()


# optimal threshold tuning

# apply threshold to positive probabilities to create labels
def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype("int")


# define thresholds
thresholds = arange(0, 1, 0.001)
# evaluate each threshold
scores = [f1_score(y_test, to_labels(probs_rf, t)) for t in thresholds]
# get best threshold
ix = argmax(scores)
print("Threshold=%.3f, F-Score=%.5f" % (thresholds[ix], scores[ix]))


# In[100]:


# optimal threshold tuning

# search thresholds for imbalanced classification
from numpy import arange
from numpy import argmax
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

# apply threshold to positive probabilities to create labels
def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype("int")


# generate dataset
X = df[["DIAGNOSIS_CD_Dummy"]]
y = df[["TEST_RESULT_Dummy"]]
# X = pd.DataFrame(df.iloc[:, -1])
# y = pd.DataFrame(df.iloc[:, :-1])


# split into train/test sets
trainX, testX, trainy, testy = train_test_split(
    X, y.values.ravel(), test_size=0.5, shuffle=True, random_state=2, stratify=y
)
# fit a model
model_rf = RandomForestClassifier()
model_rf.fit(trainX, trainy)
# predict probabilities
yhat = model_rf.predict_proba(testX)
# keep probabilities for the positive outcome only
probs = yhat[:, 1]
# define thresholds
thresholds = arange(0, 1, 0.001)
# evaluate each threshold
scores = [f1_score(testy, to_labels(probs, t)) for t in thresholds]
# get best threshold
ix = argmax(scores)
print("Threshold=%.3f, F-Score=%.5f" % (thresholds[ix], scores[ix]))


# In[96]:


# trial 3: adding cross validation (k-fold)

from numpy import arange
from numpy import argmax
from sklearn.preprocessing import MultiLabelBinarizer as mlb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from matplotlib import pyplot

# split into training and test using scikit
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y.values.ravel(), test_size=0.3, shuffle=True, random_state=2, stratify=y
)

# raw confusion matrix
df = pd.DataFrame(df, columns=["DIAGNOSIS_CD_Dummy", "TEST_RESULT_Dummy"])
confusion_matrix = pd.crosstab(
    df["TEST_RESULT_Dummy"],
    df["DIAGNOSIS_CD_Dummy"],
    rownames=["Test Result"],
    colnames=["Diagnosis"],
)
print("\n")
print(confusion_matrix)
print("\n")

from sklearn.metrics import confusion_matrix, classification_report

y_pred = model_rf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn.model_selection import cross_val_score

print(cross_val_score(model, X_train, y_train, cv=5))

import numpy as np
print(np.mean(cross_val_score(model, X_train, y_train, cv=5)))



# In[111]:


# trial 4   https://amirhessam88.github.io/finding-thresholds/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import interp
from sklearn.preprocessing import scale
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    accuracy_score,
    roc_curve,
    confusion_matrix,
    average_precision_score,
    precision_recall_curve,
)
from sklearn.model_selection import (
    cross_val_score,
    KFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from matplotlib import pyplot

import seaborn as sns

sns.set_style("ticks")
mpl.rcParams["axes.linewidth"] = 3
mpl.rcParams["lines.linewidth"] = 2
from IPython.core.display import display, HTML

display(HTML("<style>.container { width:95% !important; }</style>"))
import warnings

warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')

# Function: _model


def _clf_train(
    X_train,
    y_train,
    X_test,
    y_test,
    learning_rate=0.05,
    n_estimators=100,
    max_depth=3,
    min_child_weight=5.0,
    gamma=1,
    reg_alpha=0.0,
    reg_lambda=1.0,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="binary:logistic",
    nthread=4,
    scale_pos_weight=1.0,
    seed=1367,
    random_state=1367,
):
    """
    a RF model for training
    """

    clf = RandomForestClassifier()

    clf.fit(
        X_train,
        y_train
        # eval_metric="auc",
        # early_stopping_rounds=20,
        # verbose=True,
        # eval_set=[(X_test, y_test)],
    )

    return clf


# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Function: Finding thresholds
def _threshold_finder(model, X, y_true):
    """
    a function to find the optimal threshold for binary classification
    model: a trained model object (such as xgboost, glmnet, ...)
    X: the test set of features (pandas dataframe or numpy array)
    y_true: the true class labels (list or array of 0's and 1's)    
    """

    y_predict_proba = model.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true, y_predict_proba)
    auc = roc_auc_score(y_true, y_predict_proba)
    precision, recall, thresholds2 = precision_recall_curve(y_true, y_predict_proba)

    class_names = [0, 1]
    youden_idx = np.argmax(np.abs(tpr - fpr))
    youden_threshold = thresholds[youden_idx]
    y_pred_youden = (y_predict_proba > youden_threshold).astype(int)
    cnf_matrix = confusion_matrix(y_true, y_pred_youden)
    np.set_printoptions(precision=2)

    f1 = []
    for i in range(len(precision)):
        f1.append(2 * (precision[i] * recall[i]) / (precision[i] + recall[i]))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color="red", label=f"AUC = {auc:.3f}")
    plt.plot(
        fpr[youden_idx],
        tpr[youden_idx],
        marker="o",
        color="navy",
        ms=10,
        label=f"Youden Threshold = {youden_threshold:.2f}",
    )
    plt.axvline(
        x=fpr[youden_idx],
        ymin=fpr[youden_idx],
        ymax=tpr[youden_idx],
        color="navy",
        ls="--",
    )
    plt.plot([0, 1], [0, 1], color="black", ls="--")
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel("1 - Specificity", fontsize=12)
    plt.ylabel("Sensitivity", fontsize=12)
    plt.tick_params(axis="both", which="major", labelsize=12)
    plt.legend(prop={"size": 12}, loc=4)

    plt.subplot(1, 2, 2)
    _plot_confusion_matrix(
        cnf_matrix,
        classes=class_names,
        normalize=False,
        cmap=plt.cm.Reds,
        title=f"Youden Threshold = {youden_threshold:.2f}\nAccuracy = {accuracy_score(y_true, y_pred_youden)*100:.2f}%",
    )
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(thresholds, 1 - fpr, label="1 - Specificity")
    plt.plot(thresholds, tpr, label="Sensitivity")
    plt.xlabel("Threshold", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.legend(loc=0)
    plt.xlim([0.025, thresholds[np.argmin(abs(tpr + fpr - 1))] + 0.2])
    plt.axvline(thresholds[np.argmin(abs(tpr + fpr - 1))], color="k", ls="--")
    plt.title(
        f"Threshold = {thresholds[np.argmin(abs(tpr + fpr - 1))]:.3f}", fontsize=12
    )

    plt.subplot(1, 2, 2)
    plt.plot(thresholds2, precision[1:], label="Precision")
    plt.plot(thresholds2, recall[1:], label="Recall")
    plt.plot(thresholds2, f1[1:], label="F1-Score")
    # plt.plot(thresholds2, queue_rate, label="Queue Rate")
    plt.legend(loc=0)
    plt.xlim([0.025, thresholds2[np.argmin(abs(precision - recall))] + 0.2])
    plt.xlabel("Threshold", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.axvline(thresholds2[np.argmin(abs(precision - recall))], color="k", ls="--")
    plt.title(
        label=f"Threshold = {thresholds2[np.argmin(abs(precision-recall))]:.3f}",
        fontsize=12,
    )
    plt.show()


# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Function: Plotting Confusion Matrix
def _plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Greens
):
    from sklearn.metrics import (
        precision_score,
        recall_score,
        roc_auc_score,
        accuracy_score,
        roc_curve,
        auc,
        confusion_matrix,
    )
    import itertools

    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title, fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i, format(cm[i, j], fmt), horizontalalignment="center", color="black"
        )

    plt.ylabel("True Class", fontsize=14)
    plt.xlabel("Predicted Class", fontsize=14)

    plt.tick_params(axis="both", which="major", labelsize=14)
    plt.tight_layout()


X = df[["DIAGNOSIS_CD_Dummy"]]
y = df[["TEST_RESULT_Dummy"]]
# X = pd.DataFrame(df.iloc[:, -1])
# y = pd.DataFrame(df.iloc[:, :-1])


# split into training and test using scikit
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y.values.ravel(), test_size=0.3, random_state=1, stratify=y
)

model = _clf_train(X_train, y_train, X_test, y_test)

_threshold_finder(model=model, X=X_test, y_true=y_test)


# In[116]:


# trial 5
import numpy as np

X = df[["DIAGNOSIS_CD_Dummy"]]
y = df[["TEST_RESULT_Dummy"]]
X = (X - np.min(X)) / (np.max(X) - np.min(X))


def RandomForestClassifier(X, thres=0.5):
    predicted = np.zeros(len(X))
    for i in range(len(X)):
        if X[i] < thres:
            predicted[i] = 1
    return predicted


def calculate_metrics(predicted, actual):
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(predicted)):
        if (predicted[i] == 0) & (actual[i] == 0):
            TP += 1
        elif (predicted[i] == 0) & (actual[i] == 1):
            FP += 1
        elif (predicted[i] == 1) & (actual[i] == 1):
            TN += 1
        else:
            FN += 1

    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = (TP) / (TP + FP)
    recall = (TP) / (TP + FN)
    f1_score = (2 * precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1_score


thresh = np.linspace(0, 1, 20)
accuracy = np.zeros(len(thresh))
precision = np.zeros(len(thresh))
recall = np.zeros(len(thresh))
f1_score = np.zeros(len(thresh))

print("Threshold \t Accuracy \t Precision\t Recall \t  F1 Score ")

for i in range(len(thresh)):
    prediction = RandomForestClassifier(X, thresh[i])
    accuracy[i], precision[i], recall[i], f1_score[i] = calculate_metrics(prediction, y)
    print(
        f"{thresh[i]: .2f}\t\t {accuracy[i]: .2f}\t\t {precision[i]: .2f}\t\t {recall[i]: .2f}\t\t {f1_score[i]: .2f}"
    )

import matplotlib.pyplot as plt

plt.plot(precision, recall)
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.title("Precision versus Recall")
plt.show()

plt.plot(thresh, accuracy, "r")
plt.plot(thresh, precision, "g")
plt.plot(thresh, recall, "b")
plt.plot(thresh, f1_score, "k")

plt.legend(["Accuracy", "Precision", "Recall", "F1 Score"])
plt.xlabel("Threshold")
plt.ylabel("Scores")
plt.title("Change of Evaluation Metrics according to Threshold")

plt.show()


# In[ ]:




