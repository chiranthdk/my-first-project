import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing  import StandardScaler
from sklearn.metrics import classification_report,roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# 1.load dataset
df=pd.read_csv("Telco_Cusomer_Churn.csv")

print(df.head())


# 2.BASIC CLEANING
# drop customer ID(no predictive value)
df.drop('customerID',axis=1,inplace=True)

# convert totalcharges to numeric (kaggle issue)
df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')

df['TotalCharges'].fillna(df['TotalCharges'].median(),inplace=True)


# target encoding
df['Churn']=df['Churn'].map({'No':0,'Yes':1})

#3. encoding categorical features

categorical_cols=df.select_dtypes(include='object').columns
df=pd.get_dummies(df,columns=categorical_cols,drop_first=True)


# 4.FEATUES AND TARGET
X=df.drop('Churn',axis=1)
y=df['Churn']


# 5.TRAIN-TEST-SPLIT(STRATIFIED)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)


# 6. PIPELINE(SCALING +SMOTE+MODEL)

pipeline=Pipeline([
    ('scaler',StandardScaler()),
    ('smote',SMOTE(random_state=42)),
    ('model',RandomForestClassifier(random_state=42))
])

# 7. HYPERPARAMETER TUNING
param_grid={
    'model__n_estimators':[100,200],
    'model__max_depth':[None,10,20],
    'model__min_samples_split':[2,5],
    'model__min_samples_leaf':[1,2]
}

grid=GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

# 8.TRAIN MODEL
grid.fit(X_train,y_train)
best_model=grid.best_estimator_


# 9.PREDICTION AND EVALUATION

y_pred=best_model.predict(X_test)
y_prob=best_model.predict_proba(X_test)[:,1]

print("best parameters :\n",grid.best_params_)
print("classification report:\n",classification_report(y_test,y_pred))
print("ROC-AUC Score:",roc_auc_score(y_test,y_prob))


print("hello git")