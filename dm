import timeit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import mlxtend
import warnings
warnings.filterwarnings("ignore")
df=pd.read_csv('dataset.csv')

print("missing value")
print(df.isnull())
#print(df.shape)
print(df.describe())
df.drop(['name','equinox','pdes','id','prefix','spkid','full_name'],axis=1,inplace=True)


df.isnull().sum()
df.dropna(axis=1,inplace=True)
print("after removing missing values")
print(df.isnull())
print("unique values in the column")
print(df.pha.value_counts()) #how many belong to each class of target variable
threat=df[df.pha=='Y']
non_threat=df[df.pha=='N']
outlier_percentage=(df.pha.value_counts()[1]/df.pha.value_counts()[0])*100
print('Potential threat asteroids are: %.3f%%'%outlier_percentage)
print('Threat asteroids: ',len(threat))
print('Non-Threat asteroids: ',len(non_threat))
# Split the data into training and testing sets

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
etree=ExtraTreesClassifier(random_state=42)
rforest=RandomForestClassifier(random_state=42)
scaler=StandardScaler()
ohe=OneHotEncoder(sparse=False)
le=LabelEncoder()
features=X_train.columns.tolist()

X_train=X_train.apply(le.fit_transform)
X_test=X_test.apply(le.fit_transform)
y_train=le.fit_transform(y_train)
y_test=le.fit_transform(y_test)
X_train_scaled=scaler.fit_transform(X_train) 
X_test_scaled=scaler.fit_transform(X_test) 
#feature selection
start_time = timeit.default_timer()
mod=etree
# fit the model
mod.fit(X_train_scaled, y_train)
# get importance
importance = mod.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
df_importance=pd.DataFrame({'importance':importance},index=features)
df_importance.plot(kind='barh')
#plt.bar([x for x in range(len(importance))], importance)
elapsed = timeit.default_timer() - start_time
print('Execution Time for feature selection: %.2f minutes'%(elapsed/60))

feature_imp=list(zip(features,importance))
feature_sort=sorted(feature_imp, key = lambda x: x[1]) 
n_top_features=20
top_features=list(list(zip(*feature_sort[-n_top_features:]))[0])

X_train_sfs=X_train[top_features]
X_test_sfs=X_test[top_features]
X_train_sfs_scaled=scaler.fit_transform(X_train_sfs)
X_test_sfs_scaled=scaler.fit_transform(X_test_sfs)
models=[etree]
param_distributions=[{'criterion':['gini', 'entropy'],'n_estimators':[100,200,300]}]

for model in models:
    rand=RandomizedSearchCV(model,param_distributions=param_distributions[models.index(model)],cv=3,scoring='accuracy', n_jobs=-1, random_state=42,verbose=10)
    rand.fit(X_train_sfs_scaled,y_train)
    print(rand.best_params_,rand.best_score_) 
    

 

    
