from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
import pickle

df = pd.read_csv('Flask/students.csv')
df.sample(5)

X = df.drop(columns=['placed'])
y = df['placed']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

#train the model
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
print(X_test)
y_pred = rf.predict(X_test)

print(accuracy_score(y_test,y_pred))

#save the model in pickle format
pickle.dump(rf,open('Flask/model.pkl','wb'))