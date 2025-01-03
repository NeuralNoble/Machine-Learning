import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from KNeighborsClassifier import Knn

knn = KNeighborsClassifier(n_neighbors=5)

df = pd.read_csv('Social_Network_Ads.csv')

df = df.iloc[:,1:]
scaler = StandardScaler()
encoder = LabelEncoder()
df["Gender"] = encoder.fit_transform(df["Gender"])

X = df.iloc[:,0:3].values
Y = df.iloc[:,-1].values
X = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

knn.fit(X_train, Y_train)



y_pred = knn.predict(X_test)


print(accuracy_score(Y_test, y_pred))

apnaknn = Knn(k=5)
apnaknn.fit(X_train, Y_train)
y_pred1 = apnaknn.predict(X_test)
print(accuracy_score(Y_test, y_pred1))

