import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import joblib


df = pd.read_csv('heart.csv')
X = df.drop('output', axis=1) 
y = df['output']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
knn = KNeighborsClassifier()
model = knn.fit(x_train, y_train)
ypred = model.predict(x_test)

r2 = r2_score(y_test, ypred)
print("R2 Score:", r2)

joblib.dump(model, 'knn_model.joblib')
joblib.dump(scaler, 'scaler.joblib')