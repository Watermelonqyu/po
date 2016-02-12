import po
import numpy as np
import pandas as pd

data = po.read_csv("train.csv")

y = data['Survived']
X = data[['Age','SibSp','Fare']].fillna(0)

indices = np.arange(data.shape[0])

X_train, X_test, y_train, y_test = po.train_test_split(X, y, test_size=0.2, random_state=42)
lr = po.LogisticRegression()
lr.fit(X_train, y_train)


print("Accuracy of logistic regression (skilearn):",lr.predict(X_test), y_test)
