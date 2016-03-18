import po
import numpy as np

data = po.read_csv("train.csv")

y = data['Survived']
X = data[['Age','SibSp','Fare']].fillna(0)

indices = np.arange(data.shape[0])

# get the data you want to predict
features = data.columns[:3]
preData = data[features]
preData = preData[1:]
X_train, X_test, y_train, y_test = po.train_test_split(X, y, test_size=0.2, random_state=42)
print (data.RandomForestClassifier(X_train, y_train, preData))