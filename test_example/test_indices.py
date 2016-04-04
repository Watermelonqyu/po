import pandas
from sklearn.cross_validation import train_test_split
import sklearn.linear_model as lm
from sklearn.ensemble import RandomForestClassifier

iris = pandas.read_csv("D:\\MasterProject\\po-master\\po\\data\\iris.csv")
x_rows = iris.get(["PetalWidth", "PetalLength"])
y_rows = iris.get("Species")


# x_train_index, x_test_index, y_train_index, y_test_index = train_test_split(x_rows.index, y_rows.index, test_size=0.2)

x_train, x_test, y_train, y_test = train_test_split(x_rows, y_rows, test_size=0.2)

# get value of an index
print(x_test)

ran = RandomForestClassifier()
ran.fit(x_train, y_train)

iris['RandomForestClassifier'] = ""


for index in x_test.index:
	iris['RandomForestClassifier'].loc[index] = ran.predict(x_test.loc[index])
	print (index, "\t", x_test.loc[index])
	print ("\n")
	# iris['RandomForestClassifier'][index] = ran.predict(x_test.iloc(index))

print(iris['RandomForestClassifier'])