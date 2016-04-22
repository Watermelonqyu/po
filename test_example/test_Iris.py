import po

# get data from machineuft predict use LinearRegression and save it in L_machine
iris = po.read_csv("data\\iris.csv", encoding='utf-8')

iris.Classify(["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"], "Species", method="RandomForest", portion=0.7)

iris.to_csv("Iris_RandomForest.csv", encoding='utf-8')



# get data from machineuft predict use LinearRegression and save it in L_machine
iris = po.read_csv("data\\iris.csv", encoding='utf-8')

iris.Classify(["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"], "Species", method="SVM", portion=0.7)

iris.to_csv("Iris_SVM.csv", encoding='utf-8')



# get data from machineuft predict use LinearRegression and save it in L_machine
iris = po.read_csv("data\\iris.csv", encoding='utf-8')

iris.Classify(["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"], "Species", method="KNeighbors", n_neighbors=4, portion=0.7)

iris.to_csv("Iris_Kneighbors.csv", encoding='utf-8')