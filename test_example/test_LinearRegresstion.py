import po

iris = po.read_csv("D:\\MasterProject\\po-master\\po\\data\\iris.csv")

iris.Regression(["PetalWidth", "PetalLength"], "Species", method="LinearRegression")
