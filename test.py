import po

iris = po.read_csv("data/iris.csv")

iris.Classify(["PetalWidth", "PetalLength"], "Species", method="LinearRegression")
