import po

iris = po.read_csv("D:\\MasterProject\\po-master\\po\\data\\iris.csv")

iris.Classify(["PetalWidth", "PetalLength"], "Species", method="GaussianNB")
