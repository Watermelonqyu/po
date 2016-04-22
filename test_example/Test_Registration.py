import po

# get data from machineuft predict use LogisticRegression and save it in L_machine
iris = po.read_csv("data\\F2011-F2015_Regression_FulColumn.csv", encoding='utf-8')

iris.Classify(["ARESD", "ACT"], "REGTYPE", method="LogisticRegression", portion=0.8)

iris.to_csv("Registration_All_Logistic.csv", encoding='utf-8')


# get data from machineuft predict use RandomForest and save it in L_machine
iris = po.read_csv("data\\F2011-F2015_Regression_FulColumn.csv", encoding='utf-8')

iris.Classify(["ARESD", "ACT"], "REGTYPE", method="RandomForest", portion=0.9)

iris.to_csv("Registration_All_RandomForest.csv", encoding='utf-8')



# get data from machineuft predict use Naive-Bayes and save it in L_machine
iris = po.read_csv("D:\\MasterProject\\po-master\\po\\data\\F2011-F2015_Regression_FulColumn.csv", encoding='utf-8')

iris.Classify(["ARESD", "ACT", "HSGPA", "STATE", "RACE", "FIRSTGEN", "FATHEREDUCATION", "MOTHEREDUCATION"], "REGTYPE", method="GaussianNB", portion=0.9)

iris.to_csv("GaussianNB_Registration.csv", encoding='utf-8')



# get data from machineuft predict use SVM and save it in L_machine
iris = po.read_csv("D:\\MasterProject\\po-master\\po\\data\\F2011-F2015_Regression_FulColumn.csv", encoding='utf-8')

iris.Classify(["ARESD", "ACT", "HSGPA", "STATE", "RACE", "FIRSTGEN", "FATHEREDUCATION", "MOTHEREDUCATION"], "REGTYPE", method="SVM", portion=0.9)

iris.to_csv("SVM_Registration.csv", encoding='utf-8')


# get data from machineuft predict use KNeighbors and save it in L_machine
iris = po.read_csv("D:\\MasterProject\\po-master\\po\\data\\F2011-F2015_Regression_FulColumn.csv", encoding='utf-8')

iris.Classify(["ARESD", "ACT", "HSGPA", "STATE", "RACE", "FIRSTGEN", "FATHEREDUCATION", "MOTHEREDUCATION"], "REGTYPE", method="KNeighbors", portion=0.9)

iris.to_csv("KNeighbors_Registration.csv", encoding='utf-8')
