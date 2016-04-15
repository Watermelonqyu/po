import po

# get data from machineuft predict use LogisticRegression and save it in L_machine
iris = po.read_csv("D:\\MasterProject\\po-master\\po\\data\\F2011-F2015_Regression_FulColumn.csv", encoding='utf-8')

iris.Regression(["ARESD", "ACT", "HSGPA", "STATE", "RACE", "FIRSTGEN", "FATHEREDUCATION", "MOTHEREDUCATION"], "REGTYPE", method="LogisticRegression", portion=0.9)

iris.to_csv("L_Registration.csv", encoding='utf-8')



# get data from machineuft predict use LinearRegression and save it in L_machine
iris = po.read_csv("D:\\MasterProject\\po-master\\po\\data\\F2011-F2015_Regression_FulColumn.csv", encoding='utf-8')

iris.Regression(["ARESD", "ACT", "HSGPA", "STATE", "RACE", "FIRSTGEN", "FATHEREDUCATION", "MOTHEREDUCATION"], "REGTYPE", method="LinearRegression", portion=0.9)

iris.to_csv("Linear_Registration.csv", encoding='utf-8')


# get data from machineuft predict use RandomForest and save it in L_machine
iris = po.read_csv("D:\\MasterProject\\po-master\\po\\data\\F2011-F2015_Regression_FulColumn.csv", encoding='utf-8')

iris.Classify(["ARESD", "ACT", "HSGPA", "STATE", "RACE", "FIRSTGEN", "FATHEREDUCATION", "MOTHEREDUCATION"], "REGTYPE", method="RandomForest", portion=0.9)

iris.to_csv("RandomForest_Registration.csv", encoding='utf-8')
