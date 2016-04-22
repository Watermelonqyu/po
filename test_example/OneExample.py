import po
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 


# get data from machineuft predict use RandomForest and save it in L_machine
iris = po.read_csv("data\\F2011-F2015_Regression_FulColumn.csv", encoding='utf-8')

iris.Regression(["ACT"], "FATHEREDUCATION", method="KNeighbors", portion=0.8)




