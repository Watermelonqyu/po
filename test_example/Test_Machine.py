import po

# get data from machineuft predict use LinearRegression and save it in L_machine
iris = po.read_csv("D:\\MasterProject\\po-master\\po\\data\\machineuft.csv", encoding='utf-8')

iris.Regression(["MYCT", "MMIN", "CACH", "CHMIN", "CHMAX", "PRP"], "ERP", method="LinearRegression", portion=0.9)

iris.to_csv("L_machine.csv", encoding='utf-8')



# get data from machineuft predict use RandomForest and save it in R_machine
iris = po.read_csv("D:\\MasterProject\\po-master\\po\\data\\machineuft.csv", encoding='utf-8')

iris.Classify(["MYCT", "MMIN", "CACH", "CHMIN", "CHMAX", "PRP"], "ERP", method="RandomForest", portion=0.9)

iris.to_csv("R_machine.csv", encoding='utf-8')


# get data from machineuft predict use Naive-Bayes and save it in N_machine
iris = po.read_csv("D:\\MasterProject\\po-master\\po\\data\\machineuft.csv", encoding='utf-8')

iris.Classify(["MYCT", "MMIN", "CACH", "CHMIN", "CHMAX", "PRP"], "ERP", method="GaussianNB", portion=0.9)

iris.to_csv("N_machine.csv", encoding='utf-8')


# get data from machineuft predict use SVM and save it in S_machine
iris = po.read_csv("D:\\MasterProject\\po-master\\po\\data\\machineuft.csv", encoding='utf-8')

iris.Classify(["MYCT", "MMIN", "CACH", "CHMIN", "CHMAX", "PRP"], "ERP", method="SVM", portion=0.9 )

iris.to_csv("S_machine.csv", encoding='utf-8')


#  get data from machineuft predict use KNeighbors and save it in K_machine
iris = po.read_csv("D:\\MasterProject\\po-master\\po\\data\\machineuft.csv", encoding='utf-8')

iris.Classify(["MYCT", "MMIN", "CACH", "CHMIN", "CHMAX", "PRP"], "ERP", method="KNeighbors", portion=0.9 , n_neighbors=3)

iris.to_csv("K_machine.csv", encoding='utf-8')
