import po



#  get data from machineuft predict use KNeighbors and save it in K_machine
iris = po.read_csv("D:\\MasterProject\\po-master\\po\\data\\machineuft.csv", encoding='utf-8')

iris.Classify(["MYCT", "MMIN", "CACH", "CHMIN", "CHMAX", "PRP"], "ERP", method="KNeighbors", portion=0.9 , n_neighbors=3)

iris.to_csv("K_machine.csv", encoding='utf-8')
