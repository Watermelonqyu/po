import po

# get data from machineuft predict use LinearRegression and save it in L_machine
comp = po.read_csv("data\\computer_hardware.csv", encoding='utf-8')

comp.Regression(["MYCT", "MMIN", "CACH", "CHMIN", "CHMAX", "PRP"], "ERP", method="LinearRegression", portion=0.8)

comp.to_csv("Hardware_Linear.csv", encoding='utf-8')


# get data from machineuft predict use LinearRegression and save it in L_machine
comp = po.read_csv("data\\computer_hardware.csv", encoding='utf-8')

comp.Regression(["MYCT", "MMIN", "CACH", "CHMIN", "CHMAX", "PRP"], "ERP", method="RandomForest", portion=0.8)

comp.to_csv("Hardware_RandomForest.csv", encoding='utf-8')


# get data from machineuft predict use LinearRegression and save it in L_machine
comp = po.read_csv("data\\computer_hardware.csv", encoding='utf-8')

comp.Regression(["MYCT", "MMIN", "CACH", "CHMIN", "CHMAX", "PRP"], "ERP", method="KNeighbors", n_neighbors=4, portion=0.8)

comp.to_csv("Hardware_KN.csv", encoding='utf-8')