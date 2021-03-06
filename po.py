from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering, MeanShift
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import kneighbors_graph
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor 
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import Imputer

from sklearn import svm
import seaborn as sns
import pandas
import matplotlib.pyplot as plt
import math
import numpy
from numpy import savetxt

def read_csv(filename, **kwargs):
   return Po(pandas.read_csv(filename, **kwargs))

# added by Qiong
'''
import statsmodels.api as sm

def Logit(endog, exog, **kwargs):
   return Po(sm.Logit(endog, exog, **kwargs))
'''
import sklearn.cross_validation as cv
import sklearn.linear_model as lm


def get_dummies(data, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False):
   return Po(pandas.get_dummies(data, prefix, prefix_sep, dummy_na, columns, sparse))

class Po(pandas.core.frame.DataFrame):
   def __init__(self, df):
      super(Po, self).__init__(df)
      self.estimator = None

   def __getitem__(self, key):
      a = super(Po, self).__getitem__(key)
      return Po(a) if isinstance(a, pandas.core.frame.DataFrame) else a

   def query(self, expr, **kwargs):
      return Po(super(Po,self).query(expr, **kwargs))


   def Cluster(self, columns, **argv):
      if type(columns) != list:
         raise Exception("First parameter must be a list.")
      unknown_columns = set(columns)-set(self.keys())
      if unknown_columns != set([]):
         raise Exception("Invalid columns: " + str(unknown_columns))

      option = {}
      Estimator = dict(kmeans=KMeans, meanshift=MeanShift, dbscan=DBSCAN, hierarchical=AgglomerativeClustering, spectral=SpectralClustering)

      if argv.get('method') is None:
         method = 'meanshift' if argv.get('clusters') is None else 'kmeans'
      else:
         if argv.get('method') not in Estimator:
            raise Exception("Unknown clustering method: " + argv.get('method'))
         method = argv.get('method', 'meanshift')

      if method == 'meanshift':
         pass
      elif method == 'dbscan':
         option['eps'] = argv.get('spacing', 0.3)
      else:
         if argv.get('clusters') is None:
            raise Exception("Must specify 'clusters', which is a number greater than 1.")
         option['n_clusters'] = argv.get('clusters')
         if argv.get('method') == 'hierarchical':
            option['linkage'] = argv.get('linkage', 'average')
            option['affinity'] = argv.get('affinity', 'euclidean')
         elif argv.get('method') == 'spectral':
            option['affinity'] = argv.get('affinity', 'rbf')

      self.estimator = Estimator.get(method)(**option)

      ## Select data
      rows = self.get(columns)
      if argv.get('scaled') == True:
         rows = StandardScaler().fit_transform(rows)

      ## Cluster and store results
      labels = self.estimator.fit_predict(rows)
      self['_'+method+'_'] = labels

      if method == 'kmeans':
         p = [ self.get(c) for c in columns ]
         self['_certainty_'] = [ self.point_entropy(p, i) for i in range(len(p[0])) ]

      print("\tClustering method: ", method, "\tNumber of clusters: ", len(set(labels)))


   # if portion less than 0, portion = 0.5
   # if portion greater than 1, portion = 1
   def Regression(self, columns, predictName, portion=0.5, **argv):
      if type(columns) != list:
         raise Exception("First parameter must be a list.")
      if type(predictName) != str:
         raise Exception("Second parameter must be a string.")

      unknown_columns = set(columns)-set(self.keys())
      if unknown_columns != set([]):
         raise Exception("Invalid columns: " + str(unknown_columns))

      option = {}

      if argv.get('method') is None:
         method = 'LinearRegression'
      # perform function
      elif argv.get('method') == "LinearRegression":
         method = "LinearRegression"
      elif argv.get('method') == "RandomForest":
         method = "RandomForest"
      elif argv.get('method') == "SVM":
         method = "SVM"
      elif argv.get('method') == "KNeighbors":
         method = "KNeighbors"
      else:
         raise Exception("Unknown regression method: " + argv.get('method'))

      # specify portion
      if argv.get('portion') is None:
         portion = 0.5
      else:
         portion = argv.get('portion') 

      ## Select data
      ## need to fix the problem with NaN or infinate value
      x_rows = self.get(columns)
      # x_rows = Imputer().fit_transform(x_rows)
      y_rows = self.get(predictName)
      # y_rows = Imputer().fit_transform(y_rows)

      xRowNum = 0
      yRowNum = 0
      for x in x_rows.index:
         if numpy.any(numpy.isnan(x_rows.loc[x])):
            print("Dropped: ", x, x_rows.loc[x])
            x_rows.drop(x_rows.loc[x])
            y_rows.drop(y_rows.loc[x])


      if (xRowNum < yRowNum):
         y_rows = y_rows[:xRowNum-2]

      if (xRowNum > yRowNum):
         x_rows = x_rows[:yRowNum-2]

      # get portion
      if portion > 1:
         portion = 1
      elif portion < 0:
         portion = 0.5


      # specify neighbor
      if argv.get('n_neighbors') is None:
         n_neighbors = 3
      else:
         n_neighbors = argv.get('n_neighbors') 


      # round
      x_train, x_test, y_train, y_test = train_test_split(x_rows, y_rows, test_size = 1-portion)
      
      # perform function
      if argv.get('method') == "LinearRegression":
         lin = lm.LinearRegression()
         lin.fit(x_rows, y_rows)
         self['linearPredict'] = ""
         for index in x_test.index:
            self['linearPredict'].loc[index] = lin.predict(x_test.loc[index])
         print ("Method: ", method, "\tCoefficients: ", lin.coef_, "\tVariance score: %.2f", lin.score(x_test, y_test))

      # perform function
      elif argv.get('method') == "RandomForest":
         ran = RandomForestRegressor()
         ran.fit(x_train, y_train)
         self['RandomForestClassifier'] = ""
         for index in x_test.index:
            self['RandomForestClassifier'].loc[index] = ran.predict(x_test.loc[index])
         print ("Method: ", method, "\t Score: ", ran.score(x_test, y_test))
      
      elif argv.get('method') == "SVM":
         sssvm = svm.SVR()
         sssvm.fit(x_train, y_train)
         self['svmPredict'] = ""
         for index in x_test.index:
            self['svmPredict'].loc[index] = sssvm.predict(x_test.loc[index])
         
         print ("Method: ", method, "\t Score: ", sssvm.score(x_test, y_test))

      elif argv.get('method') == "KNeighbors":
         knn = KNeighborsRegressor(n_neighbors)
         knn.fit(x_train, y_train)
         self['KNeighborsPredict'] = ""
         for index in x_test.index:
            self['KNeighborsPredict'].loc[index] = knn.predict(x_test.loc[index])
         print ("Method: ", method, "\t Score: ", knn.score(x_test, y_test))
      

   def Classify(self, columns, predictName, **argv):
      if type(columns) != list:
         raise Exception("First parameter must be a list.")
      if type(predictName) != str:
         raise Exception("Second parameter must be a string.")

      unknown_columns = set(columns)-set(self.keys())
      if unknown_columns != set([]):
         raise Exception("Invalid columns: " + str(unknown_columns))

      option = {}

      if argv.get('method') is None:
         method = 'RandomForest'
      # perform function
      elif argv.get('method') == "LogisticRegression":
         method = "LogisticRegression"
      elif argv.get('method') == "RandomForest":
         method = "RandomForest"
      elif argv.get('method') == "SVM":
         method = "SVM"
      elif argv.get('method') == "GaussianNB":
         method = "GaussianNB"
      elif argv.get('method') == "KNeighbors":
         method = "KNeighbors"
      else:
         raise Exception("Unknown classify method: " + argv.get('method'))

      # specify portion
      if argv.get('portion') is None:
         portion = 0.5
      else:
         portion = argv.get('portion') 


      # specify neighbor
      if argv.get('n_neighbors') is None:
         n_neighbors = 3
      else:
         n_neighbors = argv.get('n_neighbors') 

      ## Select data
      x_rows = self.get(columns)
      y_rows = self.get(predictName)

      xRowNum = 0
      yRowNum = 0
      for x in x_rows.index:
         if numpy.any(numpy.isnan(x_rows.loc[x])):
            print("Dropped: ", x, x_rows.loc[x])
            x_rows.drop(x_rows.loc[x])
            y_rows.drop(y_rows.loc[x])


      if (xRowNum < yRowNum):
         y_rows = y_rows[:xRowNum-2]

      if (xRowNum > yRowNum):
         x_rows = x_rows[:yRowNum-2]


      if portion > 1:
         portion = 1
      elif portion < 0:
         portion = 0.5

      # round
      x_train, x_test, y_train, y_test = train_test_split(x_rows, y_rows, test_size = 1-portion)
      
      # perform function
      if argv.get('method') == "RandomForest":
         ran = RandomForestClassifier()
         ran.fit(x_train, y_train)
         self['RandomForestClassifier'] = ""
         for index in x_test.index:
            self['RandomForestClassifier'].loc[index] = ran.predict(x_test.loc[index])
         scores = accuracy_score(y_test, ran.predict(x_test))
         print ("Method: ", method, "\t Score: ", scores)
      
      elif argv.get('method') == "LogisticRegression":
         log = lm.LogisticRegression()
         log.fit(x_train, y_train)
         self['logisticPredict'] = ""
         for index in x_test.index:
            self['logisticPredict'].loc[index] = log.predict(x_test.loc[index])
         scores = accuracy_score(y_test, log.predict(x_test))
         print ("Method: ", method, "\tCoefficients: ", log.coef_, "\tVariance score: %.2f", scores)
      
      elif argv.get('method') == "SVM":
         ssvm = svm.SVC()
         ssvm.fit(x_train, y_train)
         self['svmPredict'] = ""
         for index in x_test.index:
            self['svmPredict'].loc[index] = ssvm.predict(x_test.loc[index])
         scores = accuracy_score(y_test, ssvm.predict(x_test))
         print ("Method: ", method, "\t Score: ", scores)
      
      elif argv.get('method') == "GaussianNB":
         gnb = GaussianNB()
         gnb.fit(x_train, y_train)
         self['GaussianNBPredict'] = ""
         for index in x_test.index:
            self['GaussianNBPredict'].loc[index] = gnb.predict(x_test.loc[index])
         scores = accuracy_score(y_test, gnb.predict(x_test))
         print ("Method: ", method, "\t Score: ", scores)

      elif argv.get('method') == "KNeighbors":
         kn = KNeighborsClassifier(n_neighbors)
         kn.fit(x_train, y_train)
         self['KNeighborsPredict'] = ""
         for index in x_test.index:
            self['KNeighborsPredict'].loc[index] = kn.predict(x_test.loc[index])
         scores = accuracy_score(y_test, kn.predict(x_test))
         print ("Method: ", method, "\t Score: ", scores)
      
   # Added by Qiong
   # Use another datafrom to train
   def ClassifyAs(self, columns, predictName, DataFrame, **argv):
      if type(columns) != list:
         raise Exception("First parameter must be a list.")
      if type(predictName) != str:
         raise Exception("Second parameter must be a string.")

      unknown_columns = set(columns)-set(self.keys())
      if unknown_columns != set([]):
         raise Exception("Invalid columns: " + str(unknown_columns))

      option = {}

      # 
      if argv.get('n_neighbors') is None:
         n_neighbors = 3
      else:
         n_neighbors = argv.get('n_neighbors')


      if argv.get('method') is None:
         method = 'RandomForest'
      # perform function
      elif argv.get('method') == "LogisticRegression":
         method = "LogisticRegression"
      elif argv.get('method') == "RandomForest":
         method = "RandomForest"
      elif argv.get('method') == "SVM":
         method = "SVM"
      elif argv.get('method') == "GaussianNB":
         method = "GaussianNB"
      elif argv.get('method') == "KNeighbors":
         method = "KNeighbors"
      else:
         if argv.get('method') not in Estimator:
            raise Exception("Unknown classify method: " + argv.get('method'))

      ## Select data
      x_train_rows = DataFrame.get(columns)
      y_train_rows = DataFrame.get(predictName)

      x_test_rows = self.get(columns)

      # perform function
      if argv.get('method') == "RandomForest":
         ran = RandomForestClassifier()
         ran.fit(x_train_rows, y_train_rows)
         self['RandomForestClassifier'] = ""
         for index in x_test_rows.index:
            self['RandomForestClassifier'].loc[index] = ran.predict(x_test_rows.loc[index])
        
      elif argv.get('method') == "LogisticRegression":
         log = lm.LogisticRegression()
         log.fit(x_train, y_train)
         self['logisticPredict'] = ""
         for index in x_test.index:
            self['logisticPredict'].loc[index] = log.predict(x_test.loc[index])
         print ("Method: ", method, "\tCoefficients: ", log.coef_, "\tVariance score: %.2f", log.score(x_test, y_test))

      elif argv.get('method') == "SVM":
         ssvm = svm.SVC()
         ssvm.fit(x_train_rows, y_train_rows)
         self['svmPredict'] = ""
         for index in x_test_rows.index:
            self['svmPredict'].loc[index] = ssvm.predict(x_test_rows.loc[index])
      
      elif argv.get('method') == "GaussianNB":
         gnb = GaussianNB()
         gnb.fit(x_train_rows, y_train_rows)
         self['GaussianNBPredict'] = ""
         for index in x_test_rows.index:
            self['GaussianNBPredict'].loc[index] = gnb.predict(x_test_rows.loc[index])

      elif argv.get('method') == "KNeighbors":
         kn = KNeighborsClassifier(n_neighbors)
         kn.fit(x_train_rows, y_train_rows)
         self['KNeighborsPredict'] = ""
         for index in x_test_rows.index:
            self['KNeighborsPredict'].loc[index] = kn.predict(x_test_rows.loc[index])

   def point_entropy(self, points, i):
      d = []

      for c in self.estimator.cluster_centers_:
         d.append(math.sqrt(sum((points[j][i]-c[j])**2 for j in range(len(points)))))
      d.sort()
      if d[0] == 0:
         entropy = 1 if d[1] != 0 else 0
      else:
         p1, p2 = float(d[0])/float(d[0]+d[1]), float(d[1])/float(d[0]+d[1])
         entropy = - (p1 * math.log(p1) + p2 * math.log(p2)) / math.log(2)

      return 1 - entropy


   def Plot(self, x, y=None, **kwargs):
      if x not in self.dtypes:
         raise Exception("Unknown column: " + x)
      if y is not None and y not in self.dtypes:
         raise Exception("Unknown column: " + y)

      if y is not None:
         if self.dtypes[x] in [int, float] and self.dtypes[y] in [int, float]:
            kwargs.setdefault("fit_reg", False)
            sns.lmplot(x=x, y=y, data=self, **kwargs)
         elif self.dtypes[x] in [object] and self.dtypes[y] in [int, float]:
            sns.factorplot(x=x, y=y, data=self, **kwargs)
         elif self.dtypes[y] in [object] and self.dtypes[x] in [int, float]:
            kwargs['orient'] = 'h'
            sns.factorplot(x=x, y=y, data=self, **kwargs)

      else:
         if self.dtypes[x] in [int, float]:
            sns.distplot(self[x], **kwargs)

      plt.show()
