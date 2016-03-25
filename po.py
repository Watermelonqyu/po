from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering, MeanShift
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import pandas
import matplotlib.pyplot as plt
import math

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


def train_test_split(*arrays, **options):
	return cv.train_test_split(*arrays, **options)
	
	
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



   def Classify(self, columns, predictName, portion=0.5, **argv):
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
      elif argv.get('method') == "LinearRegression":
         method = "LinearRegression"
      else:
         if argv.get('method') not in Estimator:
            raise Exception("Unknown classify method: " + argv.get('method'))

      # specify portion
      if argv.get('portion') is None:
         portion = 0.5
      else:
         portion = argv.get('portion') 

      ## Select data
      x_rows = self.get(columns)
      y_rows = self.get(predictName)

      # round
      actualNum = int(round(x_rows.shape[0] * portion))

      x_actualRowTrain = x_rows[:actualNum]
      y_actualRowTrain = y_rows[:actualNum] 

      x_actualRowTest = x_rows[actualNum:]
      y_actualRowTest = y_rows[actualNum:]

      # perform function
      if argv.get('method') == "LogisticRegression":
         log = lm.LinearRegression()
         log.fit(x_actualRowTrain, y_actualRowTrain)
         print(log.predict(x_actualRowTest))
      if argv.get('method') == "LinearRegression":
         lin = lm.LogisticRegression()
         lin.fit(x_actualRowTrain, y_actualRowTrain)
         print(lin.predict(x_actualRowTest))
      if argv.get('method') == "RandomForest":
         ran = RandomForestClassifier()
         ran.fit(x_actualRowTrain, y_actualRowTrain)
           
      print("\tClassify method: ", method, "\tResult: ")



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
