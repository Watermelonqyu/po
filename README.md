
Po is a Python module facilitates analysis of structured data.  It reads in a comma/tab-separated file and allows querying, selection, visualization, clustering and further analysis of the data.

Po is built on top of pandas, scikit-learn, and seaborn.

### Installation

##### Fresh install

1. Install [Anaconda distribution](http://continuum.io/downloads) with Python 3.  Po requires pandas, matplotlib, scikit-learn, and seaborn.  By default, pandas, matplotlib and scikit-learn are installed by anaconda.

2. Install seaborn:  **conda install seaborn**

3. Install po from github:  **pip install git+https://github.com/vtphan/po**

##### Upgrade to the latest version

+ **pip install -U git+https://github.com/vtphan/po**

### Read data

```
import po
iris = po.read_csv("data/iris.csv")
```

*iris* is a po instance, which is a glorified pandas data frame.  Thus, a po instance has access to all utilities available to pandas data frames.

Tab-separated files can also be read easily.
```
indels = po.read_csv("data/indels.txt", sep="\t")
```

### Query and select data

po instances have a *query* method, which wraps around pandas data frame's query method.  Po.query returns a po instance.

```
setosa = iris.query('Species == "setosa" and PetalWidth > 0.1')
```

Selecting columns by giving a list of column names. The return value is a po instance.

```
only_two_cols = iris[["Species", "PetalLength"]]
```

Consult [pandas documentation](http://pandas.pydata.org/pandas-docs/stable/indexing.html) for further information on how to select and query Pandas data frames.

##### Write a po instance (data frame) to file

```
setosa.to_csv("setosa.csv")
setosa.to_csv("setosa.csv", index=False)        # no index column
only_two_cols.to_csv("b.tsv", sep="\t")         # tab separated (default is comma separated)
```

### Cluster data

Cluster rows into 3 clusters based on petal widths and lengths.  Clustering is done using [k-means](http://scikit-learn.org/stable/modules/clustering.html#k-means).  Cluster labels are placed in a new column called *_kmeans_*.

```
iris.Cluster(["PetalWidth", "PetalLength"], clusters=3)
```

In case the number of clusters is not specified, clustering is done using [meanshift](http://scikit-learn.org/stable/modules/clustering.html#mean-shift).  Cluster labels are placed in a new column called *_meanshift_*.

```
iris.Cluster(["PetalWidth", "PetalLength"])
```

Other clusterting methods include [hierarchical clustering](http://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering), [spectral clustering](http://scikit-learn.org/stable/modules/clustering.html#spectral-clustering), and [dbscan](http://scikit-learn.org/stable/modules/clustering.html#dbscan).

```
iris.Cluster(["PetalWidth", "PetalLength"], method="hierarchical", clusters=3)
```

### Visualize data

When one variable (defined by column name) is given, it must be a numerical variable.  The plot is a distribution plot.

```
iris.Plot("SepalLength")
```

When a numerical variable is compared against another numerical variable, the plot is a scatter plot.

```
iris.Plot("SepalLength", "SepalWidth")
```

Colors can be added to differentiate rows with values belonging to a categorical variable.

```
iris.Plot("SepalLength", "SepalWidth", hue="Species")
```

Scatter plot separated into different columns and rows.
```
iris.Plot("SepalLength", "SepalWidth", col="Species")
iris.Plot("SepalLength", "SepalWidth", row="Species")
```

When a categorical variable is plot against a numerical variable, the plot can be either a *point* plot (default), *bar* plot, *strip* plot, a [*box*plot](https://en.wikipedia.org/wiki/Box_plot), or a [*violin* plot](https://en.wikipedia.org/wiki/Violin_plot).

```
iris.Plot("Species", "SepalLength")
iris.Plot("Species", "SepalLength", kind="bar")
iris.Plot("Species", "SepalLength", kind="box")
iris.Plot("Species", "SepalLength", kind="violin")
```


### Perform Logistic Regression

Here is an example for one data set:

```
# Get data
data = po.read_csv("train.csv")

y = data['Survived']
X = data[['Age','SibSp','Fare']].fillna(0)

indices = np.arange(data.shape[0])

# Train data 
X_train, X_test, y_train, y_test = po.train_test_split(X, y, test_size=0.2, random_state=42)

# Perform Logistic Regression
lr = po.LogisticRegression()
lr.fit(X_train, y_train)

# Get predict value
print("Accuracy of logistic regression (skilearn):",lr.predict(X_test), y_test)```
```

### Perform Linear Regression

```
# Get dataset from sklearn
from sklearn import datasets

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# Create linear regression object
regr = po.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# The coefficients
print('Coefficients: \n', regr.coef_)
```

