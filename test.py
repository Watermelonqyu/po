import po
import pandas

df = po.read_csv("http://www.ats.ucla.edu/stat/data/binary.csv")

df.columns = ["admit", "gre", "gpa", "prestige"]

dummy_ranks = po.get_dummies(df['prestige'], prefix = "prestige")

print (dummy_ranks)

cols_to_keep = ['admit', 'gre', 'gpa']

data = df[cols_to_keep].join(dummy_ranks.ix[:, 'prestige_2':])

data['intercept'] = 1.0

train_cols = data.columns[1:]

logit = po.Logit(data['admit'], data[train_cols])

result = logit.fit()

print (result.summary())