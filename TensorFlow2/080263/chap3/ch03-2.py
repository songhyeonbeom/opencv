import pandas as pd
df = pd.read_csv('080263/chap3/data/titanic/train.csv', index_col='PassengerId')
print(df.head())


df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = df.dropna()
X = df.drop('Survived', axis=1)
y = df['Survived']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

from sklearn import tree
model = tree.DecisionTreeClassifier()

model.fit(X_train, y_train)

y_predict = model.predict(X_test)
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_predict))
# accuracy_score(y_test, y_predict)

from sklearn.metrics import confusion_matrix
pd.DataFrame(
    confusion_matrix(y_test, y_predict),
    columns=['Predicted Not Survival', 'Predicted Survival'],
    index=['True Not Survival', 'True Survival']
)

# get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.datasets import load_digits
digits = load_digits()
print("Image Data Shape" , digits.data.shape)
print("Label Data Shape", digits.target.shape)

