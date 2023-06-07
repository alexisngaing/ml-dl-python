#print and comment
#this is comment
name = "Alex"
age = 27
print('Student Name : {one}; age : {two}'.format(one = name, two = age))
print('Student Name : {}, age : {}'.format(age, name))

#data type: number
varA = 13
varB = 21.5
varC = varA*varB-varA
varD = varC/(varB+varA)
print(varA*varB)
print(varC)
print(varD)
print(varA*varA)

#data type: String
varStr1 = 'Example'
varStr2 = "Example"
varStr3 = "Example: i'm not robot"
print(varStr1, varStr2, varStr3)

#array of character
varStr = "I Love Machine Learning"
print(varStr)
print(varStr[7:])
print(varStr + " and Data Science")
print(varStr * 2)

#list
list = ['I', 'still', "love", "you", 2023]
print(list)

#list cons
list = ['I', 'still', "love", "you", 2023]
list.append("until the end")
print(list)
print(list[1])
print(list[2:4])
print(list[:4])
list[1] = "feeling"
print(list)
len(list)

#Dictionary
myDict = {}
myDict['one'] = 1
myDict[2] = "Nothing gonna changes my love for you"
myDict["3"] = 30
print(myDict['one'])
print(myDict[2])
print(myDict.keys())
print(myDict.values())

#tuple
tp = ('no', 'time', 'to', 'worry', 1, 2, 3)
print(tp[0])
tp[2] = "wash"

#condition
varName = "Alexander the Great"

if len(varName) < 10:
  print("100.000 power")
elif len(varName) < 20 and len(varName) >= 10:
  print("500.000 power")
else:
  print("900.000 power")

#loop
list = ["one", "two", "three"]
parameter = [0, 1, 2, 3, 4, 5]

temp = 0
for number in parameter:
  print(number)
  temp = temp + number

print("Temp value : ", temp)

for n in list:
  print(n)

for item in list:
  print(item + item)

#loop cons
import numpy as np

for i in range(0, 21, 4):
  print(i)

n_test = np.arange(1, 11, 2)

print(n_test)

for item in n_test:
  print(item)

#loop cons
i = 1
while i <= 10:
  print("i is: {}".format(i))
  i = i + 1

#function
def triangle_area(base, height):
  area = 0.5 * (base * height)
  return area

a = 3
t = 4

print(triangle_area(a, t))

#module
def print_name_x_times(name, value):
  print(name * value)

name = "Your name"
value = 10

print_name_x_times(name, value)

#dataset
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine

wine = load_wine()
df_wine = pd.DataFrame(data = wine.data, columns = wine.feature_names, index = None)

df_wine.head(20)

#check if data null or NaN
print("Data null\n", df_wine.isnull().sum())
print("\n")
print("Data empty\n", df_wine.empty)
print("\n")
print("Data NaN\n", df_wine.isna().sum())

#exploratory data analysis
df_wine['target'] = wine.target.astype(int)
freq = df_wine.target.value_counts()
freq.plot(kind = 'bar')

#static desc
df_wine.describe()

#train-test split
from sklearn.model_selection import train_test_split

X = df_wine.drop('target', axis = 1)
y = df_wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

print("X_train shape", X_train.shape)
print("X_test shape", X_test.shape)

print("y_train shape", y_train.shape)
print("y_test shape", y_test.shape)

print("y_train\n\n", y_train)
print("y_test\n\n", y_train)

#training model, prediction, and evaluation model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

KNN = KNeighborsClassifier(n_neighbors = 2)
DT = DecisionTreeClassifier(random_state = 0)

KNN.fit(X_train, y_train)
DT.fit(X_train, y_train)

#prediction: prediction with input new data
# 1. Alcohol : AA
# 2. Malic acid : B.B
# 3. Ash : C.C
# 4. Alkalinity : DD
# 5. Magnesium : DDE
# 6. Total phenols : A.AB
# 7. Flavanoids : A.AB
# 8. Nonflavanoid phenols : A.AB
# 9. Proanthocyanins : A.AB
# 10. Color intensity : B.CC
# 11. Hue : B.CC
# 12. od280/od315 of diluted wines : B.CC
# 13. proline: CCDD

X_new = np.array([[20, 0.7, 1.2, 34, 345, 2.00, 2.00, 2.00, 2.00, 7.12, 7.12, 7.12, 1234]])

print("X_new that will prediction", X_new.shape)

knn_predict = KNN.predict(X_new)
print("KNN prediction label", knn_predict)

dt_predict = DT.predict(X_new)
print("DT prediction label", dt_predict)

#evaluation model: evaluation model with test set
y_pred_knn = KNN.predict(X_test)
y_pred_dt = DT.predict(X_test)

print("KNN prediction to X_test : ", y_pred_knn)
print("DT prediction to X_test : ", y_pred_dt)

#evaluation model: evaluation model with test set cons
print("Accuration KNN model prediction comparison vs label : ", format(np.mean(y_pred_knn == y_test.ravel())))
print("Accuration DT model prediction comparison vs label : ", format(np.mean(y_pred_dt == y_test.ravel())))

#model evaluation: model evaluation with test set cons
print("Accuration KNN model with function score", KNN.score(X_test, y_test))
print("Accuration DT model prediction comparison vs label", DT.score(X_test, y_test))
