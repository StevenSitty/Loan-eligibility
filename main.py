import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn import metrics, svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier

############ READING CSV FILE ##############
df = pd.read_csv('loan_prediction_dataset.csv')

# print(df.isnull().sum())
# print(df.duplicated().sum())


############ Dropping the id column ##############
df = df.drop(["Loan_ID"], axis=1)

############ handling empty cells ##############
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
df["Self_Employed"] = df["Self_Employed"].fillna(df["Self_Employed"].mode()[0])
df['LoanAmount'] = df['LoanAmount'].fillna(int(df['LoanAmount'].mean()))
df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(int(df['LoanAmount'].mean()))
df['Credit_History'] = df['Credit_History'].fillna(int(df['Credit_History'].mode()[0]))
# print(df.isnull().sum())

############ Encoding str values ##############
label = LabelEncoder()
df['Gender'] = label.fit_transform(df['Gender'])
df['Married'] = label.fit_transform(df['Married'])
df['Dependents'] = label.fit_transform(df['Dependents'])
df['Education'] = label.fit_transform(df['Education'])
df["Self_Employed"] = label.fit_transform(df["Self_Employed"])
df["Property_Area"] = label.fit_transform(df["Property_Area"])

############ getting features and classification ##############
X = df.drop(["Loan_Status"], axis=1).values
Y = df["Loan_Status"]

############ One hot encoder ##############
# onehot_encoder = OneHotEncoder(categories='auto')
# X = onehot_encoder.fit_transform(X).toarray()

############ Splitting data into train and test ##############
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=8)  # 80% training and 20% test

############ STANDARDIZATION ##############
st_x = StandardScaler()
X_train = st_x.fit_transform(X_train)
X_test = st_x.transform(X_test)

# print(X_train)

############ DECISION TREE ##############
ID3 = DecisionTreeClassifier(criterion="gini", max_depth=1)
ID3 = ID3.fit(X_train, Y_train)
Y_pred_ID3 = ID3.predict(X_test)

############ SVM ##############
clf = svm.SVC(kernel='linear')
clf.fit(X_train, Y_train)
Y_pred_SVM = clf.predict(X_test)

############ Logistic Regression ##############
model = LogisticRegression(solver='liblinear')
model.fit(X_train, Y_train)
Y_pred_LogReg = model.predict(X_test)

############ Random Forest ##############
RForest = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
RForest.fit(X_train, Y_train)
Y_pred_RForest = RForest.predict(X_test)

############ NAIVE BAYES ##############
bnb = BernoulliNB()
bnb.fit(X_train, Y_train)
Y_pred_BNB = bnb.predict(X_test)

############ KNN ##############
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, Y_train)
Y_pred_KNN = knn.predict(X_test)

############ ACCURACY PRINT ##############
# print("ID3 Accuracy:", metrics.accuracy_score(Y_test, Y_pred_ID3))
# print("SVM Accuracy:", metrics.accuracy_score(Y_test, Y_pred_SVM))
# print("Logistic Regression Accuracy:", metrics.accuracy_score(Y_test, Y_pred_LogReg))
# print("Random Forest Accuracy:", metrics.accuracy_score(Y_test, Y_pred_RForest))
# print("Naive Bayes Accuracy:", metrics.accuracy_score(Y_test, Y_pred_BNB))
# print("knn Accuracy:", metrics.accuracy_score(Y_test, Y_pred_KNN))

############ classify according to input ##################
user_input: list[int] = []

while True:
    gender = input("please enter user gender(m/f): ")
    if gender.lower() == "m":
        user_input.append(1)
        break
    elif gender.lower() == "f":
        user_input.append(0)
        break
    else:
        print("wrong input")

print(user_input)

while True:
    marry_state = input("please enter user martial status(y/n): ")
    if marry_state.lower() == "y":
        user_input.append(1)
        break
    elif marry_state.lower() == "n":
        user_input.append(0)
        break
    else:
        print("wrong input")

print(user_input)

while True:
    dependants = input("please enter user dependants: ")
    if dependants.isnumeric():
        if int(dependants) > 2:
            user_input.append(3)
            break
        else:
            user_input.append(int(dependants))
            break
    else:
        print("wrong input")

print(user_input)

while True:
    education = input("is user graduate(y/n): ")
    if education.lower() == "y":
        user_input.append(0)
        break
    elif education.lower() == "n":
        user_input.append(1)
        break
    else:
        print("wrong input")

print(user_input)

while True:
    employment = input("is user self-employed(y/n): ")
    if employment.lower() == "y":
        user_input.append(1)
        break
    elif employment.lower() == "n":
        user_input.append(0)
        break
    else:
        print("wrong input")

print(user_input)

while True:
    appincome = input("enter applicant income: ")
    if appincome.isnumeric():
        user_input.append(int(appincome))
        break
    else:
        print("wrong input")

print(user_input)

while True:
    coappincome = input("enter coapplicant income: ")
    if not coappincome.isnumeric():
        print("wrong input")
    else:
        user_input.append(int(coappincome))
        break

print(user_input)

while True:
    loanamount = input("enter loan amount: ")
    if loanamount.isnumeric():
        if int(loanamount) < 1000:
            user_input.append(int(loanamount))
            break
        else:
            print("loan amount exceeded the limit")
    else:
        print("wrong input")

print(user_input)

while True:
    LoanAmountTerm = input("enter Loan Amount Term: ")
    if LoanAmountTerm.isnumeric():
        user_input.append(int(LoanAmountTerm))
        break
    else:
        print("wrong input")

print(user_input)

while True:
    Credit_History = input("enter Credit History: ")
    if Credit_History.isnumeric():
        if int(Credit_History) == 0:
            user_input.append(int(Credit_History))
            break
        elif int(Credit_History) == 1:
            user_input.append(int(Credit_History))
            break
        else:
            print("wrong input")
    else:
        print("wrong input")

print(user_input)

while True:
    Property_Area = input("please enter user Property Area('1.Rural/2.urban/3.semi urban'): ")
    if int(Property_Area) == 1:
        user_input.append(0)
        break
    elif int(Property_Area) == 2:
        user_input.append(2)
        break
    elif int(Property_Area) == 3:
        user_input.append(1)
        break
    else:
        print("wrong input")

print(user_input)


user: ndarray = np.array(user_input)


Y_pred_KNN = knn.predict(user.reshape(1, 11))
print(Y_pred_KNN)
