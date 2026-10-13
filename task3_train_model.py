import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

#  1. Load dataset
df=pd.read_csv("/content/churnguard_data.csv")

# Clean the dataset

df=df.drop("customerID",axis=1)
df=df.drop_duplicates()

df["gender"]=df["gender"].str.strip()
df["PaymentMethod"]=df["PaymentMethod"].str.strip()

df["Churn"]=df["Churn"].str.strip().str.title()
df["PhoneService"]=df["PhoneService"].str.strip().str.title()
df["PaperlessBilling"]=df["PaperlessBilling"].str.strip().str.title()

contract_mapping={
  "Monthly":"Month-to-month",
  "Month to month":"Month-to-month",
  "Month-to-month":"Month-to-month",
  "1 year":"One year",
  "One year":"One year",
  "2 year":"Two year",
  "Two year":"Two year"
}

df["Contract"]=df["Contract"].str.strip().map(contract_mapping)

internet_mapping={
  "dsl":"DSL",
  "DSL":"DSL",
  "Fibre optic":"Fiber optic",
  "Fiber optic":"Fiber optic",
  "FiberOptic":"Fiber optic",
  "None":"No",
  "No":"No"
}

df["InternetService"]=df["InternetService"].str.strip().map(internet_mapping)


df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce")
df = df[df["tenure"]>0]

df=df[
  (df["MonthlyCharges"]>=10)&(df["MonthlyCharges"]<=200)]
  
df["MonthlyCharges"]=df["MonthlyCharges"].fillna(df["MonthlyCharges"].mean())
df["TotalCharges"]=df["TotalCharges"].fillna(df["TotalCharges"].mean())
df["tenure"]=df["tenure"].fillna(round(df["tenure"].median())).astype(int)

# // 2.Encode the target column

df["Churn"]=df["Churn"].map({
  "Yes":1,
  "No":0
})

#  3.Encode the categorical columns using get_dummies
categorical_columns =[
  "gender","PhoneService","InternetService","Contract","PaperlessBilling","PaymentMethod"
]

df=pd.get_dummies(df,columns=categorical_columns,drop_first=True)

#  4.Separate X,y
X= df.drop("Churn",axis=1)
y=df["Churn"]

#  5.Split into test and train

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)

#  6.Train Logistic Regression model

model=LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

#  7.Accuracy score

accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)

#  8.Classification report

print("\n Classifaction Report:\n")

print(classification_report(y_test,y_pred,target_names=["Stay","Churn"]))