import pandas as pd
from sklearn.linear_model import LogisticRegression

# 1. Load and clean the dataset
df = pd.read_csv("/content/churnguard_data.csv")

#  Drop customerID column
df = df.drop("customerID",axis=1)

#  Remove duplicate rows
df = df.drop_duplicates()

#  Strip whitespace from gender and PaymentMethod
df["gender"]=df["gender"].str.strip()
df["PaymentMethod"]=df["PaymentMethod"].str.strip()

#  Standardise casing
df["Churn"]=df["Churn"].str.strip().str.title()
df["PhoneService"]=df["PhoneService"].str.strip().str.title()
df["PaperlessBilling"]= df["PaperlessBilling"].str.strip().str.title()

#   Fix Contract Variations
df["Contract"] = df["Contract"].str.strip().str.lower()

contract_mapping = {
    "monthly": "Month-to-month",
    "month to month": "Month-to-month",
    "month-to-month": "Month-to-month",
    "1 year": "One year",
    "one year": "One year",
    "2 year": "Two year",
    "two year": "Two year"
}

df["Contract"]=df["Contract"].str.strip().map(contract_mapping)

#  Fix InternetService variations

internet_mapping = {
  "dsl":"DSL",
  "DSL":"DSL",
  "Fibre optic":"Fiber optic",
  "Fiber optic":"Fiber optic",
  "FiberOptic":"Fiber optic",
  "None":"No",
  "No":"No"
  }
  
df["InternetService"]=df["InternetService"].str.strip().map(internet_mapping)
  
#     Convert TotalCharges to numeric
  
df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce")
  
#    Remove rows where tenure is zero or negative
df=df[df["tenure"]>0]
  
#    Remove rows where MonthlyCharges is less than 10 or greater than 200
df=df[(df["MonthlyCharges"] >= 10) & (df["MonthlyCharges"]<=200)]
  
#    Fill missing values
df["MonthlyCharges"]=df["MonthlyCharges"].fillna(df["MonthlyCharges"].mean())
df["TotalCharges"]=df["TotalCharges"].fillna(df["TotalCharges"].mean())
df["tenure"]=df["tenure"].fillna(round(df["tenure"].median())).astype(int)

# 2.Encode 

df["Churn"]=df["Churn"].map({"Yes":1,"No":0})

contract_numeric={
    "Month-to-month":0,
    "One year":1,
    "Two year":2
}
df["Contract"]=df["Contract"].map(contract_numeric)

features=["tenure","MonthlyCharges","TotalCharges","Contract","SeniorCitizen"]
X=df[features]
y=df["Churn"]


# 3.Train model
model=LogisticRegression(max_iter=1000)
model.fit(X,y)

# 4.get user input

tenure=int(input("Enter tenure(months): "))
monthly_charges=float(input("Enter Monthly Charges: "))
total_charges=float(input("Enter Total Charges:"))
senior_citizen=int(input("Senior Citizen? (1=Yes,0=No):"))
contract=int(input("Contract type(0= Month-to-month,1= One year,2=Two year):"))

new_customer=pd.DataFrame({
    "tenure":[tenure],
    "MonthlyCharges":[monthly_charges],
    "TotalCharges":[total_charges],
    "Contract":[contract],
    "SeniorCitizen":[senior_citizen]
})
#5. predict churn

prediction = model.predict(new_customer)[0]
# 6.print result

if prediction == 1:
  print("Prediction: This Customer is likely to CHURN.")
else:
  print("Prediction:This customer is likely to STAY.")