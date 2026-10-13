import pandas as pd

#  1.Load the dataset
df = pd.read_csv("/content/churnguard_data.csv")
# 2. shape of the dataset
print("Shape of the dataset:",df.shape)

#  3. first 5 rows
print("\n First 5 rows:",df.head())

#  4. print column names and data types
print("\n Column names and datatypes:\n",df.info())

#  5.count of missing values in each column
print("\n Count of missing values in each column:\n",df.isnull().sum())

#  6.number of duplicate rows

print("\n Number of duplicate rows \n",df.duplicated().sum())

#  7. value counts of the churn column
print("\n Value counts of the churn column:\n",df["Churn"].value_counts())

#  8.unique values in the contract column
print("\n Unique values in the Contract Column: \n",df["Contract"].unique())
