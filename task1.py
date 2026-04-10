# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv("train.csv")

# Show basic info
print(df.head())
print(df.info())
print(df.isnull().sum())

# -------------------------------
# 1. Handle Missing Values
# -------------------------------
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)

# -------------------------------
# 2. Encode Categorical Data
# -------------------------------
le = LabelEncoder()

df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

# -------------------------------
# 3. Feature Scaling
# -------------------------------
scaler = StandardScaler()

df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# -------------------------------
# 4. Outlier Detection
# -------------------------------
plt.figure(figsize=(10,5))
sns.boxplot(data=df[['Age','Fare']])
plt.show()

# Remove outliers using IQR
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1

df = df[(df['Fare'] >= Q1 - 1.5*IQR) & (df['Fare'] <= Q3 + 1.5*IQR)]

print("Cleaned Data Shape:", df.shape)

# Save cleaned data
df.to_csv("cleaned_data.csv", index=False)