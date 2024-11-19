#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
df = pd.read_csv("ai4i2020.csv")
# %%
df. head()
# %%
df.info()
# %%
df.describe(include='all')
# %%
df.isna().sum()
# %%
numerical_cols = df.select_dtypes(include=[np.number])
categorical_cols = df.select_dtypes(include=["object","category"])
print("Numerical ", numerical_cols.columns)
print("Categorical ", categorical_cols.columns)
# %%
cor_matrix = df.corr()
cor_matrix
# %%
plt.figure(figsize=(10, 8))
sns.heatmap(cor_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix")
plt.show()
#%%
# Plot histograms for all numerical columns
numerical_cols.hist(bins=20, figsize=(15, 12), layout=(4, 3))
plt.suptitle("Distribution of Numerical Features")
plt.show()

# %%
targets = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF', 'Machine failure']
for target in targets:
    sns.countplot(data=df, x=target)
    plt.title(f"Distribution of {target}")
    plt.show()
#%%
#Relationship between Features and target variables
for target in targets:
    for col in numerical_cols.columns:
        sns.boxplot(data=df, x=target, y=col)
        plt.title(f"{col} vs {target}")
        plt.show()

# %%
#### Feature Engineering
# Derived features
df['Temp_Diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
df['Power'] = df['Torque [Nm]'] * (df['Rotational speed [rpm]'] * 2 * np.pi / 60)

# %%
# Create a new column 'Failure' where it's 1 if any failure type is 1, otherwise 0
df['Failure'] = df[['TWF', 'HDF', 'PWF', 'OSF', 'RNF']].max(axis=1)


# %%
# Drop columns to create model_df
model_df = df.drop(['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)
# %%
df.columns
# %%
X = model_df.drop('Failure', axis=1)  # Features
y = model_df['Failure']               # Target variable
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Define the preprocessing for numerical and categorical features
numerical_features = X.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier

# Create the model pipeline with preprocessing and classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# %%
