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


# %%
