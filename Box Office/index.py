import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('boxoffice.csv', encoding = 'latin-1')
df.head()
df.shape
df.info()
df.describe().T
to_remove = ['world_revenue', 'opening_revenue']
df.drop(to_remove, axis=1, inplace = True)
df.isnull().sum()*100/df.shape[0]
df.drop('budget', axis=1, inplace=True)

for col in ['MPAA', 'genres']:
    df[col] = df[col].fillna(df[col].mode()[0])

df.dropna(inplace = True)
df.isnull().sum().sum()
df['domestic_revenue'] = df['domestic_revenue'].astype(str).str[1]

for col in ['domestic_revenue', 'opening_theaters', 'release_days']:
    df[col] = df[col].astype(str).str.replace(',','')

    temp = (~df[col].isnull())
    df[temp][col] = df[temp][col].convert_dtypes(float)

    df[col] = pd.to_numeric(df[col], errors = 'coerce')

plt.figure(figsize=(10,5))
sb.countplot(df['MPAA'])
plt.show()

df.groupby('MPAA')['domestic_revenue'].mean()

plt.subplots(figsize=(15,5))

features = ['domestic_revenue', 'opening_theaters', 'release_days']
for i, col in enumerate(features):
    plt.subplot(1, 3, i+1)
    sb.distplot(df[col])
plt.tight_layout()
plt.show()
plt.subplots(figsize=(15,5))
for i, col in enumerate(features):
    plt.subplot(1,3,i+1)
    sb.boxplot(df[col])
plt.tight_layout()
plt.show()

for col in features:
    df[col] = df[col].apply(lambda x:np.log10(x))

plt.subplots(figsize=(15,5))
for i, col in enumerate(features):
    plt.subplot(1,3, i+1)
    sb.displot(df[col])
plt.tight_layout()
plt.show()

vectorizer = CountVectorizer() 
vectorizer.fit(df['genres']) 
features = vectorizer.transform(df['genres']).toarray() 

genres = vectorizer.get_feature_names_out() 
for i, name in enumerate(genres): 
	df[name] = features[:, i] 

df.drop('genres', axis=1, inplace=True)

removed = 0

if 'action' in df.columns and 'western' in df.columns:
    for col in df.loc[:, 'action':'western'].columns: 
        
        if (df[col] == 0).mean() > 0.95: 
            removed += 1
            df.drop(col, axis=1, inplace=True) 

print(removed) 
print(df.shape)

for col in ['distributor', 'MPAA']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

plt.figure(figsize=(8, 8))
sb.heatmap(df.select_dtypes(include=np.number).corr() > 0.8, 
            annot=True, 
            cbar=False) 
plt.show()

features = df.drop(['title', 'domestic_revenue'], axis=1) 
target = df['domestic_revenue'].values 

X_train, X_val, Y_train, Y_val = train_test_split(features, target, 
									test_size=0.1, 
									random_state=22) 
X_train.shape, X_val.shape

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

from sklearn.metrics import mean_absolute_error as mae
model = XGBRegressor()
mask = ~np.isinf(Y_train)
X_train = X_train[mask]
Y_train = Y_train[mask]
Y_train = np.nan_to_num(Y_train, nan=0.0, posinf=0.0, neginf=0.0)
Y_val = np.nan_to_num(Y_val, nan=0.0, posinf=0.0, neginf=0.0)

print("NaNs in Y_train:", np.isnan(Y_train).sum())
print("Infs in Y_train:", np.isinf(Y_train).sum())
print("Max Y_train:", np.max(Y_train))
print("Min Y_train:", np.min(Y_train))
model.fit(X_train, Y_train)

train_preds = model.predict(X_train) 
print('Training Error : ', mae(Y_train, train_preds)) 

val_preds = model.predict(X_val) 
print('Validation Error : ', mae(Y_val, val_preds)) 
print()

# --- CODE TO GENERATE CORRELATION CSV FOR TABLEAU ---

# 1. Calculate the Correlation Matrix
# This is the exact matrix used to draw the heatmap
corr_matrix = df.select_dtypes(include=np.number).corr()

# 2. Convert the square matrix into a long (unpivoted) format
# reset_index() makes the old index (Feature 1) a column
corr_long = corr_matrix.reset_index()

# pandas melt() function unpivots the remaining columns into two:
# 'Feature 2' (variable) and 'Correlation Value' (value)
corr_long = corr_long.melt(
    id_vars='index',              # The column to keep as identifier (Feature 1)
    var_name='Feature 2 (Column)', # Name for the unpivoted column headers
    value_name='Correlation Value' # Name for the unpivoted values
)

# 3. Rename the index column to be more descriptive for Tableau
corr_long.rename(columns={'index': 'Feature 1 (Row)'}, inplace=True)

# 4. Save the new long format data to a CSV file
corr_long.to_csv('correlation_matrix.csv', index=False)
print("\nGenerated: correlation_matrix.csv for Tableau Heatmap")

# --- End of Tableau Data Export Code ---