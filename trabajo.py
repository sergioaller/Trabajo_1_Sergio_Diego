import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn import preprocessing
from sklearn.decomposition import PCA

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

df=pd.read_csv('breast-cancer.csv', na_values=["?"])
# print(df.head())
# print(df.shape)
#print(df.isnull().sum()) #vemos que no falta ningún dato
#print(df['diagnosis'].value_counts()) #hay 357 B y 212 M , por lo que aplicaremos balanceo
#print(df.columns)

#pasar la clase a número: B=0 , M=1
encoder = preprocessing.LabelEncoder()
df["diagnosis"] = encoder.fit_transform(df["diagnosis"])

feature_df = df[['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']]
X_readed = np.asarray(feature_df)
#print(X_readed[0:5])
#print(feature_df.shape)
y_readed = np.asarray(df['diagnosis'])
#print(y_readed[0:5])
#for i in range(1,6):
i=1
X_train, X_test, y_train, y_test = train_test_split(X_readed, y_readed, random_state = i)

#BALANCEO
print (f'Conjunto entrenamiento original {i}', X_train.shape,  y_train.shape)
unique, counts = np.unique(y_train, return_counts=True)
#print(dict(zip(unique, counts)))

# Balanceo de datos: ejemplo de oversampling
sm_over = SMOTE(random_state=1)
X_train_over, y_train_over= sm_over.fit_resample(X_train, y_train)

# Balanceo de datos: ejemplo de undersampling
sm_under = NearMiss()
X_train_under, y_train_under= sm_under.fit_resample(X_train, y_train)


print(f'\nBalanceado con undersampling {i}:', X_train_under.shape,  y_train_under.shape)
unique_under, counts_under = np.unique(y_train_under, return_counts=True)
#print(dict(zip(unique_under, counts_under)))

print(f'\nBalanceado con oversampling {i}:', X_train_over.shape,  y_train_over.shape)
unique_over, counts_over = np.unique(y_train_over, return_counts=True)
#print(dict(zip(unique_over, counts_over)))

#normalización
scaler = preprocessing.StandardScaler()
scaler.fit(X_train_over) # fit realiza los cálculos y los almacena
scaler.fit(X_train_under) 

X_train_over = scaler.transform(X_train_over) # aplica los cálculos sobre el conjunto de datos de entrada para escalarlos
X_train_under = scaler.transform(X_train_under) 
print(X_train_over[0:5])
print(X_train_under[0:5])

