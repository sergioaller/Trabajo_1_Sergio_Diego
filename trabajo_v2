import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn import preprocessing
from sklearn.decomposition import PCA

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

# Función para cargar el conjunto de datos
def load_data():
    df = pd.read_csv('breast-cancer.csv', na_values=["?"])
    #pasar la clase a número: B=0 , M=1
    encoder = preprocessing.LabelEncoder()
    df["diagnosis"] = encoder.fit_transform(df["diagnosis"])
    feature_df = df[['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',]]
    X_read = np.asarray(feature_df)
    y_read = np.asarray(df['diagnosis'])

    return X_read, y_read

# Función para preprocesar los datos (escala, balanceo y PCA)
def preprocess_data(X,y, metodo_balanceo,pca,escalado,semilla): 
    # Dividir los datos en entrenamiento y prueba, en cada iteracion se hara una division distinta (80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=semilla)
    
    # Balanceo de datos
    if metodo_balanceo != 'n':
        if metodo_balanceo == 'o':
            sm = SMOTE(random_state=1)
            X_train, y_train = sm.fit_resample(X_train, y_train)
        elif metodo_balanceo == 'u':
            sm = NearMiss()
            X_train, y_train = sm.fit_resample(X_train, y_train)

    #Normalización
    if escalado:
        scaler = preprocessing.StandardScaler()
        scaler.fit(X_train) # fit realiza los cálculos y los almacena
        X_train = scaler.transform(X_train) # aplica los cálculos sobre el conjunto de datos de entrada para escalarlos

    # Reducción de dimensionalidad con PCA
    if pca:
        pca_model = PCA(n_components=2)
        X_train = pca_model.fit_transform(X_train)
        X_test = pca_model.transform(X_test)

    return X_train, X_test, y_train


# Función principal
def main():
    # Cargar los datos
    X, y = load_data()

    # Preguntar al usuario si quiere hacer preprocesado
    preprocess = input("¿Quieres hacer un preprocesado de los datos? (s/n): ").lower()
    
    if preprocess == 's':
        # Preguntar si quiere balanceo de datos
        balanceo = input("¿Quieres hacer balanceo de datos? (s/n): ").lower()
        
        if balanceo == 's':
            # Preguntar si quiere hacer oversampling o undersampling
            metodo_balanceo = input("¿Quieres hacer oversampling o undersampling? (o/u): ").lower()

        else:
            metodo_balanceo = 'n'
        
        # Preguntar si quiere aplicar PCA
        aplicar_pca = input("¿Quieres aplicar reducción de dimensionalidad (PCA)? (s/n): ").lower() == 's'

        # Preguntar si quiere aplicar PCA
        aplicar_escalado = input("¿Quieres hacer un escalado de los datos? (s/n): ").lower() == 's'
        
        # Preprocesar los datos
        X_train, X_test, y_train = preprocess_data(X_train, X_test, y_train, metodo_balanceo, aplicar_pca, aplicar_escalado,semilla)
    
    # Definir los modelos (SVM y k-NN)
    models = {
        "SVM": SVC(kernel='linear'),
        "k-NN": KNeighborsClassifier(n_neighbors=5)
    }
    
    # Evaluar los modelos en los datos preprocesados o originales
    for model_name, model in models.items():
        print(f"\nEvaluando modelo: {model_name}")
        accuracy, accuracy_std, time_mean, time_std = evaluate_model(model, X_train, X_test, y_train, y_test)
        
        print(f"Exactitud media: {accuracy:.4f} ± {accuracy_std:.4f}")
        print(f"Tiempo de ejecución medio: {time_mean:.4f} ± {time_std:.4f}")

if __name__ == "__main__":
    main()
