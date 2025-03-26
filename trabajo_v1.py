import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import time

# Función para cargar el conjunto de datos
def load_data():
    df = pd.read_csv('breast-cancer.csv', na_values=["?"])
    df = df.dropna(axis=0)  # Eliminar filas con valores nulos
    # Convertir las etiquetas 'M' y 'B' en 1 y 0
    df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})
    X = df[df.columns[1:-1]]  # Excluye 'ID' y 'diagnosis' para las características
    y = df['diagnosis']  # La columna objetivo es 'diagnosis'
    return X, y

# Función para preprocesar los datos (escala, balanceo y PCA)
def preprocess_data(X_train, X_test, y_train, balanceo=False, oversampling=False, undersampling=False, pca=False):
    # Escalado de datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    
    # Balanceo de datos
    if balanceo:
        if oversampling:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train_scaled, y_train)
        elif undersampling:
            rus = RandomUnderSampler(random_state=42)
            X_train, y_train = rus.fit_resample(X_train_scaled, y_train)

    # Reducción de dimensionalidad con PCA
    if pca:
        n_opt = print_variance_info(X_train_scaled, threshold=0.90)        
        pca_model = PCA(n_components=n_opt)
        X_train = pca_model.fit_transform(X_train)
        X_test = pca_model.transform(X_test)

    return X_train, X_test, y_train

# Función para entrenar y evaluar los modelos
def evaluate_model(model, X_train, X_test, y_train, y_test, n_runs=5):
    accuracies = []
    times = []
    
    for _ in range(n_runs):
        start_time = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        end_time = time.time()

        accuracies.append(accuracy_score(y_test, y_pred))
        times.append(end_time - start_time)
    
    return np.mean(accuracies), np.std(accuracies), np.mean(times), np.std(times)

def print_variance_info(X, threshold=0.90):
    pca = PCA()
    pca.fit(X)

    var_individual = pca.explained_variance_ratio_
    var_acumulada = np.cumsum(var_individual)

    print(f"{'Componente':>10} | {'Varianza Individual':>20} | {'Varianza Acumulada':>20}")
    print("-" * 60)
    for i, (var, var_acc) in enumerate(zip(var_individual, var_acumulada), 1):
        print(f"{i:10} | {var:20.6f} | {var_acc:20.6f}")

    # Buscar el número mínimo de componentes que alcanza el umbral
    n_optimo = np.argmax(var_acumulada >= threshold) + 1
    print(f"\n👉 Con {n_optimo} componentes se alcanza al menos el {threshold*100:.1f}% de varianza acumulada.")
    
    return n_optimo


# Función principal
def main():
    # Cargar los datos
    X, y = load_data()

    # Preguntar al usuario si quiere hacer preprocesado
    preprocess = input("¿Quieres hacer un preprocesado de los datos? (sí/no): ").lower()

    # Dividir los datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if preprocess == 'sí':
        # Preguntar si quiere balanceo de datos
        balanceo = input("¿Quieres hacer balanceo de datos? (sí/no): ").lower()
        
        if balanceo == 'sí':
            # Preguntar si quiere hacer oversampling o undersampling
            sampling_method = input("¿Quieres hacer oversampling o undersampling? (oversampling/undersampling): ").lower()
            oversampling = sampling_method == "oversampling"
            undersampling = sampling_method == "undersampling"
        else:
            oversampling = undersampling = False
        
        # Preguntar si quiere aplicar PCA
        apply_pca = input("¿Quieres aplicar reducción de dimensionalidad (PCA)? (sí/no): ").lower() == 'sí'
        
        # Preprocesar los datos
        X_train, X_test, y_train = preprocess_data(X_train, X_test, y_train, balanceo=True, oversampling=oversampling, undersampling=undersampling, pca=apply_pca)


    
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
