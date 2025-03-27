import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def divide_dataset(df, features, label):
    #Devuelve X e y
    return df[features].values, df[label].values

def normalization(X):
    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(X)

def print_variance_info(X, threshold):
    #Calcula la varianza por componentes y determina el número de componentes para alcanzar un umbral del 90% (por defecto)
    pca = PCA()
    pca.fit(X)
    var_individual = pca.explained_variance_ratio_
    var_acumulada = np.cumsum(var_individual)
    print(f"{'Componente':>10} | {'Varianza Individual':>20} | {'Varianza Acumulada':>20}")
    print("-" * 60)
    for i, (var, var_acc) in enumerate(zip(var_individual, var_acumulada), 1):
        print(f"{i:10} | {var:20.6f} | {var_acc:20.6f}")
    n_optimo = np.argmax(var_acumulada >= threshold) + 1
    print(f"\n Con {n_optimo} componentes se alcanza al menos el {threshold*100:.1f}% de varianza acumulada.")
    return n_optimo

def apply_pca(X, threshold):
    #Aplica reducción de dimensionalidad al conjunto de datos y calcula la pérdida de información como error cuadrático medio
    n_opt = print_variance_info(X, threshold)
    pca = PCA(n_components=n_opt)
    projected = pca.fit_transform(X)
    reconstructed = pca.inverse_transform(projected)
    loss = ((X - reconstructed) ** 2).mean()
    return projected, loss

def apply_oversampling(X, y):
    sm = SMOTE()
    return sm.fit_resample(X, y)

def apply_undersampling(X, y):
    rus = RandomUnderSampler()
    return rus.fit_resample(X, y)

def save_plot(x, y, title, xlabel, ylabel, path):
    #Guarda un gráfico a partir de dos listas, x e y.
    plt.plot(x, y)
    plt.grid(True, alpha=0.3)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(path)
    plt.close()

def save_avg_plot(x, y_list, title, xlabel, ylabel, path):
    #Hace un gráfico promedio de varias curvas.
    avg = np.mean(y_list, axis=0)
    std = np.std(y_list, axis=0)
    plt.plot(x, avg)
    plt.fill_between(x, avg - std, avg + std, alpha=0.3)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.savefig(path)
    plt.close()

def save_confusion_matrix_from_cm(cm, path):
    #Recibe la matriz suma de las 5 iteraciones de una misma combinación de parámetros y la muestra por pantalla
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benigno', 'Maligno'])
    disp.plot(cmap="Blues", colorbar=False)
    plt.title("Matriz de Confusión")
    plt.savefig(path)
    plt.close()


def knn_experiment(X, y, iterations, max_k, string_sipca, output_dir,balance_type):
    #Realiza un experimento con kNN para probar distintos valores de k y elegir el que mejor score tiene.
    #  Saca resultados con ese mejor k.
    os.makedirs(output_dir, exist_ok=True)
    k_range = list(range(1, max_k+1))
    all_scores, all_train_times, all_predict_times = [], [], []

    print(f"\n[KNN] Ejecutando experimento: {string_sipca}")

    for i in range(iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=i)

        if balance_type == 'oversample':
            X_train, y_train = apply_oversampling(X_train, y_train)
        elif balance_type == 'undersample':
            X_train, y_train = apply_undersampling(X_train, y_train)

        scores, train_times, predict_times, preds_list = [], [], [], []
        for k in k_range:
            model = KNeighborsClassifier(n_neighbors=k)

            start = time.time()
            model.fit(X_train, y_train)
            train_times.append((time.time() - start) * 1000)

            start = time.time()
            preds = model.predict(X_test)
            predict_times.append((time.time() - start) * 1000)

            preds_list.append(preds)
            scores.append(model.score(X_test, y_test))

        all_scores.append(scores)
        all_train_times.append(train_times)
        all_predict_times.append(predict_times)

        best_k_idx = np.argmax(scores)
        save_plot(k_range, scores, "KNN Score", "k", "Porcentaje acierto", f"{output_dir}/knn_scores_{string_sipca}_{i}.jpg")

    avg_scores = np.mean(all_scores, axis=0)
    final_k = k_range[np.argmax(avg_scores)]

    print(f"[KNN] Mejor k promedio: {final_k:2d} | Score medio: {avg_scores[final_k - 1]:.4f}")

    save_avg_plot(k_range, all_scores, "KNN Average Score", "k", "Porcentaje acierto", f"{output_dir}/knn_avg_scores_{string_sipca}.jpg")

    scores, train_times, predict_times = [], [], []
    final_cm = np.zeros((2, 2), dtype=float)
    for i in range(iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=i)

        if balance_type == 'oversample':
            X_train, y_train = apply_oversampling(X_train, y_train)
        elif balance_type == 'undersample':
            X_train, y_train = apply_undersampling(X_train, y_train)

        model = KNeighborsClassifier(n_neighbors=final_k)

        start = time.time()
        model.fit(X_train, y_train)
        train_times.append((time.time() - start) * 1000)

        start = time.time()
        preds = model.predict(X_test)
        predict_times.append((time.time() - start) * 1000)

        scores.append(model.score(X_test, y_test))
        final_cm += confusion_matrix(y_test, preds) / iterations

    final_cm = np.round(final_cm).astype(int)
    save_confusion_matrix_from_cm(final_cm, f"{output_dir}/knn_final_cm_{string_sipca}.jpg")

    print(f"[KNN Final] Score medio:           {np.mean(scores):.4f}")
    print(f"            Tiempo entrenamiento: {np.mean(train_times):.2f} ms")
    print(f"            Tiempo predicción:    {np.mean(predict_times):.2f} ms\n")


def svm_experiment(X, y, iterations, param_grid, string_sipca, output_dir,balance_type):
    #Aplica SVM con diferentes configuraciones de kernel y parámetros de la función RBF.
    # Saca resultados de la mejor configuración.
    os.makedirs(output_dir, exist_ok=True)
    kernels = param_grid['kernels']
    c_values = param_grid['rbf_c']
    gamma_values = param_grid['rbf_gamma']

    kernel_scores = {}
    best_combo = (None, None)

    print(f"[SVM] Ejecutando búsqueda de parámetros óptimos: {string_sipca}")

    if 'linear' in kernels:
        scores = []
        for i in range(iterations):
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=i)

            if balance_type == 'oversample':
                X_train, y_train = apply_oversampling(X_train, y_train)
            elif balance_type == 'undersample':
                X_train, y_train = apply_undersampling(X_train, y_train)

            model = SVC(kernel='linear')
            model.fit(X_train, y_train)
            scores.append(model.score(X_test, y_test))

        kernel_scores['linear'] = np.mean(scores)

    if 'rbf' in kernels:
        best_scores_by_combo = {}
        for c in c_values:
            for gamma in gamma_values:
                combo_scores = []
                for i in range(iterations):
                    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=i)

                    if balance_type == 'oversample':
                        X_train, y_train = apply_oversampling(X_train, y_train)
                    elif balance_type == 'undersample':
                        X_train, y_train = apply_undersampling(X_train, y_train)

                    model = SVC(kernel='rbf', C=c, gamma=gamma)
                    model.fit(X_train, y_train)
                    combo_scores.append(model.score(X_test, y_test))
                best_scores_by_combo[(c, gamma)] = np.mean(combo_scores)

        sorted_combos = sorted(best_scores_by_combo.items(), key=lambda x: x[1], reverse=True)
        best_combo, best_rbf_score = sorted_combos[0]
        kernel_scores['rbf'] = best_rbf_score

        print("[SVM RBF] Top 3 combinaciones:")
        for (c, g), s in sorted_combos[:3]:
            print(f"           C={c:<6} gamma={g:<6} | Score medio: {s:.4f}")

    best_kernel = max(kernel_scores, key=kernel_scores.get)
    print(f"[SVM] Mejor kernel: {best_kernel} (Score medio: {kernel_scores[best_kernel]:.4f})")

    scores, train_times, predict_times = [], [], []
    final_cm = np.zeros((2, 2), dtype=float)
    for i in range(iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=i)

        if balance_type == 'oversample':
            X_train, y_train = apply_oversampling(X_train, y_train)
        elif balance_type == 'undersample':
            X_train, y_train = apply_undersampling(X_train, y_train)

        if best_kernel == 'linear':
            model = SVC(kernel='linear')
        else:
            c, gamma = best_combo
            model = SVC(kernel='rbf', C=c, gamma=gamma)

        start = time.time()
        model.fit(X_train, y_train)
        train_times.append((time.time() - start) * 1000)

        start = time.time()
        preds = model.predict(X_test)
        predict_times.append((time.time() - start) * 1000)

        scores.append(model.score(X_test, y_test))
        final_cm += confusion_matrix(y_test, preds) / iterations

    final_cm = np.round(final_cm).astype(int)
    save_confusion_matrix_from_cm(final_cm, f"{output_dir}/svm_final_cm_{string_sipca}.jpg")

    print(f"[SVM Final] Score medio:           {np.mean(scores):.4f}")
    print(f"             Tiempo entrenamiento: {np.mean(train_times):.2f} ms")
    print(f"             Tiempo predicción:    {np.mean(predict_times):.2f} ms\n")

def main():
    # Carga del dataset sin valores nulos
    df = pd.read_csv("breast-cancer.csv", na_values=["?"])
    df = df.dropna()

    # Codificación de la variable diagnosis (Benigno/Maligno -> 0/1)
    encoder = preprocessing.LabelEncoder()
    df['diagnosis'] = encoder.fit_transform(df['diagnosis'])

    features = df.columns[2:]  # Selecciona características (omite ID y diagnosis)
    label = 'diagnosis'

    X, y = divide_dataset(df, features, label)

    #Valores que servirán de umbral de varianza para aplicar PCA con los componentes correspondientes
    pca_thresholds = [None, 0.90, 0.95]

    # Se prueban todas las combinaciones de: balanceo {ninguno, oversampling, undersampling} y uso de PCA {sí, no}
    for balance_type in ['none', 'oversample', 'undersample']:
        for threshold in pca_thresholds:
            X_proc = normalization(X)
            y_proc = y.copy()

            balance_str = {
                'none': 'noBalanceo',
                'oversample': 'Oversampling',
                'undersample': 'Undersampling'
            }[balance_type]

            # Etiqueta PCA o noPCA según el umbral
            if threshold is None:
                pca_str = "noPCA"
            else:
                X_proc, loss = apply_pca(X_proc, threshold)
                pca_str = f"PCA{int(threshold * 100)}"
                print(f"PCA applied (umbral {int(threshold * 100)}%). Reconstruction loss: {loss:.6f}")
            output_dir = f"results/{balance_str}_{pca_str}"

            # Ejecuta KNN para este escenario
            knn_experiment(
                X_proc, y_proc,
                iterations=5,
                max_k=20,
                string_sipca=f"{balance_str}_{pca_str}",
                output_dir=output_dir, balance_type=balance_type
            )

            # Define los kernel y los parámetros para SVM con kernel RBF
            svm_param_grid = {
                'kernels': ['linear', 'rbf'],
                'rbf_c': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                'rbf_gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
            }

            # Ejecuta SVM para este escenario
            svm_experiment(
                X_proc, y_proc,
                iterations=5,
                param_grid=svm_param_grid,
                string_sipca=f"{balance_str}_{pca_str}",
                output_dir=output_dir, balance_type=balance_type
            )



if __name__ == "__main__":
    main()
