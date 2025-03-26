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
    return df[features].values, df[label].values

def normalization(X):
    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(X)

def print_variance_info(X, threshold=0.90):
    pca = PCA()
    pca.fit(X)
    var_individual = pca.explained_variance_ratio_
    var_acumulada = np.cumsum(var_individual)
    print(f"{'Componente':>10} | {'Varianza Individual':>20} | {'Varianza Acumulada':>20}")
    print("-" * 60)
    for i, (var, var_acc) in enumerate(zip(var_individual, var_acumulada), 1):
        print(f"{i:10} | {var:20.6f} | {var_acc:20.6f}")
    n_optimo = np.argmax(var_acumulada >= threshold) + 1
    print(f"\n✨ Con {n_optimo} componentes se alcanza al menos el {threshold*100:.1f}% de varianza acumulada.")
    return n_optimo

def apply_pca(X, threshold=0.90):
    n_opt = print_variance_info(X, threshold)
    pca = PCA(n_components=n_opt)
    projected = pca.fit_transform(X)
    reconstructed = pca.inverse_transform(projected)
    loss = ((X - reconstructed) ** 2).mean()
    return projected, loss

def apply_oversampling(X, y):
    sm = SMOTE(random_state=42)
    return sm.fit_resample(X, y)

def apply_undersampling(X, y):
    rus = RandomUnderSampler(random_state=42)
    return rus.fit_resample(X, y)

def save_plot(x, y, title, xlabel, ylabel, path):
    plt.plot(x, y)
    plt.grid(True, alpha=0.3)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(path)
    plt.close()

def save_avg_plot(x, y_list, title, xlabel, ylabel, path):
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

def save_confusion_matrix(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    labels = np.unique(np.concatenate((y_true, y_pred)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(l) for l in labels])
    disp.plot()
    plt.title("Matriz de Confusión")
    plt.savefig(path)
    plt.close()


def knn_experiment(X, y, iterations, max_k, final_k, string_sipca, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    k_range = list(range(1, max_k+1))
    all_scores, all_train_times, all_predict_times = [], [], []

    for i in range(iterations):
        print(f"[KNN] Iteración {i+1}/{iterations}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=i)

        scores, train_times, predict_times, preds_list = [], [], [], []
        for k in k_range:
            model = KNeighborsClassifier(n_neighbors=k)

            start = time.time()
            model.fit(X_train, y_train.ravel())
            train_times.append((time.time() - start) * 1000)

            start = time.time()
            preds = model.predict(X_test)
            predict_times.append((time.time() - start) * 1000)

            preds_list.append(preds)
            scores.append(model.score(X_test, y_test))

        all_scores.append(scores)
        all_train_times.append(train_times)
        all_predict_times.append(predict_times)

        best_k = np.argmax(scores) + 1
        save_plot(k_range, scores, "KNN Score", "k", "% acierto", f"{output_dir}/knn_scores_{string_sipca}_{i}.jpg")
        save_confusion_matrix(y_test, preds_list[best_k - 1], f"{output_dir}/knn_cm_{string_sipca}_{i}.jpg")

    save_avg_plot(k_range, all_scores, "KNN Average Score", "k", "Porcentaje acierto", f"{output_dir}/knn_avg_scores_{string_sipca}.jpg")

    scores, train_times, predict_times = [], [], []
    for i in range(iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=i)
        model = KNeighborsClassifier(n_neighbors=final_k)

        start = time.time()
        model.fit(X_train, y_train.ravel())
        train_times.append((time.time() - start) * 1000)

        start = time.time()
        preds = model.predict(X_test)
        predict_times.append((time.time() - start) * 1000)

        scores.append(model.score(X_test, y_test))
        save_confusion_matrix(y_test, preds, f"{output_dir}/knn_final_cm_{string_sipca}_{i}.jpg")

    print("[KNN] Resultados finales:")
    print("  Score medio:", np.mean(scores))
    print("  Tiempo medio entrenamiento:", np.mean(train_times), "ms")
    print("  Tiempo medio predicción:", np.mean(predict_times), "ms")

def svm_experiment(X, y, iterations, param_grid, final_params, string_sipca, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    kernels = param_grid['kernels']
    c_values = param_grid['rbf_c']
    gamma_values = param_grid['rbf_gamma']

    for kernel in kernels:
        if kernel == 'linear':
            scores, train_times, predict_times = [], [], []
            for i in range(iterations):
                print(f"[SVM Linear] Iteración {i+1}/{iterations}")
                X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=i)
                model = SVC(kernel='linear')

                start = time.time()
                model.fit(X_train, y_train.ravel())
                train_times.append((time.time() - start) * 1000)

                start = time.time()
                preds = model.predict(X_test)
                predict_times.append((time.time() - start) * 1000)

                scores.append(model.score(X_test, y_test))
                save_confusion_matrix(y_test, preds, f"{output_dir}/svm_linear_cm_{string_sipca}_{i}.jpg")

            print("[SVM Linear] Score medio:", np.mean(scores))
            print("  Tiempo medio entrenamiento:", np.mean(train_times), "ms")
            print("  Tiempo medio predicción:", np.mean(predict_times), "ms")

        elif kernel == 'rbf':
            best_scores_by_combo = {}
            for c in c_values:
                for gamma in gamma_values:
                    combo_scores = []
                    for i in range(iterations):
                        print(f"[SVM RBF] Iter {i+1}, C={c}, gamma={gamma}")
                        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=i)
                        model = SVC(kernel='rbf', C=c, gamma=gamma)
                        model.fit(X_train, y_train.ravel())
                        score = model.score(X_test, y_test)
                        combo_scores.append(score)
                    best_scores_by_combo[(c, gamma)] = np.mean(combo_scores)
            sorted_combos = sorted(best_scores_by_combo.items(), key=lambda x: x[1], reverse=True)
            print("[SVM RBF] Mejores combinaciones:", sorted_combos[:5])

    kernel = final_params['kernel']
    c = final_params['c']
    gamma = final_params['gamma']
    scores, train_times, predict_times = [], [], []
    for i in range(iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=i)
        if kernel == 'linear':
            model = SVC(kernel='linear')
        else:
            model = SVC(kernel='rbf', C=c, gamma=gamma)

        start = time.time()
        model.fit(X_train, y_train.ravel())
        train_times.append((time.time() - start) * 1000)

        start = time.time()
        preds = model.predict(X_test)
        predict_times.append((time.time() - start) * 1000)

        scores.append(model.score(X_test, y_test))
        save_confusion_matrix(y_test, preds, f"{output_dir}/svm_final_cm_{string_sipca}_{i}.jpg")

    print("[SVM Final] Score medio:", np.mean(scores))
    print("  Tiempo medio entrenamiento:", np.mean(train_times), "ms")
    print("  Tiempo medio predicción:", np.mean(predict_times), "ms")


def main():
    df = pd.read_csv("breast-cancer.csv", na_values=["?"])
    df = df.dropna()
    df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})
    features = df.columns[2:]
    label = 'diagnosis'

    X, y = divide_dataset(df, features, label)

    for balance_type in ['none', 'oversample', 'undersample']:
        for apply_PCA in [False, True]:
            X_proc = normalization(X)
            y_proc = y.copy()

            balance_str = {
                'none': 'noBalanceo',
                'oversample': 'Oversampling',
                'undersample': 'Undersampling'
            }[balance_type]

            if balance_type == 'oversample':
                X_proc, y_proc = apply_oversampling(X_proc, y_proc)
            elif balance_type == 'undersample':
                X_proc, y_proc = apply_undersampling(X_proc, y_proc)

            string_sipca = "PCA" if apply_PCA else "noPCA"
            output_dir = f"results/{balance_str}_{string_sipca}"

            if apply_PCA:
                X_proc, loss = apply_pca(X_proc, threshold=0.90)
                print(f"PCA applied. Reconstruction loss: {loss}")

            knn_experiment(X_proc, y_proc, iterations=5, max_k=20, final_k=5,
                           string_sipca=f"{balance_str}_{string_sipca}", output_dir=output_dir)

            svm_param_grid = {
                'kernels': ['linear', 'rbf'],
                'rbf_c': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                'rbf_gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
            }
            svm_final_params = {'kernel': 'linear', 'c': None, 'gamma': None}
            svm_experiment(X_proc, y_proc, iterations=5, param_grid=svm_param_grid,
                           final_params=svm_final_params,
                           string_sipca=f"{balance_str}_{string_sipca}", output_dir=output_dir)


if __name__ == "__main__":
    main()
