import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from analisis import prepare_data, load_data, linear_regression, exponential_regression, calculate_rms, visualize_results

def main():
    prepare_data()
    data = load_data()

    # Memisahkan fitur dan target
    X = data[['Durasi Waktu Belajar(TB)', 'Jumlah Latihan Soal(NL)']]
    y = data['Nilai Ujian Siswa (NL)']

    # Membagi data menjadi training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fungsi untuk menjalankan analisis berdasarkan metode yang dipilih
    def run_analysis(method):
        y_pred_linear = None
        y_pred_exp = None

        if method == 'linear' or method == 'keduanya':
            y_pred_linear = linear_regression(X_train, y_train, X_test)
            rms_linear = calculate_rms(y_test, y_pred_linear)
            print(f'RMS Error for Linear Model: {rms_linear}')

        if method == 'eksponensial' or method == 'keduanya':
            y_pred_exp = exponential_regression(X_train, y_train, X_test)
            rms_exp = calculate_rms(y_test, y_pred_exp)
            print(f'RMS Error for Exponential Model: {rms_exp}')

        if method == 'keduanya':
            visualize_results(y_test, y_pred_linear, y_pred_exp)
        elif method == 'linear':
            plt.figure(figsize=(6, 6))
            plt.scatter(y_test, y_pred_linear, color='blue')
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
            plt.xlabel('Observed')
            plt.ylabel('Predicted')
            plt.title('Linear Regression')
            plt.show()
        elif method == 'eksponensial':
            plt.figure(figsize=(6, 6))
            plt.scatter(y_test, y_pred_exp, color='red')
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
            plt.xlabel('Observed')
            plt.ylabel('Predicted')
            plt.title('Exponential Regression')
            plt.show()
        else:
            print("Metode tidak dikenal. Pilih 'linear', 'eksponensial', atau 'keduanya'.")

    # Loop untuk meminta input metode dari pengguna
    while True:
        method = input("Pilih metode regresi (linear/eksponensial/keduanya) atau ketik 'keluar' untuk berhenti: ").strip().lower()
        if method == 'keluar':
            print("Program dihentikan.")
            break
        elif method in ['linear', 'eksponensial', 'keduanya']:
            run_analysis(method)
        else:
            print("Metode tidak dikenal. Pilih 'linear', 'eksponensial', 'keduanya', atau 'keluar'.")

if __name__ == "__main__":
    main()
