import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def prepare_data():
    # Baca data dari CSV
    data = pd.read_csv('Student_Performance.csv')

    # Ambil kolom yang relevan
    data = data[['Hours Studied', 'Sample Question Papers Practiced', 'Performance Index']]
    data.columns = ['Durasi Waktu Belajar(TB)', 'Jumlah Latihan Soal(NL)', 'Nilai Ujian Siswa (NL)']
    data.to_csv('processed_student_data.csv', index=False)
    print(data.head())

def load_data():
    # Baca data yang sudah diproses
    data = pd.read_csv('processed_student_data.csv')
    return data

def linear_regression(X_train, y_train, X_test):
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred_linear = linear_model.predict(X_test)
    return y_pred_linear

def exponential_regression(X_train, y_train, X_test):
    # Transformasi data menggunakan log
    X_train_exp = X_train.copy()
    X_train_exp['Durasi Waktu Belajar(TB)'] = np.log(X_train_exp['Durasi Waktu Belajar(TB)'] + 1)
    X_train_exp['Jumlah Latihan Soal(NL)'] = np.log(X_train_exp['Jumlah Latihan Soal(NL)'] + 1)

    X_test_exp = X_test.copy()
    X_test_exp['Durasi Waktu Belajar(TB)'] = np.log(X_test_exp['Durasi Waktu Belajar(TB)'] + 1)
    X_test_exp['Jumlah Latihan Soal(NL)'] = np.log(X_test_exp['Jumlah Latihan Soal(NL)'] + 1)

    exp_model = LinearRegression()
    exp_model.fit(X_train_exp, y_train)
    y_pred_exp = exp_model.predict(X_test_exp)
    return y_pred_exp

def calculate_rms(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

def visualize_results(y_test, y_pred_linear, y_pred_exp):
    # Visualisasi hasil regresi linear
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred_linear, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Observed')
    plt.ylabel('Predicted')
    plt.title('Linear Regression')

    # Visualisasi hasil regresi eksponensial
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred_exp, color='red')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Observed')
    plt.ylabel('Predicted')
    plt.title('Exponential Regression')

    plt.tight_layout()
    plt.show()
