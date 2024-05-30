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

def power_law_regression(X_train, y_train, X_test):
    # Transformasi data menggunakan log
    X_train_power = np.log(X_train + 1)
    X_test_power = np.log(X_test + 1)
    y_train_power = np.log(y_train + 1)

    power_model = LinearRegression()
    power_model.fit(X_train_power, y_train_power)
    y_pred_power_log = power_model.predict(X_test_power)
    y_pred_power = np.exp(y_pred_power_log) - 1
    return y_pred_power

def calculate_rms(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

def visualize_results(y_test, y_pred_linear, y_pred_power):
    # Visualisasi hasil regresi linear
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred_linear, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Observed')
    plt.ylabel('Predicted')
    plt.title('Linear Regression')

    # Visualisasi hasil regresi pangkat
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred_power, color='red')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Observed')
    plt.ylabel('Predicted')
    plt.title('Power Law Regression')

    plt.tight_layout()
    plt.show()
