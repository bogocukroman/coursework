import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import skew, kurtosis

def load_and_prepare_dataset(file_path: str) -> pd.DataFrame:
    """Этап 1: Безопасная загрузка данных, валидация и временная индексация."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Критическая ошибка: файл {file_path} не обнаружен.")
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df

def generate_multichannel_plots(df: pd.DataFrame, output_path: str):
    """Этап 2: Многоканальная линейная визуализация с разметкой Train/Test Split."""
    fig, axes = plt.subplots(len(df.columns), 1, figsize=(11, 8.5), sharex=True)
    split_date = pd.to_datetime('2024-06-01')
    for idx, col in enumerate(df.columns):
        axes[idx].plot(df.index, df[col], color='#1a365d', linewidth=0.7)
        axes[idx].axvline(split_date, color='#e53e3e', linestyle='--', linewidth=1.2)
        axes[idx].set_title(f"Динамика показателя {col}", weight='bold', fontsize=9)
        axes[idx].grid(True, linestyle=':', alpha=0.6)
    plt.xlabel("Временная шкала (Годы)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=250)
    plt.close()

def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Этап 3: Расчет расширенных описательных статистик."""
    stats = df.describe().T
    stats['skewness'] = df.skew()
    return stats

def compute_3sigma_outliers(df: pd.DataFrame):
    """Этап 4: Численный расчет выбросов по правилу трех сигм."""
    results = {}
    for col in df.columns:
        mean_val = df[col].mean()
        std_val = df[col].std()
        outliers = ((df[col] < (mean_val - 3*std_val)) | (df[col] > (mean_val + 3*std_val))).sum()
        results[col] = outliers
    return results

def analyze_decomposition_and_noise(df: pd.DataFrame, col: str):
    """Этап 7: Аддитивное разложение, оценка отношения сигнал/шум (SNR) и эксцесса остатков."""
    series = df[col].asfreq('B').ffill()
    result = seasonal_decompose(series, model='additive', period=252)
    resid = result.resid.dropna()
    signal = result.trend.dropna() + result.seasonal.dropna()
    snr_db = 10 * np.log10(np.var(signal) / np.var(resid))
    kurt_val = kurtosis(resid, fisher=True)
    return snr_db, kurt_val, resid, result

if __name__ == "__main__":
    # Выполнение всех этапов анализа сквозным исполняемым конвейером
    file_name = "nvidia_daily_stock_prices.csv"
    data = load_and_prepare_dataset(file_name)

    # Генерация отчетных графиков
    generate_multichannel_plots(data, "plot1_multichannel.png")

    # Индивидуальные boxplots
    fig, axes = plt.subplots(1, 5, figsize=(11, 4))
    for idx, col in enumerate(data.columns):
        sns.boxplot(y=data[col], ax=axes[idx], color='#cbd5e1', width=0.4)
        axes[idx].set_title(col, weight='bold', fontsize=10)
        axes[idx].grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig("plot2_boxplots.png", dpi=250)
    plt.close()

    # Общий boxplot в log scale
    plt.figure(figsize=(10, 4.5))
    sns.boxplot(data=data, palette="Blues_d", width=0.5)
    plt.yscale('log')
    plt.grid(True, which="both", linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig("plot3_log_boxplot.png", dpi=250)
    plt.close()

    # Матрица корреляции
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(data.corr(), annot=True, fmt=".4f", cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.7)
    plt.tight_layout()
    plt.savefig("plot4_correlation.png", dpi=250)
    plt.close()

    # Декомпозиция и гистограмма шума
    snr, kurt, noise_res, decom_obj = analyze_decomposition_and_noise(data, 'Close')

    fig, axes = plt.subplots(4, 1, figsize=(11, 8.5), sharex=True)
    axes[0].plot(decom_obj.observed.index, decom_obj.observed, color='black', linewidth=0.8)
    axes[1].plot(decom_obj.trend.index, decom_obj.trend, color='#1a365d', linewidth=1.2)
    axes[2].plot(decom_obj.seasonal.index, decom_obj.seasonal, color='#2e7d32', linewidth=0.8)
    axes[3].plot(decom_obj.resid.index, decom_obj.resid, color='#c62828', linewidth=0.5)
    for ax in axes: ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("plot5_decomposition.png", dpi=250)
    plt.close()

    plt.figure(figsize=(9, 4.5))
    sns.histplot(noise_res, kde=True, color='#7c4dff', stat="density", bins=60, edgecolor='white', alpha=0.6)
    plt.tight_layout()
    plt.savefig("plot6_noise_density.png", dpi=250)
    plt.close()

    print("Автоматизированный цикл успешно завершен. Все 6 рисунков сохранены.")
