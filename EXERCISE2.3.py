
import openmeteo_requests  # Open-Meteo API
import requests_cache      # 
import pandas as pd       # 
from retry_requests import retry  #
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from scipy import stats
import numpy as np
from scipy.optimize import curve_fit
from statsmodels.tsa.ar_model import AutoReg
from numba import njit


# 设置Open-Meteo API客户端，包含缓存和错误重试机制
# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)  # 创建缓存会话，有效期1小时
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)  # 设置重试机制，最多5次
openmeteo = openmeteo_requests.Client(session=retry_session)  # 创建API客户端


# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly
url = "https://historical-forecast-api.open-meteo.com/v1/forecast"  # API端点
params = {
    # 阿姆斯特丹区域（可以更改） / Amsterdam area (you can change)
    "latitude": 52.37,   # 纬度
    "longitude": 4.89,   # 经度
    "start_date": "2020-08-10",  # 开始日期
    "end_date": "2024-08-23",    # 结束日期
    "hourly": "temperature_2m",  # 每小时2米温度数据
    "daily": "temperature_2m_mean"  # 每日2米平均温度
}
# 发送API请求 / Send API request
responses = openmeteo.weather_api(url, params=params)

# 
# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]  # 获取第一个位置的响应


##########33
# Process hourly data. The order of variables needs to be the same as requested.
hourly = response . Hourly ()
hourly_temperature = hourly . Variables (0).ValuesAsNumpy()
hourly_data = { "date": pd.date_range(
    start = pd.to_datetime(hourly.Time(), unit = "s", utc = True), 
    end = pd.to_datetime(hourly.TimeEnd(),unit = "s", utc = True), 
    freq = pd.Timedelta(seconds = hourly.Interval()),
    inclusive = "left" 
)} 
# Process daily data. The order of variables needs to be the same as requested.

hourly_data[ "temperature_2m" ] = hourly_temperature 
hourly_dataframe = pd.DataFrame(data=hourly_data)  # Changed variable name
hourly_dataframe = hourly_dataframe.set_index(pd.to_datetime(hourly_dataframe['date'])).dropna()  # Changed to hourly


##############

# 处理每日数据 / Process daily data
daily = response.Daily()  # 从API响应中获取每日数据对象 / Get daily data object from API response

# 获取温度数据数组（第一个变量） / Get temperature data array (first variable)
daily_temperatures = daily.Variables(0).ValuesAsNumpy()

# 创建数据字典 / Create data dictionary
daily_data = {
    # 生成日期范围 / Generate date range
    "date": pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),  # 开始时间(UTC) / Start time (UTC)
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),  # 结束时间(UTC) / End time (UTC)
        freq=pd.Timedelta(seconds=daily.Interval()),  # 时间间隔 / Time interval
        inclusive="left"  # Include left boundary
    ),
    # Daily mean temperature (2m height)
    "temperature_2m_mean": daily_temperatures  
}

daily_dataframe = pd.DataFrame(data=daily_data)


daily_dataframe = daily_dataframe.set_index(pd.to_datetime(daily_dataframe['date']))
daily_dataframe['temperature_2m_mean'] = daily_dataframe['temperature_2m_mean'].interpolate(method='time', limit=3)
daily_dataframe = daily_dataframe.dropna()
###3.1
print(daily_dataframe.head())
###


# DEF

def plot_time_series(data, title, ylabel):
    """绘制时间序列图"""
    plt.figure(figsize=(12, 6))
    data.plot(title=title)
    plt.ylabel(ylabel)
    plt.xlabel('Date')
    plt.grid(True)
    plt.show()

def plot_rolling_stats(data, window=30):
    """绘制滚动统计量"""
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2,1,1)
    data.rolling(window=window).mean().plot(
        title=f'{window}-Day Rolling Mean')
    plt.ylabel('Temperature (°C)')
    plt.grid(True)
    
    plt.subplot(2,1,2)
    data.rolling(window=window).var().plot(
        title=f'{window}-Day Rolling Variance')
    plt.ylabel('Variance (°C²)')
    plt.xlabel('Date')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def decompose_time_series(data, period=365):
    """时间序列分解"""
    decomposition = seasonal_decompose(data, model='additive', period=period)
    
    # 可视化
    decomposition.plot()
    plt.suptitle('Temperature Time Series Decomposition', y=0.95)
    plt.tight_layout()
    plt.show()
    
    return decomposition.trend, decomposition.seasonal, decomposition.resid

def analyze_residuals(residuals):
    """残差分析"""
    # 分布可视化
    plt.figure(figsize=(8, 5))
    
    plt.subplot(1,2,1)
    residuals.dropna().hist(bins=50)#histagram 
    plt.title('Residual Distribution')
    plt.xlabel('Residuals (°C)')
    plt.ylabel('Frequency')
    
    plt.subplot(1,2,2)
    stats.probplot(residuals.dropna(), plot=plt)
    plt.title('Q-Q Plot of Residuals')#与正态分布重合程度
    
    plt.tight_layout()
    plt.show()
    
    # 自相关分析fig
    fig,(ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
    plot_acf(residuals.dropna(), lags=20, ax=ax1)
    ax1.set_ylim([-1.2, 1.5])  # ACF纵坐标范围
    ax1.set_title('ACF')
    plot_pacf(residuals.dropna(), lags=20, ax=ax2)
    ax2.set_ylim([-1.2, 1.5])  # PACF纵坐标范围
    ax2.set_title('PACF')
    plt.suptitle('Residuals Autocorrelation Analysis')#, y=0.95)  # y参数控制标题垂直位置
    # 调整坐标轴范围（如果需要更突出纵坐标）
    plt.tight_layout()
    plt.show()
    
    # 平稳性检验
    print("\nAugmented Dickey-Fuller Test for Residuals:")
    adf_result = adfuller(residuals.dropna())
    print(f'ADF Statistic: {adf_result[0]:.4f}')
    print(f'p-value: {adf_result[1]:.4f}')
    print('Critical Values:')
    for key, value in adf_result[4].items():
        print(f'   {key}: {value:.4f}')
    '''
    得到结果
    ADF Statistic: -7.9355 下于一下三个阀值所以拒绝原假设
    p-value: 0.0000 说明平稳 p<0时候拒绝原假设(不平稳)
    1%: -3.4378
   5%: -2.8648
   10%: -2.5685
   我们对残差进行了 ADF 单位根检验以判断其是否平稳。检验统计量为 -7.94
   显著小于 1%、5% 和 10% 水平的临界值。同时 p 值为 0
   远小于 0.05,有力拒绝原假设（序列不平稳）。因此我们可以判断 残差序列是平稳的。
    '''
    if adf_result[1] < 0.05:
        print("Conclusion: Residuals are STATIONARY")
    else:
        print("Conclusion: Residuals are NON-STATIONARY")

#主执行流程3.2

def main_analysis(daily_dataframe):
    """主执行函数"""
    print("\n" + "="*50)
    print("Part B: Constructing Stochastic Temperature Model")
    print("="*50)
    
    # Task 3.2.1: EDA
    print("\n=== Task 1: Exploratory Data Analysis ===")
    temp_series = daily_dataframe['temperature_2m_mean']
    
    plot_time_series(temp_series, 
                   'Amsterdam Daily Temperature (2m) - Raw Data',
                   'Temperature (°C)')
    
    plot_rolling_stats(temp_series)
    
    trend, seasonal, residuals = decompose_time_series(temp_series)
    
    # Task3.2. 2: 残差分析
    print("\n=== Task 2: Residual Analysis ===")
    analyze_residuals(residuals)
    
    # 将结果保存回DataFrame
    daily_dataframe['trend'] = trend
    daily_dataframe['seasonal'] = seasonal
    daily_dataframe['residuals'] = residuals
    
    return daily_dataframe





# 3.3.1
def fit_seasonal_model(daily_dataframe):
    """ 拟合 μ̄(t) = α + βt + γsin(ωt + φ) 模型 """
    # 转换为线性形式：α + βt + Asin(ωt) + Bcos(ωt)
    def linearized_model(t, alpha, beta, A, B):
        omega = 2*np.pi/365.25
        return alpha + beta*t + A*np.sin(omega*t) + B*np.cos(omega*t)
    
    # 准备数据
    t = (daily_dataframe.index - daily_dataframe.index[0]).days.values
    T = daily_dataframe['temperature_2m_mean'].values
    
    # 最小二乘拟合 
    params, _ = curve_fit(linearized_model, t, T, maxfev=10000)
    alpha, beta, A, B = params
    
    # 转换为原始参数形式
    gamma = np.sqrt(A**2 + B**2)
    phi = np.arctan2(B, A) - np.pi  # 正确处理象限

    # 计算完整季节模型
    omega = 2*np.pi/365.25
    daily_dataframe['mu_bar'] = alpha + beta*t + A*np.sin(omega*t) + B*np.cos(omega*t)# gamma*np.sin(omega*t + phi)
    
    print("\n=== Seasonal Model Parameters ===")
    print(f"α (baseline): {alpha:.4f} °C")
    print(f"β (trend): {beta:.6f} °C/day")
    print(f"γ (amplitude): {gamma:.4f} °C")
    print(f"φ (phase): {phi:.4f} rad")
    
    return alpha, beta, gamma, phi

# 3.3.2估计均值回归参数κ 
def estimate_kappa(daily_dataframe, max_lag=2):
    """ 使用残差 residuals 拟合 AR 模型，并估计均值回复速度 κ """
    # 1. residuals
    residuals = daily_dataframe['temperature_2m_mean'] - daily_dataframe['mu_bar']
    residuals = residuals.dropna()

    # 2. ADF
    adf_result = adfuller(residuals)
    print("\n=== ADF Test on Residuals ===")
    print(f"ADF statistic: {adf_result[0]:.4f}")
    print(f"p-value: {adf_result[1]:.4f}")

    if adf_result[1] > 0.05:
        print("Residuals are non-stationary. Applying first-order differencing.")
        residuals = residuals.diff().dropna()
        stationary = False
    else:
        print("Residuals are stationary.")
        stationary = True

    # 3.
    aic_values = []
    models = []
    for p in range(1, max_lag + 1):
        model = AutoReg(residuals, lags=p, old_names=False).fit()
        models.append(model)
        aic_values.append(model.aic)

    optimal_p = np.argmin(aic_values) + 1
    best_model = models[optimal_p - 1]

    print(f"\n=== AR Model Fit ===")
    print(f"Optimal AR order by AIC: {optimal_p}")
    print(best_model.summary())

    # 4.  κ，
    if stationary and optimal_p >= 1:
        phi_1 = best_model.params[1]  # AR(1)
        kappa = 1 - phi_1
        half_life = np.log(2) / kappa if kappa > 0 else np.nan
        print(f"\nEstimated κ: {kappa:.4f}")
        print(f"Half-life: {half_life:.2f} days")
    else:
        kappa = None
        print("\nκ cannot be estimated: residuals are not stationary.")

    return kappa


#  output def for 3.3
def part_c_analysis(daily_dataframe):
    # 1.  μ̄(t)
    alpha, beta, gamma, phi = fit_seasonal_model(daily_dataframe)
    
    # 2. simulate  κ
    kappa = estimate_kappa(daily_dataframe)
    
    # 3. visualize
    plt.figure(figsize=(12,6))
    daily_dataframe['temperature_2m_mean'].plot(label='Observed', alpha=0.7)
    daily_dataframe['mu_bar'].plot(label='μ̄(t) model', color='red')
    plt.title('Temperature vs Seasonal Model μ̄(t)')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.show()

    return {
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'phi': phi,
        'kappa': kappa,
        'dataframe': daily_dataframe
    }
###############33
# 3.4

def price_weather_derivatives(daily_dataframe, M=500, n=365, r=0.05, K=2500, alpha=1, C=100, F=100, K1=2400, K2=2700, beta=1):

    # 从dataframe中获取mu_bar和计算残差
    mu_bar = daily_dataframe['mu_bar']  # 从Part C得到的趋势与季节性模型
    residuals = daily_dataframe['temperature_2m_mean'] - mu_bar  # 计算残差
    
    # AR(2)模型的拟合函数
    def fit_ar2_model(residuals):
        model = AutoReg(residuals, lags=2)  # 使用滞后2阶的AR模型
        model_fitted = model.fit()
        return model_fitted.params, model_fitted.sigma2  # 返回AR模型的参数和方差
    
    # AR(2)
    ar_params, ar_sigma2 = fit_ar2_model(residuals)
    mu_bar_mean = np.mean(mu_bar)

    # 蒙特卡洛模拟算法（模拟残差项）
    def simulate_temperature_paths(mu_bar, ar_params, ar_sigma2, M, n):
        dt = 1  # 假设时间步长为1天
        paths = np.zeros((M, n))
        
        for i in range(M):
            # 初始化温度路径
            temperature_path = np.zeros(n)
            temperature_path[0] = mu_bar_mean  # 初始温度为均值
            
            for t in range(1, n):
                # 根据AR模型的残差更新温度路径
                epsilon = np.random.normal(0, np.sqrt(ar_sigma2))  # 随机生成残差
                temperature_path[t] = mu_bar[t] + ar_params[1] * (temperature_path[t-1] - mu_bar[t-1]) + ar_params[2] * (temperature_path[t-2] - mu_bar[t-2]) + epsilon
            
            paths[i, :] = temperature_path

        return paths

    # 生成模拟的温度路径
    simulated_paths = simulate_temperature_paths(mu_bar, ar_params, ar_sigma2, M, n)

    # 绘制模拟的温度路径
    plt.figure(figsize=(10, 6))
    plt.plot(simulated_paths.T, color='lightblue', alpha=0.1)
    plt.title("Simulated Temperature Paths")
    plt.xlabel("Days")
    plt.ylabel("Temperature (°C)")
    plt.show()

    # 定义期权的支付函数
    def call_option_payoff(DD, K, alpha, C):
        return np.minimum(alpha * np.maximum(DD - K, 0), C)

    def put_option_payoff(DD, K, alpha, F):
        return np.minimum(alpha * np.maximum(K - DD, 0), F)

    def collar_option_payoff(DD, K1, K2, alpha, beta, C, F):
        return np.minimum(alpha * np.maximum(DD - K1, 0), C) - np.minimum(beta * np.maximum(K2 - DD, 0), F)
    def option_price(paths, option_type, r, n, alpha, C, K, F, K1, K2, beta):
        DD = np.sum(np.maximum(18 - paths, 0), axis=1)  # Heating Degree Days (HDD)
        
        if option_type == "call":
            payoffs = call_option_payoff(DD, K, alpha, C)
        elif option_type == "put":
            payoffs = put_option_payoff(DD, K, alpha, F)
        elif option_type == "collar":
            payoffs = collar_option_payoff(DD, K1, K2, alpha, beta, C, F)
        else:
            raise ValueError("Unknown option type. Use 'call', 'put', or 'collar'.")

        discount_factor = np.exp(-r *(n/365))
        ####
        DD = np.sum(np.maximum(18 - simulated_paths, 0), axis=1)
        plt.hist(DD, bins=50)
        # plt.axvline(K, color='red', linestyle='--', label='K')
        # plt.axvline(K1, color='green', linestyle='--', label='K1')
        # plt.axvline(K2, color='blue', linestyle='--', label='K2')
        plt.title("Simulated HDD Distribution")
        plt.xlabel("HDD")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()
        ##33 check DD
        return np.mean(payoffs) * discount_factor
    call_price = option_price(simulated_paths, option_type="call", r=r, n=n,
                            alpha=alpha, C=C, K=K, F=0, K1=0, K2=0, beta=0)
    put_price = option_price(simulated_paths, option_type="put", r=r, n=n,
                            alpha=alpha, C=0, K=K, F=F, K1=0, K2=0, beta=0)
    collar_price = option_price(simulated_paths, option_type="collar", r=r, n=n,
                                alpha=alpha, C=C, K=0, F=F, K1=K1, K2=K2, beta=beta)

    return call_price, put_price, collar_price




#3.3
if __name__ == '__main__':
    # 
    result = part_c_analysis(daily_dataframe)
    
    # 
    print("\nFinal Parameters:")
    print(f"Seasonal model: T(t) = {result['alpha']:.2f} + {result['beta']:.2e}t + {result['gamma']:.2f}*sin(2πt/365.25 + {result['phi']:.2f})")
    print(f"Mean-reversion speed κ: {result['kappa']:.4f}")

'''
choose AR2 by checking the PCAF gragh the lag declay after 2-3

=== Seasonal Model Parameters ===
α (baseline): 10.6639 °C
β (trend): 0.001524 °C/day
γ (amplitude): 6.9630 °C
φ (phase): -3.7635 rad

=== ADF Test on Residuals ===
ADF statistic: -10.5188
p-value: 0.0000
Residuals are stationary.
Estimated κ: 0.0499
Half-life: 13.88 days

Final Parameters:
Seasonal model: T(t) = 10.66 + 1.52e-03t + 6.96*sin(2πt/365.25 + -3.76)
Mean-reversion speed κ: 0.0499

'''

#3.4
# 示例调用

call_price, put_price, collar_price = price_weather_derivatives(daily_dataframe, M=500, n=365, r=0.05, K=2500, alpha=1, C=100, F=100, K1=2400, K2=2700, beta=1)
print(f"Call Option Price: {call_price}")
print(f"Put Option Price: {put_price}")
print(f"Collar Option Price: {collar_price}")
'''
Call Option Price: 68.62652845897105
Put Option Price: 7.815038676819526
Collar Option Price: 32.93413756869031'''