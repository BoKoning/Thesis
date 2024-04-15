import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d


# Reading file
df = pd.read_csv('spx_eod_202301.csv')


# Creating extra variables, changing string to float
df[' [T]'] = df[' [DTE]']/365
df[' [C_MID]'] = (df[' [C_ASK]'] + df[' [C_BID]'])/2
df[' [MONEYNESS]'] = df[' [STRIKE]']/df[' [UNDERLYING_LAST]']
df[' [C_IV]'] = pd.to_numeric(df[' [C_IV]'], errors='coerce')
df[' [P_IV]'] = pd.to_numeric(df[' [C_IV]'], errors='coerce')
df[' [C_VOLUME]'] = pd.to_numeric(df[' [C_VOLUME]'], errors='coerce')
df[' [P_VOLUME]'] = pd.to_numeric(df[' [P_VOLUME]'], errors='coerce')


# Taking sample from dataset
df = df[(df[' [DTE]'] == 55) & (df[' [QUOTE_DATE]'] == ' 2023-01-04')]
df = df[(df[' [C_VOLUME]'] > 5)]
df = df.dropna(subset=[' [C_IV]'], how='all')
df = df[(df[' [C_IV]'] > 0.05) & (df[' [MONEYNESS]'] > 0.2) & (df[' [MONEYNESS]'] < 1.4)]


# Plotting IV against Strike. Shows line is not smooth for strikes lower than 3750
plt.plot(df[' [STRIKE]'], df[' [C_IV]'])
plt.xlabel('STRIKE')
plt.ylabel('IV')
plt.show()


# Interpolating IV curve
def model_func(x, a, b, c, d, e, f):
    return a * x**5 - b * x**4 + c * x**3 + d * x**2 + e * x + f


x_data = df[' [STRIKE]'].values
y_data = df[' [C_IV]'].values
valid_indices = np.logical_and(~np.isnan(x_data), ~np.isnan(y_data))
x_data = x_data[valid_indices]
y_data = y_data[valid_indices]
popt, pcov = curve_fit(model_func, x_data, y_data)
a_opt, b_opt, c_opt, d_opt, e_opt, f_opt = popt
y_fit = model_func(x_data, a_opt, b_opt, c_opt, d_opt, e_opt, f_opt)


# Plotting interpolation
plt.scatter(x_data, y_data, label='Original data')
plt.plot(x_data, y_fit, color='red', label='Fitted curve')
plt.xlabel('Strike')
plt.ylabel('Implied Volatility')
plt.title('Curve Fitting: Implied Volatility vs Strike')
plt.legend()
plt.show()


# Cumulative Risk Neutral Density Function (with interpolation)
r = 0.05
q = 0.003
T = df[' [T]'].iloc[0]
U_last = df[' [UNDERLYING_LAST]'].iloc[0]
strikes = np.linspace(2000, 6000, 4000)
C_IV_IP = model_func(strikes, a_opt, b_opt, c_opt, d_opt, e_opt, f_opt)


# For all strikes lower than minimum strike, C_IV_IP is constant, same for strikes higher than max strike
mini = df[' [STRIKE]'].min()
value_mini = model_func(mini, a_opt, b_opt, c_opt, d_opt, e_opt, f_opt)
C_IV_IP[strikes < mini] = value_mini
maxi = df[' [STRIKE]'].max()
value_maxi = model_func(maxi, a_opt, b_opt, c_opt, d_opt, e_opt, f_opt)
C_IV_IP[strikes > maxi] = value_maxi


# Plotting interpolation of C_IV
plt.plot(strikes, C_IV_IP)
plt.title('test test')
plt.show()


# Creating cdf function
top_d2 = np.log(U_last/strikes) + (r-((C_IV_IP**2)/2)) * T
bot_d2 = np.sqrt(T) * C_IV_IP
d2 = top_d2/bot_d2
cdf = 1-norm.cdf(d2)
# print(df[['spd', ' [STRIKE]', ' [D2]', ' [bot_D2]', ' [top_D2]']].head(50))


# Plotting CDF (interpolated)
plt.plot(strikes, cdf)
plt.xlabel('Strike')
plt.ylabel('CDF')
plt.title('Cumulative Density Function')
plt.show()


# Creating PDF by taking first order derivative of cdf against strike
pdf = np.gradient(cdf, strikes)


# Plotting PDF
plt.plot(strikes, pdf)
plt.xlabel('Strike')
plt.ylabel('PDF')
plt.title('Probability Density Function')
plt.show()


# State price multiplication factors
factor = 1.07
sp_mp = [1/factor**5, 1/factor**4, 1/factor**3, 1/factor**2, 1/factor, 1, factor, factor**2, factor**3, factor**4, factor**5]

# Getting the strikes list
sp_strikes = []
for i in sp_mp:
    addition = i * U_last
    sp_strikes.append(addition)

# Interpolation of pdf, getting pdf values
pdf_interp = interp1d(strikes, pdf, kind='linear')
pdf_values = pdf_interp(sp_strikes)

# Multiplying so that state prices add up to one
multiply_factor = 1/pdf_values.sum()
state_prices = []
for i in pdf_values:
    appending = i * multiply_factor
    state_prices.append(appending)

# Bar plot to show state prices
width = sp_strikes[1] - sp_strikes[0]
plt.clf()
plt.bar(sp_strikes, state_prices, 100)
plt.title('State prices')
plt.show()
