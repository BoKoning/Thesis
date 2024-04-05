import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.stats import norm


# Function for curve fitting:
def model_func(x, a, b, c, d, e, f):
    return a * x**5 - b * x**4 + c * x**3 + d * x**2 + e * x + f


# Data selection
df = pd.read_csv('spx_eod_202301.csv')
df[' [T]'] = df[' [DTE]']/365
df[' [C_MID]'] = (df[' [C_ASK]'] + df[' [C_BID]'])/2
df[' [MONEYNESS]'] = df[' [STRIKE]']/df[' [UNDERLYING_LAST]']
df[' [C_IV]'] = pd.to_numeric(df[' [C_IV]'], errors='coerce')
df[' [P_IV]'] = pd.to_numeric(df[' [C_IV]'], errors='coerce')
df[' [C_VOLUME]'] = pd.to_numeric(df[' [C_VOLUME]'], errors='coerce')
df[' [P_VOLUME]'] = pd.to_numeric(df[' [P_VOLUME]'], errors='coerce')
df = df[df[' [QUOTE_DATE]'] == ' 2023-01-04']
df = df[(df[' [C_VOLUME]'] > 2)]
df = df.dropna(subset=[' [C_IV]'], how='all')
df = df[(df[' [C_IV]'] > 0.05) & (df[' [MONEYNESS]'] > 0.2) & (df[' [MONEYNESS]'] < 2)]


# State prices function
def calculate_state_prices(DTE, Df):
    """Calculate state prices for a chosen DTE."""
    # Filter DataFrame based on DTE
    df_filtered = Df[Df[' [DTE]'] == DTE]

    # Interpolation of IV against strike
    x_data = df_filtered[' [STRIKE]'].values
    y_data = df_filtered[' [C_IV]'].values
    valid_indices = np.logical_and(~np.isnan(x_data), ~np.isnan(y_data))
    x_data = x_data[valid_indices]
    y_data = y_data[valid_indices]
    popt, pcov = curve_fit(model_func, x_data, y_data)
    a_opt, b_opt, c_opt, d_opt, e_opt, f_opt = popt

    # taking values for CDF formula
    r = 0.05
    T = df_filtered[' [T]'].iloc[0]
    u_last = df_filtered[' [UNDERLYING_LAST]'].iloc[0]
    strikes = np.linspace(2000, 6000, 4000)
    C_IV_IP = model_func(strikes, a_opt, b_opt, c_opt, d_opt, e_opt, f_opt)

    # Making IV constant for small and large strikes
    mini = df_filtered[' [STRIKE]'].min()
    value_mini = model_func(mini, a_opt, b_opt, c_opt, d_opt, e_opt, f_opt)
    C_IV_IP[strikes < mini] = value_mini
    maxi = df_filtered[' [STRIKE]'].max()
    value_maxi = model_func(maxi, a_opt, b_opt, c_opt, d_opt, e_opt, f_opt)
    C_IV_IP[strikes > maxi] = value_maxi

    # CDF & PDF function
    top_d2 = np.log(u_last/strikes) + (r-((C_IV_IP**2)/2)) * T
    bot_d2 = np.sqrt(T) * C_IV_IP
    d2 = top_d2/bot_d2
    cdf = 1-norm.cdf(d2)
    pdf = np.gradient(cdf, strikes)

    # Multiplication factors for strikes
    factor = 1.09
    sp_mp = [1/factor**5, 1/factor**4, 1/factor**3, 1/factor**2, 1/factor, 1, factor, factor**2, factor**3, factor**4, factor**5]

    # Getting strikes
    sp_strikes = [i * u_last for i in sp_mp]

    # Interpolating pdf to get its values
    pdf_interp = interp1d(strikes, pdf, kind='linear')
    pdf_values = pdf_interp(sp_strikes)

    # Multiplying so state prices add up to one. Creating State Price List
    multiply_factor = 1/pdf_values.sum()
    state_prices = [i * multiply_factor for i in pdf_values]

    return state_prices


# Using function to create state Price Matrix
def calculate_state_prices_matrix(DTE, Df):
    spm = []
    for DTE in DTE:
        state_prices = calculate_state_prices(DTE, Df)
        spm.append(state_prices)
    return np.vstack(spm)


# Testing the functions (exporting to csv)
DTEs = [33, 55, 85.96, 113.96, 134.96, 162.96, 197.96, 225.96, 253.96, 288.96, 317, 345]
state_prices_matrix = calculate_state_prices_matrix(DTEs, df)
state_prices_matrix_rounded = np.round(state_prices_matrix, 4)
df_state_prices = pd.DataFrame(state_prices_matrix_rounded.T, columns=DTEs)
df_state_prices.to_csv('state_prices_table_rounded.csv', index=False)
print(df_state_prices)
