import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.optimize import minimize


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
    factor = 1.02
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


# Testing the functions
DTEs = [33, 55, 85.96, 113.96, 134.96, 162.96, 197.96, 225.96, 253.96, 288.96, 317, 345]
# DTEs = [44, 71.96, 106.96, 134.96, 162.96, 197.96, 225.96, 253.96, 288.96, 317, 345, 380]
state_prices_matrix_e = calculate_state_prices_matrix(DTEs, df)
state_prices_matrix = state_prices_matrix_e.T
state_prices_matrix_rounded = np.round(state_prices_matrix, 3)
df_state_prices = pd.DataFrame(state_prices_matrix_rounded, columns=DTEs)
print(df_state_prices)
dd = state_prices_matrix[:, 0].tolist()
fc = np.round(dd, 3)
print(fc[:12])


# Step one: Constructing P bar
def construct_p_bar(S):
    n = S.shape[0]  # The number of states
    P_bar = np.zeros((n, n))  # Initialize P_bar with zeros
    fc = S[:, 0].tolist()
    # Fill P_bar based on the pattern provided
    for i in range(n):
        if i < n//2:  # For rows above the middle
            P_bar[i, 0] = np.sum(fc[:6-i])  # Sum of elements first column
            P_bar[i, 1:6+i] = fc[6-i:11]  # Making next columns equal to S
            P_bar[i, 6+i:11] = 0  # zeros
        elif i > n//2:  # For rows below the middle
            P_bar[i, 10] = np.sum(fc[15-i:])  # Sum of last column
            P_bar[i, -5+i:10] = fc[0:15-i]  # Diagonal elements from S
            P_bar[i, 0:-5+i] = 0  # zeros
        else:  # For the middle row
            P_bar[i] = fc  # Copy all elements from the first column of S

    return P_bar


P_bar = construct_p_bar(state_prices_matrix)


# Finding Matrix P
def objective_function(P_flat, A, B, lambda_reg, P_bar_flat, n_states):
    P = P_flat.reshape((n_states, n_states))
    P_bar = P_bar_flat.reshape((n_states, n_states))
    # Regularization term penalizes the squared difference between P and P_bar
    regularization_term = lambda_reg * np.linalg.norm(P - P_bar, ord=2)**2
    fitting_term = np.linalg.norm(A @ P - B, 'fro') ** 2
    return fitting_term + regularization_term


def find_P(state_prices, P_bar, lambda_reg):
    n_states, n_periods = state_prices.shape
    A = state_prices[:, :-1].T
    B = state_prices[:, 1:].T

    # Flatten P_bar for the optimization
    P_bar_flat = P_bar.flatten()

    # Initial guess for P (you might already have a more informed guess)
    P_init = np.ones((n_states, n_states)) / n_states
    P_init_flat = P_init.flatten()

    # Define bounds for positive values in P
    bounds = [(0, None) for _ in range(n_states ** 2)]

    # Perform optimization
    result = minimize(objective_function, P_bar_flat, args=(A, B, lambda_reg, P_bar_flat, n_states), bounds=bounds, method='SLSQP')

    if result.success:
        P_opt = result.x.reshape((n_states, n_states))
        # Normalize rows to ensure they sum to 1
        P_opt = P_opt / P_opt.sum(axis=1, keepdims=True)
        return P_opt
    else:
        raise ValueError("Optimization failed")


# Example usage:
lambda_reg = 0.2  # Example regularization parameter
P_optimized = find_P(state_prices_matrix, P_bar, lambda_reg)
Po_round = np.round(P_optimized,2)
Po_df = pd.DataFrame(Po_round)
print(Po_df)


eigenvalues, eigenvectors = np.linalg.eig(P_optimized)
y = np.max(eigenvalues)
index_of_y = np.argmax(eigenvalues)
e = eigenvectors[:, index_of_y]
print(y)
print(e)

