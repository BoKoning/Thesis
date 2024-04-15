import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sympy as sp

# Importing data
df = pd.read_csv('spx_eod_202301.csv')

# Creating extra variables
df[' [C_MID]'] = (df[' [C_ASK]'] + df[' [C_BID]']) / 2
df[' [MONEYNESS]'] = df[' [STRIKE]'] / df[' [UNDERLYING_LAST]']

# Choosing date, and assuring call volume is larger than 1
df[' [C_VOLUME]'] = pd.to_numeric(df[' [C_VOLUME]'], errors='coerce')
df[' [C_IV]'] = pd.to_numeric(df[' [C_IV]'], errors='coerce')

fdf = df[(df[' [DTE]'] == 55) & (df[' [QUOTE_DATE]'] == ' 2023-01-04') & (df[' [C_VOLUME]'] > 0) & (df[' [STRIKE]'] > 1000) & (df[' [STRIKE]'] < 4700) & (df[' [C_IV]'] > 0.05)]

# Fitting call with respect to strike
x_data = fdf[' [STRIKE]'].values
y_data = fdf[' [C_MID]'].values
valid_indices = np.logical_and(~np.isnan(x_data), ~np.isnan(y_data))
x_data = x_data[valid_indices]
y_data = y_data[valid_indices]


def model_func(x, a, b, c, d, e, f):
    return a * x ** 5 - b * x ** 4 + c * x ** 3 + d * x**2 + e * x + f  # Example model

popt, pcov = curve_fit(model_func, x_data, y_data)

# Get parameters of the fitted curve
a_opt, b_opt, c_opt, d_opt, e_opt, f_opt = popt

# Generate fitted y values based on the model and optimal parameters
y_fit = model_func(x_data, a_opt, b_opt, c_opt, d_opt, e_opt, f_opt)

# Plot the original data points and the fitted curve
plt.scatter(x_data, y_data, label='Original data')
plt.plot(x_data, y_fit, color='red', label='Fitted curve')
plt.xlabel('Strike')
plt.ylabel('Call price')
plt.title('Interpolation: Call price vs strike')
plt.legend()
plt.show()

# Getting the second order derivative
x = sp.Symbol('x')
model_sym = a_opt * x ** 5 - b_opt * x ** 4 + c_opt * x ** 3 + d_opt * x**2 + e_opt * x + f_opt
first_derivative = sp.diff(model_sym, x)
second_derivative = sp.diff(first_derivative, x)

second_derivative_func = sp.lambdify(x, second_derivative, 'numpy')
y_second_derivative = second_derivative_func(x_data)
positive_indices = np.where(y_second_derivative > 0)[0]

# Plotting for positive values
if len(positive_indices) > 0:
    # Get the start and end indices where the second derivative is positive
    start_index = positive_indices[0]
    end_index = positive_indices[-1]
    max_index = np.argmax(y_second_derivative[start_index:end_index + 1]) + start_index
    plt.plot(x_data[start_index:end_index + 1], y_second_derivative[start_index:end_index + 1], color='blue',
             linestyle='--')
    max_y = np.max(y_second_derivative[start_index:end_index + 1])
    plt.axvline(x=x_data[max_index], ymin=0, ymax=max_y / y_second_derivative[max_index], color='red', linestyle='--',)
    # Annotate the x-value of the red line
    plt.text(x_data[max_index], y_second_derivative[max_index], f'  Highest point: {x_data[max_index]:.2f}', fontsize=10,
             va='bottom', ha='left')

    plt.xlabel('Strike')
    plt.ylabel('Second derivative')
    plt.title('Second Derivative: Call price vs Strike where second derivative is positive and below blue line')
    plt.legend()
    plt.show()
else:
    print("No positive regions for second derivative of Call with respect to Strike.")
