import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.optimize import minimize
import scipy.stats as stats
import statsmodels.api as sm
from scipy.integrate import quad


# Function for data selection
def data_selection(file):
    # Importing file
    df_select = pd.read_csv(file)

    # All to numeric
    df_select[' [DTE]'] = pd.to_numeric(df_select[' [DTE]'], errors='coerce')
    df_select[' [C_IV]'] = pd.to_numeric(df_select[' [C_IV]'], errors='coerce')
    df_select[' [C_VOLUME]'] = pd.to_numeric(df_select[' [C_VOLUME]'], errors='coerce')
    df_select[' [UNDERLYING_LAST]'] = pd.to_numeric(df_select[' [UNDERLYING_LAST]'], errors='coerce')
    df_select[' [C_ASK]'] = pd.to_numeric(df_select[' [C_ASK]'], errors='coerce')
    df_select[' [C_BID]'] = pd.to_numeric(df_select[' [C_BID]'], errors='coerce')

    # Creating variable T (time to maturity in years)
    df_select[' [T]'] = df_select[' [DTE]'] / 365

    # Creating variable C_MID (average of bid and ask quotes)
    df_select[' [C_MID]'] = (df_select[' [C_ASK]'] + df_select[' [C_BID]']) / 2

    # Dropping variables for which C_VOLUME is 0/non-existent and for C_IV is lower than 0.05
    df_select = df_select[(df_select[' [C_VOLUME]'] > 0)]
    df_select = df_select.dropna(subset=[' [C_IV]'], how='all')

    # Returning filtered df
    return df_select


# Function for factors
def get_factor_list():
    # Factors for the 11 different states of the world, used later on for matrix S
    # list = [1 / mult ** 5, 1 / mult ** 4, 1 / mult ** 3, 1 / mult ** 2, 1 / mult, 1, mult, mult ** 2, mult ** 3, mult ** 4, mult ** 5]
    multipliers = [0.64, 0.70, 0.76, 0.84, 0.91, 1, 1.09, 1.2, 1.31, 1.43, 1.57]
    return multipliers


# Creating list of 12 DTEs, first one being around 30, other ones about +30 each
def get_dte(date, Df):
    num_dtes = 12
    start_dte = 30
    increment = 30
    dataframe = Df[Df[' [QUOTE_DATE]'] == date]
    list = dataframe[' [DTE]'].drop_duplicates().tolist()
    selected_dtes = []
    current_dte = start_dte
    for _ in range(num_dtes):
        closest_dte = min(list, key=lambda x: abs(x - current_dte))
        selected_dtes.append(closest_dte)
        current_dte += increment
    return selected_dtes


# Function for curve fitting:
def model_func(x, a, b, c, d):
    # Function used for interpolating the IV curve later on
    return a * x**3 - b * x**2 + c * x + d


# Finding dividend yield for given date
def get_yield(date, yield_file):
    # Read the CSV file with additional error-handling parameters
    yields_df = pd.read_csv(yield_file, encoding='utf-8', skip_blank_lines=True, engine='python')
    yields_df['Year'] = yields_df['Year'].astype(int)

    # Extract the year from the date
    year = pd.to_datetime(date.strip()).year

    # Lookup the yield for the specified year
    yield_data = yields_df.loc[yields_df['Year'] == year, 'Yield']

    # Check if a yield is available for the given year
    if yield_data.empty:
        raise ValueError(f"No yield data available for the year: {year}")

    # Return the yield value (first entry since there should only be one per year)
    return yield_data.values[0]


# Finding risk-free rate given certain date and DTE
def get_interest_rate(date, dte, rates_csv_path):
    # Import CSV file into a DataFrame
    rates_df = pd.read_csv(rates_csv_path)

    # Parse 'Date' column and format as needed
    rates_df['Date'] = pd.to_datetime(rates_df['Date'], format='%m/%d/%Y').dt.strftime('%Y/%m/%d')

    # Format the provided date for comparison
    format_date = pd.to_datetime(date.strip(), format='%Y-%m-%d').strftime('%Y/%m/%d')

    # Locate data for the specific date
    day_data = rates_df.loc[rates_df['Date'] == format_date]

    # Check if data for the specific date exists
    if day_data.empty:
        raise ValueError(f"No data available for the specified date: {format_date}")

    # Extract the available rates
    try:
        rates = {
            28: day_data['4 WEEKS BANK DISCOUNT'].values[0],
            91: day_data['13 WEEKS BANK DISCOUNT'].values[0],
            182: day_data['26 WEEKS BANK DISCOUNT'].values[0],
            364: day_data['52 WEEKS BANK DISCOUNT'].values[0]
        }
    except KeyError as e:
        raise ValueError(f"Missing expected column: {e}")

    # If DTE is greater than 364, return the 52-week rate
    if dte > 364:
        return rates[364]

    # Perform interpolation
    days = sorted(rates.keys())
    discount_rates = [rates[d] for d in days]
    interpolated_rate = np.interp(dte, days, discount_rates)

    if np.isnan(interpolated_rate) or interpolated_rate < 0:
        raise ValueError(f"Invalid interpolated interest rate: {interpolated_rate}")

    return interpolated_rate


# State prices function
def calculate_state_prices(dte, Df, date, rates_csv_path, yield_file):
    """Calculate state prices for a chosen DTE using intervals based on multipliers."""
    df_filtered = Df[Df[' [DTE]'] == dte]
    df_date = df_filtered[df_filtered[' [QUOTE_DATE]'] == date]

    if df_date.empty:
        raise ValueError(f"No data available for DTE {dte} on date {date}")

    x_data = df_date[' [STRIKE]'].values
    y_data = df_date[' [C_IV]'].values
    valid_indices = np.logical_and(~np.isnan(x_data), ~np.isnan(y_data))
    x_data = x_data[valid_indices]
    y_data = y_data[valid_indices]

    if x_data.size == 0 or y_data.size == 0:
        raise ValueError("Insufficient valid data for curve fitting")

    popt, _ = curve_fit(model_func, x_data, y_data)
    a_opt, b_opt, c_opt, d_opt = popt

    r = get_interest_rate(date, dte, rates_csv_path) / 100
    div = get_yield(date, yield_file)

    T = df_date[' [T]'].iloc[0]
    u_last = df_date[' [UNDERLYING_LAST]'].iloc[0]

    multipliers = get_factor_list()
    sp_strikes = [x * u_last for x in multipliers]

    # Calculate interval bounds
    bounds = [(sp_strikes[i] + sp_strikes[i+1])/2 for i in range(len(sp_strikes) - 1)]
    bounds = [2 * sp_strikes[0] - bounds[0]] + bounds + [2 * sp_strikes[-1] - bounds[-1]]

    strikes = np.linspace(bounds[0], bounds[-1], 50000)
    C_IV_IP = model_func(strikes, a_opt, b_opt, c_opt, d_opt)

    top_d2 = np.log(u_last / strikes) + (r - div - ((C_IV_IP ** 2) / 2)) * T
    bot_d2 = np.sqrt(T) * C_IV_IP
    d2 = top_d2 / bot_d2
    cdf = 1 - norm.cdf(d2)
    pdf = np.gradient(cdf, strikes)

    pdf_interp = interp1d(strikes, pdf, kind='linear', fill_value="extrapolate")

    # Calculate and adjust state prices
    state_prices = []
    for i in range(len(bounds) - 1):
        area, _ = quad(pdf_interp, bounds[i], bounds[i + 1])
        state_prices.append(max(area, 0))  # Ensure non-negative values

    # Normalize state prices
    total_area = sum(state_prices)
    if total_area == 0:
        raise ValueError("Total probability is zero after adjustment, cannot normalize.")
    multiply_factor = (1 - r) / total_area
    normalized_state_prices = [x * multiply_factor for x in state_prices]

    return normalized_state_prices


# Using function to create state Price Matrix
def calculate_state_prices_matrix(DTEs, Df, date, rates_csv_path, yield_file):
    spm = []
    for dte in DTEs:
        state_prices = calculate_state_prices(dte, Df, date, rates_csv_path, yield_file)
        spm.append(state_prices)
    spm_final = np.vstack(spm)
    return spm_final.T


# Constructing P bar
def construct_p_bar(S):
    # The number of states
    n = S.shape[0]

    # Initialize P_bar with zeros
    P_bar = np.zeros((n, n))
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


# Finding Matrix P
def penalty_term(Matrix_P, n_states):
    penalty = 0
    for i in range(n_states):
        for j in range(1, i):  # Check left side of the diagonal
            penalty += max(0, Matrix_P[i, j - 1] - Matrix_P[i, j])
        for j in range(i+1, n_states-1):  # Check right side of the diagonal
            penalty += max(0, Matrix_P[i, j + 1] - Matrix_P[i, j])
    return penalty


# Minimization Function (method 1)
def function_p1(P_flat, A, B, n_states):
    Matrix_P = P_flat.reshape((n_states, n_states))

    # Term for minimizing A * P - B
    fitting_term = np.linalg.norm(A @ Matrix_P - B, 'fro') ** 2

    return fitting_term


# Minimization function (method 2)
def function_p2(P_flat, A, B, P_bar_flat, n_states):
    # Reshaping matrices
    Matrix_P = P_flat.reshape((n_states, n_states))
    P_bar = P_bar_flat.reshape((n_states, n_states))

    # Term for minimizing squared difference between P and P_bar
    regularization_term = np.linalg.norm(Matrix_P - P_bar, ord=2)**2

    # Term for minimizing A * P - B
    fitting_term = np.linalg.norm(A @ Matrix_P - B, 'fro') ** 2

    return fitting_term + 0.1 * regularization_term


# Minimization function (method 3)
def function_p3(P_flat, A, B, n_states):
    # Reshaping matrix
    Matrix_P = P_flat.reshape((n_states, n_states))

    # Term for minimizing A * P - B
    fitting_term = np.linalg.norm(A @ Matrix_P - B, 'fro') ** 2

    # Term for enforcing pyramid shape around diagonal
    pyramid_penalization = 0.1 * penalty_term(Matrix_P, n_states)

    return fitting_term + pyramid_penalization


# Minimization function (method 4)
def function_p4(P_flat, A, B, P_bar_flat, n_states):
    Matrix_P = P_flat.reshape((n_states, n_states))
    P_bar = P_bar_flat.reshape((n_states, n_states))

    # Term for minimizing squared difference between P and P_bar
    regularization_term = np.linalg.norm(Matrix_P - P_bar, ord=2)**2

    # Term for minimizing A * P - B
    fitting_term = np.linalg.norm(A @ Matrix_P - B, 'fro') ** 2

    # Term for enforcing pyramid shape around diagonal
    pyramid_penalization = 0.05 * penalty_term(Matrix_P, n_states)

    return fitting_term + pyramid_penalization + 0.2 * regularization_term


# Function for finding matrix P (method 1)
def find_p1(state_prices):
    n_states, n_periods = state_prices.shape
    A = state_prices[:, :-1].T
    B = state_prices[:, 1:].T
    bounds = [(0, None) for _ in range(n_states ** 2)]

    # Guess
    guess = np.eye(11)
    guess_flatten = guess.flatten()

    # Perform optimization
    result = minimize(function_p1, guess_flatten, args=(A, B, n_states), bounds=bounds, method='SLSQP')

    if result.success:
        P_opt = result.x.reshape((n_states, n_states))
        # Normalize rows to ensure they sum to 1
        P_opt = P_opt / P_opt.sum(axis=1, keepdims=True)
        # Making middle row equal to first column S
        first_column_s = state_prices[:, 0]
        P_opt[5, :] = first_column_s
        return P_opt
    else:
        return "Optimization Failed"


# Function for finding matrix P (method 2)
def find_p2(state_prices, P_bar):
    n_states, n_periods = state_prices.shape
    A = state_prices[:, :-1].T
    B = state_prices[:, 1:].T
    P_bar_flat = P_bar.flatten()
    bounds = [(0, None) for _ in range(n_states ** 2)]

    # Perform optimization
    result = minimize(function_p2, P_bar_flat, args=(A, B, P_bar_flat, n_states), bounds=bounds, method='SLSQP')

    if result.success:
        P_opt = result.x.reshape((n_states, n_states))
        # Normalize rows to ensure they sum to 1
        P_opt = P_opt / P_opt.sum(axis=1, keepdims=True)
        first_column_s = state_prices[:, 0]
        P_opt[5, :] = first_column_s
        return P_opt
    else:
        raise ValueError("Optimization failed")


# Function for finding matrix P (method 3)
def find_p3(state_prices):
    n_states, n_periods = state_prices.shape
    A = state_prices[:, :-1].T
    B = state_prices[:, 1:].T
    bounds = [(0, None) for _ in range(n_states ** 2)]

    # Guess
    guess = np.eye(11)
    guess_flatten = guess.flatten()

    # Perform optimization
    result = minimize(function_p3, guess_flatten, args=(A, B, n_states), bounds=bounds, method='SLSQP')

    if result.success:
        P_opt = result.x.reshape((n_states, n_states))
        # Normalize rows to ensure they sum to 1
        P_opt = P_opt / P_opt.sum(axis=1, keepdims=True)
        first_column_s = state_prices[:, 0]
        P_opt[5, :] = first_column_s
        return P_opt
    else:
        raise ValueError("Optimization failed")


# Function for finding matrix P (method 4)
def find_p4(state_prices, P_bar):
    n_states, n_periods = state_prices.shape
    A = state_prices[:, :-1].T
    B = state_prices[:, 1:].T
    P_bar_flat = P_bar.flatten()
    bounds = [(0, None) for _ in range(n_states ** 2)]

    # Perform optimization
    result = minimize(function_p4, P_bar_flat, args=(A, B, P_bar_flat, n_states), bounds=bounds, method='SLSQP')

    if result.success:
        P_opt = result.x.reshape((n_states, n_states))
        # Normalize rows to ensure they sum to 1
        P_opt = P_opt / P_opt.sum(axis=1, keepdims=True)
        first_column_s = state_prices[:, 0]
        P_opt[5, :] = first_column_s
        return P_opt
    else:
        raise ValueError("Optimization failed")


# Using P to find F
def get_f(matrix_P):
    P = np.array(matrix_P, dtype=np.float64)  # Ensure P is a numpy array

    eigenvalues, eigenvectors = np.linalg.eig(P)

    # Find the index of the eigenvalue that is closest to 1 (considered the steady-state or discount factor)
    index = np.argmin(np.abs(eigenvalues - 1))
    delta = eigenvalues[index]
    principal_vector = eigenvectors[:, index]

    # Ensure the principal vector is real and non-negative
    principal_vector = np.real(principal_vector)
    if np.any(principal_vector < 0):
        principal_vector = -principal_vector
    principal_vector /= np.sum(principal_vector)  # Normalize to sum to one

    # Ensure the eigenvalue is real
    delta = np.real(delta)

    # Create matrix F using the specified transformation rule
    F = np.zeros_like(P)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            F[i, j] = (1 / delta) * (principal_vector[j] / principal_vector[i]) * P[i, j]

    # Normalize each row to sum to 1
    F = np.maximum(0, F)  # Avoid negative probabilities due to numerical issues
    F /= np.sum(F, axis=1, keepdims=True)

    return F


# Model for finding F (method 1)
def model_f1(dataframe, date, rates_csv_path, yield_file):
    # Finding list of 12 DTEs from about 30 to 360.
    DTEs = get_dte(date, dataframe)

    # Calculating state price matrix the 12 DTEs
    state_prices_matrix = calculate_state_prices_matrix(DTEs, dataframe, date, rates_csv_path, yield_file)

    # Finding matrix P
    P_opt = find_p1(state_prices_matrix)

    # Using matrix P to find matrix F
    f = get_f(P_opt)
    return f


# Model for finding F (method 2)
def model_f2(dataframe, date, rates_csv_path, yield_file):
    # Finding list of 12 DTEs from about 30 to 360.
    DTEs = get_dte(date, dataframe)

    # Calculating state price matrix the 12 DTEs
    state_prices_matrix = calculate_state_prices_matrix(DTEs, dataframe, date, rates_csv_path, yield_file)

    # Calculating P bar
    P_bar = construct_p_bar(state_prices_matrix)

    # Using minimization function to calculate matrix P

    P_opt = find_p2(state_prices_matrix, P_bar)

    # Using matrix P to find matrix F
    f = get_f(P_opt)
    return f


# Model for finding F (method 3)
def model_f3(dataframe, date, rates_csv_path, yield_file):
    # Finding list of 12 DTEs from about 30 to 360.
    DTEs = get_dte(date, dataframe)

    # Calculating state price matrix the 12 DTEs
    state_prices_matrix = calculate_state_prices_matrix(DTEs, dataframe, date, rates_csv_path, yield_file)

    # Using minimization function to calculate matrix P

    P_opt = find_p3(state_prices_matrix)

    # Using matrix P to find matrix F
    f = get_f(P_opt)
    return f


# Model for finding F (method 4)
def model_f4(dataframe, date, rates_csv_path, yield_file):
    # Finding list of 12 DTEs from about 30 to 360.
    DTEs = get_dte(date, dataframe)

    # Calculating state price matrix the 12 DTEs
    state_prices_matrix = calculate_state_prices_matrix(DTEs, dataframe, date, rates_csv_path, yield_file)

    # Calculating P bar
    P_bar = construct_p_bar(state_prices_matrix)

    # Using minimization function to calculate matrix P

    P_opt = find_p4(state_prices_matrix, P_bar)

    # Using matrix P to find matrix F
    f = get_f(P_opt)
    return f


# Function for testing P (method 1)
def model_p1(dataframe, date, rates_csv_path, yield_file):
    # Finding list of 12 DTEs from about 30 to 360.
    DTEs = get_dte(date, dataframe)

    # Calculating state price matrix the 12 DTEs
    state_prices_matrix = calculate_state_prices_matrix(DTEs, dataframe, date, rates_csv_path, yield_file)

    # Calculating P bar
    P_bar = construct_p_bar(state_prices_matrix)

    # Using minimization function to calculate matrix P
    P_opt = find_p1(state_prices_matrix)

    return P_opt


# Function for testing P (method 2)
def model_p2(dataframe, date, rates_csv_path, yield_file):
    # Finding list of 12 DTEs from about 30 to 360.
    DTEs = get_dte(date, dataframe)

    # Calculating state price matrix the 12 DTEs
    state_prices_matrix = calculate_state_prices_matrix(DTEs, dataframe, date, rates_csv_path, yield_file)

    # Calculating P bar
    P_bar = construct_p_bar(state_prices_matrix)

    # Using minimization function to calculate matrix P
    P_opt = find_p2(state_prices_matrix, P_bar)

    return P_opt


# Function for testing P (method 3)
def model_p3(dataframe, date, rates_csv_path, yield_file):
    # Finding list of 12 DTEs from about 30 to 360.
    DTEs = get_dte(date, dataframe)

    # Calculating state price matrix the 12 DTEs
    state_prices_matrix = calculate_state_prices_matrix(DTEs, dataframe, date, rates_csv_path, yield_file)

    # Using minimization function to calculate matrix P
    P_opt = find_p3(state_prices_matrix)

    return P_opt


# Function for testing P (method 4)
def model_p4(dataframe, date, rates_csv_path, yield_file):
    # Finding list of 12 DTEs from about 30 to 360.
    DTEs = get_dte(date, dataframe)

    # Calculating state price matrix the 12 DTEs
    state_prices_matrix = calculate_state_prices_matrix(DTEs, dataframe, date, rates_csv_path, yield_file)

    # Calculating P bar
    P_bar = construct_p_bar(state_prices_matrix)

    # Using minimization function to calculate matrix P
    P_opt = find_p4(state_prices_matrix, P_bar)

    return P_opt


# Function for forecasting
def forecast(dataframe, date, rates_csv_path, yield_file):
    # Calculating matrix F using function
    matrix_f = model_f4(dataframe, date, rates_csv_path, yield_file)

    # Taking middle row of matrix F to get chances of different states of the world
    prob_list = list(matrix_f[5])

    # Filtering dataset for date and taking underlying_last for that date
    filtered_df = dataframe[dataframe[' [QUOTE_DATE]'] == date]
    u_last = filtered_df[' [UNDERLYING_LAST]'].iloc[0]

    # Getting factor list and multiplying it by underlying last for strikes list
    factor_list = get_factor_list()
    strikes_list = [x * u_last for x in factor_list]

    # Multiplying strikes list by the probability list. Sum is the expected outcome
    product_list = [x * y for x, y in zip(prob_list, strikes_list)]
    exp_outcome = sum(product_list)
    return exp_outcome, u_last


# Predictions function
def generate_predictions(df, dates, rates_csv_path, yield_file):
    predictions = []

    for date in dates:
        try:
            # Call the forecast function for the current date and append the prediction to the list
            prediction, underlying = forecast(df, date, rates_csv_path, yield_file)
            predictions.append((date, prediction, underlying))
        except Exception as e:
            print(f"Error occurred for date {date}: {e}")
            continue

    # Create a DataFrame with predictions
    predictions_df = pd.DataFrame(predictions, columns=['Date', 'Predicted_Price', 'Underlying_Last'])
    predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
    return predictions_df


# Adding actual prices to df
def add_actual_prices(predictions_df, sp500_prices, days=15):
    # Convert 'Date' columns to datetime to ensure proper merging and alignment
    sp500_prices['Date'] = pd.to_datetime(sp500_prices['Date'])
    predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])

    # Create a dictionary to quickly access prices by date
    price_dict = sp500_prices.set_index('Date')['Close'].to_dict()

    actual_prices = []

    for _, row in predictions_df.iterrows():
        start_date = row['Date']
        target_date = start_date + pd.Timedelta(days=days)

        # Find the closest available date with data after the target_date
        closest_date = None
        for i in range(days + 1):
            candidate_date = target_date + pd.Timedelta(days=i)
            if candidate_date in price_dict:
                closest_date = candidate_date
                break

        if closest_date and closest_date in price_dict:
            actual_price = price_dict[closest_date]
        else:
            actual_price = None

        actual_prices.append(actual_price)

    predictions_df['Actual_Price'] = actual_prices
    return predictions_df


# Inputs
# rdf = data_selection('Data/spx_eod_201104.csv')
sp500_prices = pd.read_csv('^SPX.csv')
rdf = data_selection('2023_complete_numeric.csv')
rates_path = 'TreasuryBillComplete.csv'
yield_path = 'YearlyYieldData.csv'
# day = ' 2011-04-27'
day = ' 2023-01-04'
DTE = 30


# Testing different methods of finding P
P1 = model_p1(rdf, day, rates_path, yield_path)
P1_round = np.round(P1, 3)
print(f"Matrix P1:\n{P1_round}\n\n")

P2 = model_p2(rdf, day, rates_path, yield_path)
P2_round = np.round(P2, 3)
print(f"Matrix P2:\n{P2_round}\n\n")

P3 = model_p3(rdf, day, rates_path, yield_path)
P3_round = np.round(P3, 3)
print(f"Matrix P3:\n{P3_round}\n\n")

P4 = model_p4(rdf, day, rates_path, yield_path)
P4_round = np.round(P4, 3)
print(f"Matrix P4:\n{P4_round}\n\n")

# Testing models on data by Ross
S_ross = np.array([
    [0.005, 0.023, 0.038, 0.050, 0.058, 0.064, 0.068, 0.071, 0.073, 0.075, 0.076, 0.076],
    [0.007, 0.019, 0.026, 0.030, 0.032, 0.034, 0.034, 0.035, 0.035, 0.035, 0.034, 0.034],
    [0.018, 0.041, 0.046, 0.050, 0.051, 0.052, 0.051, 0.050, 0.050, 0.049, 0.048, 0.046],
    [0.045, 0.064, 0.073, 0.073, 0.072, 0.070, 0.068, 0.066, 0.064, 0.061, 0.058, 0.056],
    [0.164, 0.156, 0.142, 0.128, 0.118, 0.109, 0.102, 0.096, 0.091, 0.085, 0.081, 0.076],
    [0.478, 0.302, 0.234, 0.198, 0.173, 0.155, 0.141, 0.129, 0.120, 0.111, 0.103, 0.096],
    [0.276, 0.316, 0.278, 0.245, 0.219, 0.198, 0.180, 0.164, 0.151, 0.140, 0.130, 0.120],
    [0.007, 0.070, 0.129, 0.155, 0.166, 0.167, 0.164, 0.158, 0.152, 0.145, 0.137, 0.130],
    [0.000, 0.002, 0.016, 0.036, 0.055, 0.072, 0.085, 0.094, 0.100, 0.103, 0.105, 0.105],
    [0.000, 0.000, 0.001, 0.004, 0.009, 0.017, 0.026, 0.036, 0.045, 0.053, 0.061, 0.067],
    [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.001, 0.002, 0.002, 0.003, 0.003]
])

P_ross = [
    [0.671, 0.241, 0.053, 0.005, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.000],
    [0.280, 0.396, 0.245, 0.054, 0.004, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.049, 0.224, 0.394, 0.248, 0.056, 0.004, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.006, 0.044, 0.218, 0.390, 0.250, 0.057, 0.003, 0.000, 0.000, 0.000, 0.000],
    [0.006, 0.007, 0.041, 0.211, 0.385, 0.249, 0.054, 0.002, 0.000, 0.000, 0.000],
    [0.005, 0.007, 0.018, 0.045, 0.164, 0.478, 0.276, 0.007, 0.000, 0.000, 0.000],
    [0.001, 0.001, 0.001, 0.004, 0.040, 0.204, 0.382, 0.251, 0.058, 0.005, 0.000],
    [0.001, 0.001, 0.001, 0.002, 0.006, 0.042, 0.204, 0.373, 0.243, 0.055, 0.004],
    [0.002, 0.001, 0.001, 0.002, 0.003, 0.006, 0.041, 0.195, 0.361, 0.232, 0.057],
    [0.001, 0.000, 0.000, 0.001, 0.001, 0.001, 0.003, 0.035, 0.187, 0.347, 0.313],
    [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.032, 0.181, 0.875]
]

Pbar = construct_p_bar(S_ross)
P = find_p4(S_ross, Pbar)
P_round = np.round(P, 3)
print(f"Matrix P:\n{P_round}\n\n")

F = get_f(P_ross)
F_round = np.round(F, 3)
print(f"Matrix F:\n{F_round}\n\n")

# Inputs for regression analysis of predictions
df = data_selection('Complete_Merged.csv')
dates = df[' [QUOTE_DATE]'].drop_duplicates().tolist()

# Predicting using model
predictions_df = generate_predictions(df, dates, rates_path, yield_path)
predictions_df = add_actual_prices(predictions_df, sp500_prices)
predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
predictions_df.sort_values(by='Date', inplace=True)
predictions_df['Actual_Return'] = (predictions_df['Actual_Price'] - predictions_df['Underlying_Last']) / predictions_df['Underlying_Last']
predictions_df['Predicted_Return'] = (predictions_df['Predicted_Price'] - predictions_df['Underlying_Last']) / predictions_df['Underlying_Last']
print(predictions_df.head())
predictions_df.to_csv('Predictions_file.csv', index=False)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(predictions_df['Date'], predictions_df['Actual_Return'], label='Actual Return', color='blue', marker='o')
plt.plot(predictions_df['Date'], predictions_df['Predicted_Return'], label='Predicted Return', color='red', marker='x')

# Adding titles and labels
plt.title('Predicted Return vs Actual Return Over Time')
plt.xlabel('Date')
plt.ylabel('Return')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()

# Display the plot
plt.show()

# Cleaning
clean_df = predictions_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Predicted_Return', 'Actual_Return'])

# Calculate the Pearson correlation coefficient
correlation, p_value = stats.pearsonr(clean_df['Predicted_Return'], clean_df['Actual_Return'])

# Interpret the p-value
significance = "statistically significant" if p_value < 0.05 else "not statistically significant"

# Fit the regression model
X = sm.add_constant(clean_df['Predicted_Return'])
y = clean_df['Actual_Return']
model = sm.OLS(y, X).fit()

# Extract regression summary
summary_df = pd.DataFrame({
    'Statistic': ['Pearson Correlation Coefficient', 'P-value', 'Significance', 'R-squared', 'Adj. R-squared', 'F-statistic', 'Prob (F-statistic)', 'Log-likelihood', 'AIC', 'BIC'],
    'Value': [correlation, p_value, significance, model.rsquared, model.rsquared_adj, model.fvalue, model.f_pvalue, model.llf, model.aic, model.bic]
})

# Extract regression parameters
params = model.params
pvalues = model.pvalues
conf = model.conf_int()
conf['Parameter'] = params
conf.columns = ['Conf. Interval Lower', 'Conf. Interval Upper', 'Parameter']

params_df = pd.DataFrame({
    'Parameter': conf['Parameter'],
    'P-value': pvalues,
    'Conf. Interval Lower': conf['Conf. Interval Lower'],
    'Conf. Interval Upper': conf['Conf. Interval Upper']
})

# Combine summary and parameters into a single DataFrame
combined_df = pd.concat([summary_df, params_df.reset_index()], axis=1)

# Save the combined DataFrame to a CSV file
combined_df.to_csv('analysis_results.csv', index=False)

print("Results exported to analysis_results.csv")
