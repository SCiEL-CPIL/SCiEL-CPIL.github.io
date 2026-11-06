"""Bermudan put exercise policy via Monte Carlo Longstaff-Schwartz backward induction with improved regression."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s kill

    # Parse contract parameters
    S0 = float(contract.get('S0', 100.0))
    K = float(contract.get('K', 100.0))
    r = float(contract.get('r', 0.05))
    sigma = float(contract.get('sigma', 0.2))
    T = float(contract.get('T', 1.0))
    n_dates = int(contract.get('n_dates', 12))

    n_train = training_paths.shape[0]
    n_cols = training_paths.shape[1]

    # Determine time step and discount factor
    dt = T / n_dates
    discount = np.exp(-r * dt)

    # Identify which columns are exercise dates
    if n_cols == n_dates:
        date_cols = list(range(n_dates))
    elif n_cols == n_dates + 1:
        date_cols = list(range(1, n_dates + 1))
    else:
        date_cols = list(range(n_cols - n_dates, n_cols))

    # Extract prices at each exercise date
    prices = np.array([training_paths[:, date_cols[j]] for j in range(n_dates)])
    # prices shape: (n_dates, n_train)

    # Payoff at each date: max(K - S, 0) for a put
    payoffs = np.maximum(K - prices, 0.0)  # shape (n_dates, n_train)

    n = n_train

    # Backward induction with improved regression
    # Use polynomial regression on log(S) with degree 4
    deg = 4

    # V_next: value at next date
    V_next = payoffs[-1].copy()  # shape (n_train,)

    # Store coefficients for each date
    coeffs = [None] * n_dates

    # For the last date, we always exercise if payoff > 0
    coeffs[n_dates - 1] = None

    # Fit regressions from second-to-last date to first date
    for j in range(n_dates - 2, -1, -1):
        if time.time() > deadline:
            break

        S_j = prices[j]  # shape (n_train,)
        logS = np.log(S_j + 1e-10)

        # Fit polynomial regression: V_next ~ poly(logS, deg)
        try:
            c = np.polynomial.polynomial.polyfit(logS, V_next, deg)
            coeffs[j] = c
        except Exception:
            coeffs[j] = None

        # Compute fitted continuation value
        if coeffs[j] is not None:
            fitted_cont = np.polynomial.polynomial.polyval(logS, coeffs[j])
        else:
            fitted_cont = np.full(n_train, np.mean(V_next))

        # Value at date j: max(payoff, discount * continuation)
        V_j = np.maximum(payoffs[j], discount * fitted_cont)
        V_next = V_j

    # Build the exercise policy object
    class BermudanExercisePolicy:
        def __init__(self, coeffs, n_dates, K, discount, deg):
            self.coeffs = coeffs
            self.n_dates = n_dates
            self.K = K
            self.discount = discount
            self.deg = deg

        def exercise(self, current_state, time_idx):
            """
            current_state: array of spot prices (shape (n_eval,))
            time_idx: which exercise date (0-indexed, 0..n_dates-1)
            Returns: boolean array, True if exercise
            """
            if time_idx >= self.n_dates:
                return np.zeros(len(current_state), dtype=bool)

            S = np.asarray(current_state, dtype=float)
            payoff = np.maximum(self.K - S, 0.0)

            if time_idx == self.n_dates - 1:
                # Last date: exercise if payoff > 0
                return payoff > 0

            c = self.coeffs[time_idx]
            if c is None:
                return payoff > 0

            logS = np.log(S + 1e-10)
            fitted_cont = np.polynomial.polynomial.polyval(logS, c)
            cont_value = self.discount * fitted_cont
            return payoff > cont_value

    return BermudanExercisePolicy(coeffs, n_dates, K, discount, deg)

# EVOLVE-BLOCK-END
