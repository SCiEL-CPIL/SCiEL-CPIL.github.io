"""Bermudan put exercise-policy research with LSMC regression-based backward induction."""
import numpy as np
import time as _time
from scipy import stats

class ExercisePolicy:
    def __init__(self, thresholds):
        self.thresholds = thresholds  # array of shape (12,) for 12 exercise dates

    def exercise(self, current_state, time_index):
        # time_index: 0-based index into the 12 exercise dates
        if time_index < 0 or time_index >= len(self.thresholds):
            return np.zeros(len(current_state), dtype=bool)
        return current_state < self.thresholds[time_index]


def fit_exercise_policy(training_paths, contract):
    start_time = _time.time()
    deadline = start_time + 115  # safety margin before 120s kill

    # Parse contract parameters
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)
    n_paths = training_paths.shape[0]

    dt = T / n_exercise
    discount = np.exp(-r * dt)

    # training_paths shape: (n_paths, n_exercise + 1) — includes t=0
    # Exercise dates are at indices 1..n_exercise (0-based), i.e., t=dt, 2dt, ..., T
    # We work backwards from the last exercise date (index n_exercise) to the first (index 1)

    # Payoff at maturity (last exercise date, index n_exercise)
    S_last = training_paths[:, n_exercise]
    payoff = np.maximum(K - S_last, 0.0)

    # For each exercise date going backwards, we need to decide: exercise or continue
    # Continuation value = discount * E[future value | S_t]
    # We use LSMC: regress continuation value on features of S_t

    # We'll store the continuation value at each step for use in the previous step
    # cont_value[i] is the value of the option at exercise date i (0-based, i=1..n_exercise)
    # if we continue (i.e., don't exercise) at date i

    # Actually, let's think more carefully:
    # At exercise date i (1-indexed from 1 to n_exercise, corresponding to column i in training_paths),
    # the holder can exercise (get K - S_i) or continue (get discounted expected future value).
    # We go backwards: start at i = n_exercise, then n_exercise-1, ..., 1.

    # At the last date (i = n_exercise), continuation value is 0 (no more dates), so always exercise if in the money.
    # Actually at the last date, you must exercise or let it expire. If it's a Bermudan, you can exercise at the last date.
    # So at the last date, the value is max(K - S, 0) since there's no future.

    # Let's define:
    # V[i, j] = optimal value at exercise date i for path j (if optimal policy followed from date i onwards)
    # But we don't store all of them; we just need the continuation value estimate at each date.

    # Backward induction with LSMC:
    # Step n_exercise: V = max(K - S, 0) [since no continuation]
    # Step i (from n_exercise-1 down to 1):
    #   cont_value = discount * E[V[i+1] | S_i]  (estimated via regression)
    #   V[i] = max(K - S_i, cont_value)
    #   Exercise if K - S_i > cont_value

    # We'll track the exercise decision and the value at each step.

    # Start from the last exercise date
    current_value = payoff.copy()  # value at the last exercise date

    # We need to store thresholds for each exercise date
    thresholds = np.zeros(n_exercise)

    # Process backwards from the second-to-last exercise date down to the first
    # Exercise date index in training_paths: n_exercise-1 down to 1
    # 0-based index in thresholds array: (n_exercise-1 - 1) down to 0, i.e., n_exercise-2 down to 0

    for i in range(n_exercise - 1, 0, -1):
        if _time.time() > deadline - 10:
            break

        S_i = training_paths[:, i]
        S_next = training_paths[:, i + 1]

        # Features for regression: use S_i and maybe S_i^2, log(S_i), etc.
        # Simple polynomial features for the spot price
        features = np.column_stack([
            np.ones_like(S_i),
            S_i,
            S_i ** 2,
            np.log(S_i + 1e-10),
            np.exp(-S_i / 100.0)
        ])

        # Regress current_value (which is V[i+1]) on features at S_i
        # This gives E[V[i+1] | S_i]
        try:
            # Use least squares
            coef, residuals, rank, sv = np.linalg.lstsq(features, current_value, rcond=None)
            cont_estimate = features @ coef
        except:
            # Fallback: simple mean
            cont_estimate = np.full_like(current_value, current_value.mean())

        # Continuation value at date i
        continuation = discount * cont_estimate

        # Optimal value at date i
        exercise_payoff = np.maximum(K - S_i, 0.0)
        V_i = np.maximum(exercise_payoff, continuation)

        # Determine threshold: exercise if exercise_payoff > continuation
        # i.e., K - S_i > continuation (when S_i < K)
        # threshold is the S_i value where K - S_i = continuation
        # We find this by looking at the regression curve
        # For simplicity, use the median of S_i where exercise is optimal
        exercise_mask = exercise_payoff > continuation
        if np.any(exercise_mask):
            thresholds[i - 1] = np.median(S_i[exercise_mask])
        else:
            thresholds[i - 1] = 0.0  # never exercise

        # Update current_value for the next backward step
        current_value = V_i

    # Ensure thresholds are non-increasing (earlier dates should have lower or equal thresholds for a put)
    # Actually for a put, you're more likely to exercise when the price is low, and the threshold
    # typically increases as we get closer to maturity (because the time value decreases).
    # Let's enforce monotonicity: thresholds[i-1] <= thresholds[i] for i from 1 to n_exercise-1
    # In our array, thresholds[0] is for the first exercise date, thresholds[n_exercise-1] for the last.
    # We want thresholds to be non-decreasing: thresholds[0] <= thresholds[1] <= ...
    for i in range(1, n_exercise):
        if thresholds[i - 1] > thresholds[i]:
            thresholds[i - 1] = thresholds[i]

    # At the last exercise date, always exercise if in the money
    thresholds[n_exercise - 1] = K

    # Final policy
    return ExercisePolicy(thresholds)
