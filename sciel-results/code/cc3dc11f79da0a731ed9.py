"""Bermudan put exercise-policy research with backward induction on training paths."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_DEADLINE = 100.0  # safety margin under 120s limit


def _extract_contract_params(contract):
    """Extract model parameters from the contract dict (with safe defaults)."""
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)
    return S0, K, r, sigma, T, n_dates


def _backward_induction_thresholds(training_paths, K, r, T, n_dates, deadline_start):
    """
    Compute time-dependent exercise thresholds via backward induction.
    
    training_paths: shape (n_dates, n_paths) of spot prices at each exercise date.
    Returns: array of thresholds shape (n_dates,), where threshold[t] is the spot
             level below which we exercise at date t.
    """
    n_dates_actual = training_paths.shape[0]
    n_paths = training_paths.shape[1]
    dt = T / n_dates  # use nominal n_dates for discounting consistency
    
    # Discount factor per step
    df = np.exp(-r * dt)
    
    # Payoff at each date: max(K - S, 0)
    payoffs = np.maximum(K - training_paths, 0.0)
    
    # Thresholds array
    thresholds = np.zeros(n_dates_actual)
    
    # At the last exercise date, exercise iff payoff > 0, i.e., S < K
    thresholds[-1] = K
    
    # For dates from second-to-last down to first:
    # continuation_value at date t = df * E[ value_at_date[t+1] | S_t ]
    # We estimate this by regressing value_at_date[t+1] on S_t using isotonic regression.
    
    from scipy.stats import isotonic_regression
    
    # value_at_date for the last date
    value_at_date = np.zeros_like(training_paths)
    value_at_date[-1] = payoffs[-1]
    
    # For t from n_dates-2 down to 0:
    for t in range(n_dates_actual - 2, -1, -1):
        # Check deadline
        if _time.time() - deadline_start > _DEADLINE - 5:
            # Return what we have so far (fill remaining with K as fallback)
            for tt in range(t, n_dates_actual):
                thresholds[tt] = K
            return thresholds
        
        S_t = training_paths[t]
        v_next = value_at_date[t + 1]
        
        # Sort by S_t for isotonic regression
        order = np.argsort(S_t)
        S_sorted = S_t[order]
        v_sorted = v_next[order]
        
        # Isotonic regression: we want a function that is non-increasing in S.
        # isotonic_regression gives non-decreasing. So apply to -v_sorted, then negate.
        neg_v_iso = isotonic_regression(-v_sorted)
        v_iso = -neg_v_iso
        
        # Find the threshold where K - S = df * v_iso(S)
        diff = (K - S_sorted) - df * v_iso
        positive_mask = diff >= 0
        
        if not np.any(positive_mask):
            thresholds[t] = 0.0
        else:
            last_pos = np.where(positive_mask)[0][-1]
            if last_pos == len(S_sorted) - 1:
                thresholds[t] = S_sorted[-1]
            else:
                S1 = S_sorted[last_pos]
                S2 = S_sorted[last_pos + 1]
                d1 = diff[last_pos]
                d2 = diff[last_pos + 1]
                if d1 != d2:
                    frac = d1 / (d1 - d2)
                    thresholds[t] = S1 + frac * (S2 - S1)
                else:
                    thresholds[t] = S1
        
        # Compute value_at_date[t] = max(payoffs[t], df * v_iso(S_t))
        v_iso_at_S_t = np.interp(S_t, S_sorted, v_iso)
        cont_value = df * v_iso_at_S_t
        value_at_date[t] = np.maximum(payoffs[t], cont_value)
    
    return thresholds


def fit_exercise_policy(training_paths, contract):
    _fit_start = _time.time()
    
    S0, K, r, sigma, T, n_dates = _extract_contract_params(contract)
    
    # training_paths shape: (n_dates, n_paths)
    n_dates_actual = training_paths.shape[0]
    
    # Compute thresholds via backward induction
    try:
        thresholds = _backward_induction_thresholds(training_paths, K, r, T, n_dates, _fit_start)
    except Exception:
        # Fallback: use K as threshold for all dates (exercise if in-the-money)
        thresholds = np.full(n_dates_actual, K)
    
    # Ensure thresholds are finite and reasonable
    thresholds = np.where(np.isfinite(thresholds), thresholds, K)
    thresholds = np.clip(thresholds, 0.0, K)  # threshold should be <= K for a put
    
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, thresholds, K):
            self.thresholds = thresholds
            self.K = K
        
        def exercise(self, current_state, time):
            """
            current_state: array of spot prices, shape (n_paths,)
            time: int, the current exercise-date index (0-based)
            Returns: boolean array, shape (n_paths,)
            """
            t = int(time)
            if t < 0 or t >= len(self.thresholds):
                return np.zeros(len(current_state), dtype=bool)
            threshold = self.thresholds[t]
            return current_state <= threshold
    
    policy = ExercisePolicy(thresholds, K)
    return policy
# EVOLVE-BLOCK-END
