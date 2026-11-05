"""Bermudan exercise-policy research with hybrid BS-initialization and continuous coordinate ascent optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time
from scipy.stats import norm
from scipy.optimize import minimize_scalar

_DEADLINE = 115.0  # seconds; safety margin under 120s wall-clock limit (150s kill)

def _bs_put_price(S, K, r, sigma, T):
    """
    Calculate European put price using Black-Scholes formula.
    S: array of spot prices
    K: strike
    r: risk-free rate
    sigma: volatility
    T: time to maturity
    """
    if T <= 0:
        return np.maximum(K - S, 0.0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Put price = K * exp(-r*T) * N(-d2) - S * N(-d1)
    N = norm.cdf
    price = K * np.exp(-r * T) * N(-d2) - S * N(-d1)
    return price

def _find_bs_threshold(K, r, sigma, T_remaining, dt, S_grid):
    """
    Find the spot price S* such that Intrinsic(S*) = Discounted Continuation(S*).
    Intrinsic: K - S
    Continuation: exp(-r*dt) * BS_Put(S, K, r, sigma, T_remaining - dt)
    """
    T_next = T_remaining - dt
    if T_next < 0: T_next = 0
    
    # Vectorized calculation over S_grid
    intrinsic = K - S_grid
    cont_val = np.exp(-r * dt) * _bs_put_price(S_grid, K, r, sigma, T_next)
    
    # We want S* where intrinsic == cont_val
    diff = intrinsic - cont_val
    # Find sign change
    sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_change) == 0:
        # Check if always positive (exercise always) or always negative (never exercise)
        if np.all(diff > 0):
            return np.max(S_grid) 
        else:
            return np.min(S_grid)
            
    idx = sign_change[0]
    # Linear interpolation
    S1, S2 = S_grid[idx], S_grid[idx+1]
    d1, d2 = diff[idx], diff[idx+1]
    if d2 == d1:
        return S1
    S_star = S1 - d1 * (S2 - S1) / (d2 - d1)
    return S_star

class ExercisePolicy:
    def __init__(self, thresholds, r, dt):
        self.thresholds = thresholds
        self.r = r
        self.dt = dt
        self.n_dates = len(thresholds)

    def exercise(self, current_state, time):
        """
        current_state: array of stock prices
        time: integer index of the exercise date (0 to 11) or float time in [0,1]
        Returns: boolean array
        """
        # Determine the index of the exercise date
        if isinstance(time, (int, np.integer)):
            idx = int(time)
        else:
            # If time is a float, assume it's in [0, 1] representing years
            # 12 monthly dates: 1/12, 2/12, ..., 12/12
            # Index 0 -> 1/12, Index 11 -> 12/12
            idx = int(np.round(time * 12)) - 1
            if idx < 0:
                idx = 0
            if idx >= self.n_dates:
                idx = self.n_dates - 1

        if idx < 0 or idx >= self.n_dates:
            return np.zeros(len(current_state), dtype=bool)
        
        return current_state < self.thresholds[idx]

def _evaluate_policy_fast(S, K, r, T, n_dates, thresholds):
    """
    Evaluate a threshold policy on given paths.
    S: (n_dates, n_paths)
    thresholds: (n_dates,)
    Returns mean discounted payoff.
    """
    dt = T / n_dates
    n_paths = S.shape[1]
    
    # V is the value at the current step, propagated backwards
    # We start from the end.
    # At step t, if we exercise, we get intrinsic. If we don't, we get disc * V[t+1].
    # Since the policy is fixed (threshold), we just apply the rule.
    
    V = np.zeros(n_paths)
    for t in range(n_dates - 1, -1, -1):
        intrinsic = np.maximum(K - S[t], 0.0)
        ex_mask = S[t] < thresholds[t]
        # If exercising, value is intrinsic.
        # If continuing, value is discounted continuation value (which is V from next step).
        V = intrinsic * ex_mask + np.exp(-r * dt) * V * (~ex_mask)
        
    return np.mean(V)

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy.
    1. Initialize thresholds using Black-Scholes analytical boundary approximation.
    2. Refine thresholds using coordinate ascent on training paths.
    """
    t_start = _time.time()

    # Extract parameters
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)
    
    S = training_paths
    if S.ndim != 2:
        # Handle case where shape might be (n_paths, n_dates) or similar
        # Standard expectation: (n_dates, n_paths) or (n_paths, n_dates)
        # The task says "12 monthly exercise dates".
        # Usually paths are (n_paths, n_steps).
        # Let's check shape. If S.shape[0] == n_dates, it's (n_dates, n_paths).
        # If S.shape[1] == n_dates, it's (n_paths, n_dates).
        if S.shape[1] == n_dates:
            S = S.T
        else:
            S = S.reshape(n_dates, -1)
            
    n_paths = S.shape[1]
    dt = T / n_dates
    
    # 1. Initialize thresholds using BS heuristic
    thresholds = np.zeros(n_dates)
    # Use a fine grid of S values for finding the BS threshold
    # Min and Max from training data, expanded slightly
    S_min = np.min(S)
    S_max = np.max(S)
    S_grid = np.linspace(S_min * 0.5, S_max * 1.5, 200)
    
    for t in range(n_dates):
        T_remaining = (n_dates - t) * dt
        # The continuation value looks forward to t+1, so time remaining for that option is T_remaining - dt
        # But the BS formula needs the time to maturity of the option being valued.
        # At step t, we compare Intrinsic(t) vs Disc * Value(t+1).
        # Value(t+1) is approximated by BS(S, K, r, sigma, T_remaining - dt).
        thresholds[t] = _find_bs_threshold(K, r, sigma, T_remaining, dt, S_grid)
        
    # Ensure thresholds are within reasonable bounds
    thresholds = np.clip(thresholds, 0.0, K + 10.0)
    
    # 2. Coordinate Ascent Optimization
    # Phase 1: Discrete Coarse Search to find the general region
    n_candidates = 100
    candidate_thresholds = []
    for t in range(n_dates):
        # Use percentiles of the spot distribution at date t
        pcts = np.linspace(0.01, 99.99, n_candidates)
        vals = np.percentile(S[t], pcts)
        # Add 0 and large value to cover extremes
        vals = np.concatenate([[0.0], vals, [K + 100.0]])
        vals = np.sort(np.unique(vals))
        candidate_thresholds.append(vals)
        
    best_score = _evaluate_policy_fast(S, K, r, T, n_dates, thresholds)
    
    # Iterate until time runs out or no improvement
    while _time.time() - t_start < _DEADLINE - 10.0: # Leave 10s for continuous polish
        
        improved = False
        for t in range(n_dates):
            if _time.time() - t_start > _DEADLINE - 10.0:
                break
                
            candidates = candidate_thresholds[t]
            best_local_score = -np.inf
            best_local_thresh = thresholds[t]
            
            # Evaluate each candidate
            for c in candidates:
                # Temporarily set threshold t to c
                old_thresh = thresholds[t]
                thresholds[t] = c
                score = _evaluate_policy_fast(S, K, r, T, n_dates, thresholds)
                if score > best_local_score:
                    best_local_score = score
                    best_local_thresh = c
                thresholds[t] = old_thresh
                
            if best_local_thresh != thresholds[t]:
                thresholds[t] = best_local_thresh
                if best_local_score > best_score:
                    best_score = best_local_score
                    improved = True
            
        if not improved:
            # If no improvement in discrete search, proceed to continuous polish
            break
            
    # Phase 2: Continuous Coordinate Ascent (Polish)
    # Use scipy.optimize.minimize_scalar to find the precise optimal threshold for each date
    # This removes quantization error from the discrete grid
    if _time.time() - t_start < _DEADLINE - 2.0:
        # Define a helper function for the optimizer
        def objective_wrapper(t_idx):
            def obj(theta):
                old_thresh = thresholds[t_idx]
                thresholds[t_idx] = theta
                score = _evaluate_policy_fast(S, K, r, T, n_dates, thresholds)
                thresholds[t_idx] = old_thresh
                return -score # Minimize negative score
            
            return obj
        
        # Determine bounds for each threshold
        # Use a wide range to ensure we don't miss the global optimum if discrete search was bad
        # But restrict to reasonable values [0, 2*K] to avoid nonsense
        bounds = (0.0, 2.0 * K)
        
        # Iterate through coordinates multiple times
        # Each pass refines the thresholds sequentially
        while _time.time() - t_start < _DEADLINE - 2.0:
            progress = False
            for t in range(n_dates):
                if _time.time() - t_start > _DEADLINE - 2.0:
                    break
                
                current_thresh = thresholds[t]
                # Try to optimize this coordinate
                # Use 'bounded' method which is robust for 1D
                try:
                    res = minimize_scalar(objective_wrapper(t), bounds=bounds, method='bounded', 
                                          options={'xatol': 1e-4})
                    if res.success:
                        new_thresh = res.x
                        # Verify improvement
                        old_thresh = thresholds[t]
                        thresholds[t] = new_thresh
                        new_score = _evaluate_policy_fast(S, K, r, T, n_dates, thresholds)
                        if new_score > best_score:
                            best_score = new_score
                            progress = True
                        else:
                            # Revert if it didn't improve (shouldn't happen if optimizer works, but safety)
                            thresholds[t] = old_thresh
                except Exception:
                    pass
            
            if not progress:
                break

    return ExercisePolicy(thresholds, r, dt)
# EVOLVE-BLOCK-END
