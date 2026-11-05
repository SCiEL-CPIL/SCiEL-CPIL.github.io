"""Bermudan put exercise-policy research with threshold-based policy and coordinate ascent optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time
from scipy.stats import norm

_DEADLINE = 145.0  # seconds; safety margin under 150 s wall-clock limit

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
    
    diff = intrinsic - cont_val
    # Find sign change
    sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_change) == 0:
        if np.all(diff > 0):
            return np.max(S_grid)
        else:
            return np.min(S_grid)
            
    idx = sign_change[0]
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
        if isinstance(time, (int, np.integer)):
            idx = int(time)
        else:
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
    
    V = np.zeros(n_paths)
    for t in range(n_dates - 1, -1, -1):
        intrinsic = np.maximum(K - S[t], 0.0)
        ex_mask = S[t] < thresholds[t]
        V = intrinsic * ex_mask + np.exp(-r * dt) * V * (~ex_mask)
        
    return np.mean(V)

def _evaluate_policy_vectorized(S, K, r, T, n_dates, thresholds_array):
    """
    Vectorized evaluation of multiple threshold sets.
    S: (n_dates, n_paths)
    thresholds_array: (n_candidates, n_dates)
    Returns array of scores for each candidate.
    """
    dt = T / n_dates
    n_paths = S.shape[1]
    n_candidates = thresholds_array.shape[0]
    
    scores = np.zeros(n_candidates)
    for c in range(n_candidates):
        scores[c] = _evaluate_policy_fast(S, K, r, T, n_dates, thresholds_array[c])
    return scores

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy.
    1. Initialize thresholds using Black-Scholes analytical boundary approximation.
    2. Refine thresholds using coordinate ascent on training paths with finer grid.
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
        S = S.reshape(n_dates, -1)
        
    n_paths = S.shape[1]
    dt = T / n_dates
    
    # 1. Initialize thresholds using BS heuristic
    thresholds = np.zeros(n_dates)
    S_min = np.min(S)
    S_max = np.max(S)
    S_grid = np.linspace(max(S_min * 0.5, 1.0), S_max * 1.5, 300)
    
    for t in range(n_dates):
        T_remaining = (n_dates - t) * dt
        thresholds[t] = _find_bs_threshold(K, r, sigma, T_remaining, dt, S_grid)
        
    # Ensure thresholds are within reasonable bounds
    thresholds = np.clip(thresholds, 0.0, K + 10.0)
    
    # 2. Coordinate Ascent Optimization with finer candidate grid
    # Use more candidates for finer search
    n_candidates = 250
    candidate_thresholds = []
    for t in range(n_dates):
        # Use percentiles of the spot distribution at date t
        pcts = np.linspace(0.005, 99.995, n_candidates)
        vals = np.percentile(S[t], pcts)
        # Add 0 and large value to cover extremes
        vals = np.concatenate([[0.0], vals, [K + 100.0]])
        vals = np.sort(np.unique(vals))
        candidate_thresholds.append(vals)
        
    best_score = _evaluate_policy_fast(S, K, r, T, n_dates, thresholds)
    
    # Precompute discount factors
    disc = np.exp(-r * dt)
    
    # Iterate until time runs out
    max_passes = 20
    for pass_num in range(max_passes):
        if _time.time() - t_start > _DEADLINE - 5.0:
            break
            
        improved = False
        for t in range(n_dates):
            if _time.time() - t_start > _DEADLINE - 5.0:
                break
                
            candidates = candidate_thresholds[t]
            best_local_score = -np.inf
            best_local_thresh = thresholds[t]
            
            # Evaluate each candidate
            for c in candidates:
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
            break
            
    # Final refinement pass with even finer grid around current thresholds
    if _time.time() - t_start < _DEADLINE - 10.0:
        for t in range(n_dates):
            if _time.time() - t_start > _DEADLINE - 5.0:
                break
            
            # Create a finer grid around the current threshold
            current = thresholds[t]
            # Look at neighbors in the candidate list
            idx_in_candidates = np.argmin(np.abs(candidate_thresholds[t] - current))
            # Take a window of candidates around this
            window_size = 20
            start_idx = max(0, idx_in_candidates - window_size)
            end_idx = min(len(candidate_thresholds[t]), idx_in_candidates + window_size + 1)
            fine_candidates = candidate_thresholds[t][start_idx:end_idx]
            
            # Also add interpolated points
            if len(fine_candidates) > 1:
                fine_interp = np.linspace(fine_candidates[0], fine_candidates[-1], 50)
                fine_candidates = np.unique(np.concatenate([fine_candidates, fine_interp]))
            
            best_local_score = -np.inf
            best_local_thresh = thresholds[t]
            
            for c in fine_candidates:
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
    
    return ExercisePolicy(thresholds, r, dt)
# EVOLVE-BLOCK-END
