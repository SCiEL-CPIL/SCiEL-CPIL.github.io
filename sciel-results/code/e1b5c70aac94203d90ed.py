"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    # Extract parameters from contract if available, else use defaults
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_dates = 12

    if contract is not None and isinstance(contract, dict):
        K = contract.get('K', K)
        r = contract.get('r', r)
        sigma = contract.get('sigma', sigma)
        T = contract.get('T', T)

    tp = np.asarray(training_paths, dtype=np.float64)

    # Handle orientation
    if tp.ndim == 2:
        if tp.shape[0] == 4096:
            paths = tp
            n_paths, n_times = tp.shape
        elif tp.shape[1] == 4096:
            paths = tp.T
            n_times, n_paths = tp.shape
        else:
            paths = tp
            n_paths, n_times = tp.shape
    else:
        paths = tp
        n_paths, n_times = paths.shape

    dt = T / n_dates
    discount = np.exp(-r * dt)

    # We assume paths have columns for t=0, t=1/12, ..., t=12/12
    # Exercise dates are indices 1 to 12.
    # Column 0 is t=0 (no exercise).
    
    # Backward induction
    # V[i] is the value at exercise date i (index i in paths)
    # At the last date (n_dates), we must exercise (or hold to maturity, but here maturity is the last date)
    # Actually, for a Bermudan option, the last date is maturity.
    # Payoff = max(K - S, 0)
    
    S_last = paths[:, n_dates]
    V_next = np.maximum(K - S_last, 0.0)
    
    # We will store thresholds for each exercise date i in 1..n_dates
    # threshold[i] = max S such that exercise is optimal
    thresholds = np.zeros(n_dates + 1)
    
    # At maturity, we always exercise if payoff > 0.
    # The "threshold" at maturity is effectively infinity (we exercise for any S where payoff > 0)
    # But for the policy, at t=1, we just take the payoff.
    # For the purpose of the `exercise` method, if time == 1.0, we should return True for all S where K-S > 0?
    # Actually, the problem says "12 monthly exercise dates (no time-0 exercise)".
    # At the final date, it's just the payoff.
    # Let's set threshold[n_dates] to a large number so that `S <= threshold` is true for all reasonable S?
    # No, the payoff is max(K-S, 0). If S > K, payoff is 0.
    # So at maturity, we exercise if S < K.
    # So threshold[n_dates] = K.
    thresholds[n_dates] = K

    # Iterate backwards from n_dates-1 down to 1
    for i in range(n_dates - 1, 0, -1):
        if time.time() > deadline - 10:
            # Fallback: use simple heuristic
            # For puts, threshold decreases as we get earlier? No, usually increases?
            # Early exercise is more valuable when interest rates are high and stock is low.
            # Let's just break and use whatever we have or a default.
            break
            
        S_i = paths[:, i]
        immediate = np.maximum(K - S_i, 0.0)
        
        # Continuation value: discount * E[V_{i+1} | S_i]
        # We need to estimate E[V_{i+1} | S_i].
        # The current program uses `discount * V` which is WRONG because V is the vector of values at i+1 for the *same* paths, not the expectation conditional on S_i.
        # The inspiration 2 uses binning. This is a structural improvement.
        
        # Let's use a simple binning approach to estimate the conditional expectation.
        # Determine bins based on S_i distribution
        # To be fast, use a fixed number of bins, e.g., 50 or 100.
        # Since S is GBM, it's log-normal.
        
        # We need to map S_{i+1} values to bins defined by S_i? Or just bin S_i and average V_{i+1} for those paths?
        # Yes, bin S_i, and for each bin, compute the mean of V_next (which is V at i+1 for those paths).
        
        # Determine bin edges.
        # Use percentiles of S_i to ensure some data in each bin?
        # Or fixed grid?
        # Fixed grid is faster but might have empty bins.
        # Let's use a quantile-based binning for robustness, but it's slower.
        # With 4096 paths, sorting is fast.
        
        n_bins = 64
        # Create bins
        # We want to cover the range of S_i
        s_min = np.min(S_i)
        s_max = np.max(S_i)
        if s_max == s_min:
            s_max = s_min + 1.0
            
        # Linear bins
        bin_edges = np.linspace(s_min, s_max, n_bins + 1)
        
        # Bin indices for S_i
        bin_indices = np.digitize(S_i, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # Compute mean V_next for each bin
        V_cont_est = np.zeros(n_bins)
        counts = np.zeros(n_bins)
        
        # Use np.bincount for speed
        # We need sum(V_next) per bin
        sum_V = np.bincount(bin_indices, weights=V_next, minlength=n_bins)
        counts = np.bincount(bin_indices, minlength=n_bins)
        
        for b in range(n_bins):
            if counts[b] > 0:
                V_cont_est[b] = sum_V[b] / counts[b]
            else:
                # Extrapolation: if no data, use neighboring bin or 0
                # For a put, value is decreasing in S.
                # If b is high (high S), value should be low.
                # If b is low (low S), value should be high.
                if b > 0:
                    V_cont_est[b] = V_cont_est[b-1]
                elif b < n_bins - 1:
                    V_cont_est[b] = V_cont_est[b+1]
                else:
                    V_cont_est[b] = 0.0
                    
        # Interpolate V_cont for each S_i
        # We have bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        
        # Map S_i to V_cont_est using interpolation
        # S_i might be outside [s_min, s_max] due to floating point? No, s_min/s_max are from S_i.
        # But digitize puts them in bins.
        # We can just use the bin index to get the estimated continuation value.
        # This is a step function.
        V_cont = V_cont_est[bin_indices]
        
        # Continuation value = discount * V_cont
        V_cont_disc = discount * V_cont
        
        # Exercise if immediate >= V_cont_disc
        exercise_mask = immediate >= V_cont_disc
        
        # Determine threshold: max S where exercise is optimal
        # Sort S_i and exercise_mask
        # Since exercise is optimal for low S, we can just find the max S among those exercised.
        if np.any(exercise_mask):
            thresholds[i] = np.max(S_i[exercise_mask])
        else:
            thresholds[i] = 0.0

    # Check time
    if time.time() > deadline - 5:
        # If we broke out early, fill remaining thresholds with a heuristic
        # Heuristic: threshold decreases as i decreases?
        # Actually, for puts, the critical price for early exercise is lower than K.
        # It is higher for earlier dates? No.
        # Let's just use K * 0.9 for all remaining if we haven't computed them.
        for i in range(1, n_dates):
            if thresholds[i] == 0.0:
                thresholds[i] = K * 0.9

    class ExercisePolicy:
        def __init__(self, thresholds, K):
            self.thresholds = thresholds
            self.K = K

        def exercise(self, current_state, time):
            cs = np.asarray(current_state, dtype=np.float64)
            
            # Handle scalar time or array time
            if np.isscalar(time) or np.ndim(time) == 0:
                t = float(time)
                if t <= 0:
                    return np.zeros(len(cs), dtype=bool)
                if t > 1.0:
                    t = 1.0
                # Map to index 1..12
                # t=1/12 -> 1, t=1 -> 12
                i = int(round(t * 12))
                i = max(1, min(12, i))
                threshold = self.thresholds[i]
                return (cs <= threshold).astype(bool)
            else:
                time_arr = np.asarray(time, dtype=np.float64)
                result = np.zeros(len(cs), dtype=bool)
                for idx in range(len(cs)):
                    t = float(time_arr[idx])
                    if t <= 0:
                        continue
                    if t > 1.0:
                        t = 1.0
                    i = int(round(t * 12))
                    i = max(1, min(12, i))
                    threshold = self.thresholds[i]
                    if cs[idx] <= threshold:
                        result[idx] = True
                return result

    return ExercisePolicy(thresholds, K)
# EVOLVE-BLOCK-END
