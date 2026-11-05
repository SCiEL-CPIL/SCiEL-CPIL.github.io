"""Bermudan put exercise-policy research: Binned Backward Induction."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    """
    Fit an exercise policy for a Bermudan put option using binned backward induction.
    
    Parameters:
    - training_paths: array of shape (n_paths, n_dates) with simulated spot prices.
      Note: The problem states "12 monthly exercise dates (no time-0 exercise)".
      Usually, training_paths includes time 0. We need to handle indices carefully.
      If n_dates=12, the exercise dates are indices 1..12 (or 0..11 depending on convention).
      Contract says "no time-0 exercise", so we exercise at steps 1 to 12.
    - contract: dict with contract parameters.
    
    Returns:
    - An object with exercise(current_state, time) method.
    """
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)
    
    # training_paths shape: (n_paths, n_steps)
    # If n_dates=12, and no time-0 exercise, the paths likely have 13 columns (t=0 to t=12)
    # or 12 columns (t=1 to t=12).
    # The prompt says "12 monthly exercise dates".
    # Let's assume training_paths has columns for all time steps involved.
    # Standard convention: training_paths[i, j] is spot at time j.
    # If there are 12 exercise dates, and no exercise at t=0, the dates are t_1, ..., t_12.
    # The paths likely include t_0 to t_12, so shape is (N, 13).
    # Or they might only provide the exercise dates? "12 monthly exercise dates".
    # Let's inspect the shape.
    
    n_paths, n_cols = training_paths.shape
    
    # Determine the exercise indices.
    # If n_cols == 13, indices 1..12 are exercise dates.
    # If n_cols == 12, indices 0..11 are exercise dates.
    # The contract says "no time-0 exercise".
    # Let's assume the standard: t=0 is initial, t=1..12 are exercise dates.
    # So we care about columns 1 to n_cols-1.
    
    # Let's verify by checking if n_cols matches n_dates + 1 or n_dates.
    # Given "12 monthly exercise dates", it's safest to assume the last n_dates columns are exercise dates.
    # If n_cols == 13, exercise at 1, 2, ..., 12.
    # If n_cols == 12, exercise at 0, 1, ..., 11? No, "no time-0".
    # Let's assume the provided paths cover the full timeline t=0..T.
    # We will perform backward induction on the last `n_dates` columns.
    
    start_idx = n_cols - n_dates
    # Exercise dates are columns start_idx to n_cols-1.
    
    dt = T / n_dates
    disc = np.exp(-r * dt)
    
    # Binning setup
    # Spot prices in GBM are log-normal. Log-binning is more uniform.
    # Determine range of S in training data for binning.
    all_s = training_paths[:, start_idx:]
    s_min = np.min(all_s)
    s_max = np.max(all_s)
    
    # Use log-spaced bins for better resolution at low prices (where put is ITM)
    n_bins = 50
    # Ensure s_min > 0 for log
    s_min = max(s_min, 1e-6)
    log_s_min = np.log(s_min)
    log_s_max = np.log(s_max)
    
    # Create bin edges
    log_edges = np.linspace(log_s_min, log_s_max, n_bins + 1)
    s_edges = np.exp(log_edges)
    
    # Pre-compute bin indices for all paths and all exercise dates
    # This is expensive if done naively, but vectorized it's fine.
    # bin_indices[t, i] = bin index of path i at date t (relative to start_idx)
    # We'll compute this on the fly in the backward loop to save memory, 
    # or pre-compute if fast enough. 4096 * 12 is small.
    
    # V is the value function, shape (n_paths,)
    # Payoff at last date (index n_dates - 1 relative to start)
    # The last exercise date is at column n_cols - 1.
    s_last = training_paths[:, n_cols - 1]
    payoff_last = np.maximum(K - s_last, 0.0)
    
    # At the last date, we always exercise if payoff > 0 (continuation is 0)
    V = payoff_last.copy()
    
    # Store exercise decisions for each date
    # exercise_rules[t_rel] will be a function or data structure to decide exercise
    # We store the continuation values per bin for each date, and the payoff per bin?
    # Actually, we just need to decide: Exercise if Payoff >= Disc * ContVal.
    # Payoff depends on S. ContVal depends on S (via bin).
    # So for a given S and date t, we need:
    # 1. Bin index b = bin(S)
    # 2. ContVal_b = average V_{t+1} for paths in bin b at time t.
    # 3. Payoff = max(K-S, 0)
    # 4. Exercise if Payoff >= disc * ContVal_b
    
    # We will store for each relative date t (0 to n_dates-2):
    # - bin_edges (global)
    # - cont_vals[t]: array of shape (n_bins,) containing the average V_{t+1} for each bin.
    # Note: V_{t+1} is the value at the NEXT date.
    
    cont_vals_per_date = []
    
    # Backward induction
    # t_rel goes from n_dates - 2 down to 0.
    # Date index in training_paths: start_idx + t_rel
    # Next date index: start_idx + t_rel + 1
    
    for t_rel in range(n_dates - 2, -1, -1):
        col_idx = start_idx + t_rel
        s_t = training_paths[:, col_idx]
        
        # Determine bin for each path at this time
        # np.digitize returns 1..n_bins, but we want 0..n_bins-1
        # Clip to valid range
        bin_idx = np.digitize(s_t, s_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        
        # Calculate average V (which is V_{t+1}) for each bin
        # V currently holds the values at the next date (start_idx + t_rel + 1)
        # Wait, V is updated in the loop.
        # Initially V = V_last.
        # At t_rel = n_dates - 2, we compute CV using V_last.
        # Then we update V to V_t.
        
        sum_v = np.zeros(n_bins)
        counts = np.zeros(n_bins)
        
        # Efficient accumulation
        np.add.at(sum_v, bin_idx, V)
        np.add.at(counts, bin_idx, 1)
        
        # Handle empty bins: if count is 0, set cont val to 0? 
        # Or interpolate? For safety, if no paths, assume continuation is 0 
        # (which leads to exercise if ITM, which is correct for deep ITM, 
        # but for OTM it might be wrong. However, with 4096 paths and 50 bins, 
        # empty bins are unlikely for central prices. For extreme prices, 
        # put is deep ITM (exercise) or deep OTM (don't exercise).
        # If deep OTM (S >> K), V is likely 0. If deep ITM (S << K), V is likely K-S.
        # Let's use a fallback: if count == 0, use the global mean of V? 
        # Or simply 0? 
        # Actually, for a put, if S is very high, V is 0. If S is very low, V is K-S.
        # Let's just set to 0 for now, it's a conservative estimate for continuation.
        
        mask = counts > 0
        cont_vals = np.zeros(n_bins)
        cont_vals[mask] = sum_v[mask] / counts[mask]
        
        # Store continuation values for this date
        cont_vals_per_date.append(cont_vals)
        
        # Calculate new V for this date
        # Payoff at this date
        payoff_t = np.maximum(K - s_t, 0.0)
        
        # Continuation value for each path
        cont_path = cont_vals[bin_idx]
        
        # Discounted continuation
        disc_cont = disc * cont_path
        
        # Optimal value: max(payoff, disc_cont)
        V = np.maximum(payoff_t, disc_cont)
        
    # Reversing cont_vals_per_date to match time order 0..n_dates-2
    cont_vals_per_date.reverse()
    
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, K, disc, s_edges, n_bins, cont_vals_per_date, n_dates, start_idx):
            self.K = K
            self.disc = disc
            self.s_edges = s_edges
            self.n_bins = n_bins
            self.cont_vals = cont_vals_per_date
            self.n_dates = n_dates
            self.start_idx = start_idx
            
        def exercise(self, current_state, time):
            """
            current_state: array of spot prices at current time.
            time: current time index. 
                   The contract specifies 12 monthly exercise dates.
                   'time' likely corresponds to the index of the exercise date in the sequence 
                   or the global time index. 
                   Given the ambiguity, we assume 'time' is the relative index of the exercise date (0 to 11)
                   or we map global time to relative.
                   
                   Let's assume 'time' is the index in the sequence of exercise dates (0-based).
                   If the evaluator passes global time indices (1..12), we might need to adjust.
                   However, standard interfaces for such tasks often pass the step index.
                   Let's handle both by checking if time is within 0..11 or 1..12.
                   If time == 0, it's likely t=0 (no exercise).
                   If time is 1..12, it's the global index.
                   
                   Let's assume 'time' is the global index (1..12) based on "no time-0 exercise".
                   So relative index t_rel = time - 1.
                   If time is passed as 0-based relative (0..11), t_rel = time.
                   
                   To be robust:
                   If time < 1: return all False.
                   If time <= self.n_dates: t_rel = time - 1.
                   Else: t_rel = time (fallback).
            """
            if time < 1:
                return np.zeros(len(current_state), dtype=bool)
            
            # Map global time to relative index
            # Assuming time is 1..12 for 12 dates
            t_rel = time - 1
            
            if t_rel < 0 or t_rel >= self.n_dates - 1:
                # If it's the last date, we always exercise if ITM
                # But our loop only stored cont_vals for 0..n_dates-2.
                # For the last date, V = Payoff. So we exercise if Payoff > 0.
                # Wait, at the last date, there is no continuation.
                # So exercise if max(K-S, 0) > 0.
                if t_rel == self.n_dates - 1:
                    return current_state < self.K
                return np.zeros(len(current_state), dtype=bool)
            
            # Get bin indices
            bin_idx = np.digitize(current_state, self.s_edges) - 1
            bin_idx = np.clip(bin_idx, 0, self.n_bins - 1)
            
            # Get continuation values
            cont_vals = self.cont_vals[t_rel]
            cont_path = cont_vals[bin_idx]
            
            # Discounted continuation
            disc_cont = self.disc * cont_path
            
            # Payoff
            payoff = np.maximum(self.K - current_state, 0.0)
            
            # Exercise if Payoff >= Disc * Cont
            return payoff >= disc_cont
            
    return ExercisePolicy(K, disc, s_edges, n_bins, cont_vals_per_date, n_dates, start_idx)

# EVOLVE-BLOCK-END
