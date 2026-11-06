"""
Bermudan put exercise policy via backward induction on quantile-binned states.

Structural change: Instead of optimizing a fixed threshold per date (which is
essentially a 12-parameter heuristic), we use backward induction on a
quantile-binned state space to approximate the true optimal stopping rule.
This is a different modelling family: it explicitly estimates continuation
value and compares it to immediate exercise payoff at each node.

The policy is represented as a lookup table: for each date i and each state
bin j, we store a boolean 'exercise' decision. During evaluation, we map
the current spot to a bin index and return the stored decision.

We use the training paths to build the state grid (quantiles) and to
estimate conditional expectations for the backward induction.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
import math

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # Extract parameters
    if contract is None:
        contract = {}
    
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)
    dt = T / n_dates

    # training_paths: shape (n_dates, n_paths)
    # Index 0 is t_1 (first exercise date), index 11 is t_12 (maturity)
    
    n_dates_actual = training_paths.shape[0]
    n_paths = training_paths.shape[1]
    
    # Number of state bins for backward induction
    # We want a reasonable number of bins to capture the distribution
    # but not so many that we have sparse data in each bin.
    n_bins = 50
    
    # Build state grid from training paths using quantiles
    # We'll use a single grid for all dates, based on the pooled distribution
    all_prices = training_paths.flatten()
    # Use quantiles to create bins
    # Ensure we have enough data points
    if len(all_prices) > n_bins:
        bin_edges = np.quantile(all_prices, np.linspace(0, 1, n_bins + 1))
        # Ensure unique edges
        # If there are duplicates, we might have fewer bins
        unique_edges = []
        for e in bin_edges:
            if not unique_edges or e > unique_edges[-1] + 1e-12:
                unique_edges.append(e)
        # Add a small buffer at the ends
        if len(unique_edges) >= 2:
            unique_edges[0] = unique_edges[0] - 1e-6
            unique_edges[-1] = unique_edges[-1] + 1e-6
        bin_edges = unique_edges
        n_bins_actual = len(bin_edges) - 1
    else:
        # Fallback: uniform grid
        min_p = all_prices.min() if len(all_prices) > 0 else 0.0
        max_p = all_prices.max() if len(all_prices) > 0 else 2*K
        bin_edges = np.linspace(min_p, max_p, n_bins + 1)
        n_bins_actual = n_bins

    # Map each price to a bin index
    # For each date, create a matrix of bin indices: shape (n_dates, n_paths)
    bin_indices = np.zeros((n_dates_actual, n_paths), dtype=int)
    for i in range(n_dates_actual):
        # np.digitize returns indices from 1 to len(bin_edges)-1
        # We want 0 to n_bins_actual-1
        bin_indices[i] = np.digitize(training_paths[i], bin_edges, right=True) - 1
        # Clip to valid range
        bin_indices[i] = np.clip(bin_indices[i], 0, n_bins_actual - 1)

    # Backward induction
    # V[i][b] = optimal value at date i, state bin b
    # At maturity (i = n_dates_actual - 1), no future dates, so continuation value = 0
    # Actually, at the last date, you can exercise or not. If you don't exercise, payoff is 0.
    # So V[last][b] = max(K - S_b, 0)
    
    # We'll work with the bin centers as representative state values
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # V[i] is an array of length n_bins_actual
    # Initialize for the last date
    # At the last date, exercise if ITM
    # V_last[b] = max(K - bin_centers[b], 0)
    V = np.zeros((n_dates_actual, n_bins_actual))
    V[-1] = np.maximum(K - bin_centers, 0.0)
    
    # Discount factor per step
    disc = math.exp(-r * dt)
    
    # Backward induction for i from n_dates_actual - 2 down to 0
    for i in range(n_dates_actual - 2, -1, -1):
        # For each bin b at date i, we need to estimate:
        # 1. Immediate exercise payoff: max(K - S_b, 0)
        # 2. Continuation value: E[disc * V[i+1][b']] | S_i = S_b
        #    where b' is the bin index at date i+1 given S_i = S_b
        
        # To estimate continuation value, we use the training paths.
        # For paths that are in bin b at date i, look at their bin index at date i+1.
        # Continuation value for bin b = disc * mean(V[i+1][bin_indices[i+1, paths_in_bin_b]])
        
        # Build a lookup: for each bin b, collect the bin indices at next date
        # We can vectorize this using bincount or similar.
        
        # Let's create an array of continuation values for each path
        # cont_value_path[j] = disc * V[i+1][bin_indices[i+1, j]]
        cont_values = disc * V[i+1][bin_indices[i+1]]
        
        # Now, for each bin b, average cont_values over paths in bin b
        # Use np.bincount
        # We need to be careful with empty bins
        bin_sum = np.bincount(bin_indices[i], weights=cont_values, minlength=n_bins_actual)
        bin_count = np.bincount(bin_indices[i], minlength=n_bins_actual)
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            cont_bin_values = np.where(bin_count > 0, bin_sum / np.maximum(bin_count, 1), 0.0)
        
        # For bins with no data, we can interpolate or use 0. 
        # A better approach: if a bin has no data, use the value from the nearest bin with data.
        # For now, let's use 0 as a conservative estimate, but this might be bad.
        # Alternative: use a smoother approach. Let's try to fill empty bins with interpolation.
        
        # Identify empty bins
        empty_mask = bin_count == 0
        if np.any(empty_mask) and np.any(~empty_mask):
            # Interpolate from non-empty bins
            non_empty_indices = np.where(~empty_mask)[0]
            non_empty_values = cont_bin_values[non_empty_indices]
            # Use linear interpolation
            cont_bin_values[empty_mask] = np.interp(
                np.where(empty_mask)[0], 
                non_empty_indices, 
                non_empty_values
            )
        
        # Immediate exercise payoff for each bin
        exercise_payoff = np.maximum(K - bin_centers, 0.0)
        
        # Optimal value: max(exercise, continuation)
        V[i] = np.maximum(exercise_payoff, cont_bin_values)
        
        # Also store the decision: exercise if exercise_payoff >= continuation
        # We'll store the decision as a separate array
        # Actually, we can derive the decision later from V[i] and V[i+1]
    
    # Now, we need to create the exercise decision table
    # exercise_table[i][b] = True if we should exercise at date i, bin b
    exercise_table = np.zeros((n_dates_actual, n_bins_actual), dtype=bool)
    
    # At the last date, exercise if ITM
    exercise_table[-1] = (K - bin_centers) > 0
    
    # For earlier dates, exercise if immediate payoff >= continuation value
    # We need to recompute the continuation values or store them.
    # Let's recompute for clarity.
    
    for i in range(n_dates_actual - 2, -1, -1):
        cont_values = disc * V[i+1][bin_indices[i+1]]
        bin_sum = np.bincount(bin_indices[i], weights=cont_values, minlength=n_bins_actual)
        bin_count = np.bincount(bin_indices[i], minlength=n_bins_actual)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            cont_bin_values = np.where(bin_count > 0, bin_sum / np.maximum(bin_count, 1), 0.0)
        
        empty_mask = bin_count == 0
        if np.any(empty_mask) and np.any(~empty_mask):
            non_empty_indices = np.where(~empty_mask)[0]
            non_empty_values = cont_bin_values[non_empty_indices]
            cont_bin_values[empty_mask] = np.interp(
                np.where(empty_mask)[0], 
                non_empty_indices, 
                non_empty_values
            )
        
        exercise_payoff = np.maximum(K - bin_centers, 0.0)
        exercise_table[i] = exercise_payoff >= cont_bin_values
    
    # Check time
    if time.time() > deadline:
        pass  # Proceed with what we have
    
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, exercise_table, bin_edges):
            self.exercise_table = exercise_table
            self.bin_edges = bin_edges
            self.n_bins = len(bin_edges) - 1
            
        def exercise(self, current_state, time_index):
            """
            current_state: np.array of spot prices
            time_index: int (0 to n_dates-1)
            Returns: boolean vector of same shape as current_state
            """
            if time_index < 0 or time_index >= self.exercise_table.shape[0]:
                return np.zeros_like(current_state, dtype=bool)
            
            # Map current_state to bin indices
            bin_idx = np.digitize(current_state, self.bin_edges, right=True) - 1
            bin_idx = np.clip(bin_idx, 0, self.n_bins - 1)
            
            return self.exercise_table[time_index][bin_idx]
    
    return ExercisePolicy(exercise_table, bin_edges)

# EVOLVE-BLOCK-END
