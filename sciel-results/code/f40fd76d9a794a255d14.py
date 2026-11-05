"""Bermudan put exercise-policy research: Binned Backward Induction with Savitzky-Golay Smoothing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.signal import savgol_filter

def fit_exercise_policy(training_paths, contract):
    """
    Fit an exercise policy for a Bermudan put option using binned backward induction
    with Savitzky-Golay smoothing and monotonicity enforcement to reduce Monte Carlo
    noise and improve the accuracy of the exercise boundary.
    """
    start_time = time.time()
    
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)
    
    n_paths, n_cols = training_paths.shape
    
    # Determine exercise indices
    start_idx = n_cols - n_dates
    
    dt = T / n_dates
    disc = np.exp(-r * dt)
    
    # Binning setup
    all_s = training_paths[:, start_idx:]
    s_min = np.min(all_s)
    s_max = np.max(all_s)
    
    # Use 400 bins for good resolution
    n_bins = 400
    s_min_safe = max(s_min, 1e-6)
    s_max_safe = s_max * 1.02 if s_max > 0 else 1.0
    
    # Create log-spaced bin edges for better resolution across price ranges
    log_s_min = np.log(s_min_safe)
    log_s_max = np.log(s_max_safe)
    log_edges = np.linspace(log_s_min, log_s_max, n_bins + 1)
    s_edges = np.exp(log_edges)
    
    # Calculate bin centers for interpolation
    bin_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    
    # Payoff at last date
    s_last = training_paths[:, n_cols - 1]
    payoff_last = np.maximum(K - s_last, 0.0)
    V = payoff_last.copy()
    
    # Store continuation values per date
    cont_vals_per_date = []
    
    # Backward induction
    for t_rel in range(n_dates - 2, -1, -1):
        col_idx = start_idx + t_rel
        s_t = training_paths[:, col_idx]
        
        # Determine bin for each path
        bin_idx = np.digitize(s_t, s_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        
        # Calculate average V for each bin
        sum_v = np.zeros(n_bins)
        counts = np.zeros(n_bins)
        
        np.add.at(sum_v, bin_idx, V)
        np.add.at(counts, bin_idx, 1)
        
        # Handle empty bins
        mask = counts > 0
        cont_vals = np.zeros(n_bins)
        cont_vals[mask] = sum_v[mask] / counts[mask]
        
        # Fill empty bins using interpolation from nearest non-empty neighbors
        empty_mask = ~mask
        if np.any(empty_mask):
            non_empty_idx = np.where(mask)[0]
            if len(non_empty_idx) > 1:
                interp_vals = np.interp(np.arange(n_bins), non_empty_idx, cont_vals[non_empty_idx])
                cont_vals[empty_mask] = interp_vals[empty_mask]
            elif len(non_empty_idx) == 1:
                cont_vals[empty_mask] = cont_vals[non_empty_idx[0]]
        
        # Apply Savitzky-Golay filter to smooth the continuation values
        window_size = 25
        poly_order = 3
        
        if n_bins > window_size:
            try:
                cont_vals_smoothed = savgol_filter(cont_vals, window_size, poly_order)
            except Exception:
                kernel = np.ones(window_size) / window_size
                cont_vals_smoothed = np.convolve(cont_vals, kernel, mode='same')
        else:
            cont_vals_smoothed = cont_vals.copy()
        
        # Enforce monotonicity: for a put, continuation value should be non-increasing in S
        # Apply isotonic regression (pool adjacent violators) - forward pass
        for i in range(1, n_bins):
            if cont_vals_smoothed[i] > cont_vals_smoothed[i-1]:
                cont_vals_smoothed[i] = cont_vals_smoothed[i-1]
        
        # Also enforce that continuation value >= 0
        cont_vals_smoothed = np.maximum(cont_vals_smoothed, 0.0)
        
        # Ensure continuation value doesn't exceed intrinsic value at maturity
        # For a put, max continuation value is K (at S=0)
        cont_vals_smoothed = np.minimum(cont_vals_smoothed, K)
        
        cont_vals_per_date.append(cont_vals_smoothed)
        
        # Calculate new V
        payoff_t = np.maximum(K - s_t, 0.0)
        
        # Use the smoothed continuation values for the next step's binning
        cont_path = cont_vals_smoothed[bin_idx]
        disc_cont = disc * cont_path
        V = np.maximum(payoff_t, disc_cont)
        
        # Time check
        if time.time() - start_time > 130:
            break
    
    # Reverse to match time order
    cont_vals_per_date.reverse()
    
    # Pad if we broke out early
    if len(cont_vals_per_date) < n_dates - 1:
        while len(cont_vals_per_date) < n_dates - 1:
            cont_vals_per_date.append(np.zeros(n_bins))
    
    class ExercisePolicy:
        def __init__(self, K, disc, s_edges, bin_centers, n_bins, cont_vals_per_date, n_dates):
            self.K = K
            self.disc = disc
            self.s_edges = s_edges
            self.bin_centers = bin_centers
            self.n_bins = n_bins
            self.cont_vals = cont_vals_per_date
            self.n_dates = n_dates
            
        def exercise(self, current_state, time):
            if time < 1:
                return np.zeros(len(current_state), dtype=bool)
            
            t_rel = int(time) - 1
            
            if t_rel < 0 or t_rel >= self.n_dates:
                return np.zeros(len(current_state), dtype=bool)
            
            if t_rel == self.n_dates - 1:
                return current_state < self.K
            
            # Get continuation values
            cont_vals = self.cont_vals[t_rel]
            
            # Linear interpolation on bin centers
            disc_cont = self.disc * np.interp(current_state, self.bin_centers, cont_vals)
            
            payoff = np.maximum(self.K - current_state, 0.0)
            
            return payoff >= disc_cont
    
    return ExercisePolicy(K, disc, s_edges, bin_centers, n_bins, cont_vals_per_date, n_dates)

# EVOLVE-BLOCK-END
