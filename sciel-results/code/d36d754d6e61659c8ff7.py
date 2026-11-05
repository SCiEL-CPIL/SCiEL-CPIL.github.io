import numpy as np
import time as _time
from numpy.polynomial import chebyshev as C
from scipy.signal import savgol_filter

class ExercisePolicy:
    def __init__(self, thresholds):
        self.thresholds = thresholds
    
    def exercise(self, current_state, time_index):
        if time_index < 0 or time_index >= len(self.thresholds):
            return np.zeros(len(current_state), dtype=bool)
        return current_state < self.thresholds[time_index]

def fit_exercise_policy(training_paths, contract):
    start_time = _time.time()
    deadline = start_time + 110  # Safety margin before 150s kill
    
    paths = np.asarray(training_paths, dtype=np.float64)
    n_paths, n_steps = paths.shape
    n_exercise = 12  # exercise dates at steps 1..12
    
    dt = 1.0 / 12
    r = 0.05
    K = 100.0
    discount = np.exp(-r * dt)
    
    # V[j, i] stores the value of the option at exercise date j (step j+1) for path i
    V = np.zeros((n_exercise, n_paths))
    thresholds = np.zeros(n_exercise)
    
    # Use Chebyshev polynomial regression with Savitzky-Golay smoothing
    # Chebyshev polynomials are more stable than power basis for high degrees
    # Smoothing reduces noise in the targets before fitting
    cheb_degree = 4
    savgol_window = 51
    savgol_polyorder = 3
    coeffs = [None] * n_exercise
    
    # Normalization for Chebyshev: map S in [0, 200] to [-1, 1]
    S_min = 0.0
    S_max = 200.0
    
    def normalize_s(s):
        return 2 * (s - S_min) / (S_max - S_min) - 1
    
    # Last exercise date: j = 11 (index in V), step 12 in paths
    S_last = paths[:, 12]
    V[11, :] = np.maximum(K - S_last, 0.0)
    thresholds[11] = K
    
    # Backward induction for j = 10 down to 0
    for j in range(10, -1, -1):
        if _time.time() > deadline:
            break
            
        S_curr = paths[:, j+1]
        
        # Target for regression: Discounted value of next step
        targets = discount * V[j+1, :]
        
        # Sort by S for smoothing
        sort_idx = np.argsort(S_curr)
        S_sorted = S_curr[sort_idx]
        T_sorted = targets[sort_idx]
        
        # Apply Savitzky-Golay smoothing to targets
        if len(T_sorted) > savgol_window:
            T_smoothed = savgol_filter(T_sorted, savgol_window, savgol_polyorder)
        else:
            T_smoothed = T_sorted
            
        # Fit Chebyshev polynomial on smoothed data
        S_norm = normalize_s(S_sorted)
        try:
            c = C.chebfit(S_norm, T_smoothed, cheb_degree)
            coeffs[j] = c
        except:
            # Fallback to lower degree or constant
            try:
                c = C.chebfit(S_norm, T_smoothed, 2)
            except:
                c = C.chebfit(S_norm, T_smoothed, 0)
            coeffs[j] = c
            
        # Evaluate continuation value for all paths at current step
        S_curr_norm = normalize_s(S_curr)
        cont_values = C.chebval(S_curr_norm, coeffs[j])
        # Ensure continuation value is non-negative
        cont_values = np.maximum(cont_values, 0.0)
        
        # Current Payoff
        payoff = np.maximum(K - S_curr, 0.0)
        
        # Optimal Value
        V[j, :] = np.maximum(payoff, cont_values)
        
        # Extract Threshold for this step immediately
        if coeffs[j] is not None:
            # We need to find S such that K - S = P(S) for S < K
            # Let f(S) = K - S - P(S)
            # We want the largest root in [0, K]
            
            # Create a finer grid for better precision
            S_grid = np.linspace(0, K, 1000)
            S_grid_norm = normalize_s(S_grid)
            P_grid = C.chebval(S_grid_norm, coeffs[j])
            P_grid = np.maximum(P_grid, 0.0)
            
            f_vals = K - S_grid - P_grid
            
            # Find sign changes
            signs = np.sign(f_vals)
            signs[signs == 0] = 1 
            
            changes = np.where(np.diff(signs) != 0)[0]
            
            if len(changes) > 0:
                # Take the last change (highest S where we still want to exercise)
                idx = changes[-1]
                # Linear interpolation for better precision
                x1, x2 = S_grid[idx], S_grid[idx+1]
                y1, y2 = f_vals[idx], f_vals[idx+1]
                if y1 == y2:
                    thresh = x1
                else:
                    thresh = x1 - y1 * (x2 - x1) / (y2 - y1)
                    
                # Clip to valid range
                thresholds[j] = np.clip(thresh, 0, K)
            else:
                # No sign change? 
                if f_vals[0] < 0:
                    thresholds[j] = 0.0
                else:
                    thresholds[j] = K
        else:
            thresholds[j] = K

    return ExercisePolicy(thresholds)
