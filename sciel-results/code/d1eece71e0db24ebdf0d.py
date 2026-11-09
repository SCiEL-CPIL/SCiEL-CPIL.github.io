"""Bermudan exercise-policy research: combined grid-based backward induction with put-call parity refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import roots_hermite
from scipy.stats import norm

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using grid-based backward induction
    with Gauss-Hermite quadrature, refined via put-call parity where beneficial.
    
    training_paths: array of shape (n_paths, n_time_steps) with spot prices
    contract: dict with keys 'K', 'r', 'sigma', 'T', 'n_steps'
    
    Returns an object with method exercise(current_state, time) -> boolean array
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    # Extract contract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_steps = contract.get('n_steps', 12)
    
    training_paths = np.asarray(training_paths, dtype=float)
    n_paths, n_time_points = training_paths.shape
    
    dt = T / n_steps
    discount = np.exp(-r * dt)
    
    # ---- Build a dense log-space price grid ----
    s_min = max(training_paths.min() * 0.5, 1e-4)
    s_max = max(training_paths.max() * 1.5, 2.0 * K)
    
    # High-resolution grid
    n_grid = 4096
    log_edges = np.linspace(np.log(s_min), np.log(s_max), n_grid + 1)
    log_centers = 0.5 * (log_edges[:-1] + log_edges[1:])
    grid = np.exp(log_centers)
    
    # ---- Gauss-Hermite quadrature setup ----
    mu = (r - 0.5 * sigma * sigma) * dt
    s2 = sigma * sigma * dt
    
    n_q = 64
    x, w = roots_hermite(n_q)
    u = mu + np.sqrt(2.0 * s2) * x
    qw = w / np.sqrt(np.pi)
    exp_u = np.exp(u)
    
    # ---- Backward induction for PUT values ----
    V_put = np.zeros((n_steps + 1, n_grid))
    V_put[n_steps] = np.maximum(K - grid, 0.0)
    
    # ---- Backward induction for CALL values (for put-call parity) ----
    V_call = np.zeros((n_steps + 1, n_grid))
    V_call[n_steps] = np.maximum(grid - K, 0.0)
    
    def compute_continuation(V_next, grid_arr, log_centers_arr, exp_u_arr, qw_arr):
        """Compute E[V_next(S_{t+dt}) | S_t] using GH quadrature and log-space interpolation."""
        S_next = np.outer(grid_arr, exp_u_arr)  # (n_grid, n_q)
        log_S_next = np.log(S_next)
        
        # Interpolate V_next onto S_next in log-space
        idx = np.searchsorted(log_centers_arr, log_S_next, side='right') - 1
        idx = np.clip(idx, 0, len(log_centers_arr) - 2)
        
        c0 = V_next[idx]
        c1 = V_next[idx + 1]
        lc0 = log_centers_arr[idx]
        lc1 = log_centers_arr[idx + 1]
        
        t = (log_S_next - lc0) / (lc1 - lc0)
        t = np.clip(t, 0.0, 1.0)
        
        interp = c0 + t * (c1 - c0)
        
        # Weighted sum over quadrature points
        return np.sum(interp * qw_arr[None, :], axis=1)
    
    # Iterate backwards from n_steps-1 to 1
    for k in range(n_steps - 1, 0, -1):
        if time.time() > deadline:
            break
        
        # PUT continuation and optimal value
        cont_put = discount * compute_continuation(V_put[k + 1], grid, log_centers, exp_u, qw)
        ex_put = np.maximum(K - grid, 0.0)
        V_put[k] = np.maximum(ex_put, cont_put)
        
        # CALL continuation and optimal value
        cont_call = discount * compute_continuation(V_call[k + 1], grid, log_centers, exp_u, qw)
        ex_call = np.maximum(grid - K, 0.0)
        V_call[k] = np.maximum(ex_call, cont_call)
    
    # ---- Put-Call Parity Refinement ----
    # For Bermudan options, strict put-call parity does not hold due to different exercise times.
    # However, we can use the relationship: P_early >= P_european - C_european + S0 - K*exp(-rT)
    # More practically, we can compare our computed put value against a lower bound
    # derived from the European put price, and adjust if the grid-based estimate seems too low.
    
    # Compute European put prices for reference (using Black-Scholes)
    def bs_put(S, K, r, sigma, T_rem):
        if T_rem <= 0:
            return np.maximum(K - S, 0.0)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T_rem) / (sigma * np.sqrt(T_rem))
        d2 = d1 - sigma * np.sqrt(T_rem)
        return K * np.exp(-r * T_rem) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    # For each exercise date k, check if our put value is below the European lower bound
    # If so, we know the grid-based estimate may be inaccurate, and we can use
    # the parity-derived value as a floor.
    # P_bermudan(k) >= P_european(T - k*dt) + (time value from early exercise optionality)
    # Actually, P_bermudan >= P_european always (since Bermudan allows more exercise opportunities)
    # So we can use: V_put[k, grid] = max(V_put[k, grid], P_european(T - k*dt))
    
    for k in range(1, n_steps + 1):
        T_rem = T - k * dt
        if T_rem > 0:
            eu_put = bs_put(grid, K, r, sigma, T_rem)
            # The Bermudan put value should be at least the European put value
            # This provides a lower bound that can correct for grid errors
            V_put[k] = np.maximum(V_put[k], eu_put)
    
    # ---- Determine exercise thresholds from the refined put values ----
    exercise_thresholds = np.zeros(n_steps + 1)
    
    for k in range(1, n_steps):
        # Exercise if intrinsic value >= continuation value
        # We stored V[k] = max(exercise, continuation), so we need to check
        # where exercise >= continuation
        ex_grid = np.maximum(K - grid, 0.0)
        cont_grid = discount * compute_continuation(V_put[k + 1], grid, log_centers, exp_u, qw)
        
        # Exercise mask: where ex_grid >= cont_grid
        exercise_mask = ex_grid >= cont_grid
        
        # Find the threshold: highest S where we still exercise
        sort_idx = np.argsort(grid)
        grid_sorted = grid[sort_idx]
        exercise_sorted = exercise_mask[sort_idx]
        
        exercise_indices = np.where(exercise_sorted)[0]
        if len(exercise_indices) > 0:
            last_exercise_idx = exercise_indices[-1]
            if last_exercise_idx < len(grid_sorted) - 1:
                S_ex = grid_sorted[last_exercise_idx]
                S_noex = grid_sorted[last_exercise_idx + 1]
                exercise_thresholds[k] = (S_ex + S_noex) / 2.0
            else:
                exercise_thresholds[k] = grid_sorted[-1] + 1.0
        else:
            exercise_thresholds[k] = 0.0
    
    # At maturity, always exercise if in the money
    exercise_thresholds[n_steps] = K
    
    # ---- Create the policy object ----
    class ExercisePolicy:
        def __init__(self, grid, log_centers, V, n_grid, K, n_steps, thresholds):
            self.grid = grid
            self.log_centers = log_centers
            self.V = V
            self.n_grid = n_grid
            self.K = K
            self.n_steps = n_steps
            self.thresholds = thresholds
            self.dt = T / n_steps
        
        def exercise(self, current_state, time_step):
            """
            current_state: array of spot prices
            time_step: int index or float time
            
            Returns: boolean array, True if exercise
            """
            current_state = np.asarray(current_state, dtype=float)
            try:
                if isinstance(time_step, (int, np.integer)):
                    idx = int(time_step)
                else:
                    idx = int(round(time_step / self.dt))
                
                # Clamp to valid range [1, n_steps]
                idx = max(1, min(idx, self.n_steps))
                
                # At maturity (idx == n_steps), exercise if in the money
                if idx == self.n_steps:
                    return current_state < self.K
                
                # Use threshold-based decision
                threshold = self.thresholds[idx]
                return current_state < threshold
            
            except Exception:
                return np.zeros(len(current_state), dtype=bool)
    
    policy = ExercisePolicy(grid, log_centers, V_put, n_grid, K, n_steps, exercise_thresholds)
    return policy

# EVOLVE-BLOCK-END
