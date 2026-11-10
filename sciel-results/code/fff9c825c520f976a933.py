import numpy as np
import math
import time
from scipy.optimize import minimize

def fit_exercise_policy(training_paths, contract):
    t_start = time.time()
    deadline = t_start + 110.0

    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercises = 12
    
    # 1. Get initial boundaries from a fast binomial tree
    # Use 100 steps for speed
    n_steps_tree = 100
    dt = T / n_steps_tree
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r * dt) - d) / (u - d)
    df = math.exp(-r * dt)
    
    # Exercise steps in tree: 100/12 is not integer. 
    # We map exercise dates to tree steps.
    # Exercise dates are at t = 1/12, 2/12, ..., 12/12.
    # Tree steps are at k*dt.
    # We find the closest tree step for each exercise date.
    
    S_min = 0.1
    S_max = 300.0
    n_grid = 2000
    S_grid = np.linspace(S_min, S_max, n_grid)
    
    V_next = np.maximum(K - S_grid, 0.0)
    
    initial_boundaries = np.zeros(n_exercises + 1)
    # We need to run backward induction on the tree.
    # To make it faster, we only store boundaries at the exercise dates.
    # But we need to iterate through all tree steps.
    
    # Map exercise dates to tree indices
    # Exercise date i is at time (i/12)*T.
    # Tree step k is at time k*dt.
    # k_i = round((i/12)*T / dt) = round(i * n_steps_tree / 12)
    ex_steps = [int(round(i * n_steps_tree / n_exercises)) for i in range(1, n_exercises + 1)]
    ex_step_map = {step: i for i, step in enumerate(ex_steps, 1)}
    
    for step in range(n_steps_tree - 1, 0, -1):
        intrinsic = np.maximum(K - S_grid, 0.0)
        Su = S_grid * u
        Sd = S_grid * d
        V_up = np.interp(Su, S_grid, V_next)
        V_down = np.interp(Sd, S_grid, V_next)
        continuation = df * (p * V_up + (1 - p) * V_down)
        V_current = np.maximum(intrinsic, continuation)
        
        if step in ex_step_map:
            idx = ex_step_map[step]
            diff = intrinsic - continuation
            exercise_mask = diff >= 0
            if np.any(exercise_mask):
                idxs = np.where(exercise_mask)[0]
                last_idx = idxs[-1]
                if last_idx < n_grid - 1:
                    diff0 = diff[last_idx]
                    diff1 = diff[last_idx + 1]
                    if diff1 != diff0:
                        t = diff0 / (diff0 - diff1)
                        initial_boundaries[idx] = S_grid[last_idx] + t * (S_grid[last_idx + 1] - S_grid[last_idx])
                    else:
                        initial_boundaries[idx] = S_grid[last_idx]
                else:
                    initial_boundaries[idx] = S_grid[-1]
            else:
                initial_boundaries[idx] = 0.0
        
        V_next = V_current
    
    initial_boundaries[12] = K
    # Extract the 12 boundaries to optimize
    b0 = initial_boundaries[1:13]
    
    # 2. Define objective function using training paths
    # training_paths shape: (N, 12) or (N, 13). 
    # Let's assume (N, 12) corresponding to the 12 exercise dates.
    # If it's (N, 13), we take the last 12 or specific columns.
    # The prompt says "12 monthly exercise dates".
    # Let's assume the input is already aligned to the 12 dates.
    
    paths = np.array(training_paths)
    if paths.ndim == 1:
        paths = paths.reshape(-1, 12) # Guess
    # If paths has more columns, we might need to select the right ones.
    # But usually, these tasks provide paths exactly at the exercise times.
    
    def negative_objective(b):
        # b is array of 12 boundaries
        # Simulate policy on paths
        # For each path, check exercise at each step
        # If S < b, exercise, payoff = K - S, discounted.
        # If not, continue.
        
        # Vectorized simulation is hard because paths stop at different times.
        # Loop over paths is slower but N=4096 is small.
        # However, we can vectorize the "check" part.
        
        # Let's try a vectorized approach with a mask.
        # This is tricky because once exercised, the path is done.
        # We can iterate through time steps 0..11.
        # active_mask: boolean array of shape (N,)
        # payoff: float array of shape (N,)
        
        active = np.ones(paths.shape[0], dtype=bool)
        total_payoff = np.zeros(paths.shape[0])
        
        # Pre-compute discount factors for each exercise date
        # t_i = i/12
        t_times = np.arange(1, 13) / 12.0 * T
        disc_factors = np.exp(-r * t_times)
        
        for i in range(12):
            if not np.any(active):
                break
            b_i = b[i]
            # For active paths, check if S < b_i
            S_i = paths[:, i]
            ex_mask = active & (S_i < b_i)
            
            if np.any(ex_mask):
                # Payoff for these paths
                # Discount to time 0
                payoff_i = (K - S_i[ex_mask]) * disc_factors[i]
                total_payoff[ex_mask] = payoff_i
                active[ex_mask] = False
        
        # For paths that never exercised, payoff is 0? 
        # Wait, if it's a put and S > K at all times, payoff is 0.
        # If it's in the money at the end but we didn't exercise?
        # Bermudan option: you can only exercise at dates.
        # If you don't exercise, you get 0.
        # So the logic above is correct.
        
        mean_payoff = np.mean(total_payoff)
        return -mean_payoff
    
    # 3. Optimize
    # Use Powell method which is robust for non-smooth functions
    # Limit iterations and time
    t_opt_start = time.time()
    max_time = 30.0 # Leave time for safety
    
    try:
        # Bounds for boundaries: [0, K] roughly. Maybe [0, 1.2*K]
        bounds = [(0.0, 1.2 * K) for _ in range(12)]
        
        # Check time
        if time.time() - t_opt_start < max_time:
            res = minimize(negative_objective, b0, method='Powell', bounds=bounds, 
                           options={'maxiter': 50, 'ftol': 1e-6, 'maxfev': 200})
            b_opt = res.x
        else:
            b_opt = b0
    except Exception:
        b_opt = b0
    
    # 4. Construct final boundaries
    final_boundaries = np.zeros(n_exercises + 1)
    final_boundaries[1:13] = b_opt
    final_boundaries[12] = K # Ensure last is K
    
    class ExercisePolicy:
        def __init__(self, boundaries, K):
            self.boundaries = boundaries
            self.K = K
        
        def exercise(self, current_state, time):
            if isinstance(time, (int, float, np.integer, np.floating)):
                idx = int(time)
                idx = max(1, min(12, idx))
                b = self.boundaries[idx]
                return np.array(current_state) <= b
            else:
                times = np.array(time)
                times_clipped = np.clip(times, 1, 12).astype(int)
                b = np.array([self.boundaries[t] for t in times_clipped])
                return np.array(current_state) <= b
    
    return ExercisePolicy(final_boundaries, K)
