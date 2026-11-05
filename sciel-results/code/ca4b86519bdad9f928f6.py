"""Bermudan put exercise-policy research: backward induction on training paths with time-deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_DEADLINE = 145.0  # safety margin under 150s wall-clock limit


def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using backward induction on training paths.
    
    Parameters:
        training_paths: array of shape (n_paths, n_dates+1) or (n_dates+1, n_paths)
                        containing simulated spot prices
        contract: dict or object with contract parameters (K, r, sigma, T, n_dates, etc.)
    
    Returns:
        An object with exercise(current_state, time) method returning boolean vector.
    """
    t0 = _time.time()
    
    # Extract contract parameters
    if isinstance(contract, dict):
        K = contract.get('K', 100.0)
        r = contract.get('r', 0.05)
        sigma = contract.get('sigma', 0.2)
        T = contract.get('T', 1.0)
        n_dates = contract.get('n_dates', 12)
        n_paths = contract.get('n_paths', 4096)
    else:
        K = getattr(contract, 'K', 100.0)
        r = getattr(contract, 'r', 0.05)
        sigma = getattr(contract, 'sigma', 0.2)
        T = getattr(contract, 'T', 1.0)
        n_dates = getattr(contract, 'n_dates', 12)
        n_paths = getattr(contract, 'n_paths', 4096)
    
    # Ensure training_paths is 2D with shape (n_dates+1, n_paths)
    paths = np.asarray(training_paths, dtype=np.float64)
    if paths.ndim == 1:
        paths = paths.reshape(1, -1)
    if paths.shape[0] < paths.shape[1]:
        paths = paths.T
    
    # Number of time steps (exercise dates are at t_1, ..., t_n_dates)
    # paths has shape (n_steps, n_paths) where n_steps = n_dates + 1 (including t=0)
    n_steps, n_actual_paths = paths.shape
    # Exercise dates are indices 1..n_steps-1 (0-indexed), so n_exercise = n_steps - 1
    n_exercise = n_steps - 1
    
    if n_exercise <= 0:
        # No exercise dates, just return a policy that never exercises
        class _Policy:
            def exercise(self, current_state, time):
                return np.zeros(len(current_state), dtype=bool)
        return _Policy()
    
    # Time step
    dt = T / n_exercise
    
    # Discount factor per step
    disc = np.exp(-r * dt)
    
    # Continuation value at maturity (last step)
    # At the last exercise date (index n_exercise-1 in 0-indexed exercise dates,
    # which is index n_steps-1 in paths), the payoff is max(K - S, 0)
    # But wait: we have n_exercise exercise dates. Let me re-index.
    # 
    # paths[0] = t=0
    # paths[1] = t_1 (first exercise date)
    # ...
    # paths[n_exercise] = t_n_exercise (last exercise date = maturity)
    #
    # So exercise dates are at indices 1, 2, ..., n_exercise in paths.
    # The value at maturity (index n_exercise) is just the intrinsic value.
    
    # V[i] is the value at exercise date i (0-indexed, i=0..n_exercise-1)
    # V[n_exercise-1] corresponds to paths[n_exercise]
    
    V = np.zeros((n_exercise, n_actual_paths), dtype=np.float64)
    
    # Terminal condition: at the last exercise date, value = intrinsic
    V[-1] = np.maximum(K - paths[n_exercise], 0.0)
    
    # Backward induction
    # For i from n_exercise-2 down to 0:
    #   V[i] = max(intrinsic_i, disc * V[i+1])
    #   where intrinsic_i = max(K - paths[i+1], 0)
    
    for i in range(n_exercise - 2, -1, -1):
        elapsed = _time.time() - t0
        if elapsed > _DEADLINE - 5.0:
            break  # Time's up, use what we have
        
        intrinsic = np.maximum(K - paths[i + 1], 0.0)
        cont_val = disc * V[i + 1]
        V[i] = np.maximum(intrinsic, cont_val)
    
    # Now V[i] is the value at exercise date i (paths index i+1)
    # The exercise decision at exercise date i is: exercise if intrinsic >= continuation
    # i.e., max(K - S_i, 0) >= disc * V[i+1]
    # But for the last date, we always exercise (it's maturity).
    
    # Build exercise indicators for each exercise date
    # exercise_date[i] is a boolean array of shape (n_actual_paths,)
    exercise_indicators = [None] * n_exercise
    
    for i in range(n_exercise):
        elapsed = _time.time() - t0
        if elapsed > _DEADLINE - 3.0:
            # Fallback: use simple threshold for remaining dates
            for j in range(i, n_exercise):
                exercise_indicators[j] = np.maximum(K - paths[j + 1], 0.0) > 0
            break
        
        if i == n_exercise - 1:
            # Last date: always exercise (maturity)
            exercise_indicators[i] = np.ones(n_actual_paths, dtype=bool)
        else:
            intrinsic = np.maximum(K - paths[i + 1], 0.0)
            cont_val = disc * V[i + 1]
            exercise_indicators[i] = intrinsic >= cont_val
    
    # Create a threshold-based policy for generalization.
    # For each exercise date, find the threshold S* such that we exercise if S <= S*.
    # This is a simplification, but it allows generalization to new paths.
    
    thresholds = np.zeros(n_exercise, dtype=np.float64)
    for i in range(n_exercise):
        if exercise_indicators[i] is None:
            # Fallback: use BS threshold
            t_rem = T - (i + 1) * dt
            if t_rem <= 0:
                thresholds[i] = K
            else:
                # Approximate threshold using BS
                # For a put, we exercise when S is low enough
                # Use a simple heuristic
                thresholds[i] = K * 0.85  # rough guess
        else:
            S_i = paths[i + 1]
            ex_mask = exercise_indicators[i]
            if np.any(ex_mask) and np.any(~ex_mask):
                # Threshold is between max exercised S and min continued S
                max_ex = np.max(S_i[ex_mask])
                min_cont = np.min(S_i[~ex_mask])
                thresholds[i] = (max_ex + min_cont) / 2.0
            elif np.all(ex_mask):
                # Always exercise
                thresholds[i] = np.inf
            else:
                # Never exercise
                thresholds[i] = -np.inf
    
    # Build the policy object
    class ExercisePolicy:
        def __init__(self, thresholds, n_exercise):
            self.thresholds = thresholds
            self.n_exercise = n_exercise
        
        def exercise(self, current_state, time):
            """
            current_state: array of spot prices
            time: the exercise date index (0-indexed) or time value
            Returns: boolean vector
            """
            if isinstance(time, (int, np.integer)):
                idx = int(time)
            elif isinstance(time, float):
                # Convert time to exercise date index
                # time is in [0, T], exercise dates at dt, 2*dt, ..., n_exercise*dt
                idx = int(round(time / dt)) - 1
            else:
                idx = 0
            
            # Clamp index
            idx = max(0, min(idx, self.n_exercise - 1))
            
            S = np.asarray(current_state, dtype=np.float64)
            return S <= self.thresholds[idx]
    
    return ExercisePolicy(thresholds, n_exercise)
# EVOLVE-BLOCK-END
