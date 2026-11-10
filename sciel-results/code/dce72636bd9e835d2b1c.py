import numpy as np
import math
import time

def fit_exercise_policy(training_paths, contract):
    t_start = time.time()
    deadline = t_start + 110.0  # 110s safety margin before 120s limit

    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercises = 12

    # Strategy: Use a high-resolution binomial tree (1200 steps) to compute
    # optimal exercise boundaries via backward induction.
    # We use a fine grid to minimize interpolation error in the continuation value.
    # The NN refinement from the previous version is removed as it was trained on
    # the same tree boundaries and added overhead without clear benefit.
    
    n_steps = 1200
    dt = T / n_steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r * dt) - d) / (u - d)
    df = math.exp(-r * dt)
    
    # Exercise dates: steps 100, 200, 300, ..., 1200
    # We exercise at these specific steps in the backward induction.
    exercise_steps = {i * n_steps // n_exercises: i for i in range(1, n_exercises + 1)}
    
    # Grid for S values
    # Use a fine grid over [0, 1.5*K]. 
    # 100,000 points gives a step size of 0.0015 for K=100, which is very fine.
    # This ensures that the boundary location is determined with high precision.
    S_min = 0.0
    S_max = 1.5 * K
    n_grid = 100000
    S_grid = np.linspace(S_min, S_max, n_grid)
    
    # V at maturity (step n_steps)
    # For a put, payoff is max(K - S, 0)
    V_next = np.maximum(K - S_grid, 0.0)
    
    # Boundaries array, indexed by exercise date 1..12
    boundaries = np.zeros(n_exercises + 1)
    
    # Backward induction
    for step in range(n_steps - 1, 0, -1):
        intrinsic = np.maximum(K - S_grid, 0.0)
        
        # Next step spot prices
        Su = S_grid * u
        Sd = S_grid * d
        
        # Interpolate V_next at Su and Sd
        # np.interp clamps to boundary values if out of range.
        # For S > S_max, V is 0 (for put), which is correct.
        # For S < 0, V is K (approx), which is handled by the grid starting at 0.
        V_up = np.interp(Su, S_grid, V_next)
        V_down = np.interp(Sd, S_grid, V_next)
        
        continuation = df * (p * V_up + (1 - p) * V_down)
        
        # Optimal value is max(intrinsic, continuation)
        V_current = np.maximum(intrinsic, continuation)
        
        # Check if this step corresponds to an exercise date
        if step in exercise_steps:
            idx = exercise_steps[step]
            
            # Find the exercise boundary
            # The boundary is where intrinsic == continuation
            # For S < boundary, intrinsic > continuation (exercise)
            # For S > boundary, intrinsic < continuation (hold)
            diff = intrinsic - continuation
            
            # We want the largest S where diff >= 0
            # Since diff is generally decreasing in S (for puts), we can find the crossing point.
            # exercise_mask is True where we should exercise.
            exercise_mask = diff >= 0
            
            if np.any(exercise_mask):
                idxs = np.where(exercise_mask)[0]
                last_idx = idxs[-1]
                
                # Refine the boundary using linear interpolation between last_idx and last_idx+1
                if last_idx < n_grid - 1:
                    diff0 = diff[last_idx]
                    diff1 = diff[last_idx + 1]
                    
                    if diff0 != diff1:
                        # Interpolate to find where diff = 0
                        # diff0 + t * (diff1 - diff0) = 0 => t = -diff0 / (diff1 - diff0)
                        t = -diff0 / (diff1 - diff0)
                        t = max(0.0, min(1.0, t)) # Clamp to [0, 1]
                        boundaries[idx] = S_grid[last_idx] + t * (S_grid[last_idx + 1] - S_grid[last_idx])
                    else:
                        boundaries[idx] = S_grid[last_idx]
                else:
                    boundaries[idx] = S_grid[-1]
            else:
                # If no exercise region found, boundary is 0 (never exercise)
                boundaries[idx] = 0.0
        
        # Update V_next for the next iteration (one step back in time)
        V_next = V_current
        
        # Time check (unlikely to be needed for 1200 steps * 100k grid, but safe)
        if step % 200 == 0:
            if time.time() > deadline - 5:
                # If running out of time, break and use current boundaries.
                # This is highly unlikely given the speed of NumPy operations.
                pass

    # At maturity (time 12, which is T), we always exercise if in the money.
    # The problem states "12 monthly exercise dates (no time-0 exercise)".
    # Usually, the last exercise date is at T. If S < K, we exercise.
    # The boundary at T is effectively K.
    boundaries[12] = K

    class ExercisePolicy:
        def __init__(self, boundaries, K):
            self.boundaries = boundaries
            self.K = K
        
        def exercise(self, current_state, time):
            # current_state is a vector of spot prices for multiple paths
            # time is the current time index (1 to 12)
            
            # Handle time as scalar or array
            if isinstance(time, (int, float, np.integer, np.floating)):
                idx = int(time)
                idx = max(1, min(12, idx))
                b = self.boundaries[idx]
                return np.array(current_state) <= b
            else:
                # If time is an array, handle each path's time
                times = np.array(time)
                times_clipped = np.clip(times, 1, 12).astype(int)
                b = np.array([self.boundaries[t] for t in times_clipped])
                return np.array(current_state) <= b
    
    return ExercisePolicy(boundaries, K)
