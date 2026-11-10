import numpy as np
import math
import time
from scipy.interpolate import PchipInterpolator

def fit_exercise_policy(training_paths, contract):
    t_start = time.time()
    deadline = t_start + 110.0

    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercises = 12

    n_steps = 1200
    dt = T / n_steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r * dt) - d) / (u - d)
    df = math.exp(-r * dt)

    exercise_steps = {i * n_steps // n_exercises: i for i in range(1, n_exercises + 1)}

    # Log-uniform grid for better resolution at low S
    S_min = 0.01
    S_max = 1.5 * K
    n_grid = 20000
    S_grid = np.geomspace(S_min, S_max, n_grid)
    
    # V at maturity
    V_next = np.maximum(K - S_grid, 0.0)
    
    boundaries = np.zeros(n_exercises + 1)
    
    for step in range(n_steps - 1, 0, -1):
        intrinsic = np.maximum(K - S_grid, 0.0)
        
        Su = S_grid * u
        Sd = S_grid * d
        
        # Interpolate V_next at Su and Sd
        # Clip to grid range to avoid extrapolation issues, 
        # though PCHIP handles it, clipping is safer for tails
        Su_clipped = np.clip(Su, S_grid[0], S_grid[-1])
        Sd_clipped = np.clip(Sd, S_grid[0], S_grid[-1])
        
        # Create interpolator
        # PCHIP is monotonicity preserving, good for option values
        interp = PchipInterpolator(S_grid, V_next)
        
        V_up = interp(Su_clipped)
        V_down = interp(Sd_clipped)
        
        # Handle tails analytically if clipped?
        # If Su > S_max, V_up should be ~0. PCHIP extrapolates linearly from last points.
        # Last points are high S, V is 0. So extrapolation is 0. Correct.
        # If Sd < S_min, V_down should be ~K - S. 
        # First points are low S. V is ~K. PCHIP extrapolates.
        # Let's check if we need to correct.
        # For Sd < S_min, the true value is close to K - Sd.
        # PCHIP extrapolation from [S_min, S_min+dx] with V ~ K might be slightly off.
        # But S_min is 0.01, K is 100. Error is small.
        
        continuation = df * (p * V_up + (1 - p) * V_down)
        
        V_current = np.maximum(intrinsic, continuation)
        
        if step in exercise_steps:
            idx = exercise_steps[step]
            diff = intrinsic - continuation
            
            # Find boundary
            # diff is positive for low S, negative for high S
            # We want the largest S where diff >= 0
            exercise_mask = diff >= 0
            if np.any(exercise_mask):
                idxs = np.where(exercise_mask)[0]
                last_idx = idxs[-1]
                
                # Refine boundary via linear interpolation between last_idx and last_idx+1
                # If last_idx is the last point, boundary is S_grid[-1]
                if last_idx < n_grid - 1:
                    d0 = diff[last_idx]
                    d1 = diff[last_idx + 1]
                    s0 = S_grid[last_idx]
                    s1 = S_grid[last_idx + 1]
                    
                    if d0 != d1:
                        t = d0 / (d0 - d1)
                        boundaries[idx] = s0 + t * (s1 - s0)
                    else:
                        boundaries[idx] = s0
                else:
                    boundaries[idx] = S_grid[-1]
            else:
                boundaries[idx] = 0.0
        
        V_next = V_current
        
        if time.time() > deadline - 5:
            # Fallback: fill remaining with K (or current best guess)
            # Since we go backwards, we have later dates. Earlier dates are missing.
            # If we break, we should probably use the last calculated boundary or K.
            # For a put, boundary increases as time to maturity decreases.
            # So earlier dates (more time) have lower boundaries.
            # If we break at step X, we have boundaries for dates > X.
            # We should fill missing earlier dates with a conservative estimate.
            # Using the first calculated boundary (lowest) or 0 is safe.
            # Let's just break and let the class handle it? No, we need valid boundaries.
            # We'll fill remaining with the last computed boundary (which is the lowest so far? No, we go back in time, so boundaries decrease).
            # Wait, step goes from n_steps down to 1.
            # step 1200 -> date 12.
            # step 100 -> date 1.
            # So we compute date 12 first. Then 11... down to 1.
            # If we break, we have dates 12..k. Missing 1..k-1.
            # Boundary for date 1 is lower than date 2.
            # So we should fill missing (earlier) dates with the lowest boundary we have (which is the most recent one calculated, i.e., date k).
            # Actually, just filling with 0 or the last computed one is fine.
            pass

    boundaries[12] = K

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
    
    return ExercisePolicy(boundaries, K)
