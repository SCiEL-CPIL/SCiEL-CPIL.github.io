import numpy as np
import time
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

def fit_exercise_policy(training_paths, contract):
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin before 120s limit

    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercises = 12

    dt = T / n_exercises
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    df = np.exp(-r * dt)

    # Grid definition
    # Log-spaced grid is appropriate for GBM.
    # Focus on the region where the boundary is likely to be.
    # For a put, boundary < K.
    # S_min should be small enough, S_max large enough.
    S_min = 0.1
    S_max = 200.0
    n_grid = 40000  # Slightly lower than 50k to allow time for PCHIP overhead, but still high res
    
    S_grid = np.geomspace(S_min, S_max, n_grid)
    
    # V at maturity (step 12)
    V_next = np.maximum(K - S_grid, 0.0)
    
    # boundaries[i] for i in 1..12
    boundaries = [0.0] * (n_exercises + 1)
    boundaries[n_exercises] = K  # At maturity, exercise if S <= K
    
    # Backward induction from step 11 down to 1
    for j in range(n_exercises - 1, 0, -1):
        if time.time() > deadline:
            # Emergency break: return what we have.
            for jj in range(j, 0, -1):
                boundaries[jj] = boundaries[n_exercises]  # Use maturity boundary as default
            break
            
        intrinsic = np.maximum(K - S_grid, 0.0)
        
        # Create PCHIP interpolator for the current value function
        # PCHIP is monotonicity preserving, which is good for option values
        try:
            interp = PchipInterpolator(S_grid, V_next, extrapolate=False)
        except Exception:
            # Fallback to linear if PCHIP fails (e.g., duplicate points, though unlikely with geomspace)
            interp = np.interp

        # Continuation value
        Su = S_grid * u
        Sd = S_grid * d
        
        # Interpolate V_next
        # PchipInterpolator call returns an array
        V_up = interp(Su)
        V_down = interp(Sd)
        
        # Handle extrapolation if any points fall outside [S_min, S_max]
        # np.interp handles this by using endpoint values. PchipInterpolator with extrapolate=False returns NaN.
        # We should replace NaNs with 0 (value is 0 if S > S_max for a put? No, if S is very high, put value is 0.
        # If S is very low, put value is K-S.
        # Since S_grid covers [0.1, 200], and u~1.016, S*max_u ~ 203.
        # S*min_d ~ 0.098.
        # So we are slightly out of bounds.
        # Let's set bounds to be safe or handle NaNs.
        V_up = np.nan_to_num(V_up, nan=0.0)
        V_down = np.nan_to_num(V_down, nan=0.0)
        
        continuation = df * (p * V_up + (1 - p) * V_down)
        
        V_current = np.maximum(intrinsic, continuation)
        
        # Find boundary
        # diff = intrinsic - continuation
        diff = intrinsic - continuation
        
        # We want to find S such that diff(S) = 0.
        # diff is generally decreasing in S.
        # Find the index where sign changes.
        # Since diff is computed on a discrete grid, we can find the interval [S_i, S_{i+1}] where sign changes.
        sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
        
        if len(sign_changes) > 0:
            # Take the last sign change (highest S where exercise is optimal)
            idx = sign_changes[-1]
            S_low = S_grid[idx]
            S_high = S_grid[idx + 1]
            d_low = diff[idx]
            d_high = diff[idx + 1]
            
            # Use brentq for high precision root finding on the smooth function
            # Define the function to be rooted
            def objective(s):
                # We need the continuation value at s
                # Using the interpolator constructed from V_next
                # Note: We are finding the boundary for step j, which depends on V_next (step j+1)
                # The 'interp' object is based on V_next.
                su = s * u
                sd = s * d
                vu = interp(su)
                vd = interp(sd)
                vu = np.nan_to_num(vu, nan=0.0)
                vd = np.nan_to_num(vd, nan=0.0)
                cont = df * (p * vu + (1 - p) * vd)
                intr = max(K - s, 0.0)
                return intr - cont
            
            try:
                b = brentq(objective, S_low, S_high, xtol=1e-6, rtol=1e-10)
                boundaries[j] = b
            except ValueError:
                # Fallback to linear interpolation if brentq fails
                if d_high != d_low:
                    t_interp = d_low / (d_low - d_high)
                    boundaries[j] = S_low + t_interp * (S_high - S_low)
                else:
                    boundaries[j] = S_low
        else:
            # No sign change found in the grid.
            # Check if diff is always positive (exercise everywhere) or always negative (hold everywhere)
            if diff[0] > 0 and diff[-1] > 0:
                boundaries[j] = S_max
            elif diff[0] < 0 and diff[-1] < 0:
                boundaries[j] = S_min
            else:
                # Fallback: use last positive index logic
                exercise_indices = np.where(diff > 0)[0]
                if len(exercise_indices) > 0:
                    last_idx = exercise_indices[-1]
                    if last_idx < n_grid - 1:
                        d0 = diff[last_idx]
                        d1 = diff[last_idx + 1]
                        if d1 != d0:
                            t_interp = d0 / (d0 - d1)
                            boundaries[j] = S_grid[last_idx] + t_interp * (S_grid[last_idx + 1] - S_grid[last_idx])
                        else:
                            boundaries[j] = S_grid[last_idx]
                    else:
                        boundaries[j] = S_grid[-1]
                else:
                    boundaries[j] = S_min
        
        V_next = V_current
    
    class ExercisePolicy:
        def __init__(self, boundaries, K):
            self.boundaries = boundaries
            self.K = K
            
        def exercise(self, current_state, time):
            current_state = np.asarray(current_state)
            if isinstance(time, (int, float)):
                idx = int(time)
                idx = max(1, min(12, idx))
                b = self.boundaries[idx]
                return current_state <= b
            else:
                times = np.asarray(time)
                times_clipped = np.clip(times, 1, 12).astype(int)
                # Vectorized boundary lookup
                b = np.array([self.boundaries[t] for t in times_clipped])
                return current_state <= b
    
    return ExercisePolicy(boundaries, K)
