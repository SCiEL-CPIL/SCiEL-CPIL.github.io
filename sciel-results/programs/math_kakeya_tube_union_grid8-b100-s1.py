import numpy as np
import time
from scipy.optimize import minimize

def _get_breakpoints(X, N=8):
    """
    Calculate all c in (0, 1) where the order of endpoints changes.
    """
    breakpoints = set([0.0, 1.0])
    for i in range(N):
        for j in range(i + 1, N):
            denom = i - j
            # L_i = L_j
            c1 = N * (X[j] - X[i]) / denom
            if 0.0 < c1 < 1.0:
                breakpoints.add(c1)
            # L_i = R_j
            c2 = N * (X[j] + 1.0 / N - X[i]) / denom
            if 0.0 < c2 < 1.0:
                breakpoints.add(c2)
            # R_i = L_j
            c3 = N * (X[j] - X[i] - 1.0 / N) / denom
            if 0.0 < c3 < 1.0:
                breakpoints.add(c3)
            # R_i = R_j
            c4 = N * (X[j] - X[i]) / denom
            if 0.0 < c4 < 1.0:
                breakpoints.add(c4)
    return sorted(breakpoints)

def _compute_1d_cost_squared(X, N=8):
    """
    Compute integral of (Union Length)^2 over c in [0, 1].
    """
    if not np.all(np.isfinite(X)):
        return 1e10
        
    breakpoints = _get_breakpoints(X, N)
    
    total_integral = 0.0
    
    for k in range(len(breakpoints) - 1):
        c_start = breakpoints[k]
        c_end = breakpoints[k + 1]
        if c_end <= c_start:
            continue
            
        c_mid = 0.5 * (c_start + c_end)
        
        # Calculate union length and its derivative (slope) at c_mid
        # Since order is fixed, L(c) is linear.
        # We can calculate L at c_mid and L at c_mid + epsilon to get slope?
        # Or derive analytically.
        # Let's just evaluate L at c_mid and c_mid + 1e-6 to get slope numerically for robustness,
        # or better, since N is small, we can find the active intervals.
        
        # Endpoints at c_mid
        Ls = [X[i] + (i / N) * c_mid for i in range(N)]
        Rs = [X[i] + (i / N) * c_mid + 1.0 / N for i in range(N)]
        
        # Merge intervals
        intervals = sorted(zip(Ls, Rs))
        merged = []
        cur_l, cur_r = intervals[0]
        for l, r in intervals[1:]:
            if l > cur_r:
                merged.append((cur_l, cur_r))
                cur_l, cur_r = l, r
            else:
                cur_r = max(cur_r, r)
        merged.append((cur_l, cur_r))
        
        # Calculate L(c_mid)
        L_mid = sum(r - l for l, r in merged)
        
        # Calculate slope
        # The left boundary of the union is min of lefts of merged components?
        # No, the merged components are disjoint.
        # The total length is sum of (r_m - l_m).
        # Each l_m is of form X_i + (i/N)c. Slope is i/N.
        # Each r_m is of form X_j + (j/N)c + 1/N. Slope is j/N.
        # So slope of each component length is (slope_r - slope_l).
        
        # We need to identify which i and j form the boundaries of each merged component.
        # This is tricky to do robustly with just values.
        # Let's use numerical differentiation with a small step, ensuring we stay in the interval.
        delta_c = 1e-6
        c_plus = c_mid + delta_c
        if c_plus > c_end:
            c_plus = c_mid
            delta_c = 0
            slope = 0
        else:
            Ls_p = [X[i] + (i / N) * c_plus for i in range(N)]
            Rs_p = [X[i] + (i / N) * c_plus + 1.0 / N for i in range(N)]
            intervals_p = sorted(zip(Ls_p, Rs_p))
            merged_p = []
            cur_l, cur_r = intervals_p[0]
            for l, r in intervals_p[1:]:
                if l > cur_r:
                    merged_p.append((cur_l, cur_r))
                    cur_l, cur_r = l, r
                else:
                    cur_r = max(cur_r, r)
            merged_p.append((cur_l, cur_r))
            L_plus = sum(r - l for l, r in merged_p)
            slope = (L_plus - L_mid) / delta_c
            
        # Integrate (L_mid + slope * (c - c_mid))^2 from c_start to c_end
        # Let u = c - c_mid. Limits: c_start - c_mid to c_end - c_mid.
        # Since c_mid is midpoint, limits are -h to h, where h = (c_end - c_start)/2.
        h = (c_end - c_start) / 2.0
        
        # Integral of (A u + B)^2 du from -h to h
        # = [ (A u + B)^3 / (3A) ] ... wait
        # = \int_{-h}^h (A^2 u^2 + 2AB u + B^2) du
        # = 2 A^2 h^3 / 3 + 0 + 2 B^2 h
        
        A = slope
        B = L_mid
        integral_segment = (2.0 * A * A * h**3) / 3.0 + 2.0 * B * B * h
        
        total_integral += integral_segment
        
    return total_integral

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 550
    
    N = 8
    
    # Initial guesses
    inits = [
        np.zeros(N),
        np.arange(N) / N,
        -np.arange(N) / N,
        np.random.default_rng(12345).uniform(-0.5, 0.5, N),
        np.array([0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5]) / N, # Clustered
    ]
    
    best_X = None
    best_cost = float('inf')
    
    for init in inits:
        if time.time() > deadline - 30:
            break
            
        try:
            # Use Nelder-Mead for small N
            res = minimize(_compute_1d_cost_squared, init, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_cost:
                best_cost = res.fun
                best_X = res.x.copy()
        except Exception as e:
            pass
    
    if best_X is None:
        best_X = np.zeros(N)
        
    # Construct the 64x2 array
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = best_X[i_flat]
    base_y = best_X[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions
