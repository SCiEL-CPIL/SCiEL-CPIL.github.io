# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _union_length_1d(base, N):
    """Compute the length of the union of intervals at height c for a 1D problem."""
    c_samples = np.linspace(0, 1, 64)
    total_length = 0.0
    for c in c_samples:
        starts = base + c * np.arange(N) / N
        ends = starts + 1.0 / N
        order = np.argsort(starts)
        s = starts[order]
        e = ends[order]
        union_len = 0.0
        cur_s, cur_e = s[0], e[0]
        for k in range(1, N):
            if s[k] > cur_e:
                union_len += cur_e - cur_s
                cur_s, cur_e = s[k], e[k]
            else:
                cur_e = max(cur_e, e[k])
        union_len += cur_e - cur_s
        total_length += union_len
    return total_length / len(c_samples)


def _cost_3d(base_x, base_y, N):
    """Full 3D cost: product of 1D union lengths."""
    cx = _union_length_1d(base_x, N)
    cy = _union_length_1d(base_y, N)
    return cx * cy


def _smooth_cost_3d(base_x, base_y, N, beta=50.0):
    """
    Smooth surrogate for the 3D cost using soft-min to approximate union length.
    Uses a smooth approximation of the max function via log-sum-exp.
    """
    c_samples = np.linspace(0, 1, 32)
    i_vals = np.arange(N, dtype=np.float64)
    
    def smooth_union_length(base, c):
        # Intervals: [base[i] + c*i/N, base[i] + c*i/N + 1/N]
        starts = base + c * i_vals / N
        ends = starts + 1.0 / N
        
        # Sort by start
        order = np.argsort(starts)
        s = starts[order]
        e = ends[order]
        
        # Smooth union length using iterative soft-max approach
        # We approximate the union by finding the envelope
        # Use a smooth approximation: union ≈ max(0, max_i(ends_i - starts_i) + overlaps)
        # Better: use the fact that union of intervals = sum of gaps
        # Smooth approximation: use log-sum-exp for the "active" intervals
        
        # Simple approach: compute the coverage function smoothed
        # The union length = integral of indicator(covered) dx
        # Approximate by sampling x positions
        
        # For speed, use a coarser x-grid
        x_min = np.min(s)
        x_max = np.max(e)
        if x_max <= x_min:
            return 0.0
        
        # Sample x positions
        n_x = 32
        x_grid = np.linspace(x_min, x_max, n_x)
        
        # For each x, check if covered by any interval (smoothed)
        # covered(x) = max_i sigmoid(beta * (e_i - x)) * sigmoid(beta * (x - s_i))
        # But this is expensive. Instead, use the exact merge for small N.
        
        # Actually, for N=8, the exact merge is fast. Let's just use exact.
        union_len = 0.0
        cur_s, cur_e = s[0], e[0]
        for k in range(1, N):
            if s[k] > cur_e:
                union_len += cur_e - cur_s
                cur_s, cur_e = s[k], e[k]
            else:
                cur_e = max(cur_e, e[k])
        union_len += cur_e - cur_s
        return union_len
    
    total_length = 0.0
    for c in c_samples:
        total_length += smooth_union_length(base_x, c) + smooth_union_length(base_y, c)
    
    cx = total_length / len(c_samples) / 2
    cy = total_length / len(c_samples) / 2
    
    # Actually, we need separate x and y costs
    # Let me redo this properly
    
    total_x = 0.0
    total_y = 0.0
    for c in c_samples:
        total_x += smooth_union_length(base_x, c)
        total_y += smooth_union_length(base_y, c)
    
    cx = total_x / len(c_samples)
    cy = total_y / len(c_samples)
    return cx * cy


def _smooth_cost_3d_v2(base_x, base_y, N, beta=100.0):
    """
    Better smooth surrogate: use the fact that for sorted intervals,
    the union length can be expressed using smooth max operations.
    """
    c_samples = np.linspace(0, 1, 32)
    i_vals = np.arange(N, dtype=np.float64)
    
    def smooth_union_length(base, c):
        starts = base + c * i_vals / N
        ends = starts + 1.0 / N
        
        order = np.argsort(starts)
        s = starts[order]
        e = ends[order]
        
        # Exact merge (fast for N=8)
        union_len = 0.0
        cur_s, cur_e = s[0], e[0]
        for k in range(1, N):
            if s[k] > cur_e:
                union_len += cur_e - cur_s
                cur_s, cur_e = s[k], e[k]
            else:
                cur_e = max(cur_e, e[k])
        union_len += cur_e - cur_s
        return union_len
    
    total_x = 0.0
    total_y = 0.0
    for c in c_samples:
        total_x += smooth_union_length(base_x, c)
        total_y += smooth_union_length(base_y, c)
    
    cx = total_x / len(c_samples)
    cy = total_y / len(c_samples)
    return cx * cy


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    Uses simulated annealing followed by gradient-based polishing.
    """
    N = 8
    start_time = time.time()
    deadline = start_time + 600.0
    
    # Initial Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    best_x = keich.copy()
    best_y = keich.copy()
    best_cost = _cost_3d(best_x, best_y, N)
    
    # Simulated annealing phase
    T = 0.01
    T_min = 1e-6
    cooling = 0.995
    step_size = 0.02
    max_steps = 30000
    
    current_x = best_x.copy()
    current_y = best_y.copy()
    current_cost = best_cost
    
    rng = np.random.default_rng(42)
    
    for step in range(max_steps):
        if time.time() > deadline - 120.0:
            break
        
        if rng.random() < 0.5:
            idx = rng.integers(N)
            delta = rng.normal(0, step_size)
            new_x = current_x.copy()
            new_x[idx] += delta
            new_cost = _cost_3d(new_x, current_y, N)
            if new_cost < current_cost or rng.random() < np.exp(-(new_cost - current_cost) / T):
                current_x = new_x
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_x = current_x.copy()
        else:
            idx = rng.integers(N)
            delta = rng.normal(0, step_size)
            new_y = current_y.copy()
            new_y[idx] += delta
            new_cost = _cost_3d(current_x, new_y, N)
            if new_cost < current_cost or rng.random() < np.exp(-(new_cost - current_cost) / T):
                current_y = new_y
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_y = current_y.copy()
        
        T = max(T_min, T * cooling)
    
    # Gradient-based polishing phase
    if time.time() < deadline - 60.0:
        def cost_func(params):
            x = params[:N]
            y = params[N:]
            return _cost_3d(x, y, N)
        
        initial_params = np.concatenate([best_x, best_y])
        
        try:
            result = minimize(cost_func, initial_params, method='L-BFGS-B',
                             options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            if result.success or result.fun < best_cost:
                polished_params = result.x
                polished_x = polished_params[:N]
                polished_y = polished_params[N:]
                polished_cost = _cost_3d(polished_x, polished_y, N)
                if polished_cost < best_cost:
                    best_x = polished_x.copy()
                    best_y = polished_y.copy()
                    best_cost = polished_cost
        except Exception:
            pass
    
    # Build positions array
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = best_x[i_flat]
    base_y = best_y[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    return positions


# EVOLVE-BLOCK-END
