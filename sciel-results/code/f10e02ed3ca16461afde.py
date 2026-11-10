# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_1d_union_length(positions, N):
    """
    Compute the exact integral of the length of the union of N sheared intervals
    over c in [0, 1].
    positions: array of shape (N,) representing base positions x_k.
    Tube k (k=0..N-1) at height c in [0,1] is [x_k + c*k/N, x_k + c*k/N + 1/N].
    """
    N = len(positions)
    if N == 0:
        return 0.0
    
    # Collect all critical c values in [0, 1] where the order of endpoints changes.
    # Endpoint k_start = x_k + c*k/N
    # Endpoint k_end = x_k + c*k/N + 1/N
    # Order changes when x_i + c*i/N = x_j + c*j/N or similar.
    # Specifically, x_i + c*i/N = x_j + c*j/N => c = N*(x_j - x_i)/(i - j)
    # Also x_i + c*i/N + 1/N = x_j + c*j/N => c = N*(x_j - x_i - 1/N)/(i - j)
    # And x_i + c*i/N = x_j + c*j/N + 1/N => c = N*(x_j - x_i + 1/N)/(i - j)
    # And x_i + c*i/N + 1/N = x_j + c*j/N + 1/N => c = N*(x_j - x_i)/(i - j) (same as first)
    
    critical_cs = [0.0, 1.0]
    for i in range(N):
        for j in range(N):
            if i != j:
                # Case 1: start_i = start_j
                c1 = N * (positions[j] - positions[i]) / (i - j)
                if 0.0 <= c1 <= 1.0:
                    critical_cs.append(c1)
                
                # Case 2: start_i = end_j
                c2 = N * (positions[j] - positions[i] - 1.0/N) / (i - j)
                if 0.0 <= c2 <= 1.0:
                    critical_cs.append(c2)
                
                # Case 3: end_i = start_j
                c3 = N * (positions[j] - positions[i] + 1.0/N) / (i - j)
                if 0.0 <= c3 <= 1.0:
                    critical_cs.append(c3)
                
                # Case 4: end_i = end_j (same as case 1)
    
    critical_cs = np.unique(critical_cs)
    critical_cs.sort()
    
    def union_len_at_c(c_val):
        intervals = []
        for k in range(N):
            start = positions[k] + c_val * k / N
            end = start + 1.0 / N
            intervals.append((start, end))
        
        intervals.sort(key=lambda x: x[0])
        
        merged = []
        for interval in intervals:
            if not merged:
                merged.append(list(interval))
            else:
                last = merged[-1]
                if interval[0] <= last[1]:
                    last[1] = max(last[1], interval[1])
                else:
                    merged.append(list(interval))
        
        length = 0.0
        for m in merged:
            length += m[1] - m[0]
        return length
    
    total_length = 0.0
    for idx in range(len(critical_cs) - 1):
        c1 = critical_cs[idx]
        c2 = critical_cs[idx+1]
        dc = c2 - c1
        if dc <= 0:
            continue
        
        L1 = union_len_at_c(c1)
        L2 = union_len_at_c(c2)
        
        # Since the topology is constant between critical points, L(c) is linear.
        total_length += 0.5 * (L1 + L2) * dc
        
    return total_length

def _objective_1d_squared(params, N=8):
    """
    Objective function: square of the 1D union length integral.
    This is proportional to the 3D volume.
    """
    L = _compute_1d_union_length(params, N)
    return L * L

def kakeya_needle_3d8() -> np.ndarray:
    """
    Optimizes the 1D Kakeya positions using multi-start L-BFGS-B on the exact
    analytic integral of the squared 1D union length.
    """
    N = 8
    start_time = time.time()
    deadline = start_time + 600.0
    
    # Initial guess: Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    best_pos = keich.copy()
    best_score = _objective_1d_squared(best_pos, N)
    
    # Define bounds
    # We want to minimize the spread. The intervals have width 1/N.
    # A reasonable bound is [-1, 2]
    bounds = [(-1.0, 2.0)] * N
    
    # Multi-start optimization
    # We will try the Keich solution and a few random perturbations or other heuristics.
    
    # Strategy 1: Optimize from Keich
    if time.time() < deadline - 10:
        try:
            res = minimize(
                _objective_1d_squared,
                best_pos,
                args=(N,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_pos = res.x
        except Exception:
            pass
    
    # Strategy 2: Try to optimize from a "tight" initial guess
    # Maybe start with positions that are close to each other?
    # Or use a grid search for a better initial point?
    
    # Let's try a few random starts within a smaller box, e.g., [0, 1]
    rng = np.random.default_rng(42)
    num_starts = 20
    for i in range(num_starts):
        if time.time() > deadline - 30:
            break
            
        # Random start in [0, 1]
        x0 = rng.uniform(0, 1, N)
        
        try:
            res = minimize(
                _objective_1d_squared,
                x0,
                args=(N,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_pos = res.x
        except Exception:
            pass

    # Final polish with a tighter tolerance if time permits
    if time.time() < deadline - 5:
        try:
            res = minimize(
                _objective_1d_squared,
                best_pos,
                args=(N,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-15}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_pos = res.x
        except Exception:
            pass

    # Construct the 3D positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = best_pos[i_flat]
    base_y = best_pos[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions

# EVOLVE-BLOCK-END
