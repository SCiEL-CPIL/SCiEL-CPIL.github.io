# EVOLVE-BLOCK-START
import numpy as np
import time

def _union_area(x, n=64, M=20000):
    """
    Compute the area of the union of n triangles using the specified quadrature.
    """
    m = np.arange(M)
    y = (m + 0.5) / M
    i_idx = np.arange(n)
    a = (i_idx + 1) / n
    w = (1.0 - y) / n
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    L_sorted = np.sort(L, axis=0)
    d = L_sorted[1:, :] - L_sorted[:-1, :]
    min_d_w = np.minimum(d, w[np.newaxis, :])
    union_lengths = w[np.newaxis, :] + np.sum(min_d_w, axis=0)
    area = np.mean(union_lengths)
    return area

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    M_final = 20000
    deadline = time.time() + 580.0  # Safety margin for 630s limit
    
    # Initial solution: Keich construction
    k = int(np.log2(n))
    j_arr = np.arange(1, k + 1)
    x0 = np.zeros(n, dtype=np.float64)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x0[i] = np.sum((1 - j_arr) * eps * (2.0 ** -j_arr) / k)
    
    best_x = x0.copy()
    best_area = _union_area(best_x, n, M_final)
    
    # Simulated Annealing with Coarse-to-Fine Resolution Ladder
    # We use different M values to balance exploration (low M) and exploitation (high M)
    # Low M allows large steps to change the relative ordering of triangles (topology)
    # High M refines the positions for minimal area
    
    # Define the schedule: (M, steps, initial_step_size, final_step_size)
    # We will spend most time on the final resolution, but use earlier phases to escape local minima
    schedule = [
        (100, 2000, 0.5, 0.1),      # Coarse: Explore topology
        (500, 2000, 0.2, 0.05),     # Medium: Refine topology
        (2000, 3000, 0.1, 0.01),    # Fine: Approach final area
        (M_final, 5000, 0.05, 0.001) # Final: High precision refinement
    ]
    
    # Temperature schedule parameters
    T_start = 0.1
    T_end = 1e-6
    
    current_x = best_x.copy()
    current_area = best_area
    
    # We will iterate through the schedule, but we can also interleave or repeat if time permits.
    # For simplicity and robustness, we run each phase sequentially.
    
    for M, steps, s_start, s_end in schedule:
        if time.time() > deadline - 30:
            break
            
        # Pre-compute constants for this M
        m_idx = np.arange(M)
        y = (m_idx + 0.5) / M
        a = (np.arange(n) + 1) / n
        w = (1.0 - y) / n
        
        # Pre-compute L_base = x[:, None] + a[:, None] * y[None, :] is not static because x changes.
        # But a[:, None] * y[None, :] is static.
        a_y = a[:, np.newaxis] * y[np.newaxis, :]
        
        def _area_fast(x_curr):
            L = x_curr[:, np.newaxis] + a_y
            L_sorted = np.sort(L, axis=0)
            d = L_sorted[1:, :] - L_sorted[:-1, :]
            min_d_w = np.minimum(d, w[np.newaxis, :])
            union_lengths = w[np.newaxis, :] + np.sum(min_d_w, axis=0)
            return np.mean(union_lengths)
        
        T = T_start
        cooling = (T_end / T_start) ** (1.0 / steps)
        
        for step in range(steps):
            if time.time() > deadline - 10:
                break
                
            # Step size decay
            step_size = s_start * (s_end / s_start) ** (step / steps)
            
            # Generate candidate
            # We perturb a random subset of coordinates to allow larger moves
            # or perturb all with small noise. 
            # Strategy: Perturb all coordinates with Gaussian noise scaled by step_size.
            noise = np.random.randn(n) * step_size
            x_new = current_x + noise
            
            # Optional: Keep x within reasonable bounds? 
            # The problem doesn't specify bounds, but large x just shifts the union.
            # The area is translation invariant? 
            # L_i = x_i + a_i y. If we add C to all x_i, L_i shifts by C.
            # The union length depends on differences L_{j+1} - L_j.
            # So the area is invariant to global translation.
            # We can fix x_0 = 0 to reduce dimensionality? 
            # But the evaluator expects 64 floats. We can just let them drift or re-center.
            # Let's re-center to prevent numerical drift.
            x_new -= np.mean(x_new)
            
            area_new = _area_fast(x_new)
            
            # Acceptance criterion
            if area_new < current_area:
                current_x = x_new
                current_area = area_new
                
                # Update global best
                if current_area < best_area:
                    best_area = current_area
                    best_x = current_x.copy()
            else:
                # Compute delta
                delta = area_new - current_area
                # Accept with probability exp(-delta / T)
                if delta > 0:
                    prob = np.exp(-delta / T)
                    if np.random.rand() < prob:
                        current_x = x_new
                        current_area = area_new
            
            T *= cooling
    
    # Final polish with Nelder-Mead if time remains
    if time.time() < deadline - 20:
        try:
            from scipy.optimize import minimize
            result_nm = minimize(
                lambda x: _union_area(x, n, M_final),
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if result_nm.fun < best_area:
                best_x = result_nm.x.copy()
                best_area = result_nm.fun
        except Exception:
            pass

    if not np.all(np.isfinite(best_x)):
        return x0
    if best_area <= 0:
        return x0
    
    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
