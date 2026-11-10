from scipy.optimize import differential_evolution
import numpy as np
from scipy.optimize import minimize
import time

def compute_P(s, w):
    """
    Computes P[X1 + X2 + X3 < 2*X4] efficiently.
    s: support points (n,)
    w: weights (n,)
    """
    n = len(s)
    if n == 0:
        return 0.0
        
    # Pairwise sums S_ij = s_i + s_j
    S_ij = s[:, np.newaxis] + s[np.newaxis, :]
    # Pairwise weight products W_ij = w_i * w_j
    W_ij = w[:, np.newaxis] * w[np.newaxis, :]
    
    # Flatten and sort
    S_flat = S_ij.flatten()
    W_flat = W_ij.flatten()
    
    # Sort by S_ij
    sort_idx = np.argsort(S_flat)
    S_sorted = S_flat[sort_idx]
    W_sorted = W_flat[sort_idx]
    
    # Cumulative sum of weights
    W_cumsum = np.cumsum(W_sorted)
    
    P = 0.0
    for l in range(n):
        # Thresholds T_k = 2*s_l - s_k for all k
        T_k = 2 * s[l] - s
        # Find number of pairs (i,j) such that S_ij < T_k
        # np.searchsorted returns index of first element >= T_k
        # So index is the count of elements < T_k
        counts = np.searchsorted(S_sorted, T_k, side='left')
        # The sum of W_ij for S_ij < T_k is W_cumsum[counts - 1]
        # Need to handle counts == 0
        Q_l = np.zeros(n)
        mask = counts > 0
        Q_l[mask] = W_cumsum[counts[mask] - 1]
        
        # P_l = sum_k w_k * Q_l[k]
        P_l = np.dot(w, Q_l)
        P += w[l] * P_l
        
    return P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 590  # Keep 10s margin
    
    best_s = None
    best_w = None
    best_P = -1.0
    
    # We will try a few different support sizes.
    # DE is global, so smaller n might be enough to find the structure,
    # but larger n allows for more complex distributions.
    # n=6 to 10 seems reasonable for DE within time limits.
    for n in [4, 6, 8, 10]:
        if time.time() > deadline - 120:
            break
            
        # Bounds for DE
        # s in [0, 1], w in [0, 1]
        bounds = [(0, 1)] * (2 * n)
        
        def objective(x):
            s_raw = x[:n]
            w_raw = x[n:]
            
            # Sort s and reorder w to maintain correspondence
            idx = np.argsort(s_raw)
            s_sorted = s_raw[idx]
            w_sorted = w_raw[idx]
            
            # Normalize weights
            sum_w = np.sum(w_sorted)
            if sum_w < 1e-9:
                return 0.0
            w_norm = w_sorted / sum_w
            
            # Optional: Merge close support points? 
            # For now, just compute.
            try:
                p_val = compute_P(s_sorted, w_norm)
            except Exception:
                return 0.0
            return -p_val
        
        try:
            # Run Differential Evolution
            # popsize=10 is a good start, maxiter=50
            # strategy='best1bin' is standard
            res_de = differential_evolution(
                objective, 
                bounds, 
                maxiter=50, 
                popsize=10, 
                tol=1e-6,
                seed=42,
                polish=False # We will polish manually
            )
            
            if res_de.success or res_de.fun < 0:
                x_best = res_de.x
                s_de = x_best[:n]
                w_de = x_best[n:]
                
                # Sort and normalize for refinement
                idx = np.argsort(s_de)
                s_ref = s_de[idx].copy()
                w_ref = w_de[idx].copy()
                sum_w = np.sum(w_ref)
                if sum_w > 1e-9:
                    w_ref /= sum_w
                else:
                    w_ref = np.ones(n) / n
                
                p_de = -res_de.fun
                if p_de > best_P:
                    best_P = p_de
                    best_s = s_ref.copy()
                    best_w = w_ref.copy()
                    
                # Refine with SLSQP
                if time.time() < deadline - 60:
                    # SLSQP requires a smooth function. 
                    # We will optimize s and w separately or jointly?
                    # Joint is hard due to sorting discontinuity.
                    # Let's assume the order from DE is good and just optimize values.
                    # We need to ensure s stays sorted. 
                    # We can add constraints s_i <= s_{i+1}.
                    
                    def objective_slsqp(params):
                        s_opt = params[:n]
                        w_opt = params[n:]
                        
                        # Enforce sorting? 
                        # If we don't, SLSQP might cross them.
                        # Let's just sort inside, but that's non-differentiable.
                        # Instead, we optimize in the space of sorted s.
                        # This is tricky.
                        
                        # Alternative: Just optimize w given s, then s given w?
                        # Or just use the DE result as is if it's good enough.
                        # Let's try optimizing w first, keeping s fixed.
                        
                        # For simplicity in this block, we'll do a few iterations of alternating optimization.
                        # 1. Optimize w
                        def obj_w(w_params):
                            return -compute_P(s_ref, w_params)
                        
                        cons_w = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
                        bounds_w = [(0, 1)] * n
                        
                        res_w = minimize(
                            obj_w, 
                            w_ref, 
                            method='SLSQP', 
                            bounds=bounds_w,
                            constraints=cons_w,
                            options={'maxiter': 20, 'ftol': 1e-10}
                        )
                        if res_w.success:
                            w_new = np.maximum(res_w.x, 0)
                            w_new /= np.sum(w_new)
                            p_new = compute_P(s_ref, w_new)
                            if p_new > p_de:
                                p_de = p_new
                                w_ref = w_new
                                if p_de > best_P:
                                    best_P = p_de
                                    best_s = s_ref.copy()
                                    best_w = w_ref.copy()
                        
                        # 2. Optimize s
                        # We need to keep s sorted. 
                        # We can parameterize s as cumulative sums of positive steps?
                        # s_0 = 0 (fixed?), s_i = s_{i-1} + delta_i, delta_i > 0.
                        # But s_0 doesn't have to be 0.
                        # Let's just use bounds and hope SLSQP doesn't cross.
                        # Or use constraints s_i - s_{i-1} >= 0.
                        
                        def obj_s(s_params):
                            return -compute_P(s_params, w_ref)
                        
                        # Constraints for sorted s
                        cons_s = []
                        for i in range(1, n):
                            cons_s.append({
                                'type': 'ineq', 
                                'fun': lambda x, i=i: x[i] - x[i-1] - 1e-6
                            })
                        
                        bounds_s = [(0, 1)] * n
                        
                        res_s = minimize(
                            obj_s, 
                            s_ref, 
                            method='SLSQP', 
                            bounds=bounds_s,
                            constraints=cons_s,
                            options={'maxiter': 20, 'ftol': 1e-10}
                        )
                        if res_s.success:
                            s_new = res_s.x
                            # Ensure sorted
                            s_new = np.sort(s_new)
                            # Re-sort w? No, w corresponds to original s_ref order.
                            # If s_new is sorted differently, we must reorder w_ref.
                            # This is complex. Let's assume small moves.
                            # To be safe, let's just check if s_new is close to s_ref order.
                            # Actually, SLSQP with inequality constraints should maintain order.
                            p_new = compute_P(s_new, w_ref)
                            if p_new > p_de:
                                p_de = p_new
                                s_ref = s_new
                                if p_de > best_P:
                                    best_P = p_de
                                    best_s = s_ref.copy()
                                    best_w = w_ref.copy()
                                    
        except Exception as e:
            pass
            
    # Fallback
    if best_s is None:
        n = 2
        best_s = np.array([0.0, 1.0])
        best_w = np.array([0.58, 0.42]) # Approx optimal for 2-point
        
    # Final validation and cleanup
    # Ensure no NaN/Inf
    if np.any(np.isnan(best_s)) or np.any(np.isnan(best_w)):
        best_s = np.array([0.0, 1.0])
        best_w = np.array([0.5, 0.5])
        
    if np.any(np.isinf(best_s)) or np.any(np.isinf(best_w)):
        best_s = np.array([0.0, 1.0])
        best_w = np.array([0.5, 0.5])
        
    # Ensure non-negative
    best_s = np.maximum(best_s, 0)
    best_w = np.maximum(best_w, 0)
    
    # Normalize weights
    sum_w = np.sum(best_w)
    if sum_w < 1e-9:
        best_w = np.ones(len(best_w)) / len(best_w)
    else:
        best_w /= sum_w
        
    return best_s, best_w
