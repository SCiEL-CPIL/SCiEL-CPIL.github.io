"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
from scipy.optimize import linprog

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Try multiple N values, pick the best
    best_h = None
    best_score = float('inf')
    
    # Increased N range to allow better approximation of continuous optimum
    # Vectorized construction allows larger N to be handled quickly
    N_list = [16, 32, 64, 128, 256, 512, 1024, 2048]
    
    for N in N_list:
        if time.time() > deadline:
            break
        
        # Precompute step boundaries
        i_idx = np.arange(N)
        step_starts = 2.0 * i_idx / N
        step_ends = 2.0 * (i_idx + 1) / N
        
        # Candidate s values
        s_candidates = set()
        s_candidates.add(0.0)
        s_candidates.add(1.0)
        s_candidates.add(2.0)
        # Breakpoints
        k_vals = np.arange(N + 1)
        s1_vals = 2.0 * k_vals / N
        s2_vals = 1.0 - 2.0 * k_vals / N
        
        # Filter valid s values
        valid_s1 = s1_vals[(s1_vals >= 0.0) & (s1_vals <= 2.0)]
        valid_s2 = s2_vals[(s2_vals >= 0.0) & (s2_vals <= 2.0)]
        
        s_candidates.update(valid_s1.tolist())
        s_candidates.update(valid_s2.tolist())
        
        s_vals = np.array(sorted(s_candidates))
        
        num_constraints = len(s_vals)
        A_ub = np.zeros((num_constraints, N + 1))
        b_ub = np.zeros(num_constraints)
        
        # Vectorized computation of weights and constants for each s
        for idx, s in enumerate(s_vals):
            if s <= 1.0:
                # Intervals: [0, 1-s] and [1+s, 2]
                end1 = 1.0 - s
                start2 = 1.0 + s
                
                # Overlap with [0, end1]
                hi1 = np.minimum(end1, step_ends)
                lo1 = np.maximum(0.0, step_starts)
                m1 = np.maximum(0.0, hi1 - lo1)
                
                # Overlap with [start2, 2]
                hi2 = np.minimum(2.0, step_ends)
                lo2 = np.maximum(start2, step_starts)
                m2 = np.maximum(0.0, hi2 - lo2)
                
                w = m1 - m2
                d = np.sum(m2)
            else:
                # Intervals: [0, s-1] and [2-s, 2]
                end1 = s - 1.0
                start2 = 2.0 - s
                
                # Overlap with [0, end1]
                hi1 = np.minimum(end1, step_ends)
                lo1 = np.maximum(0.0, step_starts)
                m1 = np.maximum(0.0, hi1 - lo1)
                
                # Overlap with [start2, 2]
                hi2 = np.minimum(2.0, step_ends)
                lo2 = np.maximum(start2, step_starts)
                m2 = np.maximum(0.0, hi2 - lo2)
                
                w = -m1 + m2
                d = np.sum(m1)
            
            A_ub[idx, :N] = w
            A_ub[idx, N] = -1.0
            b_ub[idx] = -d
        
        # Equality constraint: sum h_i = N/2
        A_eq = np.zeros((1, N + 1))
        A_eq[0, :N] = 1.0
        b_eq = np.array([N / 2.0])
        
        # Bounds: 0 <= h_i <= 1, t >= 0
        bounds = [(0, 1)] * N + [(0, None)]
        
        # Objective: minimize t
        c = np.zeros(N + 1)
        c[N] = 1.0
        
        # Solve LP
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                            bounds=bounds, method='highs')
            if result.success:
                t_val = result.x[N]
                h_val = result.x[:N]
                # Clamp to [0,1] and ensure sum is exactly N/2
                h_val = np.clip(h_val, 0.0, 1.0)
                # Adjust to ensure exact sum
                diff = N / 2.0 - np.sum(h_val)
                if abs(diff) > 1e-12:
                    # Distribute adjustment
                    h_val += diff / N
                    h_val = np.clip(h_val, 0.0, 1.0)
                
                if t_val < best_score:
                    best_score = t_val
                    best_h = h_val
        except Exception:
            pass
    
    if best_h is not None:
        return best_h
    else:
        # Fallback to constant
        return np.full(100, 0.5)

# EVOLVE-BLOCK-END
