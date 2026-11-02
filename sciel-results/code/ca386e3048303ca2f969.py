"""Solves the Erdos minimum overlap problem using Linear Programming."""
import numpy as np
import time
from scipy.optimize import linprog

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    best_h = np.full(100, 0.5)  # Fallback
    best_score = 0.5  # Constant 0.5 gives overlap 0.5 (approx)

    # Try a few N values. 128 and 256 are good balances.
    N_values = [128, 256]

    for N in N_values:
        if time.time() > deadline - 10.0:
            break

        # Build the LP
        # Variables: h_0, ..., h_{N-1}, t
        # Minimize t
        
        # Candidate shifts s
        # The overlap function O(s) is piecewise linear in s for step functions.
        # The maximum of a set of linear functions is convex.
        # To minimize the max, we need to constrain O(s) <= t for all s.
        # It suffices to check s at the breakpoints of the piecewise linear functions.
        # Breakpoints occur when the shift boundary aligns with a step boundary.
        
        s_candidates = set()
        s_candidates.add(0.0)
        s_candidates.add(2.0)
        # Step boundaries are at 2k/N.
        # Shifts s such that s = 2k/N or 1-s = 2k/N (if defined) etc.
        # Actually, the integration limits change when 1-s = 2k/N or 1+s = 2k/N.
        # 1-s = 2k/N => s = 1 - 2k/N
        # 1+s = 2k/N => s = 2k/N - 1
        
        for k in range(N + 1):
            s1 = 2.0 * k / N
            s2 = 1.0 - 2.0 * k / N
            s3 = 2.0 * k / N - 1.0
            if 0.0 <= s1 <= 2.0:
                s_candidates.add(s1)
            if 0.0 <= s2 <= 2.0:
                s_candidates.add(s2)
            if 0.0 <= s3 <= 2.0:
                s_candidates.add(s3)
        
        s_vals = sorted(s_candidates)
        num_constraints = len(s_vals)
        
        A_ub = np.zeros((num_constraints, N + 1))
        b_ub = np.zeros(num_constraints)
        
        step_width = 2.0 / N
        
        for idx, s in enumerate(s_vals):
            d = 0.0
            for i in range(N):
                step_start = 2.0 * i / N
                step_end = 2.0 * (i + 1) / N
                
                # Calculate w_i(s) and d(s) for the linear form:
                # O(s) = sum_i w_i(s) h_i + d(s)
                
                if s <= 1.0:
                    # O(s) = int_0^{1-s} h dx + int_{1+s}^2 (1-h) dx
                    # = sum_i [m1_i h_i] + sum_i [m2_i (1-h_i)]
                    # = sum_i (m1_i - m2_i) h_i + sum_i m2_i
                    # m1_i = measure(step_i ∩ [0, 1-s])
                    # m2_i = measure(step_i ∩ [1+s, 2])
                    
                    lo1, hi1 = max(0.0, step_start), min(1.0 - s, step_end)
                    lo2, hi2 = max(1.0 + s, step_start), min(2.0, step_end)
                    m1 = max(0.0, hi1 - lo1)
                    m2 = max(0.0, hi2 - lo2)
                    
                    A_ub[idx, i] = m1 - m2
                    d += m2
                else:
                    # O(s) = int_0^{s-1} (1-h) dx + int_{2-s}^2 h dx
                    # = sum_i [m1_i (1-h_i)] + sum_i [m2_i h_i]
                    # = sum_i (m2_i - m1_i) h_i + sum_i m1_i
                    # m1_i = measure(step_i ∩ [0, s-1])
                    # m2_i = measure(step_i ∩ [2-s, 2])
                    
                    lo1, hi1 = max(0.0, step_start), min(s - 1.0, step_end)
                    lo2, hi2 = max(2.0 - s, step_start), min(2.0, step_end)
                    m1 = max(0.0, hi1 - lo1)
                    m2 = max(0.0, hi2 - lo2)
                    
                    A_ub[idx, i] = m2 - m1
                    d += m1
            
            # Constraint: sum_i w_i h_i + d - t <= 0
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
        
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                            bounds=bounds, method='highs')
            if result.success:
                t_val = result.x[N]
                h_val = result.x[:N]
                
                # Post-process: clip and adjust sum
                h_val = np.clip(h_val, 0.0, 1.0)
                diff = N / 2.0 - np.sum(h_val)
                if abs(diff) > 1e-12:
                    # Simple adjustment
                    h_val += diff / N
                    h_val = np.clip(h_val, 0.0, 1.0)
                    
                if t_val < best_score:
                    best_score = t_val
                    best_h = h_val
        except Exception:
            pass

    return best_h
