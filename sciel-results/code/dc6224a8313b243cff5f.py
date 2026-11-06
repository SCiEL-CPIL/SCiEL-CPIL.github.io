import numpy as np
import time
import random
import sys
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # 1. Generate all arithmetic progressions (APs) of length >= 2
    # Note: The task says "including singletons", but a singleton sum is just the sign itself (magnitude 1).
    # If we minimize the max over length >= 2, the max will be at least 1.
    # Usually "discrepancy" problems imply length >= 2, but let's be precise.
    # "any arithmetic progression contained in [1,256], any start, positive step and length including one."
    # If length 1 is included, the max absolute sum is at least 1.
    # Most discrepancy definitions focus on length >= 2. 
    # However, if the evaluator checks length 1, the max is at least 1.
    # Let's assume the standard interpretation where we care about the "structure" of the discrepancy.
    # If I include length 1, the constraint is trivially satisfied if max >= 1.
    # Let's generate APs of length >= 2 to find the structural minimum, 
    # but keep in mind the absolute minimum possible is 1.
    
    ap_indices = []
    for start in range(1, n + 1):
        for step in range(1, n + 1):
            terms = list(range(start - 1, n, step))
            if len(terms) >= 2:
                ap_indices.append(np.array(terms, dtype=np.int32))
                
    num_aps = len(ap_indices)
    
    if num_aps == 0:
        return [1] * n

    # 2. Strategy: Bisection on Discrepancy D using MILP
    # We want to find the smallest D such that there exists x_i in {-1, 1} 
    # such that for all APs k, -D <= sum(x_i for i in AP_k) <= D.
    
    # Variable substitution for MILP:
    # MILP requires continuous or integer variables.
    # Let y_i be binary variables (0 or 1).
    # x_i = 2*y_i - 1.
    # If y_i = 1, x_i = 1. If y_i = 0, x_i = -1.
    
    # Objective: Minimize D? 
    # MILP in scipy minimizes c @ x.
    # We can set up the problem as:
    # Minimize D
    # Subject to:
    #   sum((2*y_i - 1) for i in AP_k) <= D  for all k
    #   -sum((2*y_i - 1) for i in AP_k) <= D  for all k
    #   y_i in {0, 1}
    #   D >= 0 (D is continuous)
    
    # This is a mixed-integer linear program.
    # Variables: y_0, ..., y_{n-1}, D. Total n+1 variables.
    # y_i are integer (binary). D is continuous.
    
    # Constraints:
    # For each AP k with indices idx:
    #   sum_{i in idx} (2*y_i - 1) - D <= 0
    #   -sum_{i in idx} (2*y_i - 1) - D <= 0
    
    # Let's build the constraint matrix A for scipy.optimize.milp.
    # A_ub @ x <= b_ub
    # x = [y_0, ..., y_{n-1}, D]
    
    # The number of constraints is 2 * num_aps.
    # This might be large. num_aps is roughly n^2/2? 
    # For n=256, number of APs is significant.
    # Let's estimate: 
    # step 1: 255 APs of len 2, ..., 1 AP of len 256. Sum ~ 256^2/2 = 32768.
    # step 2: ~128^2/2 = 8192.
    # Total APs is roughly sum_{s=1}^{128} (n-s)^2/s ... actually it's less.
    # Total APs of length >= 2 is O(n^2 log n) or similar.
    # For n=256, it's probably around 50,000 - 100,000 constraints.
    # 100k constraints x 257 variables. This is a large MILP.
    # SciPy's milp uses HiGHS. HiGHS is fast, but 100k constraints might be slow for a 50s budget if we do bisection.
    
    # Alternative: Use local search first to get a good upper bound D_max, then try to prove feasibility for smaller D?
    # Or just use local search if MILP is too slow?
    # The prompt asks for a DIFFERENT strategy. MILP is the requested direction.
    # If MILP is too slow, I should implement a fallback or limit the scope.
    
    # Let's first run a quick local search to get an initial D.
    # Then use MILP to see if we can go lower?
    # Actually, bisection requires checking feasibility.
    # Feasibility check for a fixed D:
    # Minimize 0 (or 1)
    # Subject to constraints with D fixed.
    # This is a pure binary linear program (feasibility).
    
    # Let's try a hybrid: 
    # 1. Run a fast local search to get a candidate solution and its max sum D_ls.
    # 2. Try to solve MILP for D < D_ls to see if we can improve.
    # 3. If MILP is infeasible for D_ls - 1, then D_ls is optimal (or close).
    # 4. If MILP is feasible, use that solution.
    
    # But constructing the full constraint matrix for 100k APs takes time and memory.
    # Let's check the size of ap_indices first.
    
    # To speed up MILP, we can use the fact that many APs are redundant or dominated?
    # Or just try it and see if it fits in time.
    
    # Let's implement the local search first as a baseline and to get a tight upper bound.
    
    # --- Local Search Part (for initial bound and fallback) ---
    
    # Precompute pos_to_aps for local search
    pos_to_aps = [[] for _ in range(n)]
    for ap_id, idx in enumerate(ap_indices):
        for i in idx:
            pos_to_aps[i].append(ap_id)
    
    def compute_max_sum(signs):
        max_sum = 0
        for idx in ap_indices:
            s = np.sum(signs[idx])
            if abs(s) > max_sum:
                max_sum = abs(s)
        return max_sum

    def local_search(signs, max_iters, deadline, rng):
        signs = signs.copy()
        ap_sums = np.array([np.sum(signs[idx]) for idx in ap_indices], dtype=np.int32)
        
        # Histogram for max tracking
        max_possible = n
        hist = np.zeros(max_possible + 1, dtype=np.int32)
        for s in ap_sums:
            hist[abs(s)] += 1
            
        current_max = int(np.max(np.where(hist > 0, np.arange(len(hist)), 0)))
        best_signs = signs.copy()
        best_max = current_max
        
        temperature = 2.0
        cooling = 0.999
        min_temp = 0.01
        
        for _ in range(max_iters):
            if time.time() > deadline - 20.0: # Leave time for MILP
                break
                
            i = rng.randint(0, n - 1)
            old_sign = signs[i]
            new_sign = -old_sign
            delta = 2 * new_sign
            
            affected = pos_to_aps[i]
            old_max = current_max
            new_max = current_max
            
            for ap_id in affected:
                old_abs = abs(ap_sums[ap_id])
                hist[old_abs] -= 1
                ap_sums[ap_id] += delta
                new_abs = abs(ap_sums[ap_id])
                hist[new_abs] += 1
                if new_abs > new_max:
                    new_max = new_abs
            
            if hist[old_max] == 0:
                for k in range(old_max - 1, -1, -1):
                    if hist[k] > 0:
                        new_max = k
                        break
                else:
                    new_max = 0
            
            delta_max = new_max - current_max
            
            if delta_max <= 0 or (temperature > min_temp and rng.random() < np.exp(-delta_max / temperature)):
                signs[i] = new_sign
                current_max = new_max
                if current_max < best_max:
                    best_max = current_max
                    best_signs = signs.copy()
            else:
                signs[i] = old_sign
                for ap_id in affected:
                    old_abs = abs(ap_sums[ap_id])
                    hist[old_abs] -= 1
                    ap_sums[ap_id] -= delta
                    new_abs = abs(ap_sums[ap_id])
                    hist[new_abs] += 1
                current_max = old_max
                
            temperature *= cooling
            
        return best_signs, best_max

    # Run a quick local search to get a good starting point
    rng = random.Random(42)
    thue_morse = np.array([1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)], dtype=np.int8)
    best_signs_ls, best_max_ls = local_search(thue_morse, 2000, deadline, rng)
    
    # Try a few random restarts quickly
    for seed in [1, 2, 3]:
        if time.time() > deadline - 25.0:
            break
        rng_r = random.Random(seed)
        signs_r = np.array([rng_r.choice([-1, 1]) for _ in range(n)], dtype=np.int8)
        bs, bm = local_search(signs_r, 1000, deadline, rng_r)
        if bm < best_max_ls:
            best_max_ls = bm
            best_signs_ls = bs

    # --- MILP Part ---
    
    # Try to improve using MILP.
    # We will attempt to find a solution with max sum <= D_target.
    # Start with D_target = best_max_ls - 1.
    # If feasible, update best.
    # If infeasible, try D_target = best_max_ls. (Should be feasible by definition of best_max_ls, but maybe different solution?)
    # Actually, if we just want to minimize, we can set objective to minimize D.
    
    # Let's set up the MILP to minimize D directly.
    # Variables: y (n binary), D (continuous).
    # Objective: minimize D. c = [0, ..., 0, 1].
    
    # Constraints:
    # For each AP k:
    #   sum_{i in idx} (2*y_i - 1) <= D  =>  2*sum(y_i) - len(idx) - D <= 0
    #   -sum_{i in idx} (2*y_i - 1) <= D => -2*sum(y_i) + len(idx) - D <= 0
    
    # Matrix A shape: (2 * num_aps, n + 1)
    # This is huge. 100k x 257.
    # Memory: 100k * 257 * 8 bytes ~ 200 MB. Acceptable.
    # Time: HiGHS might struggle with 100k constraints in 20 seconds.
    
    # Optimization: Only include APs that are "tight" or likely to be tight?
    # Or just try it. If it takes too long, we return the local search result.
    
    try:
        # Build constraint matrix
        # To save time, we might skip building the full matrix if num_aps is too large.
        if num_aps > 20000:
            # Too large for a quick MILP solve within budget? 
            # Let's try a subset or just rely on local search.
            # Actually, 20k is small. 100k is the concern.
            # Let's check the actual count.
            pass
            
        # We'll build the matrix in chunks or use sparse? SciPy milp accepts dense.
        # Let's try to build it.
        
        A_rows = []
        b_rows = []
        
        # This loop is slow in Python.
        # Vectorization is hard because APs have different lengths.
        # We can precompute the coefficients for y.
        
        # For each AP k, we have two rows.
        # Row 1: coeffs for y are 2 if i in AP, 0 else. coeff for D is -1.
        #         RHS: len(AP)
        # Row 2: coeffs for y are -2 if i in AP, 0 else. coeff for D is -1.
        #         RHS: -len(AP)
        
        # Wait, the inequality is:
        # 2*sum(y) - L - D <= 0  =>  2*sum(y) - D <= L
        # -2*sum(y) + L - D <= 0 => -2*sum(y) - D <= -L
        
        # Let's construct A_ub and b_ub.
        
        # Given time constraints, building a 100k x 257 matrix in a Python loop might take 5-10 seconds.
        # Solving it might take more.
        
        # Let's limit the APs used in MILP to the "most critical" ones?
        # Or just use the local search result if MILP setup is too slow.
        
        # For now, let's try to solve for a specific D (feasibility) rather than optimizing D.
        # This is a Binary Linear Program (BLP) feasibility problem.
        # We can fix D = best_max_ls - 1.
        
        D_target = best_max_ls - 1
        
        if D_target < 1:
            D_target = 1
            
        # If we want to be efficient, we can just check feasibility for D_target.
        # If feasible, we get a new solution.
        
        # Build A for feasibility check (no objective, or minimize 0).
        # Actually, milp requires an objective. We can minimize 0.
        
        # Let's try to build the matrix for D_target.
        
        # To speed up, let's only include APs that currently have sum close to max?
        # No, that's heuristic.
        
        # Let's just try to run the MILP if num_aps is reasonable (< 50k).
        if num_aps < 50000:
            A_ub = np.zeros((2 * num_aps, n + 1), dtype=np.float64)
            b_ub = np.zeros(2 * num_aps, dtype=np.float64)
            
            row_idx = 0
            for k, idx in enumerate(ap_indices):
                L = len(idx)
                
                # Row 1: 2*sum(y_i) - D <= L
                # Indices in idx have coeff 2. D (index n) has coeff -1.
                A_ub[row_idx, idx] = 2.0
                A_ub[row_idx, n] = -1.0
                b_ub[row_idx] = L
                row_idx += 1
                
                # Row 2: -2*sum(y_i) - D <= -L
                A_ub[row_idx, idx] = -2.0
                A_ub[row_idx, n] = -1.0
                b_ub[row_idx] = -L
                row_idx += 1
            
            # Bounds
            # y_i in [0, 1], D in [0, n]
            bounds = Bounds(lb=[0]*n + [0], ub=[1]*n + [n])
            
            # Integrality
            integrality = np.zeros(n + 1)
            integrality[:n] = 1  # y_i are integer
            integrality[n] = 0   # D is continuous
            
            # Objective: minimize 0
            c = np.zeros(n + 1)
            
            # Constraint: A_ub @ x <= b_ub
            # We also need to ensure D <= D_target? 
            # No, the constraints already enforce the sums to be within D.
            # But we fixed D? 
            # In the formulation above, D is a variable.
            # To check feasibility for a specific D, we should fix D.
            # Or, we can add constraint D <= D_target.
            
            # Let's add D <= D_target.
            # This adds one more constraint.
            A_ub_full = np.vstack([A_ub, [[0]*n + [1]]])
            b_ub_full = np.append(b_ub, D_target)
            
            constraints = [LinearConstraint(A_ub_full, -np.inf, b_ub_full)]
            
            # Solve
            try:
                res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds, options={"time_limit": 20})
                if res.status == 0: # Optimal
                    # Extract solution
                    y_sol = res.x[:n]
                    # y_sol should be close to 0 or 1
                    x_sol = np.where(y_sol > 0.5, 1, -1).astype(np.int8)
                    
                    # Verify
                    max_sum_sol = compute_max_sum(x_sol)
                    if max_sum_sol <= D_target:
                        best_signs_ls = x_sol
                        best_max_ls = max_sum_sol
                        # If we found a better one, we could try to lower D further,
                        # but time is limited.
                        
            except Exception as e:
                pass # Fallback to local search
        
    except Exception as e:
        pass # Fallback to local search

    # Final check and return
    if best_signs_ls is None:
        return [1] * n
        
    return [int(s) for s in best_signs_ls]

# Test run
if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res[:10])
