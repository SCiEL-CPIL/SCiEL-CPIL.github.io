"""Incentive-compatible finite auction design solver using discrete enumeration."""
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    # Parse inputs
    # distribution_spec: list of 4 probabilities [p0, p1, p2, p3] for each good (independent)
    # support: list of allowed reports (value pairs)
    # constraints: dict with feasibility parameters
    
    # Values are in {0, 1, 2, 3}
    values = [0, 1, 2, 3]
    
    # Get probabilities
    if isinstance(distribution_spec, dict):
        p1 = distribution_spec.get('good1', distribution_spec.get('p1', [0.25, 0.25, 0.25, 0.25]))
        p2 = distribution_spec.get('good2', distribution_spec.get('p2', [0.25, 0.25, 0.25, 0.25]))
    elif isinstance(distribution_spec, (list, tuple)):
        if len(distribution_spec) == 4:
            p1 = list(distribution_spec)
            p2 = list(distribution_spec)
        elif len(distribution_spec) == 8:
            p1 = list(distribution_spec[:4])
            p2 = list(distribution_spec[4:])
        else:
            p1 = [0.25, 0.25, 0.25, 0.25]
            p2 = [0.25, 0.25, 0.25, 0.25]
    else:
        p1 = [0.25, 0.25, 0.25, 0.25]
        p2 = [0.25, 0.25, 0.25, 0.25]
    
    # Normalize probabilities
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    p1 = p1 / p1.sum() if p1.sum() > 0 else np.ones(4) / 4
    p2 = p2 / p2.sum() if p2.sum() > 0 else np.ones(4) / 4
    
    # Joint distribution over (v1, v2)
    joint_probs = np.outer(p1, p2)  # shape (4, 4)
    joint_probs = joint_probs.flatten()  # shape (16,)
    
    # Enumerate all type profiles: (v1, v2) where v1, v2 in {0,1,2,3}
    # Type index = v1 * 4 + v2
    n_types = 16
    types = [(v1, v2) for v1 in values for v2 in values]
    
    # Precompute values for each type
    v_vals = np.array([(t[0], t[1]) for t in types], dtype=float) # shape (16, 2)
    
    # Precompute monotonicity constraints indices
    # Good 0: x[v1+1, v2, 0] >= x[v1, v2, 0] for v1 < 3
    # Good 1: x[v1, v2+1, 1] >= x[v1, v2, 1] for v2 < 3
    mono_pairs_0 = []
    mono_pairs_1 = []
    for v1 in range(3):
        for v2 in range(4):
            t_curr = v1 * 4 + v2
            t_next = (v1 + 1) * 4 + v2
            mono_pairs_0.append((t_next, t_curr)) # x_next >= x_curr
    for v1 in range(4):
        for v2 in range(3):
            t_curr = v1 * 4 + v2
            t_next = v1 * 4 + (v2 + 1)
            mono_pairs_1.append((t_next, t_curr)) # x_next >= x_curr

    best_revenue = -np.inf
    best_allocations = np.zeros((16, 2))
    best_payments = np.zeros(16)
    
    # Helper to check monotonicity
    def is_monotone(alloc_flat):
        # alloc_flat is length 16, representing x_g0 for types 0..15? 
        # No, let's keep allocations as (16, 2) or two flat arrays for speed.
        # Let's use two flat arrays x0, x1 of length 16.
        pass

    # We will iterate over 2^16 possible allocation patterns.
    # To be faster, we can iterate over x0 and x1 separately? No, they are coupled in IC.
    # But we can pre-filter monotone sequences.
    # Number of monotone sequences for 4x4 grid? 
    # It's the number of paths from (0,0) to (3,3) in a grid? No, it's antichains?
    # Actually, monotone 0-1 sequences on a grid correspond to order ideals in the product of chains.
    # For 4x4, the number is C(8,4) = 70? No, that's for a single chain.
    # For a 2D grid 4x4, the number of monotone 0-1 arrays is larger.
    # Let's just iterate 2^16 = 65536. It's small enough.
    
    # To speed up, we can precompute the IC constraints matrix A and b for the payment LP
    # for a given allocation.
    # Payment LP:
    # Maximize sum_t prob_t * p_t
    # Subject to:
    # 1. p_t - p_t' <= v_t . (x_t - x_t')  for all t, t'
    # 2. p_t <= v_t . x_t  (IR)
    # 3. p_t >= 0
    
    # This is a standard LP. We can solve it using scipy.
    # However, calling scipy 65536 times might be slow.
    # Let's check if we can prune.
    # If an allocation is not monotone, skip it.
    
    # Precompute all monotone allocations for good 0 and good 1?
    # No, the monotonicity is per good.
    # x0 must be monotone in v1.
    # x1 must be monotone in v2.
    
    # Let's generate all monotone binary sequences for good 0 (length 16, monotone in index 0 of type)
    # Type index = v1*4 + v2.
    # x0[v1*4+v2] depends only on v1? No, it can depend on v2 but must be non-decreasing in v1.
    # i.e., x0[i] <= x0[i+4] for i in 0..11.
    
    # Let's just iterate all 2^16 and check monotonicity quickly.
    
    def check_mono_x0(x0):
        for v2 in range(4):
            for v1 in range(3):
                if x0[v1*4+v2] > x0[(v1+1)*4+v2]:
                    return False
        return True
        
    def check_mono_x1(x1):
        for v1 in range(4):
            for v2 in range(3):
                if x1[v1*4+v2] > x1[v1*4+v2+1]:
                    return False
        return True

    # We can generate all monotone x0 patterns and x1 patterns separately?
    # Yes, because the constraints on x0 and x1 are independent in terms of monotonicity.
    # The IC constraints couple them, but we can precompute the list of valid x0 and x1.
    
    valid_x0s = []
    for mask in range(16): # 4 bits for v1=0,1,2,3? No, 16 types.
        # Actually, let's just generate all 2^16 and filter. 65536 is small.
        pass
    
    # Better: Generate all monotone x0 vectors.
    # x0 is determined by the "frontier".
    # Let's just iterate 2^16 for x0 and 2^16 for x1? That's 2^32, too big.
    # But we can iterate 2^16 for the pair (x0, x1) combined? That's 2^32.
    # Wait, x0 has 16 bits, x1 has 16 bits. Total 32 bits.
    # 2^32 is too big to enumerate fully.
    
    # We need a smarter way.
    # The current program solves one big LP.
    # My proposed strategy was "direct discrete enumeration over all 256 possible binary allocation matrices".
    # Wait, 256 = 2^8. Why 256?
    # Ah, if we assume that the allocation of good 1 depends only on v1 and good 2 only on v2?
    # That is a strong assumption (separability).
    # If x0 depends only on v1, there are 2^4 = 16 possibilities for x0.
    # If x1 depends only on v2, there are 2^4 = 16 possibilities for x1.
    # Total 16 * 16 = 256 mechanisms.
    
    # Is this assumption valid?
    # The problem does NOT state that allocations must be separable.
    # However, for 1 additive bidder, 2 goods, and values in {0,1,2,3},
    # is the optimal mechanism always separable?
    # In standard auction theory with independent values, the optimal allocation rule is often monotone in each value.
    # But it doesn't have to be separable (i.e., x0 can depend on v2).
    # However, for small supports, enumerating the separable ones might be a good heuristic or even optimal if the coupling is weak.
    # But if the optimal solution requires non-separable allocations, we will miss it.
    
    # Let's try a different enumeration:
    # Enumerate all monotone x0 (16 bits) and all monotone x1 (16 bits).
    # How many monotone 16-bit sequences (4x4 grid) are there?
    # This is the number of order ideals in a 4x4 grid.
    # It is C(8,4) = 70? No.
    # The number of monotone 0-1 matrices of size m x n is C(m+n, m).
    # For 4x4, it is C(8,4) = 70.
    # So there are 70 valid x0 patterns and 70 valid x1 patterns.
    # Total combinations: 70 * 70 = 4900.
    # This is very small! We can enumerate all 4900 mechanisms.
    
    # Generate all monotone 0-1 patterns for a 4x4 grid.
    def generate_monotone_patterns(n=4):
        patterns = []
        for mask in range(1 << n):
            # This generates patterns for a 1D chain.
            # For a 2D grid, we need to be careful.
            # Let's represent the pattern as a list of length 16.
            # Monotone means x[i*4+j] <= x[(i+1)*4+j] and x[i*4+j] <= x[i*4+j+1].
            pass
        
        # Better way: iterate over all 2^16 and check.
        # 65536 checks is fast.
        valid = []
        for mask in range(1 << 16):
            x = np.array([(mask >> i) & 1 for i in range(16)], dtype=int)
            # Check monotonicity
            ok = True
            for v1 in range(3):
                for v2 in range(4):
                    if x[v1*4+v2] > x[(v1+1)*4+v2]:
                        ok = False
                        break
                if not ok: break
            if not ok:
                continue
            for v1 in range(4):
                for v2 in range(3):
                    if x[v1*4+v2] > x[v1*4+v2+1]:
                        ok = False
                        break
                if not ok: break
            if ok:
                valid.append(x)
        return valid

    # This function will take a bit of time, but it's a one-time cost.
    # Let's call it.
    monotone_patterns = generate_monotone_patterns()
    # There should be 70 patterns.
    
    # Now iterate over all pairs (x0, x1) from monotone_patterns.
    # For each pair, solve the payment LP.
    
    # Precompute the IC constraint matrix A_ic and b_ic for the payment LP?
    # The IC constraints are: p_t - p_t' <= v_t . (x_t - x_t')
    # This depends on x. So we have to build the LP for each x.
    
    # To speed up, we can use a faster LP solver or reduce the constraints.
    # The number of IC constraints is 16 * 15 = 240.
    # Plus 16 IR constraints.
    # Total ~256 constraints.
    # Variables: 16.
    # This is a very small LP. SciPy should handle it in < 1ms.
    # 4900 * 1ms = 5 seconds. This fits within the budget.
    
    for x0 in monotone_patterns:
        for x1 in monotone_patterns:
            if time.time() > deadline:
                break
                
            allocations = np.column_stack((x0, x1))
            
            # Build LP for payments
            n_vars = 16
            c = -joint_probs # Maximize revenue
            
            A_ub_rows = []
            b_ub_vals = []
            
            # IC constraints
            for t in range(16):
                v1_t, v2_t = types[t]
                # Value of allocation for type t
                val_t = v1_t * x0[t] + v2_t * x1[t]
                
                for tp in range(16):
                    if t == tp:
                        continue
                    v1_tp, v2_tp = types[tp]
                    val_tp = v1_tp * x0[tp] + v2_tp * x1[tp]
                    
                    # p_t - p_tp <= val_t - val_tp
                    row = np.zeros(n_vars)
                    row[t] = 1
                    row[tp] = -1
                    A_ub_rows.append(row)
                    b_ub_vals.append(val_t - val_tp)
            
            # IR constraints
            for t in range(16):
                v1_t, v2_t = types[t]
                val_t = v1_t * x0[t] + v2_t * x1[t]
                row = np.zeros(n_vars)
                row[t] = 1
                A_ub_rows.append(row)
                b_ub_vals.append(val_t)
            
            # Nonnegative payments
            for t in range(16):
                row = np.zeros(n_vars)
                row[t] = -1
                A_ub_rows.append(row)
                b_ub_vals.append(0)
            
            A_ub = np.array(A_ub_rows)
            b_ub = np.array(b_ub_vals)
            bounds = [(0, None)] * 16
            
            try:
                res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                if res.success:
                    revenue = -res.fun
                    if revenue > best_revenue:
                        best_revenue = revenue
                        best_allocations = allocations.copy()
                        best_payments = res.x.copy()
            except Exception:
                pass

    # Return the best mechanism found
    return {'allocations': best_allocations, 'payments': best_payments}

