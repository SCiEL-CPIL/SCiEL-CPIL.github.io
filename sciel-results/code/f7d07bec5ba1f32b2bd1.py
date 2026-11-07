"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    n = 128
    indices = list(range(1, n + 1))
    # Map index value to 0-based index in the variable vector
    val_to_idx = {v: i for i, v in enumerate(indices)}
    
    # Objective: maximize sum of x_i
    # milp minimizes c @ x, so we use c = -1
    c = np.full(n, -1.0)
    
    # Constraints: For each valid triple (a, b, c) with distinct values,
    # let s1=a+b, s2=a+c, s3=b+c.
    # If s1, s2, s3 are all in [1, n], then we cannot select all six: a, b, c, s1, s2, s3.
    # Constraint: x_a + x_b + x_c + x_s1 + x_s2 + x_s3 <= 5
    
    A_rows = []
    b_ub = []
    
    # Iterate over all distinct triples a < b < c
    for a in indices:
        if a > n // 2: break # a+b+c > n is not the only condition, but a+b must be <= n for any sum to be in range? 
        # Actually, we just need to check if the sums are in range.
        for b in indices:
            if b <= a: continue
            if a + b > n: break # Since c > b, a+c > a+b > n, so no sums will be in range? 
            # Wait, if a+b > n, then a+c > n and b+c > n. So no constraint needed.
            # However, we only need the constraint if ALL three sums are in range.
            # If a+b > n, then at least one sum is out of range, so the "all six" condition can never be met.
            # So we can break early.
            for c in indices:
                if c <= b: continue
                s1 = a + b
                s2 = a + c
                s3 = b + c
                # Check if all sums are in 1..n
                if s3 > n: 
                    continue # Since c is increasing, s3 will only get larger. But s1, s2 might be small.
                    # Actually if s3 > n, then b+c > n. The condition requires b+c to be in the set. 
                    # So if b+c > n, the constraint is vacuously satisfied (we can't select b+c).
                    # So we only need to add constraint if s1, s2, s3 are ALL <= n.
                if s1 <= n and s2 <= n and s3 <= n:
                    # Get indices
                    i_a = val_to_idx[a]
                    i_b = val_to_idx[b]
                    i_c = val_to_idx[c]
                    i_s1 = val_to_idx[s1]
                    i_s2 = val_to_idx[s2]
                    i_s3 = val_to_idx[s3]
                    
                    # Create row for A
                    row = np.zeros(n)
                    row[i_a] = 1
                    row[i_b] = 1
                    row[i_c] = 1
                    row[i_s1] = 1
                    row[i_s2] = 1
                    row[i_s3] = 1
                    A_rows.append(row)
                    b_ub.append(5)
    
    if not A_rows:
        return indices
        
    A = np.array(A_rows)
    b_ub = np.array(b_ub)
    
    # Bounds: x_i in [0, 1]
    integrality = np.ones(n)
    bounds = Bounds(lb=0, ub=1)
    constraints = LinearConstraint(A, ub=b_ub)
    
    # Solve
    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    
    if not res.success:
        # Fallback: return all numbers if solver fails (though likely invalid, better to return a valid heuristic? 
        # But task says invalid result scores 0. Let's try to construct a valid set if it fails.
        # A safe valid set is just {1..64} maybe? Or use a simpler heuristic.
        # For now, if it fails, return the all-ones solution which is INVALID. 
        # Better to return a known valid subset. 
        # The set of numbers > n/2 has no sums in range? 
        # If a,b,c > n/2, then a+b > n. So no pair sums are in range. 
        # So any subset of {n/2 + 1, ..., n} is valid.
        # Size is n/2 = 64.
        # The current score is 1.03, implying size > 64 (since seed was likely 64? or random?).
        # Let's just return the top half if solver fails.
        return list(range(n // 2 + 1, n + 1))
        
    # Extract solution
    x = res.x
    selected = [indices[i] for i in range(n) if x[i] > 0.5]
    
    # Ensure types are int
    return [int(v) for v in selected]

# EVOLVE-BLOCK-END
