def construct():
    from fractions import Fraction
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds
    import math
    
    denoms = list(range(2, 49))
    n = len(denoms)
    vals = [Fraction(1, d) for d in denoms]
    one = Fraction(1, 1)
    
    # Calculate LCM
    L = 1
    for d in denoms:
        L = L * d // math.gcd(L, d)
        
    # Coefficients c_i = L / d_i
    c_ints = [L // d for d in denoms]
    
    # Objective: Maximize sum(c_i * x_i) -> Minimize sum(-c_i * x_i)
    c_obj = np.array([-float(ci) for ci in c_ints])
    
    # Constraints:
    # 1. sum(c_i * x_i) <= L - 1
    # 2. sum(x_i) >= 1
    
    # For constraint 1:
    A1 = np.array([float(ci) for ci in c_ints])
    lb1 = 0.0
    ub1 = float(L - 1)
    
    # For constraint 2:
    A2 = np.ones(n)
    lb2 = 1.0
    ub2 = np.inf
    
    constraints = [
        LinearConstraint(A1, lb1, ub1),
        LinearConstraint(A2, lb2, ub2)
    ]
    
    integrality = np.ones(n)
    bounds = Bounds(0, 1)
    
    try:
        res = milp(c=c_obj, constraints=constraints, integrality=integrality, bounds=bounds)
        if res.status == 0:
            x = res.x
            indices = [i for i in range(n) if x[i] > 0.5]
            
            # Verify with exact arithmetic
            if indices:
                s = set(indices)
                tot = Fraction(0)
                for i in s:
                    tot += vals[i]
                
                if tot < one:
                    return list(indices)
                else:
                    # If invalid, try to fix by removing largest denominator (smallest value)
                    # Sort by value ascending (largest denominator first)
                    sorted_indices = sorted(indices, key=lambda i: denoms[i])
                    # Actually, to reduce sum, remove the smallest value (largest denominator)
                    # Wait, removing an element reduces the sum.
                    # We want to get below 1.
                    # Just remove elements until sum < 1.
                    # Greedy removal: remove the one that causes the largest drop? No, any drop works.
                    # Let's just remove the one with the largest denominator (smallest value) to preserve the "larger" fractions?
                    # Actually, we just need *a* valid set.
                    # The ILP might have returned a set that sums to >= 1 due to float error in constraint.
                    # Let's remove elements until valid.
                    while tot >= one and indices:
                        # Remove index with largest denominator (smallest value) to minimize loss of "mass" near 1?
                        # Or just remove the last one?
                        # Let's remove the one with the largest denominator.
                        idx_to_remove = max(indices, key=lambda i: denoms[i])
                        indices.remove(idx_to_remove)
                        tot -= vals[idx_to_remove]
                    
                    if indices:
                        return list(indices)
            else:
                # Empty set, fallback
                return [0]
        else:
            # Fallback
            return [0]
    except Exception:
        return [0]
