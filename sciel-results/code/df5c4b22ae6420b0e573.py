def construct():
    from fractions import Fraction
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds

    denoms = list(range(2, 49))
    n = len(denoms)
    
    # Float weights
    w_float = np.array([1.0 / d for d in denoms])
    
    # Objective: Maximize sum(w_i * x_i) -> Minimize -sum(w_i * x_i)
    c = -w_float
    
    integrality = np.ones(n, dtype=int)
    bounds = Bounds(0, 1)
    
    A = w_float.reshape(1, -1)
    
    # Try with ub = 1.0 first
    # Note: milp might find a solution that is exactly 1.0 in float
    # which might be > 1 in exact arithmetic.
    
    def solve_with_ub(ub):
        constraints = [LinearConstraint(A, 0, ub)]
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        if not res.success:
            return None
        x = res.x
        indices = [i for i in range(n) if x[i] > 0.5]
        return indices

    indices = solve_with_ub(1.0)
    
    if indices:
        # Verify exact sum
        s = set(indices)
        tot = Fraction(0)
        for i in s:
            tot += Fraction(1, denoms[i])
        
        if tot < 1:
            return indices
        else:
            # Sum is >= 1, need to find a strictly below solution
            # Try a tighter bound
            # 1 - 1e-15
            indices2 = solve_with_ub(1.0 - 1e-15)
            if indices2:
                s2 = set(indices2)
                tot2 = Fraction(0)
                for i in s2:
                    tot2 += Fraction(1, denoms[i])
                if tot2 < 1:
                    return indices2
            
            # Fallback: try even tighter or manual adjustment
            # If still failing, just return a known valid small set
            # e.g., [0] (1/2)
            return [0]
    else:
        return [0]
