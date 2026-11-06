"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    from math import gcd
    from scipy.optimize import milp, LinearConstraint, Bounds
    
    N = 128
    
    # 1. Identify all forbidden triples {a, b, c} with 1 <= a < b < c <= N
    # such that gcd(a,b) == gcd(a,c) == gcd(b,c)
    forbidden = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gcd(a, b)
            for c in range(b + 1, N + 1):
                if gcd(a, c) == g_ab and gcd(b, c) == g_ab:
                    forbidden.append((a - 1, b - 1, c - 1))
    
    # 2. Formulate as Integer Linear Programming (MILP)
    # Maximize sum(x_i) subject to x_a + x_b + x_c <= 2 for all forbidden triples
    # Minimize -sum(x_i)
    
    c_obj = -np.ones(N)
    
    if not forbidden:
        return list(range(1, N + 1))
        
    # Build constraint matrix
    # Number of constraints M = len(forbidden)
    M = len(forbidden)
    A_ub = np.zeros((M, N))
    for i, (a, b, c) in enumerate(forbidden):
        A_ub[i, a] = 1.0
        A_ub[i, b] = 1.0
        A_ub[i, c] = 1.0
    
    b_ub = np.ones(M) * 2.0
    
    constraints = LinearConstraint(A_ub, -np.inf, b_ub)
    bounds = Bounds(0, 1)
    integrality = np.ones(N)
    
    try:
        res = milp(c_obj, constraints=constraints, integrality=integrality, bounds=bounds)
        if res.success:
            selected = [i + 1 for i in range(N) if res.x[i] > 0.5]
            if selected:
                return selected
    except Exception:
        pass
    
    # 3. Fallback: Greedy Heuristic on full set
    # Sort by number of conflicts (degree in hypergraph) or just random
    import random
    
    def is_valid(new_elem, current_set):
        """Check if adding new_elem to current_set maintains the no-equal-GCD-triple property."""
        # Pre-computed check is faster, but for fallback we can just check pairs
        # Since this is fallback, we can be slightly slower but correct
        for a in current_set:
            g_an = gcd(a, new_elem)
            for b in current_set:
                if a == b:
                    continue
                g_ab = gcd(a, b)
                g_bn = gcd(b, new_elem)
                if g_ab == g_an == g_bn:
                    return False
        return True

    best_selected = []
    
    # Try a few random orders
    for _ in range(10):
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        sel = []
        for c in candidates:
            if is_valid(c, sel):
                sel.append(c)
        if len(sel) > len(best_selected):
            best_selected = sel
            
    return best_selected
# EVOLVE-BLOCK-END
