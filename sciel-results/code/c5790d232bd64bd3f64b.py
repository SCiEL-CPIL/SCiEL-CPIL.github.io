def construct():
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds
    
    # 1. Generate forbidden triples
    triples = []
    for b in range(1, 181):
        for c in range(b+1, 181):
            s = b + c
            prod = b * c
            if prod % s == 0:
                a = prod // s
                if 1 <= a <= 180 and a != b and a != c:
                    triples.append(tuple(sorted((a, b, c))))
    
    # Remove duplicates
    triples = list(set(triples))
    
    n = 180
    
    # If no triples, return all
    if not triples:
        return list(range(1, 181))
        
    # 2. Formulate ILP
    # Maximize sum(x_i)
    # Equivalent to Minimize -sum(x_i)
    c_obj = -np.ones(n)
    
    # Constraints: for each triple (a,b,c), x_a + x_b + x_c <= 2
    # We need to build the A matrix (sparse or dense).
    # Number of triples M.
    M = len(triples)
    
    # Using sparse matrix for efficiency
    from scipy.sparse import lil_matrix
    A = lil_matrix((M, n), dtype=np.float64)
    b_ub = np.full(M, 2.0)
    
    for i, (a, b, c) in enumerate(triples):
        # Indices are 0-based in ILP, 1-based in problem
        A[i, a-1] = 1
        A[i, b-1] = 1
        A[i, c-1] = 1
        
    A = A.tocsr()
    
    # LinearConstraint: A @ x <= b_ub
    # In scipy.optimize.milp, constraints are defined as:
    # lb <= A @ x <= ub
    # So we set lb = -inf, ub = 2
    constraints = LinearConstraint(A, lb=-np.inf, ub=b_ub)
    
    # Bounds for x_i: 0 <= x_i <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: all variables are integers (binary)
    integrality = np.ones(n)
    
    try:
        res = milp(c=c_obj, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0: # Optimal
            # Extract solution
            x = res.x
            # x are floats, close to 0 or 1
            selected = [i+1 for i in range(n) if x[i] > 0.5]
            if selected:
                return sorted(selected)
            else:
                # Fallback
                return [1]
        else:
            # Fallback to greedy if ILP fails
            pass
    except Exception:
        pass

    # Fallback: Greedy Removal (from original code) to ensure valid output
    from collections import defaultdict
    def greedy_remove(triple_list):
        remaining = set(range(1, 181))
        while True:
            active = [t for t in triple_list if all(e in remaining for e in t)]
            if not active:
                break
            cnt = defaultdict(int)
            for t in active:
                for e in t:
                    cnt[e] += 1
            worst = max(cnt.keys(), key=lambda e: (cnt[e], -e))
            remaining.remove(worst)
        return len(remaining), remaining
        
    size1, set1 = greedy_remove(triples)
    return list(sorted(set1))
