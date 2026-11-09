def construct():
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import lil_matrix
    from math import gcd

    N = 512
    vertices = list(range(1, N + 1))
    n = len(vertices)
    
    # 1. Build conflict list for ILP
    conflicts = []
    for i_idx in range(n):
        i_val = vertices[i_idx]
        for j_idx in range(i_idx + 1, n):
            j_val = vertices[j_idx]
            g = gcd(i_val, j_val)
            if i_val * j_val > N * g:
                conflicts.append((i_idx, j_idx))
                
    m = len(conflicts)
    
    # 2. Formulate ILP
    # Maximize sum(x_i) subject to x_i + x_j <= 1 for conflicts, x_i in {0,1}
    # Minimize -sum(x_i)
    c = -np.ones(n)
    
    # Sparse constraint matrix A (m x n)
    A = lil_matrix((m, n), dtype=np.float64)
    for k, (i_idx, j_idx) in enumerate(conflicts):
        A[k, i_idx] = 1
        A[k, j_idx] = 1
    A = A.tocsr()
    
    bounds = Bounds(lb=0, ub=1)
    integrality = np.ones(n)
    lin_cons = LinearConstraint(A, lb=-np.inf, ub=np.ones(m))
    
    # 3. Solve
    try:
        res = milp(c=c, constraints=lin_cons, integrality=integrality, bounds=bounds)
        if res.status == 0 and res.success:
            solution_indices = np.where(res.x > 0.5)[0]
            solution_values = [vertices[i] for i in solution_indices]
            
            # Validate
            if solution_values:
                valid = True
                for i in range(len(solution_values)):
                    for j in range(i+1, len(solution_values)):
                        a = solution_values[i]
                        b = solution_values[j]
                        if a * b // gcd(a, b) > N:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    return solution_values
    except Exception:
        pass

    # Fallback: Divisors of 504 (highly composite number <= 512)
    # 504 has 24 divisors. This is a valid set.
    divisors = []
    for i in range(1, 505):
        if 504 % i == 0:
            divisors.append(i)
    return divisors
