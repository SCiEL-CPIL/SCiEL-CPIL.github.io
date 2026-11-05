import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import itertools
import scipy.sparse

def _seed_solution(q, n, radius):
    # Direct sum of the ternary tetracode (9 words of length 4, a perfect code of covering
    # radius 1) with every word on the other n - 4 coordinates: 9 * 3**(n - 4) words.
    if q != 3 or n < 4:
        # Generic fallback: return all words? No, too many.
        # Return a simple grid?
        # For safety, just return a small set if it's not the specific case.
        # But the task is specific to n=6.
        return [list(t) for t in itertools.product(range(q), repeat=n)]
        
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

def construct(q, n, radius):
    N = q ** n
    
    # If the problem is too large for ILP, use the seed heuristic
    if N > 5000:
        return _seed_solution(q, n, radius)

    # Generate all words
    words = [list(t) for t in itertools.product(range(q), repeat=n)]
    
    # Build the incidence matrix A
    # A[i][j] = 1 if dist(words[i], words[j]) <= radius
    A = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        wi = words[i]
        for j in range(N):
            wj = words[j]
            dist = sum(1 for k in range(n) if wi[k] != wj[k])
            if dist <= radius:
                A[i][j] = 1
    
    A_sparse = scipy.sparse.csr_matrix(A)
    
    # Objective: minimize sum(x)
    c = np.ones(N)
    
    # Constraints: A x >= 1
    constraint = LinearConstraint(A_sparse, lb=np.ones(N), ub=np.inf)
    
    # Bounds: 0 <= x <= 1
    bounds = Bounds(lb=np.zeros(N), ub=np.ones(N))
    
    # Integrality: all variables are integers
    integrality = np.ones(N)
    
    try:
        res = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
        
        if res.status == 0 or res.status == 1: # Optimal or Feasible
            x = (res.x > 0.5).astype(int)
            indices = np.where(x == 1)[0]
            if len(indices) > 0:
                return [words[i] for i in indices]
    except Exception:
        pass
        
    # Fallback to seed solution if ILP fails or returns empty
    return _seed_solution(q, n, radius)
