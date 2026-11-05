import numpy as np
import time
import itertools
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave margin
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Precompute valid indices for the "1" constraints
    # (u, v, w) such that u=(i,j), v=(j,k), w=(k,i)
    valid_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                valid_indices.append((u, v, w))
                
    all_indices = [(u, v, w) for u in range(nm) for v in range(mp) for w in range(pn)]
    # valid_indices is a subset of all_indices
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        for c, (i, j, k) in enumerate([(i, j, k) for i in range(n) for j in range(m) for k in range(p)]):
            U[i * m + j][c] = 1.0
            V[j * p + k][c] = 1.0
            W[k * n + i][c] = 1.0
        return U, V, W

    def verify(U, V, W):
        # Check if U, V, W define the tensor exactly
        # U, V, W are assumed to be half-integers
        # We check all 900 constraints
        R = U.shape[1]
        for u in range(nm):
            for v in range(mp):
                for w in range(pn):
                    s = 0.0
                    for r in range(R):
                        s += U[u, r] * V[v, r] * W[w, r]
                    # Determine expected value
                    # Check if (u,v,w) is in valid_indices
                    # u = i*m+j, v = j'*p+k', w = k''*n+i''
                    i = u // m
                    j = u % m
                    j_prime = v // p
                    k_prime = v % p
                    k_double_prime = w // n
                    i_double_prime = w % n
                    
                    expected = 1.0 if (j == j_prime and k_prime == k_double_prime and i_double_prime == i) else 0.0
                    
                    # Since entries are half-integers, s should be close to expected
                    if abs(s - expected) > 1e-9:
                        return False
        return True

    def solve_rank(r):
        # Variables: U (nm x r), V (mp x r), W (pn x r)
        # Total vars: (nm + mp + pn) * r
        total_vars = (nm + mp + pn) * r
        x0 = np.random.randn(total_vars) * 0.5
        
        # Fix gauge: set U[0,0] = 1?
        # Or just rely on the optimizer.
        
        def residuals(x):
            U = x[:nm*r].reshape(nm, r)
            V = x[nm*r:(nm+mp)*r].reshape(mp, r)
            W = x[(nm+mp)*r:].reshape(pn, r)
            
            res = np.zeros(nm * mp * pn)
            idx = 0
            for u in range(nm):
                for v in range(mp):
                    for w in range(pn):
                        s = np.sum(U[u, :] * V[v, :] * W[w, :])
                        # Expected value
                        i = u // m
                        j = u % m
                        j_prime = v // p
                        k_prime = v % p
                        k_double_prime = w // n
                        i_double_prime = w % n
                        expected = 1.0 if (j == j_prime and k_prime == k_double_prime and i_double_prime == i) else 0.0
                        res[idx] = s - expected
                        idx += 1
            return res

        try:
            # Use trf for bounds
            # Bounds: [-5, 5] is likely enough for half-integers
            res = least_squares(residuals, x0, method='trf', bounds=([-5]*total_vars, [5]*total_vars), max_nfev=1000)
            
            x_sol = res.x
            U = x_sol[:nm*r].reshape(nm, r)
            V = x_sol[nm*r:(nm+mp)*r].reshape(mp, r)
            W = x_sol[(nm+mp)*r:].reshape(pn, r)
            
            # Round to nearest 0.5
            U = np.round(U * 2) / 2
            V = np.round(V * 2) / 2
            W = np.round(W * 2) / 2
            
            if verify(U, V, W):
                return U, V, W
            else:
                return None
        except Exception:
            return None

    # Try rank 28
    best_U, best_V, best_W = None, None, None
    for attempt in range(10):
        if time.time() > deadline:
            break
        np.random.seed(attempt)
        result = solve_rank(28)
        if result:
            best_U, best_V, best_W = result
            break
            
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Fallback to schoolbook
        U, V, W = schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
