import numpy as np
import time
import random

def construct():
    n, m, p = 2, 3, 4
    start_time = time.time()
    deadline = start_time + 115.0
    
    def _validate(U, V, W):
        """Check if U, V, W form a valid decomposition. Vectorized."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        r = U.shape[1]
        if U.shape != (n*m, r) or V.shape != (m*p, r) or W.shape != (p*n, r):
            return False
        # T[(i,j),(jp,k),(k2,i2)] = sum_s U[i*m+j,s] * V[jp*p+k,s] * W[k2*n+i2,s]
        # Should be 1 if i==i2, j==jp, k==k2 else 0
        # Build tensor T of shape (n*m, m*p, p*n)
        # T[a,b,c] = sum_s U[a,s] * V[b,s] * W[c,s]
        # This is like a 3-way tensor product
        T = np.einsum('ar,br,cr->abc', U, V, W)
        # Check diagonal: T[i*m+j, jp*p+k, k*n+i] should be 1 if j==jp else 0
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        expected = 1.0 if j == jp else 0.0
                        if abs(T[i*m+j, jp*p+k, k*n+i] - expected) > 1e-6:
                            return False
        # Check off-diagonal: T[i*m+j, jp*p+k, k2*n+i2] should be 0 if not (i==i2 and j==jp and k==k2)
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        for k2 in range(p):
                            for i2 in range(n):
                                if i == i2 and j == jp and k == k2:
                                    continue
                                if abs(T[i*m+j, jp*p+k, k2*n+i2]) > 1e-6:
                                    return False
        return True

    def _try_rank(r, max_iters=2000):
        """Try to find a rank-r decomposition."""
        if time.time() > deadline - 3.0:
            return None
        
        # Strategy: random search with half-integer values
        vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
        
        for it in range(max_iters):
            if time.time() > deadline - 2.0:
                return None
            
            # Random initialization
            U = np.random.choice(vals, size=(n*m, r))
            V = np.random.choice(vals, size=(m*p, r))
            W = np.random.choice(vals, size=(p*n, r))
            
            # Check validity
            if _validate(U, V, W):
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try a few ranks with limited iterations
    for r in range(18, 25):
        if time.time() > deadline - 5.0:
            break
        
        result = _try_rank(r, max_iters=500)
        if result is not None:
            return result
    
    # Return schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
