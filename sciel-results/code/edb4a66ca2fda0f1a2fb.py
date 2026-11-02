import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 6]

def _verify_exact(U, V, W, n, m, p):
    """Verify exact decomposition using integer arithmetic (entries are half-integers)."""
    r = len(U[0])
    # Multiply by 2 to work with integers
    U2 = [[2 * U[i][j] for j in range(r)] for i in range(n*m)]
    V2 = [[2 * V[i][j] for j in range(r)] for i in range(m*p)]
    W2 = [[2 * W[i][j] for j in range(r)] for i in range(p*n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U2[i*m+j][c] * V2[j*p+k][c] * W2[k*n+i][c]
                if s != 8:  # 2*2*2=8
                    return False
    return True

def _try_rank(n, m, p, r, timeout=10):
    """Try to find a rank r decomposition using optimization."""
    start_time = time.time()
    # Flatten U, V, W into a single vector
    size_u = n * m * r
    size_v = m * p * r
    size_w = p * n * r
    size = size_u + size_v + size_w
    
    # Random initialization
    x0 = np.random.randn(size) * 0.1
    
    def objective(x):
        if time.time() - start_time > timeout:
            return np.inf
        u = x[:size_u].reshape(n*m, r)
        v = x[size_u:size_u+size_v].reshape(m*p, r)
        w = x[size_u+size_v:].reshape(p*n, r)
        
        # Compute tensor T = sum_c u_c \otimes v_c \otimes w_c
        # T[i, j, k] = sum_c u[i*m+j, c] * v[j*p+k, c] * w[k*n+i, c]
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += u[i*m+j, c] * v[j*p+k, c] * w[k*n+i, c]
                    # Target is 1 if i,j,k match, 0 otherwise?
                    # No, the tensor entries are 1 for the "diagonal" and 0 otherwise.
                    # The multiplication tensor has T_{(i,j), (j',k), (k',i')} = 1 if i=i', j=j', k=k'.
                    # So for fixed i,j,k, the only non-zero term is when the indices match.
                    # In our indexing:
                    # U index: i*m + j
                    # V index: j*p + k
                    # W index: k*n + i
                    # The term is non-zero only if we are computing C[i,k] from A[i,j] and B[j,k].
                    # The tensor component is 1.
                    # So we want s to be 1.
                    error += (s - 1.0) ** 2
        return error

    # Use L-BFGS-B or similar
    try:
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'maxfun': 10000})
        if res.fun < 1e-6:
            # Reconstruct U, V, W
            u = res.x[:size_u].reshape(n*m, r)
            v = res.x[size_u:size_u+size_v].reshape(m*p, r)
            w = res.x[size_u+size_v:].reshape(p*n, r)
            
            # Round to half-integers
            U = [[u[i][j] for j in range(r)] for i in range(n*m)]
            V = [[v[i][j] for j in range(r)] for i in range(m*p)]
            W = [[w[i][j] for j in range(r)] for i in range(p*n)]
            
            # Round to nearest 0.5
            U = [[round(U[i][j] * 2) / 2 for j in range(r)] for i in range(n*m)]
            V = [[round(V[i][j] * 2) / 2 for j in range(r)] for i in range(m*p)]
            W = [[round(W[i][j] * 2) / 2 for j in range(r)] for i in range(p*n)]
            
            if _verify_exact(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try ranks 20, 21, 22
    for r in [20, 21, 22]:
        if time.time() - start_time > 120:
            break
        result = _try_rank(n, m, p, r, timeout=30)
        if result:
            return result
            
    # Fallback: schoolbook decomposition
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
