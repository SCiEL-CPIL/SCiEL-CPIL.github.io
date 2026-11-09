"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the factors correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _try_rank_r(n, m, p, r, max_iters=100, n_trials=20):
    """Try to find a rank-r decomposition via ALS on the all-ones tensor."""
    np.random.seed(42)
    best_err = float('inf')
    best_U = best_V = best_W = None
    
    for trial in range(n_trials):
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        for it in range(max_iters):
            # Update U: for each (i,j), solve for U[i*m+j,:]
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, r))
                    for k in range(p):
                        for c in range(r):
                            A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    b = np.ones(p)
                    try:
                        u_new, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except:
                        continue
                    U[i*m+j, :] = u_new
            
            # Update V: for each (j,k), solve for V[j*p+k,:]
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, r))
                    for i in range(n):
                        for c in range(r):
                            A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    b = np.ones(n)
                    try:
                        v_new, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except:
                        continue
                    V[j*p+k, :] = v_new
            
            # Update W: for each (i,k), solve for W[k*n+i,:]
            for i in range(n):
                for k in range(p):
                    A = np.zeros((m, r))
                    for j in range(m):
                        for c in range(r):
                            A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    b = np.ones(m)
                    try:
                        w_new, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except:
                        continue
                    W[k*n+i, :] = w_new
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(r):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        err += (val - 1.0)**2
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < 1e-10:
                break
        
        if best_err < 1e-10:
            break
    
    if best_err < 1e-10 and best_U is not None:
        U_list = [[float(best_U[i, c]) for c in range(r)] for i in range(n*m)]
        V_list = [[float(best_V[i, c]) for c in range(r)] for i in range(m*p)]
        W_list = [[float(best_W[i, c]) for c in range(r)] for i in range(p*n)]
        
        # Round to half-integers
        U_list = [[round(v*2)/2 for v in row] for row in U_list]
        V_list = [[round(v*2)/2 for v in row] for row in V_list]
        W_list = [[round(v*2)/2 for v in row] for row in W_list]
        
        if _verify(U_list, V_list, W_list, n, m, p):
            return U_list, V_list, W_list
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try ranks from lower bound up
    # Lower bound for tensor rank is max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    for r in range(30, 91):
        result = _try_rank_r(n, m, p, r)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
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
# EVOLVE-BLOCK-END
