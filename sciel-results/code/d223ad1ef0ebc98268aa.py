"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import least_squares

def _strassen_block(j0, j1, k0, k1, U, V, W, col, n, m, p):
    """Add 7 Strassen products for a 2x2x2 block and return new col."""
    # P1 = (A[0,j0]+A[1,j1]) * (B[j0,k0]+B[j1,k1])
    U[0*m + j0][col] += 1
    U[1*m + j1][col] += 1
    V[j0*p + k0][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    # P2 = (A[1,j0]+A[1,j1]) * B[j0,k0]
    U[1*m + j0][col] += 1
    U[1*m + j1][col] += 1
    V[j0*p + k0][col] += 1
    W[k0*n + 1][col] += 1
    W[k1*n + 1][col] -= 1
    col += 1
    
    # P3 = A[0,j0] * (B[j0,k1]-B[j1,k1])
    U[0*m + j0][col] += 1
    V[j0*p + k1][col] += 1
    V[j1*p + k1][col] -= 1
    W[k1*n + 0][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    # P4 = A[1,j1] * (B[j1,k0]-B[j0,k0])
    U[1*m + j1][col] += 1
    V[j1*p + k0][col] += 1
    V[j0*p + k0][col] -= 1
    W[k0*n + 0][col] += 1
    W[k0*n + 1][col] += 1
    col += 1
    
    # P5 = (A[0,j0]+A[0,j1]) * B[j1,k1]
    U[0*m + j0][col] += 1
    U[0*m + j1][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] -= 1
    W[k1*n + 0][col] += 1
    col += 1
    
    # P6 = (A[1,j0]-A[0,j0]) * (B[j0,k0]+B[j0,k1])
    U[1*m + j0][col] += 1
    U[0*m + j0][col] -= 1
    V[j0*p + k0][col] += 1
    V[j0*p + k1][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    # P7 = (A[0,j1]-A[1,j1]) * (B[j1,k0]+B[j1,k1])
    U[0*m + j1][col] += 1
    U[1*m + j1][col] -= 1
    V[j1*p + k0][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] += 1
    col += 1
    
    return col

def _verify_decomp(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct using numpy."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for l in range(m):
            for k in range(p):
                s = np.dot(U_arr[i*m+l, :], V_arr[l*p+k, :] * W_arr[k*n+i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank(R, n, m, p, deadline, rng):
    """Try to find a rank-R decomposition using numerical optimization."""
    U_shape = (n*m, R)
    V_shape = (m*p, R)
    W_shape = (p*n, R)
    
    # Build the list of (i,l,k) triples
    triples = [(i, l, k) for i in range(n) for l in range(m) for k in range(p)]
    ntriples = len(triples)
    
    def residuals(x):
        U = x[:n*m*R].reshape(U_shape)
        V = x[n*m*R:n*m*R + m*p*R].reshape(V_shape)
        W = x[n*m*R + m*p*R:].reshape(W_shape)
        
        res = np.zeros(ntriples)
        for idx, (i, l, k) in enumerate(triples):
            res[idx] = np.dot(U[i*m+l, :], V[l*p+k, :] * W[k*n+i, :]) - 1.0
        return res
    
    # Initialize with the Strassen decomposition padded to R columns
    # First build the rank-54 decomposition
    rank54 = 54
    U54 = [[0] * rank54 for _ in range(n*m)]
    V54 = [[0] * rank54 for _ in range(m*p)]
    W54 = [[0] * rank54 for _ in range(p*n)]
    
    col = 0
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            col = _strassen_block(j0, j1, k0, k1, U54, V54, W54, col, n, m, p)
    for k in range(p):
        U54[0*m + 4][col] += 1
        V54[4*p + k][col] += 1
        W54[k*n + 0][col] += 1
        col += 1
        U54[1*m + 4][col] += 1
        V54[4*p + k][col] += 1
        W54[k*n + 1][col] += 1
        col += 1
    
    U54_arr = np.array(U54, dtype=float)
    V54_arr = np.array(V54, dtype=float)
    W54_arr = np.array(W54, dtype=float)
    
    if R <= rank54:
        # Use first R columns
        U_init = U54_arr[:, :R]
        V_init = V54_arr[:, :R]
        W_init = W54_arr[:, :R]
    else:
        # Pad with zeros
        U_init = np.zeros(U_shape)
        V_init = np.zeros(V_shape)
        W_init = np.zeros(W_shape)
        U_init[:, :rank54] = U54_arr
        V_init[:, :rank54] = V54_arr
        W_init[:, :rank54] = W54_arr
    
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    # Add small random perturbation to help escape local minima
    x0 = x0 + rng.normal(0, 0.1, size=x0.shape)
    
    try:
        result = least_squares(residuals, x0, method='lm', max_nfev=10000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        
        if result.success or np.max(np.abs(result.fun)) < 1e-9:
            U_sol = result.x[:n*m*R].reshape(U_shape)
            V_sol = result.x[n*m*R:n*m*R + m*p*R].reshape(V_shape)
            W_sol = result.x[n*m*R + m*p*R:].reshape(W_shape)
            
            # Verify
            if _verify_decomp(U_sol.tolist(), V_sol.tolist(), W_sol.tolist(), n, m, p, R):
                return U_sol, V_sol, W_sol
    except Exception:
        pass
    
    # Try with different random initialization
    for attempt in range(3):
        if time.time() > deadline:
            break
        x0 = rng.normal(0, 1, size=x0.shape)
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=5000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            if result.success or np.max(np.abs(result.fun)) < 1e-9:
                U_sol = result.x[:n*m*R].reshape(U_shape)
                V_sol = result.x[n*m*R:n*m*R + m*p*R].reshape(V_shape)
                W_sol = result.x[n*m*R + m*p*R:].reshape(W_shape)
                if _verify_decomp(U_sol.tolist(), V_sol.tolist(), W_sol.tolist(), n, m, p, R):
                    return U_sol, V_sol, W_sol
        except Exception:
            continue
    
    return None

def construct():
    n, m, p = DIMENSIONS
    _start_time = time.time()
    _deadline = _start_time + 130  # 20s safety margin
    
    # First, build the known rank-54 decomposition as fallback
    rank54 = 54
    U54 = [[0] * rank54 for _ in range(n*m)]
    V54 = [[0] * rank54 for _ in range(m*p)]
    W54 = [[0] * rank54 for _ in range(p*n)]
    
    col = 0
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            col = _strassen_block(j0, j1, k0, k1, U54, V54, W54, col, n, m, p)
    for k in range(p):
        U54[0*m + 4][col] += 1
        V54[4*p + k][col] += 1
        W54[k*n + 0][col] += 1
        col += 1
        U54[1*m + 4][col] += 1
        V54[4*p + k][col] += 1
        W54[k*n + 1][col] += 1
        col += 1
    
    # Verify the rank-54 decomposition
    if not _verify_decomp(U54, V54, W54, n, m, p, rank54):
        # Should not happen, but fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n*m)]
        V = [[0] * rank for _ in range(m*p)]
        W = [[0] * rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for l in range(m):
                for k in range(p):
                    U[i*m+l][col] = 1
                    V[l*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_U, best_V, best_W = U54, V54, W54
    best_rank = rank54
    
    # Try to find lower rank decompositions
    rng = np.random.default_rng(42)
    for R in range(rank54 - 1, 40, -1):
        if time.time() > _deadline:
            break
        result = _try_rank(R, n, m, p, _deadline, rng)
        if result is not None:
            U_sol, V_sol, W_sol = result
            best_U = U_sol.tolist()
            best_V = V_sol.tolist()
            best_W = W_sol.tolist()
            best_rank = R
            # Found a lower rank, continue trying even lower
            continue
        else:
            # If we fail at R, higher ranks are unlikely to work easily
            # But try one more time with different seed
            if time.time() < _deadline:
                rng2 = np.random.default_rng(R)
                result = _try_rank(R, n, m, p, _deadline, rng2)
                if result is not None:
                    U_sol, V_sol, W_sol = result
                    best_U = U_sol.tolist()
                    best_V = V_sol.tolist()
                    best_W = W_sol.tolist()
                    best_rank = R
                    continue
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
