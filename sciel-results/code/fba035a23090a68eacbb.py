"""
Matrix multiplication tensor rank search for 2x4 x 4x5.
Uses Strassen-based blocks + targeted local search to reduce rank.
"""
DIMENSIONS = [2, 4, 5]
import numpy as np
import time
import random

def _build_target_tensor():
    """Build the target matrix multiplication tensor T[a,b,c] where
    a = i*m + j, b = j*p + k, c = k*n + i.
    T[a,b,c] = 1 if the indices are consistent (same i,j,k), else 0."""
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
    return T

def _tensor_from_factors(U, V, W):
    """Compute T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    T = np.zeros((U.shape[0], V.shape[0], W.shape[0]), dtype=np.float64)
    for ri in range(r):
        T += np.einsum('a,b,c->abc', U[:, ri], V[:, ri], W[:, ri])
    return T

def _verify(U, V, W, tol=1e-9):
    """Check if decomposition is exact."""
    n, m, p = DIMENSIONS
    T = _tensor_from_factors(U, V, W)
    Texp = _build_target_tensor()
    return np.allclose(T, Texp, atol=tol)

def _strassen_block(U, V, W, col, i1, j1, k1):
    """Add a 2x2x2 Strassen block at offset (i1,j1,k1). Returns new col index."""
    n, m, p = DIMENSIONS
    terms = [
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
    ]
    for u_pat, v_pat, w_pat in terms:
        for (di, dj), coeff in u_pat.items():
            U[(i1+di)*m + (j1+dj)][col] += coeff
        for (dj, dk), coeff in v_pat.items():
            V[(j1+dj)*p + (k1+dk)][col] += coeff
        for (di, dk), coeff in w_pat.items():
            W[(k1+dk)*n + (i1+di)][col] += coeff
        col += 1
    return col

def _schoolbook_2x2x1(U, V, W, col, i1, j1, k1):
    """Add schoolbook terms for 2x2 x 1 block."""
    n, m, p = DIMENSIONS
    for di in range(2):
        for dj in range(2):
            i = i1 + di
            j = j1 + dj
            k = k1
            U[i*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
    return col

def _build_base_decomposition():
    """Build the base rank-36 decomposition using 4 Strassen blocks + 2 schoolbook blocks."""
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0.0] * rank for _ in range(n*m)]
    V = [[0.0] * rank for _ in range(m*p)]
    W = [[0.0] * rank for _ in range(p*n)]
    col = 0
    col = _strassen_block(U, V, W, col, 0, 0, 0)
    col = _strassen_block(U, V, W, col, 0, 2, 0)
    col = _strassen_block(U, V, W, col, 0, 0, 2)
    col = _strassen_block(U, V, W, col, 0, 2, 2)
    col = _schoolbook_2x2x1(U, V, W, col, 0, 0, 4)
    col = _schoolbook_2x2x1(U, V, W, col, 0, 2, 4)
    return U, V, W, rank

def _try_remove_column(U, V, W, target_rank, n_attempts=30, tol=1e-9):
    """
    Try to find a valid decomposition of rank target_rank by removing one column
    from a rank-(target_rank+1) decomposition and adjusting via local perturbation.
    """
    n, m, p = DIMENSIONS
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    current_rank = len(U[0])
    
    if current_rank != target_rank + 1:
        return None
    
    for attempt in range(n_attempts):
        # Pick a random column to try removing
        col_idx = random.randint(0, current_rank - 1)
        
        # Remove the column
        U_test = [row[:col_idx] + row[col_idx+1:] for row in U]
        V_test = [row[:col_idx] + row[col_idx+1:] for row in V]
        W_test = [row[:col_idx] + row[col_idx+1:] for row in W]
        
        if _verify(U_test, V_test, W_test, tol):
            return U_test, V_test, W_test
        
        # Try small perturbations of the remaining columns to fix the error
        # This is a crude local search
        for _ in range(5):
            U_pert = [row[:] for row in U_test]
            V_pert = [row[:] for row in V_test]
            W_pert = [row[:] for row in W_test]
            
            # Add small random perturbations
            for a in range(len(U_pert)):
                for r in range(target_rank):
                    U_pert[a][r] += random.gauss(0, 0.1)
                    V_pert[a][r] += random.gauss(0, 0.1)
                    W_pert[a][r] += random.gauss(0, 0.1)
            
            if _verify(U_pert, V_pert, W_pert, tol):
                return U_pert, V_pert, W_pert
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe
    
    n, m, p = DIMENSIONS
    U, V, W, rank = _build_base_decomposition()
    
    # Verify base decomposition
    if not _verify(U, V, W):
        # Fallback to full schoolbook
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n*m)]
        V = [[0.0] * rank for _ in range(m*p)]
        W = [[0.0] * rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1.0
                    V[j*p + k][col] = 1.0
                    W[k*n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    U_best, V_best, W_best, best_rank = U, V, W, rank
    
    # Try to reduce rank one step at a time
    consecutive_failures = 0
    while time.time() < deadline - 10:
        target = best_rank - 1
        if target < 20:  # Don't go too low
            break
        
        result = _try_remove_column(U_best, V_best, W_best, target, n_attempts=30)
        if result is not None:
            U_best, V_best, W_best = result
            best_rank = target
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures >= 3:
                break
    
    return {"U": U_best, "V": V_best, "W": W_best}
