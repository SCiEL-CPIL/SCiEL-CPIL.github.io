"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = 2, 2, 15
    deadline = time.time() + 140  # 10s safety margin
    
    # Strategy:
    # 1. Build the rank-53 baseline (6 Strassen blocks + 1 schoolbook block)
    # 2. Try to find a rank-10 decomposition for 2x2x3 to replace the last
    #    Strassen (rank 7) + schoolbook (rank 4) = 11 with rank 10
    # 3. If found, total rank = 6*7 + 10 = 52
    # 4. Otherwise, return rank 53
    
    U53, V53, W53 = _build_strassen_53(n, m, p)
    
    # Verify the baseline
    if _verify_int(U53, V53, W53, n, m, p):
        best_rank = 53
        best_U, best_V, best_W = U53, V53, W53
    else:
        # Fallback to schoolbook
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    # Try to find rank-10 for 2x2x3
    if time.time() < deadline - 5:
        U10, V10, W10 = _try_find_2x2x3_rank10(deadline)
        if U10 is not None:
            # Build the combined rank-52 decomposition
            U52, V52, W52 = _build_combined_52(n, m, p, U10, V10, W10)
            if _verify_int(U52, V52, W52, n, m, p):
                best_rank = 52
                best_U, best_V, best_W = U52, V52, W52
    
    return {"U": best_U, "V": best_V, "W": best_W}

def _build_strassen_53(n, m, p):
    """Build the rank-53 Strassen-based decomposition for 2x2x15."""
    n, m, p = 2, 2, 15
    blocks = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14,)]
    
    all_U = []
    all_V = []
    all_W = []

    for block in blocks:
        if len(block) == 2:
            k0, k1 = block
            terms = _strassen_terms(k0, k1, n, m, p)
        else:
            k = block[0]
            terms = _schoolbook_terms(k, n, m, p)
        
        for (u_dict, v_dict, w_dict) in terms:
            u_entries = [(i * m + j, c) for (i, j), c in u_dict.items()]
            v_entries = [(j * p + k, c) for (j, k), c in v_dict.items()]
            w_entries = [(k * n + i, c) for (k, i), c in w_dict.items()]
            
            all_U.append(u_entries)
            all_V.append(v_entries)
            all_W.append(w_entries)

    rank = len(all_U)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c in range(rank):
        for (row, val) in all_U[c]:
            U[row][c] = val
        for (row, val) in all_V[c]:
            V[row][c] = val
        for (row, val) in all_W[c]:
            W[row][c] = val

    return U, V, W

def _strassen_terms(k0, k1, n, m, p):
    """Return Strassen terms for a 2x2x2 block."""
    terms = [
        ({(0,0): 1, (1,1): 1}, {(0, k0): 1, (1, k1): 1}, {(k0, 0): 1, (k1, 1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0, k0): 1}, {(k0, 1): 1}),
        ({(0,0): 1}, {(1, k0): 1, (1, k1): -1}, {(k1, 0): 1}),
        ({(1,1): 1}, {(0, k1): 1, (0, k0): -1}, {(k0, 0): 1, (k0, 1): 1}),
        ({(0,0): 1, (0,1): 1}, {(1, k1): 1}, {(k0, 0): -1, (k1, 0): 1}),
        ({(1,0): 1, (0,0): -1}, {(0, k0): 1, (0, k1): 1}, {(k1, 1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1, k0): 1, (1, k1): 1}, {(k0, 0): 1}),
    ]
    return terms

def _schoolbook_terms(k, n, m, p):
    """Return schoolbook terms for a 2x2x1 block."""
    terms = []
    for i in range(n):
        for j in range(m):
            terms.append((
                {(i, j): 1},
                {(j, k): 1},
                {(k, i): 1}
            ))
    return terms

def _build_schoolbook(n, m, p):
    """Build the schoolbook rank-60 decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _verify_int(U, V, W, n, m, p):
    """Verify decomposition using exact integer arithmetic."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                val = 0
                for c in range(rank):
                    val += U[u_row][c] * V[v_row][c] * W[w_row][c]
                if val != 1:
                    return False
    return True

def _build_combined_52(n, m, p, U10, V10, W10):
    """
    Build the rank-52 decomposition using 6 Strassen blocks (rank 42) 
    and a rank-10 decomposition for the last 2x2x3 block.
    U10: (4, 10), V10: (6, 10), W10: (6, 10) for 2x2x3
    """
    n, m, p = 2, 2, 15
    # First 6 Strassen blocks cover columns 0-11
    blocks = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
    
    all_U = []
    all_V = []
    all_W = []

    for block in blocks:
        k0, k1 = block
        terms = _strassen_terms(k0, k1, n, m, p)
        for (u_dict, v_dict, w_dict) in terms:
            u_entries = [(i * m + j, c) for (i, j), c in u_dict.items()]
            v_entries = [(j * p + k, c) for (j, k), c in v_dict.items()]
            w_entries = [(k * n + i, c) for (k, i), c in w_dict.items()]
            all_U.append(u_entries)
            all_V.append(v_entries)
            all_W.append(w_entries)
    
    # Add the rank-10 terms for columns 12, 13, 14
    # U10 has shape (4, 10) where row index is i*m+j for 2x2
    # V10 has shape (6, 10) where row index is j*p3+k for 2x2x3 (p3=3)
    # W10 has shape (6, 10) where row index is k3*n+i for 2x2x3
    # We need to map the local indices to global indices
    # For the 2x2x3 block at columns 12,13,14:
    # Local k' in {0,1,2} maps to global k = 12 + k'
    
    for r in range(10):
        u_entries = []
        v_entries = []
        w_entries = []
        
        for i in range(n):
            for j in range(m):
                val = U10[i * m + j, r]
                if val != 0:
                    u_entries.append((i * m + j, val))
        
        for j in range(m):
            for k_local in range(3):
                val = V10[j * 3 + k_local, r]
                if val != 0:
                    k_global = 12 + k_local
                    v_entries.append((j * p + k_global, val))
        
        for k_local in range(3):
            for i in range(n):
                val = W10[k_local * n + i, r]
                if val != 0:
                    k_global = 12 + k_local
                    w_entries.append((k_global * n + i, val))
        
        all_U.append(u_entries)
        all_V.append(v_entries)
        all_W.append(w_entries)
    
    rank = len(all_U)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c in range(rank):
        for (row, val) in all_U[c]:
            U[row][c] = val
        for (row, val) in all_V[c]:
            V[row][c] = val
        for (row, val) in all_W[c]:
            W[row][c] = val

    return U, V, W

def _try_find_2x2x3_rank10(deadline):
    """
    Attempt to find a rank-10 decomposition for 2x2x3 matrix multiplication.
    Returns (U, V, W) as numpy arrays if found, None otherwise.
    Uses ALS with multiple restarts.
    """
    n, m, p = 2, 2, 3
    rank = 10
    
    best_error = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    np.random.seed(42)
    
    for restart in range(200):
        if time.time() > deadline - 10:
            break
        
        # Random initialization with small values
        U = np.random.randn(4, rank) * 0.5
        V = np.random.randn(6, rank) * 0.5
        W = np.random.randn(6, rank) * 0.5
        
        # ALS iterations
        for it in range(200):
            if time.time() > deadline - 10:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    B = np.zeros((rank, p))
                    for r in range(rank):
                        for k in range(p):
                            B[r, k] = V[j * p + k, r] * W[k * n + i, r]
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(B, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    C = np.zeros((rank, n * m))
                    for r in range(rank):
                        for i in range(n):
                            C[r, i * m + j] = U[i * m + j, r] * W[k * n + i, r]
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(C, np.ones(n * m), rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    D = np.zeros((rank, m))
                    for r in range(rank):
                        for j in range(m):
                            D[r, j] = U[i * m + j, r] * V[j * p + k, r]
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(D, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Check error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(rank):
                            val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                        error += (val - 1.0) ** 2
            
            if error < best_error:
                best_error = error
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if error < 1e-12:
                break
    
    # Check if we found a valid solution
    if best_U is not None and best_error < 1e-10:
        # Try to round to integers or half-integers
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2
        
        # Verify the rounded solution
        if _verify_2x2x3(U_rounded, V_rounded, W_rounded):
            return U_rounded, V_rounded, W_rounded
        
        # Try rounding to integers
        U_rounded = np.round(best_U)
        V_rounded = np.round(best_V)
        W_rounded = np.round(best_W)
        
        if _verify_2x2x3(U_rounded, V_rounded, W_rounded):
            return U_rounded, V_rounded, W_rounded
    
    return None, None, None

def _verify_2x2x3(U, V, W, tol=1e-9):
    """
    Verify that the decomposition exactly decomposes the 2x2x3 matrix multiplication tensor.
    U: (4, rank), V: (6, rank), W: (6, rank)
    """
    n, m, p = 2, 2, 3
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(actual - expected) > tol:
                    return False
    return True
# EVOLVE-BLOCK-END
