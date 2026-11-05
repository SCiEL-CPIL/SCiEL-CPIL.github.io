"""Matrix multiplication 5x5x5 tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U, V, W are lists of 7 vectors, each of length 4.
    Indexing: A[i,j] -> i*2+j, B[j,k] -> j*2+k, C[k,i] -> k*2+i
    """
    # Standard Strassen 2x2
    # P1 = (A00-A11)(B11+B12) -- Wait, let's use the verified one from inspiration or standard.
    # Let's use the one from the inspiration which was verified to work for 4x4.
    U = [
        [1, 0, 0, 1],   # A00+A11
        [0, 0, 1, 1],   # A10+A11
        [1, 0, 0, 0],   # A00
        [0, 0, 0, 1],   # A11
        [1, 0, 1, 0],   # A00+A10
        [-1, 1, 0, 0],  # A01-A00
        [0, 0, 1, -1],  # A10-A11
    ]
    V = [
        [1, 0, 0, 1],   # B00+B11
        [1, 0, 0, 0],   # B00
        [0, 0, 1, -1],  # B10-B11
        [-1, 1, 0, 0],  # B01-B00
        [0, 0, 0, 1],   # B11
        [1, 0, 1, 0],   # B00+B10
        [0, 1, 0, 1],   # B01+B11
    ]
    W = [
        [1, 0, 0, 1],   # C00+C11
        [0, 0, 1, -1],  # C10-C11
        [0, 1, 0, 1],   # C01+C11
        [1, 0, 1, 0],   # C00+C10
        [-1, 1, 0, 0],  # -C00+C01
        [0, 0, 0, 1],   # C11
        [1, 0, 0, 0],   # C00
    ]
    return U, V, W

def _build_4x4_strassen():
    """
    Builds 4x4x4 Strassen decomposition (rank 49).
    Returns U, V, W as lists of 49 vectors, each of length 16.
    Indexing: A[i,j] -> i*4+j, B[j,k] -> j*4+k, C[k,i] -> k*4+i
    """
    U2, V2, W2 = _strassen_2x2()
    
    U4 = []
    V4 = []
    W4 = []
    
    # The 4x4 tensor is composed of 2x2 blocks.
    # The rank is 7*7 = 49.
    # Each term (r, s) corresponds to a product of a term from the top-level Strassen
    # and a term from the bottom-level Strassen.
    for r in range(7):
        for s in range(7):
            u = [0] * 16
            v = [0] * 16
            w = [0] * 16
            
            # For A[i,j]: i = i1*2+i2, j = j1*2+j2
            # Coeff = U2[r][i1*2+j1] * U2[s][i2*2+j2]
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            u[i * 4 + j] = U2[r][i1 * 2 + j1] * U2[s][i2 * 2 + j2]
            
            # For B[j,k]: j = j1*2+j2, k = k1*2+k2
            # Coeff = V2[r][j1*2+k1] * V2[s][j2*2+k2]
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            v[j * 4 + k] = V2[r][j1 * 2 + k1] * V2[s][j2 * 2 + k2]
            
            # For C[i,k]: i = i1*2+i2, k = k1*2+k2
            # Coeff = W2[r][i1*2+k1] * W2[s][i2*2+k2]
            # W index in 4x4 is k*4+i
            for i1 in range(2):
                for k1 in range(2):
                    for i2 in range(2):
                        for k2 in range(2):
                            i = i1 * 2 + i2
                            k = k1 * 2 + k2
                            w[k * 4 + i] = W2[r][i1 * 2 + k1] * W2[s][i2 * 2 + k2]
            
            U4.append(u)
            V4.append(v)
            W4.append(w)
    
    return U4, V4, W4

def _verify_float(n, m, p, U, V, W, tol=1e-9):
    """Verify using floating point."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                dot = np.dot(U_np[i * m + j], V_np[j * p + k] * W_np[k * n + i])
                if abs(dot - 1.0) > tol:
                    return False
    return True

def _verify_exact(n, m, p, U, V, W):
    """Exact verification using Fraction arithmetic."""
    rank = len(U[0]) if U else 0
    U_f = [[Fraction(x) for x in row] for row in U]
    V_f = [[Fraction(x) for x in row] for row in V]
    W_f = [[Fraction(x) for x in row] for row in W]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += U_f[i * m + j][c] * V_f[j * p + k][c] * W_f[k * n + i][c]
                if total != Fraction(1):
                    return False
    return True

def _schoolbook(n, m, p):
    """Construct the standard rank n*m*p decomposition."""
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
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Fallback: Schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Strategy: Embed 4x4 Strassen (rank 49) into 5x5
    # The 4x4 core handles indices 0..3 in all dimensions.
    # The remaining 54 terms (boundary) are handled by schoolbook.
    # Total rank: 49 + 54 = 103.
    
    try:
        U4, V4, W4 = _build_4x4_strassen()
        
        core_rank = len(U4)
        boundary_terms = []
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    if i < 4 and j < 4 and k < 4:
                        continue
                    boundary_terms.append((i, j, k))
        
        total_rank = core_rank + len(boundary_terms)
        
        U5 = [[0] * total_rank for _ in range(25)]
        V5 = [[0] * total_rank for _ in range(25)]
        W5 = [[0] * total_rank for _ in range(25)]
        
        # 1. Fill core (4x4)
        for c in range(core_rank):
            for ii in range(16):
                if U4[c][ii] != 0:
                    i, j = divmod(ii, 4)
                    U5[i * 5 + j][c] = U4[c][ii]
            for jj in range(16):
                if V4[c][jj] != 0:
                    j, k = divmod(jj, 4)
                    V5[j * 5 + k][c] = V4[c][jj]
            for kk in range(16):
                if W4[c][kk] != 0:
                    k, i = divmod(kk, 4)
                    W5[k * 5 + i][c] = W4[c][kk]
        
        # 2. Fill boundary (schoolbook)
        col = core_rank
        for i, j, k in boundary_terms:
            U5[i * 5 + j][col] = 1
            V5[j * 5 + k][col] = 1
            W5[k * 5 + i][col] = 1
            col += 1
            
        # Verify
        if _verify_float(n, m, p, U5, V5, W5):
            if _verify_exact(n, m, p, U5, V5, W5):
                if total_rank < best_rank:
                    best = {"U": U5, "V": V5, "W": W5}
                    best_rank = total_rank
                    # print(f"Improved to rank {best_rank}")
        else:
            # If verification fails, we have a bug in the 4x4 construction.
            # Fall back to the previous best (schoolbook or 2x2 embed if we had it)
            pass
            
    except Exception as e:
        # If anything goes wrong, return the schoolbook fallback
        pass
        
    return best

# EVOLVE-BLOCK-END
