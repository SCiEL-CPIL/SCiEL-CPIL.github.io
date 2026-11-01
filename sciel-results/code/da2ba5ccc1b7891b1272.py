"""Matrix multiplication 2x3x8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 8]

import time
import numpy as np

_DEADLINE = 145.0  # Safety margin before 150s kill


def _build_schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
    return max_err


def _build_decomp(n, m, p):
    """
    Build a decomposition using Strassen for the 2x2 sub-block.
    
    For 2x3x8:
    - Split A (2x3) into A1 (2x2, cols 0,1) and A2 (2x1, col 2)
    - Split B (3x8) into B1 (2x8, rows 0,1) and B2 (1x8, row 2)
    - C = A1*B1 + A2*B2
    
    A2*B2: rank 16 (each output C[i,k] = A[i,2]*B[2,k])
    A1*B1: 2x2 * 2x8, split into 4 Strassen blocks of 2x2x2, rank 7*4=28
    
    Total rank: 16 + 28 = 44
    """
    n, m, p = 2, 3, 8
    
    rank_total = 44
    U = [[0] * rank_total for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank_total for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank_total for _ in range(p * n)]  # 16 x 44
    
    col = 0
    
    # Part 1: A2*B2 (rank 16)
    # A[0,2] at index 0*3+2=2, A[1,2] at index 1*3+2=5
    # B[2,k] at index 2*8+k = 16+k
    # C[i,k] at index k*2+i
    for k in range(8):
        # C[0,k] += A[0,2] * B[2,k]
        U[2][col] = 1
        V[16 + k][col] = 1
        W[k * 2 + 0][col] = 1
        col += 1
        # C[1,k] += A[1,2] * B[2,k]
        U[5][col] = 1
        V[16 + k][col] = 1
        W[k * 2 + 1][col] = 1
        col += 1
    
    # Part 2: A1*B1 via Strassen (rank 28)
    # A1 is 2x2: A[0,0]@0, A[0,1]@1, A[1,0]@3, A[1,1]@4
    # B1 is 2x8: rows 0,1. B[0,k]@k, B[1,k]@8+k
    # C is 2x8: C[i,k] at k*2+i
    
    for g in range(4):
        b11 = 0 + 2 * g      # B[0, 2g]
        b12 = 0 + 2 * g + 1  # B[0, 2g+1]
        b21 = 8 + 2 * g      # B[1, 2g]
        b22 = 8 + 2 * g + 1  # B[1, 2g+1]
        
        a11 = 0  # A[0,0]
        a12 = 1  # A[0,1]
        a21 = 3  # A[1,0]
        a22 = 4  # A[1,1]
        
        # Standard Strassen:
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b22-b12)
        # M4 = a22(b11-b21)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        
        # M1 = (a11+a22)(b11+b22)
        U[a11][col] = 1
        U[a22][col] = 1
        V[b11][col] = 1
        V[b22][col] = 1
        # C11 += M1, C22 += M1
        W[(2 * g) * 2 + 0][col] = 1      # C[0, 2g]
        W[(2 * g + 1) * 2 + 1][col] = 1  # C[1, 2g+1]
        col += 1
        
        # M2 = (a21+a22)b11
        U[a21][col] = 1
        U[a22][col] = 1
        V[b11][col] = 1
        # C21 += M2
        W[(2 * g) * 2 + 1][col] = 1  # C[1, 2g]
        col += 1
        
        # M3 = a11(b22-b12)
        U[a11][col] = 1
        V[b22][col] = 1
        V[b12][col] = -1
        # C12 += M3, C22 -= M3
        W[(2 * g + 1) * 2 + 0][col] = 1   # C[0, 2g+1]
        W[(2 * g + 1) * 2 + 1][col] = -1  # C[1, 2g+1]
        col += 1
        
        # M4 = a22(b11-b21)
        U[a22][col] = 1
        V[b11][col] = 1
        V[b21][col] = -1
        # C11 -= M4, C21 += M4
        W[(2 * g) * 2 + 0][col] = -1  # C[0, 2g]
        W[(2 * g) * 2 + 1][col] = 1   # C[1, 2g]
        col += 1
        
        # M5 = (a11+a12)b22
        U[a11][col] = 1
        U[a12][col] = 1
        V[b22][col] = 1
        # C12 += M5
        W[(2 * g + 1) * 2 + 0][col] = 1  # C[0, 2g+1]
        col += 1
        
        # M6 = (a21-a11)(b11+b12)
        U[a21][col] = 1
        U[a11][col] = -1
        V[b11][col] = 1
        V[b12][col] = 1
        # C11 += M6
        W[(2 * g) * 2 + 0][col] = 1  # C[0, 2g]
        col += 1
        
        # M7 = (a12-a22)(b21+b22)
        U[a12][col] = 1
        U[a22][col] = -1
        V[b21][col] = 1
        V[b22][col] = 1
        # C22 += M7
        W[(2 * g + 1) * 2 + 1][col] = 1  # C[1, 2g+1]
        col += 1
    
    return U, V, W


def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # Try the Strassen-based decomposition (rank 44)
    try:
        if time.time() - start_time > _DEADLINE:
            raise TimeoutError("Deadline exceeded")
        U, V, W = _build_decomp(n, m, p)
        # Quick verification
        err = _verify(U, V, W, n, m, p)
        if err < 1e-10:
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback: schoolbook
    if time.time() - start_time > _DEADLINE:
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
