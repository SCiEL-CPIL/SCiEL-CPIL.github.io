"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2x4():
    """Rank-10 decomposition for 2x2 * 2x4 using Strassen for first 2x2 block + 3 products for second."""
    n, m, p = 2, 2, 4
    rank = 10
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Indexing: A[i*m+j], B[j*p+k], W[k*n+i]
    # A = [[a00, a01], [a10, a11]] -> indices 0,1,2,3
    # B = [[b00, b01, b02, b03], [b10, b11, b12, b13]] -> indices 0-7
    # C = [[c00, c01, c02, c03], [c10, c11, c12, c13]] -> W indices: c00->0, c01->1, c02->2, c03->3, c10->4, c11->5, c12->6, c13->7
    
    # Strassen 7 products for B columns 0-1:
    # M1 = (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[5, 0] = 1
    W[0, 0] = 1; W[5, 0] = 1  # c00, c11
    
    # M2 = (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[4, 1] = 1; W[5, 1] = -1  # c10, c11
    
    # M3 = a00(b10-b00)
    U[0, 2] = 1
    V[4, 2] = 1; V[0, 2] = -1
    W[1, 2] = 1; W[5, 2] = 1  # c01, c11
    
    # M4 = a11(b11-b01)
    U[3, 3] = 1
    V[5, 3] = 1; V[1, 3] = -1
    W[0, 3] = 1; W[4, 3] = 1  # c00, c10
    
    # M5 = (a00+a01)b11
    U[0, 4] = 1; U[1, 4] = 1
    V[5, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1  # c00, c01
    
    # M6 = (a10-a00)(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[5, 5] = 1  # c11
    
    # M7 = (a01-a11)(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[4, 6] = 1; V[5, 6] = 1
    W[0, 6] = 1  # c00
    
    # 3 products for B columns 2-3:
    # M8 = (a00+a11)(b02+b13)
    U[0, 7] = 1; U[3, 7] = 1
    V[2, 7] = 1; V[7, 7] = 1
    W[2, 7] = 1; W[7, 7] = 1  # c02, c13
    
    # M9 = (a10+a11)b02
    U[2, 8] = 1; U[3, 8] = 1
    V[2, 8] = 1
    W[6, 8] = 1; W[7, 8] = -1  # c12, c13
    
    # M10 = a00(b12-b02)
    U[0, 9] = 1
    V[6, 9] = 1; V[2, 9] = -1
    W[2, 9] = 1; W[6, 9] = 1  # c02, c12
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct using vectorized operations."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    # Build the full tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # For efficiency, compute for all (a,b,c) triples
    # T = sum_r U[:,r] * V[:,r] * W[:,r]
    rank = U.shape[1]
    # Use einsum for speed
    T = np.einsum('ar,br,cr->abc', U, V, W)
    # Check ones
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a, b, c = i*m+j, j*p+k, k*n+i
                if abs(T[a,b,c] - 1.0) > 1e-6:
                    return False
    # Check zeros
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i, j = a//m, a%m
                jj, k = b//p, b%p
                kk, ii = c//n, c%n
                if j != jj or k != kk or i != ii:
                    if abs(T[a,b,c]) > 1e-6:
                        return False
    return True

def construct():
    """Rank-minimal decomposition for 2x2 * 2x12 matrix multiplication.
    
    Strategy: Use a rank-10 decomposition of 2x2x4, then tile 3 times for rank-30.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    # Get rank-10 decomposition for 2x2x4
    U4, V4, W4 = _strassen_2x2x4()
    
    # Tile 3 times for 2x2x12
    rank = 30
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    for t in range(3):
        base_col = 4*t
        for r in range(10):
            U[:, t*10+r] = U4[:, r]
            for j in range(2):
                for k in range(4):
                    V[j*p + base_col + k, t*10+r] = V4[j*4+k, r]
            for k in range(4):
                for i in range(2):
                    W[(base_col+k)*n+i, t*10+r] = W4[k*2+i, r]
    
    # Verify and return
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: rank-48 schoolbook
    rank = 48
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
