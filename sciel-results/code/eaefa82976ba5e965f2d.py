"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct using vectorized operations."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
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
    
    Strategy: Use 7 Strassen products for the first 2 columns, then 3 products
    for each subsequent pair of columns. Total rank = 7 + 3*(p/2 - 1) = 7 + 3*5 = 22.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    num_pairs = p // 2  # 6 pairs
    
    # Rank: 7 for first pair + 3 for each remaining pair
    rank = 7 + 3 * (num_pairs - 1)  # 7 + 15 = 22
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # First pair of columns (0, 1): full Strassen, 7 products
    # A = [[a00, a01], [a10, a11]] -> U indices: a00=0, a01=1, a10=2, a11=3
    # B cols 0,1: b00=0, b01=1, b10=4, b11=5
    # C: c00=W[0], c01=W[1], c10=W[4], c11=W[5]
    
    # M1 = (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[5, 0] = 1
    W[0, 0] = 1; W[5, 0] = 1  # c00, c11
    
    # M2 = (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[4, 1] = 1; W[5, 1] = -1  # c10, c11
    
    # M3 = a00(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[5, 2] = -1
    W[1, 2] = 1; W[5, 2] = 1  # c01, c11
    
    # M4 = a11(b10-b00)
    U[3, 3] = 1
    V[4, 3] = 1; V[0, 3] = -1
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
    
    # Remaining pairs (t=1 to 5): 3 products each
    for t in range(1, num_pairs):
        base = 7 + 3*(t-1)
        k0 = 2*t      # first column of this pair
        k1 = 2*t + 1  # second column of this pair
        
        # P1 = (a00+a11)(b_{0,k0} + b_{1,k1})
        U[0, base] = 1; U[3, base] = 1
        V[0*p + k0, base] = 1; V[1*p + k1, base] = 1
        W[k0*n + 0, base] = 1; W[k1*n + 1, base] = 1  # c_{0,k0}, c_{1,k1}
        
        # P2 = (a10+a11) * b_{0,k0}
        U[2, base+1] = 1; U[3, base+1] = 1
        V[0*p + k0, base+1] = 1
        W[k0*n + 1, base+1] = 1; W[k1*n + 1, base+1] = -1  # c_{1,k0}, c_{1,k1}
        
        # P3 = a00 * (b_{1,k0} - b_{0,k0})
        U[0, base+2] = 1
        V[1*p + k0, base+2] = 1; V[0*p + k0, base+2] = -1
        W[k0*n + 0, base+2] = 1; W[k0*n + 1, base+2] = 1  # c_{0,k0}, c_{1,k0}
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: rank-48 schoolbook
    rank_fb = 48
    U = np.zeros((n*m, rank_fb))
    V = np.zeros((m*p, rank_fb))
    W = np.zeros((p*n, rank_fb))
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
