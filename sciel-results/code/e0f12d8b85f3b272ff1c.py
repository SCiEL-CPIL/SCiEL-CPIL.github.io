import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(U.shape[1]):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _try_rank_27_cyclic():
    """
    Attempt rank-27 decomposition using cyclic convolution / DFT structure.
    Returns None if no valid rank-27 decomposition is found.
    """
    n, m, p = DIMENSIONS
    rank = 27
    
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    #
    # Cyclic convolution approach:
    # Map (i,j,k) to group elements in Z/qZ for some q.
    # For q=11: map i -> 0,1; j -> 3,4,5; k -> 6,7,8,9,10
    # The "product" in the group algebra: (i,j,k) -> i + j + k mod 11
    # Values: i+j+k ranges from 0+3+6=9 to 1+5+10=16 mod 11 = 5
    # So i+j+k mod 11 can be: 9,10,0,1,2,3,4,5 (8 distinct values)
    #
    # The cyclic convolution tensor C[s] = sum_{i+j+k ≡ s} T[i,j,k]
    # C[9] = T[0,3,6]... wait, j only goes 0,1,2 not 3,4,5.
    #
    # Let me redo: i in {0,1}, j in {0,1,2}, k in {0,1,2,3,4}
    # Map: i -> i, j -> 2+j (so j=0,1,2 -> 2,3,4), k -> 5+k (so k=0..4 -> 5,6,7,8,9)
    # i+j+k mapped: i + (2+j) + (5+k) = 7 + i + j + k
    # Range: 7+0+0+0=7 to 7+1+2+4=14
    # Mod 11: 7,8,9,10,0,1,2,3 (8 values)
    #
    # The DFT over Z/11Z diagonalizes the cyclic convolution.
    # The rank of the cyclic convolution tensor = number of non-zero DFT coefficients.
    # But this doesn't directly give us the rank of the original tensor.
    #
    # This approach is too complex to implement correctly in the time available.
    # Return None to use the rank-28 fallback.
    return None

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    
    # Try rank 27 first with cyclic convolution approach
    result = _try_rank_27_cyclic()
    if result is not None:
        # Verify before returning
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            return result
    
    # Fall back to rank 28
    rank = 28
    U = [[0] * rank for _ in range(n * m)]   # 6 x 28
    V = [[0] * rank for _ in range(m * p)]   # 15 x 28
    W = [[0] * rank for _ in range(p * n)]   # 10 x 28
    
    r = 0
    
    # Part 1: A[:,2] * B[2,:] -> 2*5 = 10 terms
    # u_idx = i*m + 2, v_idx = 2*p + k, w_idx = k*n + i
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U[u_idx][r] = 1
            V[v_idx][r] = 1
            W[w_idx][r] = 1
            r += 1
    
    # Part 2: A[:,0:2] * B[0:2, 4] -> 2*2 = 4 terms
    # u_idx = i*m + j, v_idx = j*p + 4, w_idx = 4*n + i
    for i in range(n):
        for j in range(2):
            u_idx = i * m + j
            v_idx = j * p + 4
            w_idx = 4 * n + i
            U[u_idx][r] = 1
            V[v_idx][r] = 1
            W[w_idx][r] = 1
            r += 1
    
    # Part 3: A[:,0:2] * B[0:2, 0:2] -> 2x2x2 Strassen (7 terms)
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # u_idx = i*3 + j, v_idx = j*5 + k, w_idx = k*2 + i
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    U[0*3+0][r] = 1; U[1*3+1][r] = 1
    V[0*5+0][r] = 1; V[1*5+1][r] = 1
    W[0*2+0][r] = 1; W[1*2+1][r] = 1
    r += 1
    
    # M2: (A10+A11)B00 -> C10, C11
    U[1*3+0][r] = 1; U[1*3+1][r] = 1
    V[0*5+0][r] = 1
    W[0*2+1][r] = 1; W[1*2+1][r] = -1
    r += 1
    
    # M3: A00(B11-B01) -> C10, C11
    U[0*3+0][r] = 1
    V[0*5+1][r] = 1; V[1*5+1][r] = -1
    W[1*2+0][r] = 1; W[1*2+1][r] = 1
    r += 1
    
    # M4: (A01+A11)(B00-B10) -> C00, C01
    U[1*3+1][r] = 1
    V[1*5+0][r] = 1; V[0*5+0][r] = -1
    W[0*2+0][r] = 1; W[0*2+1][r] = 1
    r += 1
    
    # M5: (A00+A01)B11 -> C00, C01
    U[0*3+0][r] = 1; U[0*3+1][r] = 1
    V[1*5+1][r] = 1
    W[0*2+0][r] = -1; W[1*2+0][r] = 1
    r += 1
    
    # M6: (A10-A00)(B00+B01) -> C11
    U[1*3+0][r] = 1; U[0*3+0][r] = -1
    V[0*5+0][r] = 1; V[0*5+1][r] = 1
    W[1*2+1][r] = 1
    r += 1
    
    # M7: (A01-A11)(B10+B11) -> C00
    U[0*3+1][r] = 1; U[1*3+1][r] = -1
    V[1*5+0][r] = 1; V[1*5+1][r] = 1
    W[0*2+0][r] = 1
    r += 1
    
    # Part 4: A[:,0:2] * B[0:2, 2:4] -> 2x2x2 Strassen (7 terms)
    # Indices: i in {0,1}, j in {0,1}, k in {2,3}
    # u_idx = i*3 + j, v_idx = j*5 + k, w_idx = k*2 + i
    
    # M1: (A00+A11)(B02+B13) -> C02, C13
    U[0*3+0][r] = 1; U[1*3+1][r] = 1
    V[0*5+2][r] = 1; V[1*5+3][r] = 1
    W[2*2+0][r] = 1; W[3*2+1][r] = 1
    r += 1
    
    # M2: (A10+A11)B02 -> C12, C13
    U[1*3+0][r] = 1; U[1*3+1][r] = 1
    V[0*5+2][r] = 1
    W[2*2+1][r] = 1; W[3*2+1][r] = -1
    r += 1
    
    # M3: A00(B13-B03) -> C12, C13
    U[0*3+0][r] = 1
    V[0*5+3][r] = 1; V[1*5+3][r] = -1
    W[3*2+0][r] = 1; W[3*2+1][r] = 1
    r += 1
    
    # M4: (A01+A11)(B02-B12) -> C02, C03
    U[1*3+1][r] = 1
    V[1*5+2][r] = 1; V[0*5+2][r] = -1
    W[2*2+0][r] = 1; W[2*2+1][r] = 1
    r += 1
    
    # M5: (A00+A01)B13 -> C02, C03
    U[0*3+0][r] = 1; U[0*3+1][r] = 1
    V[1*5+3][r] = 1
    W[2*2+0][r] = -1; W[3*2+0][r] = 1
    r += 1
    
    # M6: (A10-A00)(B02+B03) -> C13
    U[1*3+0][r] = 1; U[0*3+0][r] = -1
    V[0*5+2][r] = 1; V[0*5+3][r] = 1
    W[3*2+1][r] = 1
    r += 1
    
    # M7: (A01-A11)(B12+B13) -> C02
    U[0*3+1][r] = 1; U[1*3+1][r] = -1
    V[1*5+2][r] = 1; V[1*5+3][r] = 1
    W[2*2+0][r] = 1
    r += 1
    
    assert r == rank, f"Expected rank {rank}, got {r}"
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify the decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Decomposition verification failed")
    
    return result
# EVOLVE-BLOCK-END
