"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 35 decomposition using 5 blocks of Strassen (2x2x2)
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 5 blocks of 2 columns
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Strassen terms for block b
        # Term 1: (a+d)(e+h)
        # U: a(0), d(3)
        # V: e(0, k0), h(1, k1)
        # W: C00(k0, 0) +, C11(k1, 1) +
        U[0][col] = 1; U[3][col] = 1
        V[0 * 10 + k0][col] = 1; V[1 * 10 + k1][col] = 1
        W[k0 * 2 + 0][col] = 1; W[k1 * 2 + 1][col] = 1
        col += 1
        
        # Term 2: (c+d)e
        # U: c(2), d(3)
        # V: e(0, k0)
        # W: C10(k0, 1) +, C11(k1, 1) -
        U[2][col] = 1; U[3][col] = 1
        V[0 * 10 + k0][col] = 1
        W[k0 * 2 + 1][col] = 1; W[k1 * 2 + 1][col] = -1
        col += 1
        
        # Term 3: a(f-h)
        # U: a(0)
        # V: f(0, k1), h(1, k1)
        # W: C01(k1, 0) +, C11(k1, 1) +
        U[0][col] = 1
        V[0 * 10 + k1][col] = 1; V[1 * 10 + k1][col] = -1
        W[k1 * 2 + 0][col] = 1; W[k1 * 2 + 1][col] = 1
        col += 1
        
        # Term 4: d(g-e)
        # U: d(3)
        # V: g(1, k0), e(0, k0)
        # W: C00(k0, 0) +, C10(k0, 1) +
        U[3][col] = 1
        V[1 * 10 + k0][col] = 1; V[0 * 10 + k0][col] = -1
        W[k0 * 2 + 0][col] = 1; W[k0 * 2 + 1][col] = 1
        col += 1
        
        # Term 5: (a+b)h
        # U: a(0), b(1)
        # V: h(1, k1)
        # W: C00(k0, 0) -, C01(k1, 0) +
        U[0][col] = 1; U[1][col] = 1
        V[1 * 10 + k1][col] = 1
        W[k0 * 2 + 0][col] = -1; W[k1 * 2 + 0][col] = 1
        col += 1
        
        # Term 6: (c-a)(e+f)
        # U: c(2), a(0)
        # V: e(0, k0), f(0, k1)
        # W: C11(k1, 1) +
        U[2][col] = 1; U[0][col] = -1
        V[0 * 10 + k0][col] = 1; V[0 * 10 + k1][col] = 1
        W[k1 * 2 + 1][col] = 1
        col += 1
        
        # Term 7: (b-d)(g+h)
        # U: b(1), d(3)
        # V: g(1, k0), h(1, k1)
        # W: C00(k0, 0) +
        U[1][col] = 1; U[3][col] = -1
        V[1 * 10 + k0][col] = 1; V[1 * 10 + k1][col] = 1
        W[k0 * 2 + 0][col] = 1
        col += 1

    # Verification step
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails (should not happen)
        return _schoolbook(n, m, p)

    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1 if j==k else 0
    Note: The tensor for matrix multiplication C = AB is T[i,j,k] = delta_{j, ?}
    Actually, C[i,k] = sum_j A[i,j] B[j,k].
    The tensor T is in V_A x V_B x V_C.
    T[i, j, k] = 1 if the term A[i,j] B[j,k] contributes to C[i,k].
    In the decomposition, we sum rank-1 tensors.
    The condition is:
    For all i in [0,n), j in [0,m), k in [0,p):
    sum_{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 if j is the summation index?
    No.
    The standard identity is:
    C_{ik} = \sum_{j} A_{ij} B_{jk}
    This corresponds to the tensor T = \sum_{i,j,k} e_{ij} \otimes e_{jk} \otimes e_{ik}.
    So T_{i,j,k} = 1 for all valid i,j,k.
    Wait, if T_{i,j,k} = 1, then sum_r ... should be 1 for ALL i,j,k.
    Let's check the Strassen construction.
    In the Strassen block for b=0 (k0=0, k1=1):
    It decomposes the 2x2x2 tensor.
    The 2x2x2 tensor M_{2,2,2} has entries T_{i,j,k} = 1 for i,j,k in {0,1}.
    So the sum should be 1 for all i,j,k in the block.
    Let's verify this property.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    rank = len(U[0])
    
    # Convert to numpy for speed
    Ua = np.array(U)
    Va = np.array(V)
    Wa = np.array(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                # Compute sum over r
                # sum_r Ua[idx_u, r] * Va[idx_v, r] * Wa[idx_w, r]
                val = np.dot(np.multiply(Ua[idx_u, :], Va[idx_v, :]), Wa[idx_w, :])
                # Expected value is 1.0
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
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
# EVOLVE-BLOCK-END
