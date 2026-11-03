"""Tensor decomposition for 4x4 x 4x9 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Returns U, V, W for M_{2,2,2} using Strassen's algorithm.
    Rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # C indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    # M1: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3: A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4: A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1
    
    # M5: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # M6: (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _winograd_2x2x1():
    """
    Decomposition of M_{2,2,1} using rank 4 (Winograd-style).
    Multiply 2x2 A by 2x1 B to get 2x1 C.
    Rank 4.
    U: (4, 4), V: (2, 4), W: (2, 4)
    
    C0 = A00*B0 + A01*B1
    C1 = A10*B0 + A11*B1
    
    Winograd approach:
    m1 = A00 + A01, m2 = A10 + A11
    n1 = B0 + B1, n2 = B0 - B1
    p1 = m1*n1, p2 = m2*n2
    C0 = (p1 + p2)/2 + (A00-A01)*B1 ... 
    
    Actually let's just use a simple rank-4 decomposition:
    C0 = A00*B0 + A01*B1
    C1 = A10*B0 + A11*B1
    
    Term 1: A00 * B0 -> C0
    Term 2: A01 * B1 -> C0
    Term 3: A10 * B0 -> C1
    Term 4: A11 * B1 -> C1
    """
    rank = 4
    U = np.zeros((4, rank))
    V = np.zeros((2, rank))
    W = np.zeros((2, rank))
    
    # Term 0: A00 * B0 -> C0
    U[0, 0] = 1
    V[0, 0] = 1
    W[0, 0] = 1
    
    # Term 1: A01 * B1 -> C0
    U[1, 1] = 1
    V[1, 1] = 1
    W[0, 1] = 1
    
    # Term 2: A10 * B0 -> C1
    U[2, 2] = 1
    V[0, 2] = 1
    W[1, 2] = 1
    
    # Term 3: A11 * B1 -> C1
    U[3, 3] = 1
    V[1, 3] = 1
    W[1, 3] = 1
    
    return U, V, W

def _winograd_2x2x1_optimal():
    """
    Optimal rank-3 decomposition of M_{2,2,1}? 
    Actually M_{2,2,1} has rank 4 (it's just the standard bilinear form).
    Wait, let me reconsider. The tensor for 2x2 * 2x1 -> 2x1 is:
    T[i,j,k] where i in {0,1}, j in {0,1}, k in {0,1}
    T[i,j,k] = 1 if the (i,j) entry of A times the j-th entry of B contributes to C_i
    
    Actually the multiplication tensor M_{n,m,p} has indices (i,j,k) where
    i is row of A (and C), j is col of A (and row of B), k is col of B (and C).
    T[i,j,k] = 1 means A[i,j] * B[j,k] contributes to C[i,k].
    
    For 2x2 * 2x1: n=2, m=2, p=1
    T[0,0,0] = 1 (A[0,0]*B[0,0] -> C[0,0])
    T[0,1,0] = 1 (A[0,1]*B[1,0] -> C[0,0])
    T[1,0,0] = 1 (A[1,0]*B[0,0] -> C[1,0])
    T[1,1,0] = 1 (A[1,1]*B[1,0] -> C[1,0])
    
    This is rank 4 (it's a sum of 4 rank-1 tensors that can't be combined).
    Actually wait - can we do better? The rank of M_{2,2,1} is 4.
    """
    return _winograd_2x2x1()

def _decomp_4x4x4_strassen():
    """
    Decomposition of M_{4,4,4} using 2-level Strassen.
    Rank 49.
    U: (16, 49), V: (16, 49), W: (16, 49)
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    inner_U, inner_V, inner_W = _strassen_2x2()
    
    outer_A = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 1, 0, -1]
    ]
    
    outer_B = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    outer_C = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [-1, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    for t in range(7):
        col_offset = t * 7
        for r in range(7):
            col = col_offset + r
            for li in range(2):
                for lj in range(2):
                    b = li * 2 + lj
                    coeff_U = outer_A[t][b] * inner_U[li*2 + lj, r]
                    coeff_V = outer_B[t][b] * inner_V[li*2 + lj, r]
                    coeff_W = outer_C[t][b] * inner_W[li*2 + lj, r]
                    
                    for ei in range(2):
                        for ej in range(2):
                            g_i = 2 * (b // 2) + ei
                            g_j = 2 * (b % 2) + ej
                            U[g_i * 4 + g_j, col] = coeff_U
                            V[g_j * 4 + g_i, col] = coeff_V
                            W[g_i * 4 + g_j, col] = coeff_W
    
    return U, V, W

def _decomp_4x4x1():
    """
    Decomposition of M_{4,4,1} using rank 16 (schoolbook).
    U: (16, 16), V: (4, 16), W: (4, 16)
    """
    n, m, p = 4, 4, 1
    rank = 16
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            U[i * m + j, col] = 1
            V[j, col] = 1
            W[i, col] = 1
            col += 1
    return U, V, W

def _verify_fast(U, V, W, n, m, p):
    """Vectorized verification of the decomposition."""
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    i_idx, j_idx, k_idx = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    i_idx = i_idx.ravel()
    j_idx = j_idx.ravel()
    k_idx = k_idx.ravel()
    
    u_vec = U_r[i_idx, j_idx, :]
    v_vec = V_r[j_idx, k_idx, :]
    w_vec = W_r[k_idx, i_idx, :]
    
    prod = u_vec * v_vec * w_vec
    sums = prod.sum(axis=1)
    
    return np.allclose(sums, 1.0, atol=1e-9)

def _schoolbook_nm_p(n, m, p):
    """Schoolbook decomposition for M_{n,m,p}. Rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _decomp_4x4x8_kron():
    """
    Decomposition of M_{4,4,8} using Kronecker product of Strassen 2x2x2 (rank 7)
    and a 2x2x4 decomposition.
    
    Actually, let's think about this differently.
    4x4 x 4x8: split into 2x2 blocks.
    A is 4x4 = 2x2 blocks of 2x2
    B is 4x8 = 2x2 blocks of 2x4 (4 blocks horizontally)
    C is 4x8 = 2x2 blocks of 2x4
    
    For each pair of A-block (li,lj) and B-block (lj, mk), we need to compute
    the contribution to C-block (li, mk).
    
    Using Strassen at the top level:
    7 terms, each involving a 2x2 * 2x4 multiplication.
    
    For 2x2 * 2x4 -> 2x4:
    This is M_{2,2,4}. We can decompose this as 2x2 * 2x2 (rank 7) + 2x2 * 2x2 (rank 7) = rank 14.
    Or we can use a more clever decomposition.
    
    Actually, M_{2,2,4} = M_{2,2,2} ⊕ M_{2,2,2} (block diagonal in the k dimension),
    so rank is 7 + 7 = 14.
    
    Total rank: 7 * 14 = 98 for the 4x4x8 part.
    
    Wait, that's not right either. Let me reconsider.
    
    The Kronecker product approach:
    M_{4,4,8} can be seen as M_{2,2,2} ⊗ M_{2,2,4} in some sense.
    But the tensor structure is:
    T[i,j,k] where i in 0..3, j in 0..3, k in 0..7
    
    If we write i = 2*i1 + i0, j = 2*j1 + j0, k = 2*k1 + k0,
    then T[2*i1+i0, 2*j1+j0, 2*k1+k0] = 1 if and only if
    the (i1,j1,k1) and (i0,j0,k0) conditions are both satisfied.
    
    Actually, T[i,j,k] = 1 always (for valid indices). The multiplication tensor
    M_{n,m,p} has T[i,j,k] = 1 for all valid (i,j,k).
    
    So M_{4,4,8} has T[i,j,k] = 1 for i in 0..3, j in 0..3, k in 0..7.
    This is the tensor product M_{2,2,2} ⊗ M_{2,2,4}? No, that's not quite right.
    
    Actually, M_{4,4,8} = M_{2,2,2} ⊗ M_{2,2,4} as tensors? Let's check dimensions:
    M_{2,2,2}: (2*2) x (2*2) x (2*2) = 4 x 4 x 4
    M_{2,2,4}: (2*2) x (2*4) x (4*2) = 4 x 8 x 8
    
    Kronecker product: (4*4) x (4*8) x (4*8) = 16 x 32 x 32
    But M_{4,4,8} is: (4*4) x (4*8) x (8*4) = 16 x 32 x 32. Yes!
    
    So M_{4,4,8} = M_{2,2,2} ⊗ M_{2,2,4} and rank(M_{4,4,8}) <= rank(M_{2,2,2}) * rank(M_{2,2,4}) = 7 * 14 = 98.
    
    But wait, is rank(M_{2,2,4}) = 14? M_{2,2,4} is 2x2 * 2x4 -> 2x4.
    T[i,j,k] = 1 for i in 0..1, j in 0..1, k in 0..3.
    This is just 4 copies of the identity in some sense. Actually:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    For k=0,1: C[:,0:2] = A * B[:,0:2] (2x2 * 2x2)
    For k=2,3: C[:,2:4] = A * B[:,2:4] (2x2 * 2x2)
    
    So M_{2,2,4} = M_{2,2,2} ⊕ M_{2,2,2} and rank = 7 + 7 = 14.
    
    Therefore rank(M_{4,4,8}) <= 7 * 14 = 98.
    
    For M_{4,4,9} = M_{4,4,8} ⊕ M_{4,4,1}:
    rank <= 98 + 16 = 114.
    
    This is the same as the current approach! The current approach uses
    49 + 49 + 16 = 114, which is the same total rank.
    
    But the Kronecker approach might be cleaner to implement. Let's use it.
    """
    n, m, p = 4, 4, 8
    rank = 98
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    sU, sV, sW = _strassen_2x2()  # rank 7
    # For 2x2x4, we use two copies of Strassen 2x2x2
    # sU2, sV2, sW2 for 2x2x4: rank 14
    sU2 = np.zeros((4, 14))
    sV2 = np.zeros((8, 14))
    sW2 = np.zeros((8, 14))
    # First 2x2x2 block (k=0,1)
    sU2[:, 0:7] = sU
    sV2[:, 0:7] = sV
    sW2[:, 0:7] = sW
    # Second 2x2x2 block (k=2,3)
    sU2[:, 7:14] = sU
    sV2[0:4, 7:14] = sV  # B rows for k=2,3 are B[0,2], B[0,3], B[1,2], B[1,3]
    sV2[4:8, 7:14] = sV
    sW2[0:4, 7:14] = sW  # C rows for k=2,3
    sW2[4:8, 7:14] = sW
    
    # Now Kronecker product
    # M_{4,4,8} = M_{2,2,2} ⊗ M_{2,2,4}
    # U: (4*4, 7*14) = (16, 98)
    # V: (4*8, 7*14) = (32, 98)
    # W: (8*4, 7*14) = (32, 98)
    
    for r1 in range(7):
        for r2 in range(14):
            col = r1 * 14 + r2
            # U: index (i1,i0,j1,j0) -> (2*i1+i0)*4 + (2*j1+j0)
            # sU index: (i0, j0) -> i0*2 + j0
            # sU2 index: (i1, j1) -> i1*2 + j1
            for i0 in range(2):
                for j0 in range(2):
                    for i1 in range(2):
                        for j1 in range(2):
                            val = sU[i0*2+j0, r1] * sU2[i1*2+j1, r2]
                            U[(2*i1+i0)*4 + (2*j1+j0), col] = val
            
            # V: index (j1,j0,k1,k0) -> (2*j1+j0)*8 + (2*k1+k0)
            # sV index: (j0, k0) -> j0*2 + k0
            # sV2 index: (j1, k1) -> j1*2 + k1
            for j0 in range(2):
                for k0 in range(2):
                    for j1 in range(2):
                        for k1 in range(2):
                            val = sV[j0*2+k0, r1] * sV2[j1*2+k1, r2]
                            V[(2*j1+j0)*8 + (2*k1+k0), col] = val
            
            # W: index (k1,k0,i1,i0) -> (2*k1+k0)*4 + (2*i1+i0)
            # sW index: (k0, i0) -> k0*2 + i0
            # sW2 index: (k1, i1) -> k1*2 + i1
            for k0 in range(2):
                for i0 in range(2):
                    for k1 in range(2):
                        for i1 in range(2):
                            val = sW[k0*2+i0, r1] * sW2[k1*2+i1, r2]
                            W[(2*k1+k0)*4 + (2*i1+i0), col] = val
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130
    
    # Approach 1: Kronecker product M_{4,4,8} (rank 98) + M_{4,4,1} (rank 16) = rank 114
    U1, V1, W1 = _decomp_4x4x8_kron()
    U2, V2, W2 = _decomp_4x4x1()
    
    rank_total = 98 + 16
    
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    U[:, 0:98] = U1
    U[:, 98:114] = U2
    
    # V1: (32, 98) for B columns 0..7
    for j in range(m):
        for k in range(8):
            V[j * p + k, 0:98] = V1[j * 8 + k, :]
    
    # V2: (4, 16) for B column 8
    for j in range(m):
        V[j * p + 8, 98:114] = V2[j, :]
    
    # W1: (32, 98) for C columns 0..7
    for k in range(8):
        for i in range(n):
            W[k * n + i, 0:98] = W1[k * 4 + i, :]
    
    # W2: (4, 16) for C column 8
    for i in range(n):
        W[8 * n + i, 98:114] = W2[i, :]
    
    if time.time() > deadline - 5:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if _verify_fast(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        Us, Vs, Ws = _schoolbook_nm_p(n, m, p)
        return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

# EVOLVE-BLOCK-END
