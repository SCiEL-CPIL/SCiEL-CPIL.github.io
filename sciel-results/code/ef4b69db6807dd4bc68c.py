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
    
    # Strassen formulas:
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # C indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    # M1: A00+A11, B00+B11
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: A10+A11, B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3: A00, B01-B11
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4: A11, B10-B00
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1
    
    # M5: A00+A01, B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # M6: A10-A00, B00+B01
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: A01-A11, B10+B11
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _decomp_4x4x4():
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
    
    # Outer Strassen coefficients
    outer_A = [
        [1, 0, 0, 1],   # M1: A00+A11
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00+A01
        [0, 0, 1, -1],  # M6: A10-A00
        [0, 1, 0, -1]   # M7: A01-A11
    ]
    
    outer_B = [
        [1, 0, 0, 1],   # M1: B00+B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01-B11
        [0, 0, 1, -1],  # M4: B10-B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00+B01
        [0, 0, 1, 1]    # M7: B10+B11
    ]
    
    outer_C = [
        [1, 0, 0, 1],   # M1 -> C00, C11
        [0, 0, 1, -1],  # M2 -> C10, C11(-1)
        [0, 1, 0, 1],   # M3 -> C01, C11
        [-1, 0, 1, 0],  # M4 -> C00(-1), C10
        [1, 1, 0, 0],   # M5 -> C00, C01
        [0, 0, 0, 1],   # M6 -> C11
        [1, 0, 0, 0]    # M7 -> C00
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

def _schoolbook_nm_p(n, m, p):
    """
    Schoolbook decomposition for M_{n,m,p}.
    Rank n*m*p.
    U: (n*m, n*m*p), V: (m*p, n*m*p), W: (p*n, n*m*p)
    """
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

def _decomp_2x4x4_rank16():
    """
    Rank-16 decomposition for M_{2,4,4}: (2x4) * (4x4) -> (2x4).
    Uses Strassen on the 2x2 block structure after reshaping.
    A: 2x4, B: 4x4, C: 2x4.
    Split A into [A1 | A2] where A1, A2 are 2x2.
    Split B into [[B11, B12], [B21, B22]] where each is 2x2.
    C = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    
    This gives 4 sub-problems of 2x2x2, each rank 7 = 28 total.
    But we can share: C1 = A1*B11 + A2*B21, C2 = A1*B12 + A2*B22.
    
    Actually, let's use a different approach. View as:
    C[i,k] = sum_j A[i,j]*B[j,k]
    
    For rank 16, we need a clever structure. Let's use:
    Split the 4 inner dimension into two 2s.
    A = [A0 | A1] (each 2x2), B = [[B00, B01], [B10, B11]] (each 2x2)
    C = A0*B00 + A1*B10 (left 2 cols) and A0*B01 + A1*B11 (right 2 cols)
    
    Each 2x2x2 product has rank 7. But we have 4 such products.
    However, we can combine: let S1 = A0 + A1, S2 = A0 - A1, T1 = B00 + B10, T2 = B00 - B10, etc.
    
    Actually, the simplest rank-16 construction: just use 16 rank-1 terms.
    For each (i,j) pair in A (2x4 = 8 entries) and each k (4 cols), that's 32.
    
    Let me just use schoolbook for 2x4x4 which is rank 32, and see if we can do better.
    
    Alternative: use the fact that M_{2,4,4} can be decomposed as
    two independent M_{1,4,4} problems, each of rank 16. Total rank 32.
    
    For now, use schoolbook rank 32.
    """
    n, m, p = 2, 4, 4
    rank = n * m * p  # 32 schoolbook
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

def _decomp_4x4x9_rank41():
    """
    Attempt rank-41 decomposition for M_{4,4,9}.
    Partition B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1).
    Split A (4x4) into A_top (2x4) and A_bot (2x4).
    
    C_top = A_top * B  (2x9)
    C_bot = A_bot * B  (2x9)
    
    For C_top: split into A_top*B1 (2x4), A_top*B2 (2x4), A_top*B3 (2x1)
    For C_bot: split into A_bot*B1 (2x4), A_bot*B2 (2x4), A_bot*B3 (2x1)
    
    M_{2,4,4} schoolbook rank = 32 each -> too high.
    M_{2,4,1} schoolbook rank = 8 each.
    
    This approach gives 4*32 + 2*8 = 144. Not helpful.
    
    Let me try a different partitioning that might yield lower rank.
    """
    # This is a placeholder - the actual rank-41 construction needs more work.
    # For now, fall back to the known 114-rank solution.
    return None

def _decomp_4x4x9_direct():
    """
    Direct construction of M_{4,4,9} decomposition.
    Split k into k0 (0..3) and k1 (4..8).
    k0 uses 4x4x4 Strassen (rank 49).
    k1 uses 4x4x5 with 4x4x4 Strassen (rank 49) + 4x4x1 schoolbook (rank 16).
    Total rank: 49 + 49 + 16 = 114.
    """
    n, m, p = 4, 4, 9
    total_rank = 114
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # Part 1: k=0..3, 4x4x4 Strassen
    U1, V1, W1 = _decomp_4x4x4()
    U[:, 0:49] = U1
    for j in range(4):
        for k in range(4):
            V[j*9 + k, 0:49] = V1[j*4 + k, :]
    for k in range(4):
        for i in range(4):
            W[k*4 + i, 0:49] = W1[k*4 + i, :]
    
    # Part 2: k=4..7, 4x4x4 Strassen (for k1=0..3 of the 4x4x5)
    U2, V2, W2 = _decomp_4x4x4()
    U[:, 49:98] = U2
    for j in range(4):
        for k in range(4):
            V[j*9 + (4+k), 49:98] = V2[j*4 + k, :]
    for k in range(4):
        for i in range(4):
            W[(4+k)*4 + i, 49:98] = W2[k*4 + i, :]
    
    # Part 3: k=8, 4x4x1 schoolbook
    U3, V3, W3 = _schoolbook_nm_p(4, 4, 1)
    U[:, 98:114] = U3
    for j in range(4):
        V[j*9 + 8, 98:114] = V3[j, :]
    for i in range(4):
        W[8*4 + i, 98:114] = W3[i, :]
    
    return U, V, W

def _decomp_4x4x9_improved():
    """
    Improved decomposition using 4x4x4 Strassen for two blocks and
    a more efficient 4x4x1 block.
    Total rank: 49 + 49 + 16 = 114 (same as direct, but let's try
    combining the two 4x4x4 blocks differently).
    
    Actually, let's try: use 4x4x8 via two 4x4x4 Strassen = 98,
    plus 4x4x1 schoolbook = 16. Total = 114.
    
    Alternative: can we do 4x4x9 more cleverly?
    4x4x9 = 4x4x(8+1). 4x4x8 = 2 * 4x4x4 (rank 49 each) = 98.
    4x4x1 = rank 16 schoolbook.
    Total: 114.
    
    Let's try a different split: 4x4x9 = 4x4x(4+4+1) is what we have.
    Or: 4x4x9 = 4x4x(3+3+3) with 4x4x3 blocks.
    4x4x3 schoolbook = 48 each, total 144. Worse.
    
    Or: use 4x4x5 + 4x4x4. 4x4x5 = 4x4x4 + 4x4x1 = 49+16 = 65.
    Total: 65 + 49 = 114. Same.
    """
    n, m, p = 4, 4, 9
    total_rank = 114
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # Part 1: k=0..3, 4x4x4 Strassen (rank 49)
    U1, V1, W1 = _decomp_4x4x4()
    U[:, 0:49] = U1
    for j in range(4):
        for k in range(4):
            V[j*9 + k, 0:49] = V1[j*4 + k, :]
    for k in range(4):
        for i in range(4):
            W[k*4 + i, 0:49] = W1[k*4 + i, :]
    
    # Part 2: k=4..7, 4x4x4 Strassen (rank 49)
    U2, V2, W2 = _decomp_4x4x4()
    U[:, 49:98] = U2
    for j in range(4):
        for k in range(4):
            V[j*9 + (4+k), 49:98] = V2[j*4 + k, :]
    for k in range(4):
        for i in range(4):
            W[(4+k)*4 + i, 49:98] = W2[k*4 + i, :]
    
    # Part 3: k=8, 4x4x1 schoolbook (rank 16)
    U3, V3, W3 = _schoolbook_nm_p(4, 4, 1)
    U[:, 98:114] = U3
    for j in range(4):
        V[j*9 + 8, 98:114] = V3[j, :]
    for i in range(4):
        W[8*4 + i, 98:114] = W3[i, :]
    
    return U, V, W

def _check_decomp_fast(U, V, W, n, m, p, tol=1e-10):
    """
    Vectorized verification of the tensor decomposition.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    Check that for all (i,j,k): sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] == 1
    """
    r = U.shape[1]
    # Reshape to 3D
    U3 = U.reshape(n, m, r)  # (n, m, r)
    V3 = V.reshape(m, p, r)  # (m, p, r)
    W3 = W.reshape(p, n, r)  # (p, n, r)
    
    # For each t, compute the outer product contribution
    # Result[i,j,k] = sum_t U3[i,j,t] * V3[j,k,t] * W3[k,i,t]
    
    # Use einsum for efficient computation
    # U3: (n, m, r), V3: (m, p, r), W3: (p, n, r)
    # We want: sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    
    # Step 1: For each r, compute A_r[i,j,k] = U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # This is tricky with einsum due to the cyclic index.
    
    # Alternative: iterate over r but vectorize over (i,j,k)
    result = np.zeros((n, m, p))
    for t in range(r):
        # U3[:, :, t]: (n, m)
        # V3[:, :, t]: (m, p)
        # W3[:, :, t]: (p, n)
        # Contribution: U3[i,j,t] * V3[j,k,t] * W3[k,i,t]
        # = sum over j: (U3[i,j,t] * W3[?,i,t]) ... not quite
        
        # Let's do it explicitly:
        # For fixed t: contrib[i,j,k] = U3[i,j,t] * V3[j,k,t] * W3[k,i,t]
        # = U3[:, :, t][:, :, np.newaxis] * V3[:, :, t][np.newaxis, :, :] * W3[:, :, t][np.newaxis, np.newaxis, :]
        # But W3[k,i,t] needs index swapping: W3[:, :, t] is (p, n), so W3[k,i] = W3[:, :, t][k, i]
        # We need: for each (i,j,k): U3[i,j] * V3[j,k] * W3[k,i]
        
        U_t = U3[:, :, t]  # (n, m)
        V_t = V3[:, :, t]  # (m, p)
        W_t = W3[:, :, t]  # (p, n)
        
        # contrib[i,j,k] = U_t[i,j] * V_t[j,k] * W_t[k,i]
        # = sum over nothing, just element-wise with index alignment
        # U_t: (n, m) -> need (n, m, 1)
        # V_t: (m, p) -> need (1, m, p)
        # W_t: (p, n) -> need (p, 1, n) but we need (1, 1, p) * ... 
        # Actually: W_t[k,i] so we need W_t transposed: W_t.T[i,k]
        # contrib[i,j,k] = U_t[i,j] * V_t[j,k] * W_t.T[i,k]
        
        UT = U_t[:, :, np.newaxis]  # (n, m, 1)
        VT = V_t[np.newaxis, :, :]  # (1, m, p)
        WT = W_t.T[:, np.newaxis, :]  # (n, 1, p)
        
        result += UT * VT * WT
    
    # Check all entries equal 1
    return np.allclose(result, 1.0, atol=tol)

def _check_decomp(U, V, W, n, m, p, tol=1e-10):
    """Verify the decomposition using tensor contraction."""
    return _check_decomp_fast(U, V, W, n, m, p, tol)

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    limit = t0 + 140.0
    
    # Try improved decomposition first (same rank 114 but with fast verification)
    U, V, W = _decomp_4x4x9_improved()
    
    if time.time() < limit - 10.0:
        ok = _check_decomp(U, V, W, n, m, p)
        if ok:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook (rank 144)
    t1 = time.time()
    if t1 < limit - 5.0:
        Us, Vs, Ws = _schoolbook_nm_p(n, m, p)
        return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}
    
    # Last resort: minimal valid output
    r = n * m * p
    Uf = np.zeros((n*m, r))
    Vf = np.zeros((m*p, r))
    Wf = np.zeros((p*n, r))
    for c in range(r):
        i = c // (m*p)
        j = (c // p) % m
        k = c % p
        Uf[i*m + j, c] = 1
        Vf[j*p + k, c] = 1
        Wf[k*n + i, c] = 1
    return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}

# EVOLVE-BLOCK-END
