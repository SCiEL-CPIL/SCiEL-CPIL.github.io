"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2_2x2():
    """Return U, V, W (as 2D lists) for 2x2x2 matrix mult via Strassen (rank 7).
    Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] with n=m=p=2.
    """
    n = m = p = 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]  # 4 x 7
    V = [[0]*rank for _ in range(m*p)]  # 4 x 7
    W = [[0]*rank for _ in range(p*n)]  # 4 x 7

    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # M2 = (A10+A11)B00 -> C10 += M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1

    # M3 = A00(B10-B11) -> C01 += M3
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1

    # M4 = A11(B01-B00) -> C00 += M4, C10 += M4
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1

    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1

    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1

    return U, V, W


def _strassen_4x4_4x4():
    """Return U, V, W for 4x4x4 matrix mult using recursive Strassen (rank 49).
    A is 4x4, B is 4x4, C = A*B is 4x4.
    Split into 2x2 blocks of size 2x2, apply Strassen 2x2x2 at block level.
    Each block multiplication uses the 2x2x2 Strassen (rank 7).
    Block-level Strassen: 7 block-multiplications, each rank 7 -> total rank 49.
    """
    n = m = p = 4
    rank = 49
    U = [[0]*rank for _ in range(n*m)]  # 16 x 49
    V = [[0]*rank for _ in range(m*p)]  # 16 x 49
    W = [[0]*rank for _ in range(p*n)]  # 16 x 49

    # 2x2x2 Strassen base
    U2, V2, W2 = _strassen_2x2_2x2()

    # Block indices for 4x4 split into 2x2 blocks of 2x2:
    # Block (bi, bj) of A covers rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # Block (bj, bk) of B covers rows 2*bj..2*bj+1, cols 2*bk..2*bk+1
    # Block (bi, bk) of C covers rows 2*bi..2*bi+1, cols 2*bk..2*bk+1
    #
    # Strassen at block level (2x2 blocks):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # For each block-level Strassen term, we need to compute the product of
    # sums of 2x2 blocks. The key insight: at the element level, each "block
    # multiplication" of two 2x2 blocks is itself a 2x2x2 matrix mult (rank 7).
    # But the Strassen combinations involve SUMS and DIFFERENCES of blocks.
    #
    # Actually, the correct recursive approach:
    # The 4x4x4 tensor is the "tensor product" (Kronecker) of the 2x2x2 tensor
    # with itself. If T_2 is the 2x2x2 tensor (rank 7), then T_4 = T_2 ⊗ T_2
    # has rank at most 7*7 = 49.
    #
    # The Kronecker product decomposition: if T_2[i,j,k] = sum_c u2[i,c]*v2[j,c]*w2[k,c],
    # then T_4[(i1,i2),(j1,j2),(k1,k2)] = sum_{c1,c2} u2[i1,c1]*u2[i2,c2] * v2[j1,c1]*v2[j2,c2] * w2[k1,c1]*w2[k2,c2]
    #
    # So U_4[(i1*2+i2), (c1*7+c2)] = u2[i1,c1] * u2[i2,c2]
    #    V_4[(j1*2+j2), (c1*7+c2)] = v2[j1,c1] * v2[j2,c2]
    #    W_4[(k1*2+k2), (c1*7+c2)] = w2[k1,c1] * w2[k2,c2]

    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i1 in range(2):
                for i2 in range(2):
                    i = i1 * 2 + i2
                    U[i][c] = U2[i1][c1] * U2[i2][c2]
            for j1 in range(2):
                for j2 in range(2):
                    j = j1 * 2 + j2
                    V[j][c] = V2[j1][c1] * V2[j2][c2]
            for k1 in range(2):
                for k2 in range(2):
                    k = k1 * 2 + k2
                    W[k][c] = W2[k1][c1] * W2[k2][c2]

    return U, V, W


def _strassen_4x4_4x3():
    """Return U, V, W for 4x4x3 matrix mult (A 4x4, B 4x3, C 4x3).
    Strategy: Split B into B1 (4x2) + B2 (4x1).
    For A*B1 (4x4 * 4x2): use Kronecker Strassen for the 4x4x2 part.
    For A*B2 (4x4 * 4x1): use a rank-4 schoolbook-like approach.
    
    Actually, let's use a cleaner approach:
    4x4 * 4x3: Split into 2x2 blocks.
    A = [[A00, A01], [A10, A11]] (2x2 blocks of 2x2)
    B = [[B00, B01, B02], [B10, B11, B12]] (2x3 blocks of 2x2)
    C = [[C00, C01, C02], [C10, C11, C12]] (2x3 blocks of 2x2)
    
    Each C_ij = A_i0 * B_0j + A_i1 * B_1j (sum of two 2x2*2x2 products)
    
    For each (i,j) pair, we need to compute A_i0*B_0j + A_i1*B_1j.
    This is a 2x2x2 + 2x2x2 problem. We can use the "sum of two matrix products"
    which has rank at most 7+7=14, but we can do better.
    
    Actually, the simplest correct approach:
    For 4x4x3, use the Kronecker product idea partially.
    4x4x3 = (2x2x1) ⊗ (2x2x3) ... no, that doesn't work cleanly.
    
    Let me use a direct construction:
    4x4 * 4x3: The tensor has 4*4*3 = 48 entries, all = 1.
    We can decompose as: for each output column k (0,1,2), compute A * B[:,k].
    A * B[:,k] is a 4x1 vector = A @ (4x1 vector).
    This is rank 4 for each column (schoolbook for 4x4*4x1).
    Total rank = 3 * 4 = 12? No, that's not right either.
    
    Actually for 4x4 * 4x1: C[i,0] = sum_j A[i,j]*B[j,0]
    This is a bilinear form in A and B with 16 A-entries and 4 B-entries.
    The tensor is T[i,j,k] where i in 0..3, j in 0..3, k=0.
    T[i,j,0] = 1 for all i,j.
    Rank of this tensor: it's the all-ones tensor of size 4x4x1.
    The all-ones 4x4x1 tensor has rank 4 (it's the outer product of (1,1,1,1) in A-space
    and (1,1,1,1) in B-space... wait, no.
    
    T[i,j,k] = delta_{k,0} * 1. This is just the 4x4 all-ones matrix as a "tensor".
    As a tensor in R^{4x4x1}, it factors as: for each (i,j), T[i,j,0]=1.
    We need sum_c U[i,j,c]*V[j,0,c]*W[0,i,c] = 1 for all i,j.
    Simplest: rank 4. U[i,j,c] = 1 if i==c else 0, V[j,0,c]=1, W[0,i,c]=1.
    Check: sum_c [i==c] * 1 * 1 = 1. Yes! Rank 4 for 4x4x1.
    
    So 4x4x3: split B into 3 columns, each gives rank 4, total rank 12.
    But we can do better by sharing structure.
    
    Actually, let's just use the Kronecker approach properly.
    4x4x3: Think of it as (2x2) x (2x2) x (2x2) for the 4x4x4 part,
    but with only 3 output columns instead of 4.
    
    Let me just do the schoolbook for the 4x4x3 part: rank 48.
    Total: 49 + 48 = 97. That's better than 112.
    
    But we can do much better. Let me use the Kronecker product for 4x4x2 (rank 28)
    and 4x4x1 (rank 4): total 32.
    
    4x4x2 via Kronecker: T_4x4x2 = T_2x2x1 ⊗ T_2x2x2? No.
    Actually, 4x4x2 = (2x2)⊗(2x2) for A, (2x2)⊗(2x1) for B... this is getting complicated.
    
    Let me just implement the direct Kronecker for 4x4x4 (rank 49) and then
    for the 4x4x3 part, use a simple rank-12 construction (3 columns x rank 4 each).
    Total rank: 49 + 12 = 61.
    """
    n = m = 4
    p = 3
    # For 4x4x3, use the fact that it's 3 separate 4x4x1 problems.
    # Each 4x4x1 has rank 4 (as shown above).
    # Total rank = 12.
    rank = 12
    U = [[0]*rank for _ in range(n*m)]  # 16 x 12
    V = [[0]*rank for _ in range(m*p)]  # 12 x 12
    W = [[0]*rank for _ in range(p*n)]  # 12 x 12

    for k in range(3):
        base = k * 4
        for i in range(4):
            for j in range(4):
                # U[i*m+j, base+i] = 1
                U[i*m+j][base+i] = 1
                # V[j*p+k, base+i] = 1
                V[j*p+k][base+i] = 1
                # W[k*n+i, base+i] = 1
                W[k*n+i][base+i] = 1

    return U, V, W


def _validate(U, V, W, n, m, p):
    """Validate a decomposition."""
    rank = len(U[0])
    if len(U) != n*m or len(V) != m*p or len(W) != p*n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                if abs(total - 1) > 1e-9:
                    return False
    # Check integer/half-integer
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    return True


def construct():
    """
    Find a low-rank tensor decomposition for 4x4 * 4x7 matrix multiplication.
    Uses Kronecker product of Strassen 2x2x2 tensors.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130

    # Build 4x4x4 decomposition (rank 49) and 4x4x3 decomposition (rank 12)
    U4, V4, W4 = _strassen_4x4_4x4()
    U3, V3, W3 = _strassen_4x4_4x3()

    # Combine: B is 4x7 = [B_left (4x4), B_right (4x3)]
    # C = A * B_left + A * B_right
    # The decompositions are additive: just concatenate the factor matrices.
    # U is shared (same A), V and W are concatenated along the rank dimension.
    
    rank4 = len(U4[0])  # 49
    rank3 = len(U3[0])  # 12
    total_rank = rank4 + rank3  # 61

    # U: same for both parts (A is the same 4x4 matrix)
    # U4 has shape (16, 49), U3 has shape (16, 12)
    # Combined U: (16, 61)
    U_combined = [row4 + row3 for row4, row3 in zip(U4, U3)]

    # V: V4 has shape (16, 49) for B_left (4x4), V3 has shape (12, 12) for B_right (4x3)
    # B_left indices: j*p+k where p=4, so j in 0..3, k in 0..3 -> 16 entries
    # B_right indices: j*p+k where p=7, so j in 0..3, k in 4..6 -> 12 entries
    # V_combined has shape (28, 61)
    V_combined = [[0]*total_rank for _ in range(m*p)]  # 28 x 61
    # Fill B_left part (rows 0..15)
    for j in range(4):
        for k in range(4):
            v_idx = j * 4 + k  # index in V4: j*p4+k where p4=4
            V_combined[j * 7 + k] = V4[v_idx][:]  # first 49 entries
            # Last 12 entries are 0 (B_right doesn't contribute to B_left rows)
            V_combined[j * 7 + k] = V4[v_idx][:] + [0]*rank3
    # Fill B_right part (rows 16..27)
    for j in range(4):
        for k in range(3):
            v_idx = j * 3 + k  # index in V3: j*p3+k where p3=3
            V_combined[j * 7 + (k + 4)] = [0]*rank4 + V3[v_idx][:]

    # W: W4 has shape (16, 49) for C_left (4x4), W3 has shape (12, 12) for C_right (4x3)
    # C_left indices: k*n+i where n=4, k in 0..3, i in 0..3 -> 16 entries
    # C_right indices: k*n+i where n=4, k in 4..6, i in 0..3 -> 12 entries
    # W_combined has shape (28, 61)
    W_combined = [[0]*total_rank for _ in range(p*n)]  # 28 x 61
    # Fill C_left part (rows 0..15)
    for k in range(4):
        for i in range(4):
            w_idx = k * 4 + i  # index in W4: k*n4+i where n4=4
            W_combined[k * 4 + i] = W4[w_idx][:] + [0]*rank3
    # Fill C_right part (rows 16..27)
    for k in range(3):
        for i in range(4):
            w_idx = k * 4 + i  # index in W3: k*n3+i where n3=4
            W_combined[(k + 4) * 4 + i] = [0]*rank4 + W3[w_idx][:]

    # Validate
    if time.time() < deadline:
        valid = _validate(U_combined, V_combined, W_combined, n, m, p)
        if valid:
            return {"U": U_combined, "V": V_combined, "W": W_combined}

    # Fallback: schoolbook (rank 112)
    rank_school = n * m * p
    U_school = [[0]*rank_school for _ in range(n*m)]
    V_school = [[0]*rank_school for _ in range(m*p)]
    W_school = [[0]*rank_school for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1

    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
