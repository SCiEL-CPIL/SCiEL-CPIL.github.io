"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
TIME_BUDGET = 115.0

def _time_left():
    return max(0.0, TIME_BUDGET - (time.time() - _start_time))

def _verify(U, V, W, n, m, p):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-8:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
    r = n*m*p
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 * 2x2.
    U: shape (4, 7), V: shape (4, 7), W: shape (4, 7)
    Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1  # A00, A11
    V[0, 0] = 1; V[3, 0] = 1  # B00, B11
    W[0, 0] = 1; W[3, 0] = 1  # C00, C11
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1  # A10, A11
    V[0, 1] = 1  # B00
    W[2, 1] = 1; W[3, 1] = 1  # C10, C11
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1  # A00
    V[1, 2] = 1; V[3, 2] = -1  # B01, -B11
    W[1, 2] = 1; W[3, 2] = 1  # C01, C11
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1  # A11
    V[2, 3] = 1; V[0, 3] = -1  # B10, -B00
    W[0, 3] = 1; W[2, 3] = 1  # C00, C10
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1  # A00, A01
    V[3, 4] = 1  # B11
    W[0, 4] = -1; W[1, 4] = 1  # -C00, C01
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1  # A10, -A00
    V[0, 5] = 1; V[1, 5] = 1  # B00, B01
    W[3, 5] = 1  # C11
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1  # A01, -A11
    V[2, 6] = 1; V[3, 6] = 1  # B10, B11
    W[0, 6] = 1  # C00
    
    return U.tolist(), V.tolist(), W.tolist()

def _construct_strassen_3x3x8():
    """
    Construct a decomposition for 3x3 * 3x8 using Strassen blocks.
    
    Strategy: Split the 3x3 matrix into 2x2 and 1x1 blocks, and the 3x8 into
    2x4 and 1x4 blocks. Use Strassen for the 2x2*2x4 part (which can be done
    with rank 7*4=28 by applying Strassen to each pair of columns), and 
    schoolbook for the remaining parts.
    
    Actually, let me think more carefully.
    
    A is 3x3, B is 3x8.
    Split A into:
      A11 (2x2), A12 (2x1), A21 (1x2), A22 (1x1)
    Split B into:
      B11 (2x8), B21 (1x8)
    
    C = A*B = [A11*B11 + A12*B21, A21*B11 + A22*B21]
    
    For the 2x2 * 2x8 part: We can use Strassen on each pair of columns.
    8 columns = 4 pairs of 2 columns. Each 2x2*2x2 uses rank 7.
    So 2x2*2x8 uses rank 7*4 = 28.
    
    For the 2x1 * 1x8 part: This is just 2 scalar-vector products, rank 2.
    For the 1x2 * 2x8 part: This is just 2 scalar-vector products, rank 2.
    For the 1x1 * 1x8 part: This is just 1 scalar-vector product, rank 8.
    
    Total: 28 + 2 + 2 + 8 = 40.
    
    But wait, we need to be more careful. The Strassen decomposition for 2x2*2x2
    gives us the 4 output entries. For 2x2*2x8, we have 4 output entries per
    pair of columns, so 4*4=16 output entries total.
    
    Let me construct this properly.
    
    Actually, let me try a different approach. Let me use the fact that:
    3x3 * 3x8 can be seen as 3 rows of (1x3 * 3x8).
    Each 1x3 * 3x8 is a vector-matrix product, which has rank 3 (since it's
    just 3 scalar multiplications for each of the 8 outputs, but they share
    the same 3 multipliers... wait, no.
    
    For a 1x3 vector a = (a0, a1, a2) and a 3x8 matrix B:
    c[k] = a0*B[0,k] + a1*B[1,k] + a2*B[2,k] for k=0..7
    
    This requires 3 multiplications per output, so 24 total for one row.
    For 3 rows: 72 (schoolbook).
    
    But we can do better by sharing across rows using Strassen.
    
    Let me try the block approach more carefully.
    
    A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    B = [[B1], [B2]] where B1 is 2x8, B2 is 1x8
    
    C = [[A11*B1 + A12*B2], [A21*B1 + A22*B2]]
    
    C is 3x8:
    C[0:2, :] = A11*B1 + A12*B2
    C[2, :] = A21*B1 + A22*B2
    
    For A11*B1 (2x2 * 2x8): Use Strassen on 4 pairs of columns, rank 7*4=28
    For A12*B2 (2x1 * 1x8): rank 2 (each of the 2 rows needs 1 mult per col, but they share B2)
      Actually, A12 is 2x1, so A12*B2 means:
      row0: A12[0,0] * B2[0,:]  (8 outputs, 1 mult each, but same mult for all 8)
      row1: A12[1,0] * B2[0,:]  (8 outputs, 1 mult each)
      So rank 2.
    For A21*B1 (1x2 * 2x8):
      C[2,k] = A21[0,0]*B1[0,k] + A21[0,1]*B1[1,k]
      This is 2 mults per column, 8 columns, so rank 16? No, it's just 2 scalar-vector products.
      Rank 2.
    For A22*B2 (1x1 * 1x8):
      C[2,k] += A22[0,0] * B2[0,k]
      Rank 1 (one multiplier shared across all 8 columns).
    
    Total: 28 + 2 + 2 + 1 = 33.
    
    Wait, that's not right. The rank of a scalar-vector product (1x1 * 1x8) is 1,
    because it's just one multiplication. But the output has 8 entries, all equal
    to the same product. So the rank-1 term would be:
    U[2, c] = A22[0,0] (but U depends on the input, which is not allowed)
    
    I'm confusing the algorithm with the tensor decomposition again.
    
    In a tensor decomposition, the coefficients U, V, W are FIXED. They don't
    depend on the input matrices. The decomposition must satisfy:
    For all inputs A, B: C = A*B = sum_c U_c * (outer product of V_c and W_c)
    
    More precisely, the tensor T[i,j,k] (where i indexes A, j indexes B, k indexes C)
    must equal sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]... 
    
    Wait, let me re-read the convention. The output factor is W[k*n+i, r].
    So the tensor is T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    For the 1x1 * 1x8 case:
    T[i,j,k] where i is the single row of A, j is the single column of B (well, B is 1x8,
    so j indexes the 8 columns of B... wait, no.
    
    Let me re-index. A is n x m, B is m x p, C is n x p.
    T[i,j,k] = 1 if this entry of A times this entry of B contributes to C[i,k].
    Specifically, C[i,k] = sum_j A[i,j] * B[j,k].
    
    So T[i,j,k] = 1 for all valid (i,j,k), and the decomposition is:
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    For the 1x1 * 1x8 sub-problem: n=1, m=1, p=8.
    T[0,0,k] = 1 for k=0..7.
    We need: sum_c U[0, c] * V[k, c] * W[k, c] = 1 for all k.
    
    If we set U[0,c] = 1 for all c, V[k,c] = 1 for all k,c, W[k,c] = 1 for all k,c,
    then sum_c 1*1*1 = rank, which is not 1.
    
    To get rank 1: U[0,0] = 1, V[k,0] = 1 for all k, W[k,0] = 1 for all k.
    Then sum_c U[0,c]*V[k,c]*W[k,c] = 1*1*1 = 1 for all k. Yes! Rank 1.
    
    OK so my earlier analysis was correct. Let me redo the block construction.
    
    For 3x3 * 3x8:
    Split A into 2x2 + 2x1 + 1x2 + 1x1 blocks.
    Split B into 2x8 + 1x8 blocks.
    
    C[0:2, :] = A11 * B1 + A12 * B2
    C[2, :] = A21 * B1 + A22 * B2
    
    Block 1: A11 (2x2) * B1 (2x8) -> C1 (2x8)
    Use Strassen: split B1 into 4 pairs of 2 columns.
    For each pair (columns 2t, 2t+1), use Strassen 2x2*2x2 with rank 7.
    Total rank: 7*4 = 28.
    
    Block 2: A12 (2x1) * B2 (1x8) -> C2 (2x8)
    C2[i,k] = A12[i,0] * B2[0,k]
    This is 2 independent scalar-vector products.
    Rank 2: one rank-1 term for each row of A12.
    Term 1: U[0*3+2, c] = 1 (A12[0,0] is at position i=0, j=2 in the full A), 
            V[2*8+k, c] = 1 for all k (B2[0,k] is at position j=2, k in full B),
            W[k*3+0, c] = 1 for all k (C[0,k] is at position k, i=0 in full C)
    Term 2: U[1*3+2, c] = 1, V[2*8+k, c] = 1, W[k*3+1, c] = 1
    
    Block 3: A21 (1x2) * B1 (2x8) -> C3 (1x8)
    C3[0,k] = A21[0,0]*B1[0,k] + A21[0,1]*B1[1,k]
    This is a dot product of length 2 for each k.
    Rank 2: one term for each j in {0,1}.
    Term 1: U[2*3+0, c] = 1, V[0*8+k, c] = 1, W[k*3+2, c] = 1
    Term 2: U[2*3+1, c] = 1, V[1*8+k, c] = 1, W[k*3+2, c] = 1
    
    Block 4: A22 (1x1) * B2 (1x8) -> C4 (1x8)
    C4[0,k] = A22[0,0] * B2[0,k]
    Rank 1: U[2*3+2, c] = 1, V[2*8+k, c] = 1, W[k*3+2, c] = 1
    
    Total rank: 28 + 2 + 2 + 1 = 33.
    
    But wait, I need to make sure the Strassen terms are correctly embedded.
    
    For Strassen 2x2*2x2 on columns 2t and 2t+1:
    The local indices are:
    A local: rows 0,1; cols 0,1 -> global: rows 0,1; cols 0,1
    B local: rows 0,1; cols 0,1 -> global: rows 0,1; cols 2t, 2t+1
    C local: rows 0,1; cols 0,1 -> global: rows 0,1; cols 2t, 2t+1
    
    For each Strassen term c_local in 0..6:
    U_global[i_local*2 + j_local, c_global] = U_strassen[i_local*2 + j_local, c_local]
    V_global[j_local*2 + k_local, c_global] = V_strassen[j_local*2 + k_local, c_local]
    W_global[k_local*2 + i_local, c_global] = W_strassen[k_local*2 + i_local, c_local]
    
    But we need to map to the full 3x3 * 3x8 indices:
    U index: i*m + j = i*3 + j (since m=3)
    V index: j*p + k = j*8 + k (since p=8)
    W index: k*n + i = k*3 + i (since n=3)
    
    For the Strassen block on rows 0,1 of A and cols 0,1 of A (which are the same as rows 0,1 of B),
    and cols 2t, 2t+1 of B:
    
    U[i*3 + j, c] for i in {0,1}, j in {0,1}
    V[j*8 + k, c] for j in {0,1}, k in {2t, 2t+1}
    W[k*3 + i, c] for k in {2t, 2t+1}, i in {0,1}
    
    Let me code this up.
    """
    n, m, p = 3, 3, 8
    
    U_strassen, V_strassen, W_strassen = _strassen_2x2()
    
    U_dict = {}  # key: (row, col) in U, value: dict of term_index -> value
    V_dict = {}
    W_dict = {}
    
    term_count = 0
    def add_term(u_entries, v_entries, w_entries):
        nonlocal term_count
        c = term_count
        for idx, val in u_entries.items():
            if idx not in U_dict:
                U_dict[idx] = {}
            U_dict[idx][c] = val
        for idx, val in v_entries.items():
            if idx not in V_dict:
                V_dict[idx] = {}
            V_dict[idx][c] = val
        for idx, val in w_entries.items():
            if idx not in W_dict:
                W_dict[idx] = {}
            W_dict[idx][c] = val
        term_count += 1
    
    # Block 1: A11 (2x2) * B1 (2x8) using Strassen on 4 pairs of columns
    for t in range(4):
        k_offset = 2 * t
        for c_local in range(7):
            u_entries = {}
            v_entries = {}
            w_entries = {}
            for i_local in range(2):
                for j_local in range(2):
                    val = U_strassen[i_local*2 + j_local, c_local]
                    if val != 0:
                        u_entries[i_local*3 + j_local] = val
            for j_local in range(2):
                for k_local in range(2):
                    val = V_strassen[j_local*2 + k_local, c_local]
                    if val != 0:
                        k_global = k_offset + k_local
                        v_entries[j_local*8 + k_global] = val
            for k_local in range(2):
                for i_local in range(2):
                    val = W_strassen[k_local*2 + i_local, c_local]
                    if val != 0:
                        k_global = k_offset + k_local
                        w_entries[k_global*3 + i_local] = val
            add_term(u_entries, v_entries, w_entries)
    
    # Block 2: A12 (2x1) * B2 (1x8)
    # A12[i,0] * B2[0,k] for i in {0,1}, k in {0..7}
    for i in range(2):
        u_entries = {i*3 + 2: 1}
        v_entries = {}
        w_entries = {}
        for k in range(8):
            v_entries[2*8 + k] = 1
            w_entries[k*3 + i] = 1
        add_term(u_entries, v_entries, w_entries)
    
    # Block 3: A21 (1x2) * B1 (2x8)
    # A21[0,j] * B1[j,k] for j in {0,1}, k in {0..7}
    for j in range(2):
        u_entries = {2*3 + j: 1}
        v_entries = {}
        w_entries = {}
        for k in range(8):
            v_entries[j*8 + k] = 1
            w_entries[k*3 + 2] = 1
        add_term(u_entries, v_entries, w_entries)
    
    # Block 4: A22 (1x1) * B2 (1x8)
    # A22[0,0] * B2[0,k] for k in {0..7}
    u_entries = {2*3 + 2: 1}
    v_entries = {}
    w_entries = {}
    for k in range(8):
        v_entries[2*8 + k] = 1
        w_entries[k*3 + 2] = 1
    add_term(u_entries, v_entries, w_entries)
    
    r = term_count
    U_mat = [[0] * r for _ in range(n * m)]
    V_mat = [[0] * r for _ in range(m * p)]
    W_mat = [[0] * r for _ in range(p * n)]
    
    for idx, cols in U_dict.items():
        for c, val in cols.items():
            U_mat[idx][c] = val
    for idx, cols in V_dict.items():
        for c, val in cols.items():
            V_mat[idx][c] = val
    for idx, cols in W_dict.items():
        for c, val in cols.items():
            W_mat[idx][c] = val
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}

def construct():
    n, m, p = DIMENSIONS
    best = _schoolbook()
    best_rank = len(best["U"][0])
    
    if _time_left() > 5.0:
        result = _construct_strassen_3x3x8()
        if _verify(result["U"], result["V"], result["W"], n, m, p):
            r = len(result["U"][0])
            if r < best_rank:
                best = result
                best_rank = r
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
# EVOLVE-BLOCK-END
