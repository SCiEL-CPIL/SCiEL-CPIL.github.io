"""Matrix multiplication 3,3,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction as Fr

def _build_trivial(n, m, p):
    """Build the trivial (schoolbook) decomposition of rank n*m*p."""
    r = n * m * p
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _build_strassen_222():
    """Build rank-7 decomposition for 2x2x2 matrix multiplication."""
    # Indices for 2x2:
    # U index: i*2 + j (0..3)
    # V index: j*2 + k (0..3)
    # W index: k*2 + i (0..3)
    
    # Strassen terms:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # We define U, V, W vectors for each term.
    # U vector corresponds to coefficients of a_ij in the first factor.
    # V vector corresponds to coefficients of b_jk in the second factor.
    # W vector corresponds to coefficients of c_ki in the result.
    
    # Note: The tensor T[a,b,c] = sum_t U_t[a] V_t[b] W_t[c]
    # For a term M_t = (sum u_a a_a) * (sum v_b b_b), it contributes to c_c = sum w_c * M_t ?
    # No, the decomposition is T = sum U_t \otimes V_t \otimes W_t.
    # This means T[a,b,c] = sum_t U_t[a] V_t[b] W_t[c].
    # In matmul, C = AB. C_{ki} = sum_j A_{ij} B_{jk}.
    # The tensor is T[A_idx, B_idx, C_idx] = 1 if valid.
    # So for a rank-1 term, we specify which A entries are multiplied, which B entries, and which C entry is produced.
    # Actually, a rank-1 term is of the form (u \cdot A) (v \cdot B) w^T?
    # No, it's (u^T A v) * w?
    # The standard decomposition is:
    # C_{ki} = sum_t (sum_j U_t[i,j] A_{ij}) (sum_l V_t[j,l] B_{jl}) W_t[k,i] ?
    # No, the factors are vectors.
    # U_t is a vector of size nm. V_t size mp. W_t size pn.
    # Term t contributes: (sum_{i,j} U_t[i,j] A_{ij}) * (sum_{j,k} V_t[j,k] B_{jk}) * W_t[k,i] ?
    # This doesn't look right because the indices j must match.
    
    # Correct Tensor Definition:
    # T[i*m+j, j*p+k, k*n+i] = 1.
    # Decomposition: T = sum_t U_t \otimes V_t \otimes W_t.
    # U_t is vector of length nm.
    # V_t is vector of length mp.
    # W_t is vector of length pn.
    # The product is sum_t U_t[i*m+j] * V_t[j*p+k] * W_t[k*n+i].
    
    # For Strassen M1 = (a11+a22)(b11+b22):
    # This term is used in C11, C12, C21, C22.
    # C11 += M1. C12 += M1. C21 += M1. C22 += M1.
    # So W_t has 1s at positions corresponding to C11, C12, C21, C22.
    # U_t has 1s at A11, A22.
    # V_t has 1s at B11, B22.
    
    def idx_u(i, j): return i * 2 + j
    def idx_v(j, k): return j * 2 + k
    def idx_w(k, i): return k * 2 + i
    
    terms = []
    
    # M1
    u1 = [0]*4; v1 = [0]*4; w1 = [0]*4
    u1[idx_u(0,0)] = 1; u1[idx_u(1,1)] = 1
    v1[idx_v(0,0)] = 1; v1[idx_v(1,1)] = 1
    w1[idx_w(0,0)] = 1; w1[idx_w(0,1)] = 1; w1[idx_w(1,0)] = 1; w1[idx_w(1,1)] = 1
    terms.append((u1, v1, w1))
    
    # M2 = (a21+a22)b11
    # C21 += M2. C22 -= M2.
    u2 = [0]*4; v2 = [0]*4; w2 = [0]*4
    u2[idx_u(1,0)] = 1; u2[idx_u(1,1)] = 1
    v2[idx_v(0,0)] = 1
    w2[idx_w(0,1)] = 1; w2[idx_w(1,1)] = -1
    terms.append((u2, v2, w2))
    
    # M3 = a11(b12-b22)
    # C12 += M3. C22 += M3.
    u3 = [0]*4; v3 = [0]*4; w3 = [0]*4
    u3[idx_u(0,0)] = 1
    v3[idx_v(0,1)] = 1; v3[idx_v(1,1)] = -1
    w3[idx_w(0,1)] = 1; w3[idx_w(1,1)] = 1
    terms.append((u3, v3, w3))
    
    # M4 = a22(b21-b11)
    # C11 += M4. C21 += M4.
    u4 = [0]*4; v4 = [0]*4; w4 = [0]*4
    u4[idx_u(1,1)] = 1
    v4[idx_v(1,0)] = 1; v4[idx_v(0,0)] = -1
    w4[idx_w(0,0)] = 1; w4[idx_w(0,1)] = 1
    terms.append((u4, v4, w4))
    
    # M5 = (a11+a12)b22
    # C11 -= M5. C12 += M5.
    u5 = [0]*4; v5 = [0]*4; w5 = [0]*4
    u5[idx_u(0,0)] = 1; u5[idx_u(0,1)] = 1
    v5[idx_v(1,1)] = 1
    w5[idx_w(0,0)] = -1; w5[idx_w(0,1)] = 1
    terms.append((u5, v5, w5))
    
    # M6 = (a21-a11)(b11+b12)
    # C22 += M6.
    u6 = [0]*4; v6 = [0]*4; w6 = [0]*4
    u6[idx_u(1,0)] = 1; u6[idx_u(0,0)] = -1
    v6[idx_v(0,0)] = 1; v6[idx_v(0,1)] = 1
    w6[idx_w(1,1)] = 1
    terms.append((u6, v6, w6))
    
    # M7 = (a12-a22)(b21+b22)
    # C11 += M7.
    u7 = [0]*4; v7 = [0]*4; w7 = [0]*4
    u7[idx_u(0,1)] = 1; u7[idx_u(1,1)] = -1
    v7[idx_v(1,0)] = 1; v7[idx_v(1,1)] = 1
    w7[idx_w(0,0)] = 1
    terms.append((u7, v7, w7))
    
    U = np.array([t[0] for t in terms]).T # Shape (4, 7)
    V = np.array([t[1] for t in terms]).T # Shape (4, 7)
    W = np.array([t[2] for t in terms]).T # Shape (4, 7)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _build_rect_trivial(n, m, p):
    """Trivial decomposition for rectangular n x m x p."""
    r = n * m * p
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _combine_decompositions(decomps):
    """Combine a list of decompositions into one."""
    total_rank = sum(len(d["U"][0]) for d in decomps)
    U_rows = len(decomps[0]["U"])
    V_rows = len(decomps[0]["V"])
    W_rows = len(decomps[0]["W"])
    
    U = [[0] * total_rank for _ in range(U_rows)]
    V = [[0] * total_rank for _ in range(V_rows)]
    W = [[0] * total_rank for _ in range(W_rows)]
    
    col_offset = 0
    for d in decomps:
        r = len(d["U"][0])
        for i in range(U_rows):
            for t in range(r):
                U[i][col_offset + t] = d["U"][i][t]
        for i in range(V_rows):
            for t in range(r):
                V[i][col_offset + t] = d["V"][i][t]
        for i in range(W_rows):
            for t in range(r):
                W[i][col_offset + t] = d["W"][i][t]
        col_offset += r
        
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS # 3, 3, 5
    
    # Decompose M_{3,3,5} into M_{3,3,3} and M_{3,3,2}
    
    # 1. M_{3,3,3}
    # Split A(3x3) and B(3x3) into 2x2 blocks
    # A = [A11 A12; A21 A22]
    # B = [B11 B12; B21 B22]
    # C = [C11 C12; C21 C22]
    
    # Terms:
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    
    # We need to construct decompositions for each product and sum them.
    # However, the indices in the global U, V, W must be mapped correctly.
    
    # Let's build the decompositions for the sub-matrices and then embed them.
    
    # Sub-decomposition 1: A11 (2x2) * B11 (2x2) -> C11 (2x2)
    # This is M_{2,2,2}. Use Strassen.
    d_11 = _build_strassen_222()
    
    # Sub-decomposition 2: A12 (2x1) * B21 (1x2) -> C11 (2x2)
    # This is M_{2,1,2}. Trivial rank 4.
    d_12 = _build_rect_trivial(2, 1, 2)
    
    # Sub-decomposition 3: A11 (2x2) * B12 (2x2) -> C12 (2x2)
    # This is M_{2,2,2}. Use Strassen? 
    # Wait, B12 is 2x2? 
    # B is 3x3. B12 is top-right 2x1? 
    # Let's check partition:
    # A = [ A11 (2x2) | A12 (2x1) ]
    #     [ A21 (1x2) | A22 (1x1) ]
    # B = [ B11 (2x2) | B12 (2x1) ]
    #     [ B21 (1x2) | B22 (1x1) ]
    
    # C11 (2x2) = A11(2x2)B11(2x2) + A12(2x1)B21(1x2)
    # C12 (2x1) = A11(2x2)B12(2x1) + A12(2x1)B22(1x1)
    # C21 (1x2) = A21(1x2)B11(2x2) + A22(1x1)B21(1x2)
    # C22 (1x1) = A21(1x2)B12(2x1) + A22(1x1)B22(1x1)
    
    # So:
    # T1: M_{2,2,2} (A11, B11, C11) -> Rank 7
    # T2: M_{2,1,2} (A12, B21, C11) -> Rank 4
    # T3: M_{2,2,1} (A11, B12, C12) -> Rank 4 (Trivial)
    # T4: M_{2,1,1} (A12, B22, C12) -> Rank 2 (Trivial)
    # T5: M_{1,2,2} (A21, B11, C21) -> Rank 4 (Trivial)
    # T6: M_{1,1,2} (A22, B21, C21) -> Rank 2 (Trivial)
    # T7: M_{1,2,1} (A21, B12, C22) -> Rank 2 (Trivial)
    # T8: M_{1,1,1} (A22, B22, C22) -> Rank 1 (Trivial)
    
    # Total Rank for M_{3,3,3}: 7+4+4+2+4+2+2+1 = 26.
    # Wait, earlier I estimated 24. Let's re-verify T3, T5.
    # T3: A11(2x2) * B12(2x1). Result 2x1. M_{2,2,1}. Rank 4. Correct.
    # T5: A21(1x2) * B11(2x2). Result 1x2. M_{1,2,2}. Rank 4. Correct.
    
    # So Rank 26 for M_{3,3,3}.
    
    # Now M_{3,3,2}
    # B is 3x2.
    # A = [ A11 (2x2) | A12 (2x1) ]
    #     [ A21 (1x2) | A22 (1x1) ]
    # B = [ B1 (2x2) ]
    #     [ B2 (1x2) ]
    # Note: B1 is top 2 rows of B (2x2). B2 is bottom 1 row of B (1x2).
    
    # C (3x2) = [ C_top (2x2) ]
    #            [ C_bot (1x2) ]
    
    # C_top = A11(2x2)B1(2x2) + A12(2x1)B2(1x2)
    # C_bot = A21(1x2)B1(2x2) + A22(1x1)B2(1x2)
    
    # Terms:
    # T9: M_{2,2,2} (A11, B1, C_top) -> Rank 7
    # T10: M_{2,1,2} (A12, B2, C_top) -> Rank 4
    # T11: M_{1,2,2} (A21, B1, C_bot) -> Rank 4
    # T12: M_{1,1,2} (A22, B2, C_bot) -> Rank 2
    
    # Total Rank for M_{3,3,2}: 7+4+4+2 = 17.
    
    # Total Rank: 26 + 17 = 43.
    # This is better than 45.
    
    # We need to construct the global U, V, W.
    
    # Helper to embed a decomposition into the global frame.
    # We need to map local indices to global indices.
    
    # Global U (3x3=9), V (3x5=15), W (5x3=15).
    
    # For M_{3,3,3} (uses B cols 0,1,2 and C cols 0,1,2):
    # Local U (2x2=4, etc) maps to Global U.
    # Local V (2x2=4, etc) maps to Global V (rows 0-5 for B11, 6-7 for B12, etc).
    # Local W (2x2=4, etc) maps to Global W (rows 0-5 for C11, 6-7 for C12, etc).
    
    # This embedding is complex. I will write a helper function.
    
    def embed(decomp, n_loc, m_loc, p_loc, i_off, j_off, k_off):
        """
        Embed a local decomposition of size n_loc x m_loc x p_loc 
        into the global frame.
        i_off: offset for i index in A (rows)
        j_off: offset for j index in A (cols) / B (rows)
        k_off: offset for k index in B (cols) / C (rows)
        """
        r = len(decomp["U"][0])
        U_loc = np.array(decomp["U"])
        V_loc = np.array(decomp["V"])
        W_loc = np.array(decomp["W"])
        
        U_g = np.zeros((9, r))
        V_g = np.zeros((15, r))
        W_g = np.zeros((15, r))
        
        # Map U indices: local (i,j) -> global (i+i_off, j+j_off)
        for i in range(n_loc):
            for j in range(m_loc):
                li = i * m_loc + j
                gi = (i + i_off) * 3 + (j + j_off)
                U_g[gi, :] = U_loc[li, :]
                
        # Map V indices: local (j,k) -> global (j+j_off, k+k_off)
        for j in range(m_loc):
            for k in range(p_loc):
                lj = j * p_loc + k
                gj = (j + j_off) * 5 + (k + k_off)
                V_g[gj, :] = V_loc[lj, :]
                
        # Map W indices: local (k,i) -> global (k+k_off, i+i_off)
        for k in range(p_loc):
            for i in range(n_loc):
                lk = k * n_loc + i
                gk = (k + k_off) * 3 + (i + i_off)
                W_g[gk, :] = W_loc[lk, :]
                
        return {"U": U_g.tolist(), "V": V_g.tolist(), "W": W_g.tolist()}
        
    decomp_list = []
    
    # M_{3,3,3} parts
    # T1: A11(0,0) B11(0,0) C11(0,0) -> n=2,m=2,p=2, off=(0,0,0)
    decomp_list.append(embed(_build_strassen_222(), 2, 2, 2, 0, 0, 0))
    
    # T2: A12(0,2) B21(1,0) C11(0,0) -> n=2,m=1,p=2, off=(0,2,0)
    # A12 is rows 0-1, col 2. B21 is row 2, cols 0-1.
    # Wait, B21 in local coords of M_{2,1,2} is (j,k) where j=0 (local), k=0-1.
    # Global j = 2 (row of B), k = 0,1.
    # So j_off = 2? No, B21 is row 2 of B. In global V, row index is j*5+k.
    # So j=2, k=0,1.
    # A12 is col 2 of A. In global U, col index is i*3+j.
    # So i=0,1, j=2.
    # C11 is cols 0,1 of C. In global W, row index is k*3+i.
    # So k=0,1, i=0,1.
    
    # My embed function takes i_off, j_off, k_off.
    # For A12: i_off=0, j_off=2.
    # For B21: j_off=2, k_off=0.
    # For C11: k_off=0, i_off=0.
    # This is consistent.
    
    decomp_list.append(embed(_build_rect_trivial(2, 1, 2), 2, 1, 2, 0, 2, 0))
    
    # T3: A11(0,0) B12(0,2) C12(0,1) -> n=2,m=2,p=1, off=(0,0,2)
    # B12 is cols 2 of B? No, B is 3x3. B12 is top-right 2x1.
    # So k=2.
    # C12 is col 1 of C? No, C is 3x3. C12 is top-right 2x1.
    # So k=1? No, C12 is columns 1?
    # C = [C11 C12; C21 C22].
    # C11 is cols 0,1. C12 is col 2?
    # If A is 3x3, B is 3x3.
    # A11 is 2x2 (cols 0,1). A12 is 2x1 (col 2).
    # B11 is 2x2 (rows 0,1). B12 is 2x1 (row 2? No, B12 is cols 2 of B?
    # B = [B11 B12; B21 B22].
    # B11 is 2x2 (rows 0,1, cols 0,1).
    # B12 is 2x1 (rows 0,1, col 2).
    # So for B12, k=2.
    # C12 is 2x1 (rows 0,1, col 2).
    # So for C12, k=2?
    # C = [C11 C12; C21 C22].
    # C11 is 2x2 (rows 0,1, cols 0,1).
    # C12 is 2x1 (rows 0,1, col 2).
    # So k=2 for C12.
    
    decomp_list.append(embed(_build_rect_trivial(2, 2, 1), 2, 2, 1, 0, 0, 2))
    
    # T4: A12(0,2) B22(1,1) C12(0,1) -> n=2,m=1,p=1, off=(0,2,2)
    # B22 is row 2, col 2. So j=2, k=2.
    # C12 is row 0,1, col 2. So k=2, i=0,1.
    decomp_list.append(embed(_build_rect_trivial(2, 1, 1), 2, 1, 1, 0, 2, 2))
    
    # T5: A21(1,0) B11(0,0) C21(1,0) -> n=1,m=2,p=2, off=(1,0,0)
    # A21 is row 2, cols 0,1. So i=2, j=0,1.
    # B11 is rows 0,1, cols 0,1. So j=0,1, k=0,1.
    # C21 is row 2, cols 0,1. So k=0,1, i=2.
    decomp_list.append(embed(_build_rect_trivial(1, 2, 2), 1, 2, 2, 1, 0, 0))
    
    # T6: A22(1,1) B21(1,0) C21(1,0) -> n=1,m=1,p=2, off=(1,2,0)
    # A22 is row 2, col 2.
    # B21 is row 2, cols 0,1.
    # C21 is row 2, cols 0,1.
    decomp_list.append(embed(_build_rect_trivial(1, 1, 2), 1, 1, 2, 1, 2, 0))
    
    # T7: A21(1,0) B12(0,2) C22(1,1) -> n=1,m=2,p=1, off=(1,0,2)
    # A21 is row 2, cols 0,1.
    # B12 is rows 0,1, col 2.
    # C22 is row 2, col 2.
    decomp_list.append(embed(_build_rect_trivial(1, 2, 1), 1, 2, 1, 1, 0, 2))
    
    # T8: A22(1,1) B22(1,1) C22(1,1) -> n=1,m=1,p=1, off=(1,2,2)
    decomp_list.append(embed(_build_rect_trivial(1, 1, 1), 1, 1, 1, 1, 2, 2))
    
    # M_{3,3,2} parts
    # B is 3x2. Cols 3,4 of global B.
    # k_off = 3.
    
    # T9: A11(0,0) B1(0,0) C_top(0,0) -> n=2,m=2,p=2, off=(0,0,3)
    decomp_list.append(embed(_build_strassen_222(), 2, 2, 2, 0, 0, 3))
    
    # T10: A12(0,2) B2(1,0) C_top(0,0) -> n=2,m=1,p=2, off=(0,2,3)
    decomp_list.append(embed(_build_rect_trivial(2, 1, 2), 2, 1, 2, 0, 2, 3))
    
    # T11: A21(1,0) B1(0,0) C_bot(1,0) -> n=1,m=2,p=2, off=(1,0,3)
    decomp_list.append(embed(_build_rect_trivial(1, 2, 2), 1, 2, 2, 1, 0, 3))
    
    # T12: A22(1,1) B2(1,0) C_bot(1,0) -> n=1,m=1,p=2, off=(1,2,3)
    decomp_list.append(embed(_build_rect_trivial(1, 1, 2), 1, 1, 2, 1, 2, 3))
    
    result = _combine_decompositions(decomp_list)
    
    return result
# EVOLVE-BLOCK-END
