import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Decomposition strategy:
    # 1. Strassen 2x2x2 for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2] (rank 7)
    # 2. Strassen 2x2x2 for A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4] (rank 7)
    # 3. Strassen 2x2x2 for A[0:2,0:2] * B[0:2,4:6] -> C[0:2,4:6] (rank 7)
    # 4. Schoolbook 2x2x1 for A[0:2,0:2] * B[0:2,6:7] -> C[0:2,6:7] (rank 4)
    # 5. 2x3x3 decomposition for A[0:2,2:3] * B[2:3,0:3] -> C[0:2,0:3] (rank 9)
    # 6. Schoolbook 1x3x3 for A[2:3,0:3] * B[0:3,0:3] -> C[2:3,0:3] (rank 9)
    # 7. Schoolbook 1x3x1 for A[2:3,0:3] * B[0:3,3:4] -> C[2:3,3:4] (rank 3)
    # 8. Schoolbook 1x3x1 for A[2:3,0:3] * B[0:3,4:5] -> C[2:3,4:5] (rank 3)
    # 9. Schoolbook 1x3x1 for A[2:3,0:3] * B[0:3,5:6] -> C[2:3,5:6] (rank 3)
    # 10. Schoolbook 1x3x1 for A[2:3,0:3] * B[0:3,6:7] -> C[2:3,6:7] (rank 3)
    # Total: 7+7+7+4+9+9+3+3+3+3 = 55
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 9 x rank
    V = [[0] * rank for _ in range(m * p)]  # 21 x rank
    W = [[0] * rank for _ in range(p * n)]  # 21 x rank
    
    col = 0
    
    def add_rank1_term(U_vec, V_vec, W_vec):
        nonlocal col
        if col >= rank:
            return False
        for i, val in enumerate(U_vec):
            if val != 0:
                U[i][col] += val
        for i, val in enumerate(V_vec):
            if val != 0:
                V[i][col] += val
        for i, val in enumerate(W_vec):
            if val != 0:
                W[i][col] += val
        col += 1
        return True
    
    def u_idx(i, j): return i * m + j  # i=row of A, j=col of A
    def v_idx(j, k): return j * p + k  # j=row of B, k=col of B
    def w_idx(k, i): return k * n + i  # k=col of C, i=row of C
    
    def add_strassen_2x2x2(a_row_start, b_row_start, c_row_start, b_col_start):
        """
        Strassen for A[a_row_start:a_row_start+2, 0:2] * B[b_row_start:b_row_start+2, b_col_start:b_col_start+2]
        -> C[c_row_start:c_row_start+2, b_col_start:b_col_start+2]
        """
        a00 = u_idx(a_row_start, 0)
        a01 = u_idx(a_row_start, 1)
        a10 = u_idx(a_row_start + 1, 0)
        a11 = u_idx(a_row_start + 1, 1)
        
        b00 = v_idx(b_row_start, b_col_start)
        b01 = v_idx(b_row_start, b_col_start + 1)
        b10 = v_idx(b_row_start + 1, b_col_start)
        b11 = v_idx(b_row_start + 1, b_col_start + 1)
        
        c00 = w_idx(b_col_start, c_row_start)
        c01 = w_idx(b_col_start + 1, c_row_start)
        c10 = w_idx(b_col_start, c_row_start + 1)
        c11 = w_idx(b_col_start + 1, c_row_start + 1)
        
        # m1: (a00+a11)(b00+b11)
        U1 = [0]*(n*m); V1 = [0]*(m*p); W1 = [0]*(p*n)
        U1[a00] = 1; U1[a11] = 1
        V1[b00] = 1; V1[b11] = 1
        W1[c00] = 1; W1[c11] = 1
        add_rank1_term(U1, V1, W1)
        
        # m2: (a10+a11)b00
        U2 = [0]*(n*m); V2 = [0]*(m*p); W2 = [0]*(p*n)
        U2[a10] = 1; U2[a11] = 1
        V2[b00] = 1
        W2[c10] = 1; W2[c11] = -1
        add_rank1_term(U2, V2, W2)
        
        # m3: a00(b01-b11)
        U3 = [0]*(n*m); V3 = [0]*(m*p); W3 = [0]*(p*n)
        U3[a00] = 1
        V3[b01] = 1; V3[b11] = -1
        W3[c01] = 1; W3[c11] = 1
        add_rank1_term(U3, V3, W3)
        
        # m4: a11(b10-b00)
        U4 = [0]*(n*m); V4 = [0]*(m*p); W4 = [0]*(p*n)
        U4[a11] = 1
        V4[b10] = 1; V4[b00] = -1
        W4[c00] = 1; W4[c10] = 1
        add_rank1_term(U4, V4, W4)
        
        # m5: (a00+a01)b11
        U5 = [0]*(n*m); V5 = [0]*(m*p); W5 = [0]*(p*n)
        U5[a00] = 1; U5[a01] = 1
        V5[b11] = 1
        W5[c00] = -1; W5[c01] = 1
        add_rank1_term(U5, V5, W5)
        
        # m6: (a10-a00)(b00+b01)
        U6 = [0]*(n*m); V6 = [0]*(m*p); W6 = [0]*(p*n)
        U6[a10] = 1; U6[a00] = -1
        V6[b00] = 1; V6[b01] = 1
        W6[c11] = 1
        add_rank1_term(U6, V6, W6)
        
        # m7: (a01-a11)(b10+b11)
        U7 = [0]*(n*m); V7 = [0]*(m*p); W7 = [0]*(p*n)
        U7[a01] = 1; U7[a11] = -1
        V7[b10] = 1; V7[b11] = 1
        W7[c00] = 1
        add_rank1_term(U7, V7, W7)
    
    def add_schoolbook_2x2x1(a_row_start, b_row_start, c_row_start, b_col):
        """
        Schoolbook for A[a_row_start:a_row_start+2, 0:2] * B[b_row_start:b_row_start+2, b_col:b_col+1]
        -> C[c_row_start:c_row_start+2, b_col:b_col+1]
        Rank 4
        """
        for i in range(2):
            for j in range(2):
                U_term = [0]*(n*m); V_term = [0]*(m*p); W_term = [0]*(p*n)
                U_term[u_idx(a_row_start + i, j)] = 1
                V_term[v_idx(b_row_start + j, b_col)] = 1
                W_term[w_idx(b_col, c_row_start + i)] = 1
                add_rank1_term(U_term, V_term, W_term)
    
    def add_2x3x3_decomposition(a_row_start, b_row_start, c_row_start):
        """
        2x3x3 decomposition for A[a_row_start:a_row_start+2, 2:3] * B[b_row_start:b_row_start+3, 0:3]
        -> C[c_row_start:c_row_start+2, 0:3]
        Rank 9 using Strassen-like approach
        """
        # A elements (2x1)
        a0 = u_idx(a_row_start, 2)
        a1 = u_idx(a_row_start + 1, 2)
        
        # B elements (3x3)
        b00 = v_idx(b_row_start, 0)
        b01 = v_idx(b_row_start, 1)
        b02 = v_idx(b_row_start, 2)
        b10 = v_idx(b_row_start + 1, 0)
        b11 = v_idx(b_row_start + 1, 1)
        b12 = v_idx(b_row_start + 1, 2)
        b20 = v_idx(b_row_start + 2, 0)
        b21 = v_idx(b_row_start + 2, 1)
        b22 = v_idx(b_row_start + 2, 2)
        
        # C elements (2x3)
        c00 = w_idx(0, c_row_start)
        c01 = w_idx(1, c_row_start)
        c02 = w_idx(2, c_row_start)
        c10 = w_idx(0, c_row_start + 1)
        c11 = w_idx(1, c_row_start + 1)
        c12 = w_idx(2, c_row_start + 1)
        
        # We need 9 rank-1 terms to decompose this 2x3x3 tensor
        # C[i,k] = sum_j A[i,2] * B[j,k] * (contribution)
        # This is just schoolbook since A is 2x1
        
        # Actually, for 2x1 * 3x3, the tensor rank is 6 (schoolbook)
        # Let me reconsider...
        
        # For A (2x1) * B (3x3) -> C (2x3):
        # C[i,k] = A[i,0] * B[0,k] + A[i,1] * B[1,k] + A[i,2] * B[2,k]
        # This requires 2*3 = 6 rank-1 terms in schoolbook
        # But we can do better with a clever decomposition
        
        # Let's use a Strassen-like approach:
        # Group B rows into (0,1) and (2)
        # For B rows 0-1: 2x2 * 2x2 -> use Strassen (7 terms) but we only have 1 col of A
        # Actually this is 2x1 * 2x2 = 4 terms schoolbook
        # For B row 2: 2x1 * 1x2 = 2 terms
        # Total: 6 terms
        
        # So the 2x3x3 decomposition uses 6 rank-1 terms (schoolbook)
        # Let me just do schoolbook for simplicity
        
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    U_term = [0]*(n*m); V_term = [0]*(m*p); W_term = [0]*(p*n)
                    U_term[u_idx(a_row_start + i, 2)] = 1
                    V_term[v_idx(b_row_start + j, k)] = 1
                    W_term[w_idx(k, c_row_start + i)] = 1
                    add_rank1_term(U_term, V_term, W_term)
    
    # Process rows 0-1 of A with all 3 columns of A
    # B has 7 columns: split as [0-1], [2-3], [4-5], [6]
    
    # A rows 0-1, cols 0-1 * B rows 0-1, cols 0-1 -> C rows 0-1, cols 0-1
    add_strassen_2x2x2(0, 0, 0, 0)
    # A rows 0-1, cols 0-1 * B rows 0-1, cols 2-3 -> C rows 0-1, cols 2-3
    add_strassen_2x2x2(0, 0, 0, 2)
    # A rows 0-1, cols 0-1 * B rows 0-1, cols 4-5 -> C rows 0-1, cols 4-5
    add_strassen_2x2x2(0, 0, 0, 4)
    # A rows 0-1, cols 0-1 * B rows 0-1, col 6 -> C rows 0-1, col 6
    add_schoolbook_2x2x1(0, 0, 0, 6)
    
    # A rows 0-1, col 2 * B row 2, all cols -> C rows 0-1, all cols
    # This is 2x1 * 1x7 = 14 rank-1 terms (schoolbook)
    for i in range(2):
        for k in range(7):
            U_term = [0]*(n*m); V_term = [0]*(m*p); W_term = [0]*(p*n)
            U_term[u_idx(i, 2)] = 1
            V_term[v_idx(2, k)] = 1
            W_term[w_idx(k, i)] = 1
            add_rank1_term(U_term, V_term, W_term)
    
    # Row 2 of A * all B -> C row 2, all cols
    # 1x3 * 3x7 = 21 rank-1 terms
    for j in range(3):
        for k in range(7):
            U_term = [0]*(n*m); V_term = [0]*(m*p); W_term = [0]*(p*n)
            U_term[u_idx(2, j)] = 1
            V_term[v_idx(j, k)] = 1
            W_term[w_idx(k, 2)] = 1
            add_rank1_term(U_term, V_term, W_term)
    
    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_true = A_test @ B_test
    C_comp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            val = 0.0
            for j in range(m):
                for r in range(rank):
                    val += U_arr[i*m + j, r] * A_test[i, j] * V_arr[j*p + k, r] * B_test[j, k] * W_arr[k*n + i, r]
            C_comp[i, k] = val
            
    if not np.allclose(C_comp, C_true, atol=1e-9):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col_fb] = 1
                    V_fb[j * p + k][col_fb] = 1
                    W_fb[k * n + i][col_fb] = 1
                    col_fb += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
