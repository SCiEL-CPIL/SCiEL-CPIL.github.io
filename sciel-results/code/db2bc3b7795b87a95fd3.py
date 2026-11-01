"""Rewritten program for matrix multiplication 2,5,6 rank minimization."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]
DEADLINE = 120.0

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # We will construct a rank 56 decomposition using Strassen's algorithm recursively.
    # Structure:
    # M(2,5,6) <= M(2,5,5) + M(2,5,1)
    # M(2,5,5) <= M(2,4,5) + M(2,1,5)
    # M(2,4,5) <= M(2,4,4) + M(2,4,1)
    # M(2,4,4) <= M(2,2,4) + M(2,2,4)
    # M(2,2,4) <= M(2,2,2) + M(2,2,2)
    # M(2,2,2) = 7 (Strassen)
    
    # Total Rank = 10 (2,5,1) + 10 (2,1,5) + 8 (2,4,1) + 2 * (2 * 7) = 28 + 28 = 56.
    
    rank_total = 56
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    col_offset = 0
    
    # Helper to add a block multiplication
    # i_offset, j_offset, k_offset are indices in the global matrices A, B, C
    # A is n x m, B is m x p, C is p x n (Wait, C is n x p in standard, but W index is k*n+i? 
    # Task says W[k*n+i, r]. C[i,k] = sum A[i,j] B[j,k].
    # So W index corresponds to C index.
    # A index: i*m + j.
    # B index: j*p + k.
    # C index: k*n + i.
    
    def add_block(i_off, j_off, k_off, n_sub, m_sub, p_sub, U_sub, V_sub, W_sub):
        nonlocal col_offset
        for r in range(U_sub.shape[1]):
            # U_sub is (n_sub * m_sub, rank_sub)
            # Map to global U
            for i_sub in range(n_sub):
                for j_sub in range(m_sub):
                    val = U_sub[i_sub * m_sub + j_sub, r]
                    if val != 0:
                        U[(i_off + i_sub) * m + (j_off + j_sub), col_offset + r] = val
                        
            # V_sub is (m_sub * p_sub, rank_sub)
            for j_sub in range(m_sub):
                for k_sub in range(p_sub):
                    val = V_sub[j_sub * p_sub + k_sub, r]
                    if val != 0:
                        V[(j_off + j_sub) * p + (k_off + k_sub), col_offset + r] = val
                        
            # W_sub is (p_sub * n_sub, rank_sub)
            for k_sub in range(p_sub):
                for i_sub in range(n_sub):
                    val = W_sub[k_sub * n_sub + i_sub, r]
                    if val != 0:
                        W[(k_off + k_sub) * n + (i_off + i_sub), col_offset + r] = val
        col_offset += U_sub.shape[1]

    # 1. M(2,5,1) part: A (2x5) * B_right (5x1)
    # This is just 10 mults.
    U_sub_1 = np.zeros((2*5, 10))
    V_sub_1 = np.zeros((5*1, 10))
    W_sub_1 = np.zeros((1*2, 10))
    idx = 0
    for i in range(2):
        for j in range(5):
            U_sub_1[i*5+j, idx] = 1
            V_sub_1[j*1+0, idx] = 1
            W_sub_1[0*2+i, idx] = 1
            idx += 1
    add_block(0, 0, 5, 2, 5, 1, U_sub_1, V_sub_1, W_sub_1) # B cols 5..5 (index 5)
    
    # 2. M(2,1,5) part: A_right (2x1) * B_bottom (1x5)
    # A rows 0..1, col 4. B row 4, cols 0..4.
    U_sub_2 = np.zeros((2*1, 10))
    V_sub_2 = np.zeros((1*5, 10))
    W_sub_2 = np.zeros((5*2, 10))
    idx = 0
    for i in range(2):
        for k in range(5):
            U_sub_2[i*1+0, idx] = 1
            V_sub_2[0*5+k, idx] = 1
            W_sub_2[k*2+i, idx] = 1
            idx += 1
    add_block(0, 4, 0, 2, 1, 5, U_sub_2, V_sub_2, W_sub_2) # A col 4, B row 4
    
    # 3. M(2,4,1) part: A_left (2x4) * B_mid_right (4x1)
    # A rows 0..1, cols 0..3. B rows 0..3, col 4.
    U_sub_3 = np.zeros((2*4, 8))
    V_sub_3 = np.zeros((4*1, 8))
    W_sub_3 = np.zeros((1*2, 8))
    idx = 0
    for i in range(2):
        for j in range(4):
            U_sub_3[i*4+j, idx] = 1
            V_sub_3[j*1+0, idx] = 1
            W_sub_3[0*2+i, idx] = 1
            idx += 1
    add_block(0, 0, 4, 2, 4, 1, U_sub_3, V_sub_3, W_sub_3) # B col 4
    
    # 4. M(2,4,4) part: A_left (2x4) * B_top_left (4x4)
    # This is split into two M(2,2,4)
    # M(2,2,4) is split into two M(2,2,2)
    
    def get_strassen_2x2():
        # Returns U, V, W for 2x2x2 with rank 7
        U_s = np.zeros((4, 7))
        V_s = np.zeros((4, 7))
        W_s = np.zeros((4, 7))
        
        # M1 = (A11+A22)(B11+B22)
        U_s[0, 0] = 1; U_s[3, 0] = 1 # A11, A22
        V_s[0, 0] = 1; V_s[3, 0] = 1 # B11, B22
        W_s[0, 0] = 1; W_s[3, 0] = 1 # C11, C22
        W_s[1, 0] = -1; W_s[2, 0] = 1 # C12, C21 ? No, C is 2x2.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let's use standard Strassen coefficients
        # M1: A11+A22, B11+B22
        # M2: A21+A22, B11
        # M3: A11, B12-B22
        # M4: A22, B21-B11
        # M5: A11+A12, B22
        # M6: A21-A11, B11+B12
        # M7: A12-A22, B21+B22
        
        U_s[0, 0] = 1; U_s[3, 0] = 1
        V_s[0, 0] = 1; V_s[3, 0] = 1
        W_s[0, 0] = 1; W_s[3, 0] = 1
        
        U_s[1, 1] = 1; U_s[3, 1] = 1
        V_s[0, 1] = 1
        W_s[2, 1] = 1; W_s[3, 1] = 1
        
        U_s[0, 2] = 1
        V_s[1, 2] = 1; V_s[3, 2] = -1
        W_s[1, 2] = 1; W_s[3, 2] = 1
        
        U_s[3, 3] = 1
        V_s[2, 3] = 1; V_s[0, 3] = -1
        W_s[0, 3] = 1; W_s[2, 3] = 1
        
        U_s[0, 4] = 1; U_s[1, 4] = 1
        V_s[3, 4] = 1
        W_s[0, 4] = -1; W_s[1, 4] = 1
        
        U_s[1, 5] = 1; U_s[0, 5] = -1
        V_s[0, 5] = 1; V_s[1, 5] = 1
        W_s[3, 5] = 1
        
        U_s[1, 6] = 1; U_s[3, 6] = -1
        V_s[2, 6] = 1; V_s[3, 6] = 1
        W_s[0, 6] = 1; W_s[1, 6] = 1
        
        return U_s, V_s, W_s

    U_s, V_s, W_s = get_strassen_2x2()
    
    # Block 4a: A(0..1, 0..1) * B(0..1, 0..1) -> C(0..1, 0..1)
    add_block(0, 0, 0, 2, 2, 2, U_s, V_s, W_s)
    
    # Block 4b: A(0..1, 0..1) * B(0..1, 2..3) -> C(2..3, 0..1)
    # Note: B cols 2,3. C cols 2,3.
    add_block(0, 0, 2, 2, 2, 2, U_s, V_s, W_s)
    
    # Block 4c: A(0..1, 2..3) * B(2..3, 0..1) -> C(0..1, 0..1)
    # Wait, M(2,4,4) = [A1 A2] * [B11 B12; B21 B22]
    # = [A1 B11 + A2 B21, A1 B12 + A2 B22]
    # We did A1 B11 and A1 B12.
    # Now need A2 B21 and A2 B22.
    
    # Block 4d: A(0..1, 2..3) * B(2..3, 0..1) -> C(0..1, 0..1)
    add_block(0, 2, 0, 2, 2, 2, U_s, V_s, W_s)
    
    # Block 4e: A(0..1, 2..3) * B(2..3, 2..3) -> C(2..3, 0..1)
    add_block(0, 2, 2, 2, 2, 2, U_s, V_s, W_s)
    
    # Check time
    if time.time() - start_time > DEADLINE - 5:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Verify rank
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
