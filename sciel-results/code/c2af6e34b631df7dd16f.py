"""Matrix multiplication 2x3 x 3x5 with exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """
    Return (U, V, W) for Strassen's rank-7 decomposition of 2x2 * 2x2.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _schoolbook_2x2x1():
    """
    Return (U, V, W) for schoolbook rank-4 decomposition of 2x2 * 2x1.
    A = [[a, b], [c, d]], B = [[e], [f]]
    C = [[ae+bf], [ce+df]]
    M1=ae, M2=bf, M3=ce, M4=df
    """
    U = np.zeros((4, 4))
    V = np.zeros((2, 4))
    W = np.zeros((2, 4))
    
    # M1 = a*e
    U[0, 0] = 1
    V[0, 0] = 1
    W[0, 0] = 1
    
    # M2 = b*f
    U[1, 1] = 1
    V[1, 1] = 1
    W[0, 1] = 1
    
    # M3 = c*e
    U[2, 2] = 1
    V[0, 2] = 1
    W[1, 2] = 1
    
    # M4 = d*f
    U[3, 3] = 1
    V[1, 3] = 1
    W[1, 3] = 1
    
    return U, V, W

def _schoolbook_2x1x5():
    """
    Return (U, V, W) for schoolbook rank-10 decomposition of 2x1 * 1x5.
    A = [[a], [b]], B = [[e, f, g, h, i]]
    C = [[ae, af, ag, ah, ai], [be, bf, bg, bh, bi]]
    """
    U = np.zeros((2, 10))
    V = np.zeros((5, 10))
    W = np.zeros((10, 10))
    
    # C[i,k] = A[i,0] * B[0,k]
    # Indices:
    # U: A[i,0] -> index i
    # V: B[0,k] -> index k
    # W: C[i,k] -> index k*2 + i
    
    for k in range(5):
        for i in range(2):
            col = k * 2 + i
            U[i, col] = 1.0
            V[k, col] = 1.0
            W[k * 2 + i, col] = 1.0
            
    return U, V, W

def _verify_full_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify the tensor decomposition is correct."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    """
    Constructs the tensor decomposition for matrix multiplication 2x3 * 3x5.
    
    Strategy:
    Split A (2x3) into A_L (2x2) and A_R (2x1).
    Split B (3x5) into B_T (2x5) and B_B (1x5).
    
    C = A_L * B_T + A_R * B_B
    
    Decompose A_L * B_T (2x2 * 2x5):
    Split B_T into B_T1 (2x2) and B_T2 (2x3).
    A_L * B_T1 (2x2 * 2x2): Rank 7 (Strassen)
    A_L * B_T2 (2x2 * 2x3):
        Split B_T2 into B_T2a (2x2) and B_T2b (2x1).
        A_L * B_T2a (2x2 * 2x2): Rank 7 (Strassen)
        A_L * B_T2b (2x2 * 2x1): Rank 4 (Schoolbook)
        Total for A_L * B_T: 7 + 7 + 4 = 18
    
    Decompose A_R * B_B (2x1 * 1x5): Rank 10 (Schoolbook)
    
    Total Rank: 18 + 10 = 28
    """
    n, m, p = DIMENSIONS  # 2, 3, 5
    
    # Total rank
    rank_total = 28
    
    U = np.zeros((n * m, rank_total)) # 6 x 28
    V = np.zeros((m * p, rank_total)) # 15 x 28
    W = np.zeros((p * n, rank_total)) # 10 x 28
    
    col_offset = 0
    
    # --- Block 1: A_L * B_T1 (2x2 * 2x2) ---
    # A_L: rows 0,1; cols 0,1. Indices in A: 0,1,3,4
    # B_T1: rows 0,1; cols 0,1. Indices in B: 0,1,5,6
    # C: rows 0,1; cols 0,1. Indices in C: 0,1,5,6
    
    U_s, V_s, W_s = _strassen_2x2x2()
    
    a_map = [0, 1, 3, 4]
    b_map = [0, 1, 5, 6]
    c_map = [0, 1, 5, 6]
    
    for r in range(7):
        for s_idx, f_idx in enumerate(a_map):
            U[f_idx, col_offset + r] = U_s[s_idx, r]
        for s_idx, f_idx in enumerate(b_map):
            V[f_idx, col_offset + r] = V_s[s_idx, r]
        for s_idx, f_idx in enumerate(c_map):
            W[f_idx, col_offset + r] = W_s[s_idx, r]
            
    col_offset += 7
    
    # --- Block 2: A_L * B_T2a (2x2 * 2x2) ---
    # B_T2a: rows 0,1; cols 2,3. Indices in B: 2,3,7,8
    # C: rows 0,1; cols 2,3. Indices in C: 2,3,7,8
    
    U_s2, V_s2, W_s2 = _strassen_2x2x2()
    
    a_map2 = [0, 1, 3, 4]
    b_map2 = [2, 3, 7, 8]
    c_map2 = [2, 3, 7, 8]
    
    for r in range(7):
        for s_idx, f_idx in enumerate(a_map2):
            U[f_idx, col_offset + r] = U_s2[s_idx, r]
        for s_idx, f_idx in enumerate(b_map2):
            V[f_idx, col_offset + r] = V_s2[s_idx, r]
        for s_idx, f_idx in enumerate(c_map2):
            W[f_idx, col_offset + r] = W_s2[s_idx, r]
            
    col_offset += 7
    
    # --- Block 3: A_L * B_T2b (2x2 * 2x1) ---
    # B_T2b: rows 0,1; col 4. Indices in B: 4, 9
    # C: rows 0,1; col 4. Indices in C: 4, 9
    
    U_sb, V_sb, W_sb = _schoolbook_2x2x1()
    
    a_map3 = [0, 1, 3, 4]
    b_map3 = [4, 9]
    c_map3 = [4, 9]
    
    for r in range(4):
        for s_idx, f_idx in enumerate(a_map3):
            U[f_idx, col_offset + r] = U_sb[s_idx, r]
        for s_idx, f_idx in enumerate(b_map3):
            V[f_idx, col_offset + r] = V_sb[s_idx, r]
        for s_idx, f_idx in enumerate(c_map3):
            W[f_idx, col_offset + r] = W_sb[s_idx, r]
            
    col_offset += 4
    
    # --- Block 4: A_R * B_B (2x1 * 1x5) ---
    # A_R: rows 0,1; col 2. Indices in A: 2, 5
    # B_B: row 2; cols 0-4. Indices in B: 10, 11, 12, 13, 14
    # C: rows 0,1; cols 0-4. Indices in C: 0-9
    
    U_sr, V_sr, W_sr = _schoolbook_2x1x5()
    
    a_map4 = [2, 5]
    b_map4 = [10, 11, 12, 13, 14]
    c_map4 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    for r in range(10):
        for s_idx, f_idx in enumerate(a_map4):
            U[f_idx, col_offset + r] = U_sr[s_idx, r]
        for s_idx, f_idx in enumerate(b_map4):
            V[f_idx, col_offset + r] = V_sr[s_idx, r]
        for s_idx, f_idx in enumerate(c_map4):
            W[f_idx, col_offset + r] = W_sr[s_idx, r]
            
    col_offset += 10
    
    # Verify
    if not _verify_full_decomposition(U, V, W, n, m, p, tol=1e-8):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
