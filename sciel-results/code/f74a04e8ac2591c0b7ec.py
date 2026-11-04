import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank is 36 based on Strassen for 2x2x2 and hybrid for 2x2x3
    # 7 + 7 + 11 + 11 = 36
    
    # Helper to create Strassen 2x2x2 decomposition
    # U, V, W are 4x7
    def get_strassen_222():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00 + A11)(B00 + B11)
        U[:, 0] = [1, 0, 0, 1]
        V[:, 0] = [1, 0, 0, 1]
        W[:, 0] = [1, 0, 0, 0] # C00 += M1
        
        # M2 = (A10 + A11) B00
        U[:, 1] = [0, 0, 1, 1]
        V[:, 1] = [1, 0, 0, 0]
        W[:, 1] = [0, 0, 0, 1] # C11 -= M2
        
        # M3 = A00 (B10 - B11)
        U[:, 2] = [1, 0, 0, 0]
        V[:, 2] = [0, 0, 1, -1]
        W[:, 2] = [0, 1, 0, 0] # C01 += M3
        
        # M4 = A11 (B01 - B00)
        U[:, 3] = [0, 0, 0, 1]
        V[:, 3] = [0, 1, -1, 0]
        W[:, 3] = [1, 0, 0, 0] # C00 += M4
        
        # M5 = (A00 + A01) B11
        U[:, 4] = [1, 1, 0, 0]
        V[:, 4] = [0, 0, 0, 1]
        W[:, 4] = [0, 0, 0, 1] # C11 += M5 (Wait, C11 = M1 - M2 + M3 + M6? No)
        
        # Let's re-verify Strassen formulas carefully.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B10 - B11)
        # M4 = A11(B01 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # My W assignments above were just guesses. Let's do it properly.
        # W for M1: contributes to C00 (+1), C11 (+1)
        # W for M2: contributes to C10 (+1), C11 (-1)
        # W for M3: contributes to C01 (+1), C11 (+1)
        # W for M4: contributes to C00 (+1), C10 (+1)
        # W for M5: contributes to C01 (+1), C00 (-1)
        # W for M6: contributes to C11 (+1)
        # W for M7: contributes to C00 (+1)
        
        U[:, 0] = [1, 0, 0, 1] # A00+A11
        V[:, 0] = [1, 0, 0, 1] # B00+B11
        W[:, 0] = [1, 0, 0, 1] # C00+C11
        
        U[:, 1] = [0, 0, 1, 1] # A10+A11
        V[:, 1] = [1, 0, 0, 0] # B00
        W[:, 1] = [0, 0, 1, -1] # C10-C11
        
        U[:, 2] = [1, 0, 0, 0] # A00
        V[:, 2] = [0, 0, 1, -1] # B10-B11
        W[:, 2] = [0, 1, 0, 1] # C01+C11
        
        U[:, 3] = [0, 0, 0, 1] # A11
        V[:, 3] = [0, 1, -1, 0] # B01-B00
        W[:, 3] = [1, 0, 1, 0] # C00+C10
        
        U[:, 4] = [1, 1, 0, 0] # A00+A01
        V[:, 4] = [0, 0, 0, 1] # B11
        W[:, 4] = [-1, 1, 0, 0] # -C00+C01
        
        U[:, 5] = [0, 0, 1, -1] # A10-A00
        V[:, 5] = [1, 1, 0, 0] # B00+B01
        W[:, 5] = [0, 0, 0, 1] # C11
        
        U[:, 6] = [0, 1, 0, -1] # A01-A11
        V[:, 6] = [0, 0, 1, 1] # B10+B11
        W[:, 6] = [1, 0, 0, 0] # C00
        
        return U, V, W

    # Helper to create Schoolbook 2x1x1 decomposition (for the 3rd column of 2x2x3)
    # A is 2x2 (indices 0..3), B is 2x1 (indices 0..1), C is 2x1 (indices 0..1)
    # We only care about the 4th column of B and C in the 2x2x3 context, 
    # but here we just build the 4x4, 2x4, 2x4 blocks.
    def get_schoolbook_211():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        
        # C00 = A00*B00 + A01*B10
        # C10 = A10*B00 + A11*B10
        
        # Term 0: A00 B00 -> C00
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # Term 1: A01 B10 -> C00
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        # Term 2: A10 B00 -> C10
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        # Term 3: A11 B10 -> C10
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        
        return U, V, W

    # Combine Strassen 2x2 and Schoolbook 2x1 to make 2x2x3 (Rank 11)
    def get_223_decomp():
        Us, Vs, Ws = get_strassen_222()
        Ue, Ve, We = get_schoolbook_211()
        
        U_223 = np.hstack([Us, Ue]) # 4x11
        
        # V needs to be 6x11 (2 rows x 3 cols)
        V_223 = np.zeros((6, 11))
        # First 7 cols: Strassen on B[:, 0:2]
        V_223[0:2, 0:7] = Vs
        # Last 4 cols: Schoolbook on B[:, 2]
        V_223[0:2, 7:11] = Ve
        
        # W needs to be 6x11 (2 rows x 3 cols)
        W_223 = np.zeros((6, 11))
        # First 7 cols: Strassen on C[:, 0:2]
        W_223[0:2, 0:7] = Ws
        # Last 4 cols: Schoolbook on C[:, 2]
        W_223[0:2, 7:11] = We
        
        return U_223, V_223, W_223

    # Construct full 2x4x5
    # U: 8xR, V: 20xR, W: 10xR
    
    # Sub-blocks:
    # 1. A1(2x2) * B11(2x2) -> C11(2x2). Rank 7.
    # 2. A2(2x2) * B21(2x2) -> C11(2x2). Rank 7.
    # 3. A1(2x2) * B12(2x3) -> C12(2x3). Rank 11.
    # 4. A2(2x2) * B22(2x3) -> C12(2x3). Rank 11.
    
    total_rank = 7 + 7 + 11 + 11
    U_full = np.zeros((8, total_rank))
    V_full = np.zeros((20, total_rank))
    W_full = np.zeros((10, total_rank))
    
    U_s, V_s, W_s = get_strassen_222()
    U_223, V_223, W_223 = get_223_decomp()
    
    col_offset = 0
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    U_full[0:2, col_offset:col_offset+7] = U_s
    V_full[0:2, col_offset:col_offset+7] = V_s
    W_full[0:2, col_offset:col_offset+7] = W_s
    col_offset += 7
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    U_full[0:2, col_offset:col_offset+7] = U_s
    V_full[4:6, col_offset:col_offset+7] = V_s # B rows 2,3
    W_full[0:2, col_offset:col_offset+7] = W_s
    col_offset += 7
    
    # Block 3: A[0:2, 0:2] * B[0:2, 2:5] -> C[0:2, 2:5]
    U_full[0:2, col_offset:col_offset+11] = U_223
    V_full[0:2, col_offset:col_offset+11] = V_223
    W_full[4:6, col_offset:col_offset+11] = W_223 # C cols 2,3,4
    col_offset += 11
    
    # Block 4: A[0:2, 2:4] * B[2:4, 2:5] -> C[0:2, 2:5]
    U_full[0:2, col_offset:col_offset+11] = U_223
    V_full[4:6, col_offset:col_offset+11] = V_223
    W_full[4:6, col_offset:col_offset+11] = W_223
    col_offset += 11
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
# EVOLVE-BLOCK-END
