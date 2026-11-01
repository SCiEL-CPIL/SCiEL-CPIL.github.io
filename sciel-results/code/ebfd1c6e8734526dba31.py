import time
import numpy as np

def construct():
    n, m, p = 2, 5, 6
    
    # Total rank
    # 3 blocks of 2x2x2 (rank 7)
    # 3 blocks of 2x3x2 (rank 11)
    total_rank = 3 * 7 + 3 * 11
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    # Helper to add a sub-decomposition to global matrices
    def add_subblock(U_sub, V_sub, W_sub, u_offset, v_offset, w_offset, rank_sub):
        """
        Adds subblock matrices to global matrices.
        u_offset: (i_min, j_min) relative to subblock A indices
        v_offset: (j_min, k_min) relative to subblock B indices
        w_offset: (k_min, i_min) relative to subblock C indices
        """
        i_min, j_min = u_offset
        j_min_v, k_min = v_offset
        k_min_w, i_min_w = w_offset
        
        # Note: j_min from U and j_min_v from V must be consistent with the global j index.
        # In our setup, j is the shared index.
        
        for r in range(rank_sub):
            # U
            for i in range(U_sub.shape[0]):
                for j in range(U_sub.shape[1]):
                    val = U_sub[i, j, r]
                    if val != 0:
                        gi = i_min + i
                        gj = j_min + j
                        U[gi * m + gj, r] = val
            
            # V
            for j in range(V_sub.shape[0]):
                for k in range(V_sub.shape[1]):
                    val = V_sub[j, k, r]
                    if val != 0:
                        gj = j_min_v + j
                        gk = k_min + k
                        V[gj * p + gk, r] = val
            
            # W
            for k in range(W_sub.shape[0]):
                for i in range(W_sub.shape[1]):
                    val = W_sub[k, i, r]
                    if val != 0:
                        gk = k_min_w + k
                        gi = i_min_w + i
                        W[gk * n + gi, r] = val

    # Define Strassen 2x2x2
    def get_strassen_222():
        # Returns U, V, W of shape (2,2,7), (2,2,7), (2,2,7)
        U_s = np.zeros((2, 2, 7), dtype=int)
        V_s = np.zeros((2, 2, 7), dtype=int)
        W_s = np.zeros((2, 2, 7), dtype=int)
        
        # m1 = (A00+A11)(B00+B11) -> C00, C11
        U_s[0,0,0] = 1; U_s[1,1,0] = 1
        V_s[0,0,0] = 1; V_s[1,1,0] = 1
        W_s[0,0,0] = 1; W_s[1,1,0] = 1
        
        # m2 = (A10+A11)B00 -> -C00, C10
        U_s[1,0,1] = 1; U_s[1,1,1] = 1
        V_s[0,0,1] = 1
        W_s[0,0,1] = -1; W_s[0,1,1] = 1
        
        # m3 = A01(B10-B11) -> C01, -C11
        U_s[0,1,2] = 1
        V_s[1,0,2] = 1; V_s[1,1,2] = -1
        W_s[1,0,2] = 1; W_s[1,1,2] = -1
        
        # m4 = A00(B11-B00) -> C00, C11
        U_s[0,0,3] = 1
        V_s[1,1,3] = 1; V_s[0,0,3] = -1
        W_s[0,0,3] = 1; W_s[1,1,3] = 1
        
        # m5 = (A00+A10)B11 -> C01, C10
        U_s[0,0,4] = 1; U_s[1,0,4] = 1
        V_s[1,1,4] = 1
        W_s[1,0,4] = 1; W_s[0,1,4] = 1
        
        # m6 = (A11-A10)(B00+B01) -> C00, C01
        U_s[1,1,5] = 1; U_s[1,0,5] = -1
        V_s[0,0,5] = 1; V_s[0,1,5] = 1
        W_s[0,0,5] = 1; W_s[1,0,5] = 1
        
        # m7 = (A00-A01)(B10+B11) -> -C10, C11
        U_s[0,0,6] = 1; U_s[0,1,6] = -1
        V_s[1,0,6] = 1; V_s[1,1,6] = 1
        W_s[0,1,6] = -1; W_s[1,1,6] = 1
        
        return U_s, V_s, W_s

    # Define 2x3x2 decomposition (Strassen 2x2x2 + 2x1x2)
    def get_232():
        # A is 2x3, B is 3x2, C is 2x2
        # Split A = [A_L (2x2) | A_R (2x1)]
        # Split B = [B_T (2x2); B_B (1x2)]
        # C = A_L B_T + A_R B_B
        
        U_s = np.zeros((2, 3, 11), dtype=int)
        V_s = np.zeros((3, 2, 11), dtype=int)
        W_s = np.zeros((2, 2, 11), dtype=int)
        
        # Part 1: A_L B_T (2x2x2) using Strassen
        U_str, V_str, W_str = get_strassen_222()
        # Map A_L indices (0,1) to A indices (0,1) in cols 0,1
        U_s[0:2, 0:2, 0:7] = U_str
        # Map B_T indices (0,1) to B indices (0,1) in rows 0,1
        V_s[0:2, 0:2, 0:7] = V_str
        # Map C indices to C
        W_s[:, :, 0:7] = W_str
        
        # Part 2: A_R B_B (2x1x2)
        # A_R is col 2 of A. Indices i in 0,1. j=2.
        # B_B is row 2 of B. Indices j=2. k in 0,1.
        # C is 2x2.
        
        # Term 7: A_R[0] * B_B[0] -> C[0,0]
        U_s[0, 2, 7] = 1
        V_s[2, 0, 7] = 1
        W_s[0, 0, 7] = 1
        
        # Term 8: A_R[0] * B_B[1] -> C[0,1]
        U_s[0, 2, 8] = 1
        V_s[2, 1, 8] = 1
        W_s[1, 0, 8] = 1
        
        # Term 9: A_R[1] * B_B[0] -> C[1,0]
        U_s[1, 2, 9] = 1
        V_s[2, 0, 9] = 1
        W_s[0, 1, 9] = 1
        
        # Term 10: A_R[1] * B_B[1] -> C[1,1]
        U_s[1, 2, 10] = 1
        V_s[2, 1, 10] = 1
        W_s[1, 1, 10] = 1
        
        return U_s, V_s, W_s

    # Assemble
    current_rank = 0
    
    # Block 1: A1 (2x2) * B1a (2x2) -> C1a (2x2)
    # A cols 0,1. B rows 0,1, cols 0,1. C cols 0,1.
    U1, V1, W1 = get_strassen_222()
    add_subblock(U1, V1, W1, (0,0), (0,0), (0,0), 7)
    current_rank += 7
    
    # Block 2: A1 (2x2) * B1b (2x2) -> C1b (2x2)
    # A cols 0,1. B rows 0,1, cols 2,3. C cols 2,3.
    U2, V2, W2 = get_strassen_222()
    add_subblock(U2, V2, W2, (0,0), (0,2), (2,0), 7)
    current_rank += 7
    
    # Block 3: A1 (2x2) * B1c (2x2) -> C1c (2x2)
    # A cols 0,1. B rows 0,1, cols 4,5. C cols 4,5.
    U3, V3, W3 = get_strassen_222()
    add_subblock(U3, V3, W3, (0,0), (0,4), (4,0), 7)
    current_rank += 7
    
    # Block 4: A2 (2x3) * B2a (3x2) -> C2a (2x2)
    # A cols 2,3,4. B rows 2,3,4, cols 0,1. C cols 0,1.
    U4, V4, W4 = get_232()
    add_subblock(U4, V4, W4, (0,2), (2,0), (0,0), 11)
    current_rank += 11
    
    # Block 5: A2 (2x3) * B2b (3x2) -> C2b (2x2)
    # A cols 2,3,4. B rows 2,3,4, cols 2,3. C cols 2,3.
    U5, V5, W5 = get_232()
    add_subblock(U5, V5, W5, (0,2), (2,2), (2,0), 11)
    current_rank += 11
    
    # Block 6: A2 (2x3) * B2c (3x2) -> C2c (2x2)
    # A cols 2,3,4. B rows 2,3,4, cols 4,5. C cols 4,5.
    U6, V6, W6 = get_232()
    add_subblock(U6, V6, W6, (0,2), (2,4), (4,0), 11)
    current_rank += 11
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
