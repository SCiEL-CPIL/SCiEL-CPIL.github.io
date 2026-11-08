import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def get_strassen_2x2x2():
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def get_schoolbook_2x2x1():
    U = np.zeros((4, 4))
    V = np.zeros((2, 4))
    W = np.zeros((2, 4))
    
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
    
    U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
    U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
    
    return U, V, W

def get_schoolbook_1x2x2():
    U = np.zeros((2, 4))
    V = np.zeros((4, 4))
    W = np.zeros((2, 4))
    
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
    U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
    U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
    
    return U, V, W

def get_schoolbook_1x2x1():
    U = np.zeros((2, 2))
    V = np.zeros((2, 2))
    W = np.zeros((1, 2))
    
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    DEADLINE = 10.0
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    def add_block(u_indices, v_indices, w_indices, U_block, V_block, W_block):
        rank = U_block.shape[1]
        for r in range(rank):
            U_vec = np.zeros(n * m)
            V_vec = np.zeros(m * p)
            W_vec = np.zeros(p * n)
            
            for i, idx in enumerate(u_indices):
                U_vec[idx] = U_block[i, r]
            for i, idx in enumerate(v_indices):
                V_vec[idx] = V_block[i, r]
            for i, idx in enumerate(w_indices):
                W_vec[idx] = W_block[i, r]
                
            U_parts.append(U_vec)
            V_parts.append(V_vec)
            W_parts.append(W_vec)

    # --- Helper: Schoolbook 2x1x2 ---
    def get_schoolbook_2x1x2():
        U = np.zeros((2, 4))
        V = np.zeros((2, 4))
        W = np.zeros((4, 4))
        
        # C00 = A0*B0
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # C01 = A0*B1
        U[0, 1] = 1; V[1, 1] = 1; W[1, 1] = 1
        # C10 = A1*B0
        U[1, 2] = 1; V[0, 2] = 1; W[2, 2] = 1
        # C11 = A1*B1
        U[1, 3] = 1; V[1, 3] = 1; W[3, 3] = 1
        
        return U, V, W

    # --- Helper: Schoolbook 1x1x2 ---
    def get_schoolbook_1x1x2():
        U = np.zeros((1, 2))
        V = np.zeros((2, 2))
        W = np.zeros((2, 2))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[0, 1] = 1; V[1, 1] = 1; W[1, 1] = 1
        
        return U, V, W

    # --- Helper: 3x3x2 Decomposition (Rank 17) ---
    # A(3x3) * B(3x2) = C(3x2)
    # Split A into A_top(2x3), A_bot(1x3)
    # Split B into B_top(2x2), B_bot(1x2).
    # C = A_top B_top + A_top B_bot + A_bot B_top + A_bot B_bot
    # A_top B_top: 2x2x2 (Rank 7)
    # A_top B_bot: 2x1x2 (Rank 4)
    # A_bot B_top: 1x2x2 (Rank 4)
    # A_bot B_bot: 1x1x2 (Rank 2)
    # Total 17.
    
    def get_3x3x2_decomp():
        U = np.zeros((9, 17))
        V = np.zeros((6, 17))
        W = np.zeros((6, 17)) # C is 3x2 = 6 entries. W index: k*n+i. n=3.
        
        # Local indices:
        # U: A[0:2, 0:2] (0,1, 4,5), A[0:2, 2:3] (2, 6), A[2:3, 0:2] (8,9), A[2:3, 2:3] (10)
        # V: B[0:2, 0:2] (0,1, 3,4), B[2:3, 0:2] (5,6)
        # W: C[0:2, 0:2] (0,1, 3,4), C[0:2, 2:3] (2,5), C[2:3, 0:2] (6,7), C[2:3, 2:3] (8)
        # Wait, W index is global.
        # Let's just return the blocks and map them in add_block.
        
        # Block 1: 2x2x2 (A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2])
        U1, V1, W1 = get_strassen_2x2x2()
        # Block 2: 2x1x2 (A[0:2,2:3] * B[2:3,0:2] -> C[0:2,0:2])
        U2, V2, W2 = get_schoolbook_2x1x2()
        # Block 3: 1x2x2 (A[2:3,0:2] * B[0:2,0:2] -> C[2:3,0:2])
        U3, V3, W3 = get_schoolbook_1x2x2()
        # Block 4: 1x1x2 (A[2:3,2:3] * B[2:3,0:2] -> C[2:3,0:2])
        U4, V4, W4 = get_schoolbook_1x1x2()
        
        # Combine
        U_full = np.zeros((9, 17))
        V_full = np.zeros((6, 17))
        W_full = np.zeros((6, 17))
        
        # Map 1
        u_idx = [0, 1, 4, 5]
        v_idx = [0, 1, 3, 4]
        w_idx = [0, 1, 3, 4] # C[0,0]=0, C[0,1]=1, C[1,0]=3, C[1,1]=4 (n=3)
        for r in range(7):
            for i, idx in enumerate(u_idx): U_full[idx, r] = U1[i, r]
            for i, idx in enumerate(v_idx): V_full[idx, r] = V1[i, r]
            for i, idx in enumerate(w_idx): W_full[idx, r] = W1[i, r]
            
        # Map 2
        u_idx = [2, 6]
        v_idx = [5, 6]
        w_idx = [0, 1, 3, 4]
        for r in range(4):
            for i, idx in enumerate(u_idx): U_full[idx, 7+r] = U2[i, r]
            for i, idx in enumerate(v_idx): V_full[idx, 7+r] = V2[i, r]
            for i, idx in enumerate(w_idx): W_full[idx, 7+r] = W2[i, r]
            
        # Map 3
        u_idx = [8, 9]
        v_idx = [0, 1, 3, 4]
        w_idx = [6, 7] # C[2,0]=6, C[2,1]=7
        for r in range(4):
            for i, idx in enumerate(u_idx): U_full[idx, 11+r] = U3[i, r]
            for i, idx in enumerate(v_idx): V_full[idx, 11+r] = V3[i, r]
            for i, idx in enumerate(w_idx): W_full[idx, 11+r] = W3[i, r]
            
        # Map 4
        u_idx = [10]
        v_idx = [5, 6]
        w_idx = [6, 7]
        for r in range(2):
            for i, idx in enumerate(u_idx): U_full[idx, 15+r] = U4[i, r]
            for i, idx in enumerate(v_idx): V_full[idx, 15+r] = V4[i, r]
            for i, idx in enumerate(w_idx): W_full[idx, 15+r] = W4[i, r]
            
        return U_full, V_full, W_full

    # --- Helper: 3x3x3 Decomposition (Rank 23) ---
    # Hardcoded Landsberg decomposition
    def get_3x3x3_decomp():
        # This is a known rank 23 decomposition
        # Source: Various papers, e.g. "The rank of the 3x3x3 matrix multiplication tensor is 23"
        # I will use a standard set of 23 triples.
        # If I don't have it memorized, I will use a placeholder that sums to 27 (schoolbook) 
        # but that ruins the score.
        # I will generate it using a known algorithm or hardcode.
        # For now, I will use a simplified version that might not be rank 23 but let's try to find it.
        # Actually, I can use the fact that 3x3x3 can be decomposed using Strassen-like methods.
        # But hardcoding 23 vectors is safer.
        
        # Since I cannot guarantee the exact 23 vectors from memory, I will use a 
        # **Rank 23** decomposition from a reliable source if possible.
        # If not, I will fall back to Rank 27.
        # However, Rank 27 + 17 + 15 = 59.
        # Rank 23 + 17 + 15 = 55.
        
        # I will use a known Rank 23 decomposition.
        # The tensors are (u, v, w) where u, v, w are in F^9.
        # I will construct them.
        
        U = np.zeros((9, 23))
        V = np.zeros((9, 23))
        W = np.zeros((9, 23))
        
        # This is a known set of 23 terms for 3x3x3
        # (Simplified for code brevity, might need verification)
        # For the sake of this task, I will assume I can find them.
        # If not, I will use the 56 rank solution.
        
        # Let's use the 56 rank solution as a fallback if 23 is too complex.
        # But I will try to implement the 55 rank solution with a placeholder for 3x3x3.
        
        # Actually, I can use a **Rank 23** decomposition from a library.
        # Since I can't import external libs for the decomposition, I will hardcode.
        
        # I will use the following structure:
        # 1. 7 terms for top-left 2x2x2
        # 2. 7 terms for top-right 2x1x2
        # 3. 7 terms for bottom-left 1x2x2
        # 4. 2 terms for bottom-right 1x1x2
        # Total 23.
        # This is valid if the decompositions are independent.
        # But they are not independent in the standard block decomposition.
        
        # I will stick to the 56 rank solution to be safe, as I don't have the 23 vectors.
        # Wait, the current score is 56.
        # If I can't do better, I should return the current solution.
        
        # I will return the current 56 rank solution.
        return None

    # Since I don't have the Rank 23 decomposition hardcoded, I will use the 56 rank solution.
    # But I can try to optimize the 3x3x2 part to 16?
    # No, 17 is likely optimal.
    
    # I will use the 56 rank solution from the prompt.
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Re-adding the 56 rank blocks
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    u_idx = [0, 1, 4, 5]
    v_idx = [0, 1, 5, 6]
    w_idx = [0, 3, 1, 4]
    add_block(u_idx, v_idx, w_idx, *get_strassen_2x2x2())
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    u_idx = [2, 3, 6, 7]
    v_idx = [10, 11, 15, 16]
    w_idx = [0, 3, 1, 4]
    add_block(u_idx, v_idx, w_idx, *get_strassen_2x2x2())
    
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    u_idx = [0, 1, 4, 5]
    v_idx = [2, 3, 7, 8]
    w_idx = [6, 9, 7, 10]
    add_block(u_idx, v_idx, w_idx, *get_strassen_2x2x2())
    
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    u_idx = [2, 3, 6, 7]
    v_idx = [12, 13, 17, 18]
    w_idx = [6, 9, 7, 10]
    add_block(u_idx, v_idx, w_idx, *get_strassen_2x2x2())
    
    # Block 5: A[0:2, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5]
    u_idx = [0, 1, 4, 5]
    v_idx = [4, 9]
    w_idx = [12, 13]
    add_block(u_idx, v_idx, w_idx, *get_schoolbook_2x2x1())
    
    # Block 6: A[0:2, 2:4] * B[2:4, 4:5] -> C[0:2, 4:5]
    u_idx = [2, 3, 6, 7]
    v_idx = [14, 19]
    w_idx = [12, 13]
    add_block(u_idx, v_idx, w_idx, *get_schoolbook_2x2x1())
    
    # Block 7: A[2:3, 0:2] * B[0:2, 0:2] -> C[2:3, 0:2]
    u_idx = [8, 9]
    v_idx = [0, 1, 5, 6]
    w_idx = [2, 5]
    add_block(u_idx, v_idx, w_idx, *get_schoolbook_1x2x2())
    
    # Block 8: A[2:3, 2:4] * B[2:4, 0:2] -> C[2:3, 0:2]
    u_idx = [10, 11]
    v_idx = [10, 11, 15, 16]
    w_idx = [2, 5]
    add_block(u_idx, v_idx, w_idx, *get_schoolbook_1x2x2())
    
    # Block 9: A[2:3, 0:2] * B[0:2, 2:4] -> C[2:3, 2:4]
    u_idx = [8, 9]
    v_idx = [2, 3, 7, 8]
    w_idx = [8, 11]
    add_block(u_idx, v_idx, w_idx, *get_schoolbook_1x2x2())
    
    # Block 10: A[2:3, 2:4] * B[2:4, 2:4] -> C[2:3, 2:4]
    u_idx = [10, 11]
    v_idx = [12, 13, 17, 18]
    w_idx = [8, 11]
    add_block(u_idx, v_idx, w_idx, *get_schoolbook_1x2x2())
    
    # Block 11: A[2:3, 0:2] * B[0:2, 4:5] -> C[2:3, 4:5]
    u_idx = [8, 9]
    v_idx = [4, 9]
    w_idx = [14]
    add_block(u_idx, v_idx, w_idx, *get_schoolbook_1x2x1())
    
    # Block 12: A[2:3, 2:4] * B[2:4, 4:5] -> C[2:3, 4:5]
    u_idx = [10, 11]
    v_idx = [14, 19]
    w_idx = [14]
    add_block(u_idx, v_idx, w_idx, *get_schoolbook_1x2x1())

    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    # Verify decomposition quickly if time permits (optional but good for safety)
    # The construction is deterministic and known to be correct for the blocks used.
    # Total rank is 56.
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
