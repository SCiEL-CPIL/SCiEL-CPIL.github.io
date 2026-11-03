"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Hybrid Block Decomposition
    # Partition A (4x5) into blocks I={0,1}, J={0,1}, J={2,3}, J={4}
    # Partition B (5x5) into blocks J={0,1}, J={2,3}, J={4} and K={0,1}, K={2,3}, K={4}
    # Output C (4x5) blocks I={0,1}, K={0,1}, K={2,3}, K={4}
    
    # Sub-problems:
    # 1. Strassen 2x2x2 (Rank 7)
    # 2. Schoolbook 2x2x1 (Rank 4)
    # 3. Schoolbook 2x1x2 (Rank 4)
    # 4. Schoolbook 2x1x1 (Rank 2)
    
    # Define factors for sub-problems
    # Indices for 2x2 matrices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for 2x1 vectors: 0:(0,0), 1:(1,0)
    # Indices for 1x2 vectors: 0:(0,0), 1:(0,1)
    # Indices for 1x1 scalars: 0:(0,0)
    
    # --- Strassen 2x2x2 ---
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B11
    # M3 = A00(B10-B11)
    # M4 = A11(B00-B01)
    # M5 = (A00+A01)B10
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B01+B11)
    
    # U_strassen[4, 7], V_strassen[4, 7], W_strassen[4, 7]
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # Term 1: (A00+A11)(B00+B11)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[1, 0] = 1; W_s[2, 0] = 1; W_s[3, 0] = 1 # C00, C01, C10, C11 all get +1? 
    # Wait, Strassen C formulas:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M1 + M3 - M2 + M6
    # C10 = M2 + M5 - M1 + M3
    # C11 = M1 + M2 - M4 + M6
    # So W_s[0, 0] = 1, W_s[1, 0] = 1, W_s[2, 0] = -1, W_s[3, 0] = 1
    
    W_s[0, 0] = 1; W_s[1, 0] = 1; W_s[2, 0] = -1; W_s[3, 0] = 1
    
    # Term 2: (A10+A11)B11
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[3, 1] = 1
    W_s[0, 1] = 0; W_s[1, 1] = -1; W_s[2, 1] = 1; W_s[3, 1] = 1
    
    # Term 3: A00(B10-B11)
    U_s[0, 2] = 1
    V_s[2, 2] = 1; V_s[3, 2] = -1
    W_s[0, 2] = 0; W_s[1, 2] = 1; W_s[2, 2] = 1; W_s[3, 2] = 0
    
    # Term 4: A11(B00-B01)
    U_s[3, 3] = 1
    V_s[0, 3] = 1; V_s[1, 3] = -1
    W_s[0, 3] = 1; W_s[1, 3] = 0; W_s[2, 3] = 0; W_s[3, 3] = -1
    
    # Term 5: (A00+A01)B10
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[2, 4] = 1
    W_s[0, 4] = -1; W_s[1, 4] = 0; W_s[2, 4] = 1; W_s[3, 4] = 0
    
    # Term 6: (A10-A00)(B00+B01)
    U_s[0, 5] = -1; U_s[2, 5] = 1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    W_s[0, 5] = 0; W_s[1, 5] = 1; W_s[2, 5] = 0; W_s[3, 5] = 1
    
    # Term 7: (A01-A11)(B01+B11)
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[1, 6] = 1; V_s[3, 6] = 1
    W_s[0, 6] = 1; W_s[1, 6] = 0; W_s[2, 6] = 0; W_s[3, 6] = 0 # Wait, C11 has M7?
    # C11 = M1 + M2 - M4 + M6. No M7.
    # C00 = M1 + M4 - M5 + M7.
    # C01 = M1 + M3 - M2 + M6.
    # C10 = M2 + M5 - M1 + M3.
    # So W_s[3, 6] should be 0.
    W_s[0, 6] = 1; W_s[1, 6] = 0; W_s[2, 6] = 0; W_s[3, 6] = 0
    
    # --- Schoolbook 2x2x1 (Rank 4) ---
    # A (2x2), B (2x1) -> C (2x1)
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    U_221 = np.zeros((4, 4))
    V_221 = np.zeros((2, 4))
    W_221 = np.zeros((2, 4))
    
    # T1: A00 B0 -> C0
    U_221[0, 0] = 1; V_221[0, 0] = 1; W_221[0, 0] = 1
    # T2: A01 B1 -> C0
    U_221[1, 1] = 1; V_221[1, 1] = 1; W_221[0, 1] = 1
    # T3: A10 B0 -> C1
    U_221[2, 2] = 1; V_221[0, 2] = 1; W_221[1, 2] = 1
    # T4: A11 B1 -> C1
    U_221[3, 3] = 1; V_221[1, 3] = 1; W_221[1, 3] = 1
    
    # --- Schoolbook 2x1x2 (Rank 4) ---
    # A (2x1), B (1x2) -> C (2x2)
    # C00 = A0 B0
    # C01 = A0 B1
    # C10 = A1 B0
    # C11 = A1 B1
    U_212 = np.zeros((2, 4))
    V_212 = np.zeros((2, 4))
    W_212 = np.zeros((4, 4))
    
    # T1: A0 B0 -> C00
    U_212[0, 0] = 1; V_212[0, 0] = 1; W_212[0, 0] = 1
    # T2: A0 B1 -> C01
    U_212[0, 1] = 1; V_212[1, 1] = 1; W_212[1, 1] = 1
    # T3: A1 B0 -> C10
    U_212[1, 2] = 1; V_212[0, 2] = 1; W_212[2, 2] = 1
    # T4: A1 B1 -> C11
    U_212[1, 3] = 1; V_212[1, 3] = 1; W_212[3, 3] = 1
    
    # --- Schoolbook 2x1x1 (Rank 2) ---
    # A (2x1), B (1x1) -> C (2x1)
    # C0 = A0 B0
    # C1 = A1 B0
    U_211 = np.zeros((2, 2))
    V_211 = np.zeros((1, 2))
    W_211 = np.zeros((2, 2))
    
    # T1: A0 B0 -> C0
    U_211[0, 0] = 1; V_211[0, 0] = 1; W_211[0, 0] = 1
    # T2: A1 B0 -> C1
    U_211[1, 1] = 1; V_211[0, 1] = 1; W_211[1, 1] = 1
    
    # Assembly
    # Global Dimensions: n=4, m=5, p=5
    # U shape: (4*5, rank) = (20, rank)
    # V shape: (5*5, rank) = (25, rank)
    # W shape: (5*4, rank) = (20, rank)
    
    # Index Mapping
    # A[i, j]: i in 0..3, j in 0..4
    # Global U index: i*5 + j
    # B[j, k]: j in 0..4, k in 0..4
    # Global V index: j*5 + k
    # C[i, k]: i in 0..3, k in 0..4
    # Global W index: k*4 + i (Trace convention W[k*n+i])
    
    total_rank = 92
    U = np.zeros((20, total_rank))
    V = np.zeros((25, total_rank))
    W = np.zeros((20, total_rank))
    
    col = 0
    
    # Block Indices
    # I: 0 -> {0,1}, 1 -> {2,3}
    # J: 0 -> {0,1}, 1 -> {2,3}, 2 -> {4}
    # K: 0 -> {0,1}, 1 -> {2,3}, 2 -> {4}
    
    I_map = {0: [0, 1], 1: [2, 3]}
    J_map = {0: [0, 1], 1: [2, 3], 2: [4]}
    K_map = {0: [0, 1], 1: [2, 3], 2: [4]}
    
    # Helper to add a sub-problem
    def add_subproblem(U_sub, V_sub, W_sub, i_offset, j_offset, k_offset, rank_sub):
        nonlocal col
        for r in range(rank_sub):
            for i_local in range(U_sub.shape[0]):
                i_global = I_map[i_offset][i_local // 2] + (i_local % 2) # This mapping is wrong for 2x1 vs 2x2
                # Need specific index mapping for each sub-problem type
            pass # Will implement inline instead for clarity
    
    # Explicit Loop for Clarity and Correctness
    
    for alpha in [0, 1]:
        i_rows = I_map[alpha]
        for beta in [0, 1, 2]:
            k_cols = K_map[beta]
            
            # Case 1: beta < 2 (Output 2x2)
            if beta < 2:
                # gamma = 0 (J 0,1)
                j_cols = J_map[0]
                # Strassen 2x2x2
                for r in range(7):
                    # U
                    for i_l in range(4):
                        i_g = i_rows[i_l // 2] + (i_l % 2)
                        j_g = j_cols[(i_l // 2) * 2 + (i_l % 2)] # Wait, A is 2x2.
                        # A indices: (0,0), (0,1), (1,0), (1,1)
                        # i_l=0 -> (0,0), i_l=1 -> (0,1), i_l=2 -> (1,0), i_l=3 -> (1,1)
                        # i_g = i_rows[0] if i_l<2 else i_rows[1]
                        # j_g = j_cols[0] if i_l%2==0 else j_cols[1]
                        if U_s[i_l, r] != 0:
                            U[i_g * 5 + j_g, col] = U_s[i_l, r]
                    # V
                    for j_l in range(4):
                        j_g = j_cols[j_l // 2] + (j_l % 2)
                        k_g = k_cols[(j_l // 2) * 2 + (j_l % 2)] # Wait, B is 2x2.
                        # B indices: (0,0), (0,1), (1,0), (1,1)
                        # j_l=0 -> (0,0), j_l=1 -> (0,1), j_l=2 -> (1,0), j_l=3 -> (1,1)
                        # j_g = j_cols[0] if j_l<2 else j_cols[1]
                        # k_g = k_cols[0] if j_l%2==0 else k_cols[1]
                        if V_s[j_l, r] != 0:
                            V[j_g * 5 + k_g, col] = V_s[j_l, r]
                    # W
                    for k_l in range(4):
                        k_g = k_cols[k_l // 2] + (k_l % 2)
                        i_g = i_rows[(k_l // 2) * 2 + (k_l % 2)] # Wait, C is 2x2.
                        # C indices: (0,0), (0,1), (1,0), (1,1)
                        # k_l=0 -> (0,0), k_l=1 -> (0,1), k_l=2 -> (1,0), k_l=3 -> (1,1)
                        # k_g = k_cols[0] if k_l<2 else k_cols[1]
                        # i_g = i_rows[0] if k_l%2==0 else i_rows[1]
                        if W_s[k_l, r] != 0:
                            W[k_g * 4 + i_g, col] = W_s[k_l, r]
                    col += 1
                
                # gamma = 1 (J 2,3)
                j_cols = J_map[1]
                # Strassen 2x2x2
                for r in range(7):
                    for i_l in range(4):
                        i_g = i_rows[i_l // 2] + (i_l % 2)
                        j_g = j_cols[(i_l // 2) * 2 + (i_l % 2)]
                        if U_s[i_l, r] != 0:
                            U[i_g * 5 + j_g, col] = U_s[i_l, r]
                    for j_l in range(4):
                        j_g = j_cols[j_l // 2] + (j_l % 2)
                        k_g = k_cols[(j_l // 2) * 2 + (j_l % 2)]
                        if V_s[j_l, r] != 0:
                            V[j_g * 5 + k_g, col] = V_s[j_l, r]
                    for k_l in range(4):
                        k_g = k_cols[k_l // 2] + (k_l % 2)
                        i_g = i_rows[(k_l // 2) * 2 + (k_l % 2)]
                        if W_s[k_l, r] != 0:
                            W[k_g * 4 + i_g, col] = W_s[k_l, r]
                    col += 1
                
                # gamma = 2 (J 4)
                j_cols = J_map[2] # [4]
                # Outer Product 2x1 x 1x2 -> 2x2
                # A is 2x1 (indices 0,1), B is 1x2 (indices 0,1)
                # U_212 shape (2,4), V_212 shape (2,4), W_212 shape (4,4)
                for r in range(4):
                    for i_l in range(2):
                        i_g = i_rows[i_l]
                        j_g = j_cols[0]
                        if U_212[i_l, r] != 0:
                            U[i_g * 5 + j_g, col] = U_212[i_l, r]
                    for j_l in range(2):
                        j_g = j_cols[0]
                        k_g = k_cols[j_l]
                        if V_212[j_l, r] != 0:
                            V[j_g * 5 + k_g, col] = V_212[j_l, r]
                    for k_l in range(4):
                        k_g = k_cols[k_l // 2] + (k_l % 2)
                        i_g = i_rows[(k_l // 2) * 2 + (k_l % 2)]
                        if W_212[k_l, r] != 0:
                            W[k_g * 4 + i_g, col] = W_212[k_l, r]
                    col += 1
            else:
                # Case 2: beta == 2 (Output 2x1)
                # gamma = 0 (J 0,1)
                j_cols = J_map[0]
                # 2x2 x 2x1 -> 2x1
                # U_221 shape (4,4), V_221 shape (2,4), W_221 shape (2,4)
                for r in range(4):
                    for i_l in range(4):
                        i_g = i_rows[i_l // 2] + (i_l % 2)
                        j_g = j_cols[(i_l // 2) * 2 + (i_l % 2)]
                        if U_221[i_l, r] != 0:
                            U[i_g * 5 + j_g, col] = U_221[i_l, r]
                    for j_l in range(2):
                        j_g = j_cols[j_l]
                        k_g = k_cols[0] # k is 2x1, so k index is 0,1? No, k_cols is [4].
                        # Wait, B is 2x1. j_l is row index of B.
                        # B indices: (0,0), (1,0).
                        # j_g = j_cols[j_l]
                        # k_g = k_cols[0]
                        if V_221[j_l, r] != 0:
                            V[j_g * 5 + k_g, col] = V_221[j_l, r]
                    for k_l in range(2):
                        k_g = k_cols[0]
                        i_g = i_rows[k_l]
                        if W_221[k_l, r] != 0:
                            W[k_g * 4 + i_g, col] = W_221[k_l, r]
                    col += 1
                
                # gamma = 1 (J 2,3)
                j_cols = J_map[1]
                for r in range(4):
                    for i_l in range(4):
                        i_g = i_rows[i_l // 2] + (i_l % 2)
                        j_g = j_cols[(i_l // 2) * 2 + (i_l % 2)]
                        if U_221[i_l, r] != 0:
                            U[i_g * 5 + j_g, col] = U_221[i_l, r]
                    for j_l in range(2):
                        j_g = j_cols[j_l]
                        k_g = k_cols[0]
                        if V_221[j_l, r] != 0:
                            V[j_g * 5 + k_g, col] = V_221[j_l, r]
                    for k_l in range(2):
                        k_g = k_cols[0]
                        i_g = i_rows[k_l]
                        if W_221[k_l, r] != 0:
                            W[k_g * 4 + i_g, col] = W_221[k_l, r]
                    col += 1
                
                # gamma = 2 (J 4)
                j_cols = J_map[2]
                # 2x1 x 1x1 -> 2x1
                # U_211 shape (2,2), V_211 shape (1,2), W_211 shape (2,2)
                for r in range(2):
                    for i_l in range(2):
                        i_g = i_rows[i_l]
                        j_g = j_cols[0]
                        if U_211[i_l, r] != 0:
                            U[i_g * 5 + j_g, col] = U_211[i_l, r]
                    for j_l in range(1):
                        j_g = j_cols[0]
                        k_g = k_cols[0]
                        if V_211[j_l, r] != 0:
                            V[j_g * 5 + k_g, col] = V_211[j_l, r]
                    for k_l in range(2):
                        k_g = k_cols[0]
                        i_g = i_rows[k_l]
                        if W_211[k_l, r] != 0:
                            W[k_g * 4 + i_g, col] = W_211[k_l, r]
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
