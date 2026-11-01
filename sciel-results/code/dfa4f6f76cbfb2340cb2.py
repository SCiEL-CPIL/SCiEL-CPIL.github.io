import numpy as np
import time
from scipy.optimize import linprog

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 125  # 120s limit, keep 5s safety margin for return and overhead
    
    n, m, p = DIMENSIONS
    
    # Fallback: Schoolbook (Rank 32)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    # Strategy: Block Strassen for Rank 28
    # We view A as 2x4 (two 2x2 blocks side by side) and B as 4x4 (four 2x2 blocks).
    # C = A * B is 2x4 (two 2x2 blocks side by side).
    # Let A = [A11 A12] and B = [[B11 B12], [B21 B22]] where blocks are 2x2.
    # C = [C1 C2] where:
    # C1 = A11*B11 + A12*B21
    # C2 = A11*B12 + A12*B22
    
    # Each 2x2 multiplication can be done with rank 7 (Strassen).
    # Total rank = 2 (for C1, C2 blocks? No, C1 and C2 are 2x2 matrices).
    # Actually, C is 2x4. It has two 2x2 blocks.
    # C1 = A11 B11 + A12 B21. This is the sum of two 2x2 matrix products.
    # The tensor for sum of two products is the direct sum of the two tensors.
    # So rank(C1) = rank(2x2) + rank(2x2) = 7 + 7 = 14.
    # Similarly rank(C2) = 7 + 7 = 14.
    # Total rank = 14 + 14 = 28.
    
    # We will construct the 28-rank decomposition by combining 4 independent 2x2 Strassen decompositions.
    
    # Standard Strassen for 2x2 matrices:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M2 - M6
    
    # We need to map these to the global indices.
    
    def add_strassen_block(U, V, W, offset, a_map, b_map, c_map):
        """
        Adds a 2x2 Strassen decomposition to the global matrices.
        offset: starting column index in U, V, W.
        a_map: dict mapping local 2x2 index (0..3) to global A index (0..7)
        b_map: dict mapping local 2x2 index (0..3) to global B index (0..15)
        c_map: dict mapping local 2x2 index (0..3) to global C index (0..7)
        """
        # Local indices: 0=A11, 1=A12, 2=A21, 3=A22
        # B indices: 0=B11, 1=B12, 2=B21, 3=B22
        # C indices: 0=C11, 1=C12, 2=C21, 3=C22
        
        # M1 = (A11+A22)(B11+B22)
        U[:, offset + 0] += [1, 0, 0, 1] # A11, A22
        V[:, offset + 0] += [1, 0, 0, 0] # B11? No, B11+B22 -> B[0]=1, B[3]=1? 
        # Wait, b_map maps local B index to global.
        # Local B indices: 0:B11, 1:B12, 2:B21, 3:B22.
        # M1 uses B11 and B22.
        V[b_map[0], offset + 0] += 1
        V[b_map[3], offset + 0] += 1
        # W for C11, C12, C21, C22
        # C11 = M1 ...
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M2 - M6
        
        # Let's be systematic.
        # We will fill U, V, W for the 7 terms.
        
        # Term 0: M1
        U[a_map[0], offset+0] += 1
        U[a_map[3], offset+0] += 1
        V[b_map[0], offset+0] += 1
        V[b_map[3], offset+0] += 1
        W[c_map[0], offset+0] += 1  # C11
        W[c_map[3], offset+0] += 1  # C22
        
        # Term 1: M2 = (A21+A22)B11
        U[a_map[2], offset+1] += 1
        U[a_map[3], offset+1] += 1
        V[b_map[0], offset+1] += 1
        W[c_map[2], offset+1] += 1  # C21
        W[c_map[3], offset+1] += 1  # C22
        
        # Term 2: M3 = A11(B12-B22)
        U[a_map[0], offset+2] += 1
        V[b_map[1], offset+2] += 1
        V[b_map[3], offset+2] -= 1
        W[c_map[1], offset+2] += 1  # C12
        W[c_map[3], offset+2] -= 1  # C22
        
        # Term 3: M4 = A22(B21-B11)
        U[a_map[3], offset+3] += 1
        V[b_map[2], offset+3] += 1
        V[b_map[0], offset+3] -= 1
        W[c_map[0], offset+3] -= 1  # C11
        W[c_map[2], offset+3] += 1  # C21
        
        # Term 4: M5 = (A11+A12)B22
        U[a_map[0], offset+4] += 1
        U[a_map[1], offset+4] += 1
        V[b_map[3], offset+4] += 1
        W[c_map[0], offset+4] += 1  # C11
        W[c_map[1], offset+4] += 1  # C12
        
        # Term 5: M6 = (A21-A11)(B11+B12)
        U[a_map[2], offset+5] += 1
        U[a_map[0], offset+5] -= 1
        V[b_map[0], offset+5] += 1
        V[b_map[1], offset+5] += 1
        W[c_map[3], offset+5] -= 1  # C22
        
        # Term 6: M7 = (A12-A22)(B21+B22)
        U[a_map[1], offset+6] += 1
        U[a_map[3], offset+6] -= 1
        V[b_map[2], offset+6] += 1
        V[b_map[3], offset+6] += 1
        W[c_map[0], offset+6] += 1  # C11
        # Note: C21 does not use M7. C12 does not use M7. C22 does not use M7.
        # Check standard Strassen:
        # C11 = M1 - M4 + M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M2 - M6
        # So M7 only appears in C11. Correct.

    # Global Indices:
    # A (2x4):
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=0, j=2 -> 2
    # i=0, j=3 -> 3
    # i=1, j=0 -> 4
    # i=1, j=1 -> 5
    # i=1, j=2 -> 6
    # i=1, j=3 -> 7
    
    # B (4x4):
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=0, j=2 -> 2
    # i=0, j=3 -> 3
    # i=1, j=0 -> 4
    # i=1, j=1 -> 5
    # i=1, j=2 -> 6
    # i=1, j=3 -> 7
    # i=2, j=0 -> 8
    # i=2, j=1 -> 9
    # i=2, j=2 -> 10
    # i=2, j=3 -> 11
    # i=3, j=0 -> 12
    # i=3, j=1 -> 13
    # i=3, j=2 -> 14
    # i=3, j=3 -> 15
    
    # C (2x4):
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=0, j=2 -> 2
    # i=0, j=3 -> 3
    # i=1, j=0 -> 4
    # i=1, j=1 -> 5
    # i=1, j=2 -> 6
    # i=1, j=3 -> 7

    # Blocks:
    # A11: A[0:2, 0:2] -> indices 0, 1, 4, 5. (Row 0: 0,1; Row 1: 4,5)
    # A12: A[0:2, 2:4] -> indices 2, 3, 6, 7. (Row 0: 2,3; Row 1: 6,7)
    # B11: B[0:2, 0:2] -> indices 0, 1, 4, 5.
    # B12: B[0:2, 2:4] -> indices 2, 3, 6, 7.
    # B21: B[2:4, 0:2] -> indices 8, 9, 12, 13.
    # B22: B[2:4, 2:4] -> indices 10, 11, 14, 15.
    
    # C1: C[0:2, 0:2] -> indices 0, 1, 4, 5.
    # C2: C[0:2, 2:4] -> indices 2, 3, 6, 7.

    # Map for A11: Local (0,1,2,3) -> Global (0, 1, 4, 5)
    # Local 0 (11) -> 0
    # Local 1 (12) -> 1
    # Local 2 (21) -> 4
    # Local 3 (22) -> 5
    a_map_A11 = {0: 0, 1: 1, 2: 4, 3: 5}
    
    # Map for A12: Local (0,1,2,3) -> Global (2, 3, 6, 7)
    a_map_A12 = {0: 2, 1: 3, 2: 6, 3: 7}
    
    # Map for B11: Local (0,1,2,3) -> Global (0, 1, 4, 5)
    b_map_B11 = {0: 0, 1: 1, 2: 4, 3: 5}
    
    # Map for B12: Local (0,1,2,3) -> Global (2, 3, 6, 7)
    b_map_B12 = {0: 2, 1: 3, 2: 6, 3: 7}
    
    # Map for B21: Local (0,1,2,3) -> Global (8, 9, 12, 13)
    b_map_B21 = {0: 8, 1: 9, 2: 12, 3: 13}
    
    # Map for B22: Local (0,1,2,3) -> Global (10, 11, 14, 15)
    b_map_B22 = {0: 10, 1: 11, 2: 14, 3: 15}
    
    # Map for C1: Local (0,1,2,3) -> Global (0, 1, 4, 5)
    c_map_C1 = {0: 0, 1: 1, 2: 4, 3: 5}
    
    # Map for C2: Local (0,1,2,3) -> Global (2, 3, 6, 7)
    c_map_C2 = {0: 2, 1: 3, 2: 6, 3: 7}

    # Initialize Rank 28 matrices
    U_28 = np.zeros((n * m, 28))
    V_28 = np.zeros((m * p, 28))
    W_28 = np.zeros((p * n, 28))
    
    # Block 1: A11 * B11 -> C1
    add_strassen_block(U_28, V_28, W_28, 0, a_map_A11, b_map_B11, c_map_C1)
    
    # Block 2: A12 * B21 -> C1
    add_strassen_block(U_28, V_28, W_28, 7, a_map_A12, b_map_B21, c_map_C1)
    
    # Block 3: A11 * B12 -> C2
    add_strassen_block(U_28, V_28, W_28, 14, a_map_A11, b_map_B12, c_map_C2)
    
    # Block 4: A12 * B22 -> C2
    add_strassen_block(U_28, V_28, W_28, 21, a_map_A12, b_map_B22, c_map_C2)
    
    # Verify Rank 28
    def verify_decomposition(U, V, W):
        # T_recon[k, i, j] = sum_r W[k, r] U[i, r] V[j, r]
        # i in 0..7 (A), j in 0..15 (B), k in 0..7 (C)
        # T_true[k, i, j] = 1 if C_k contains A_i B_j.
        # C_k = sum_j A_ij B_jk.
        # Flat C index k_global = k_col * n + i_row = k_col * 2 + i_row.
        # Flat A index i_global = i_row * m + j_col = i_row * 4 + j_col.
        # Flat B index j_global = j_row * p + k_col = j_row * 4 + k_col.
        
        T_recon = np.einsum('kr,ir,jr->kij', W, U, V)
        
        T_true = np.zeros((p * n, n * m, m * p))
        for i_row in range(n):
            for j_col in range(m):
                for k_col in range(p):
                    i_global = i_row * m + j_col
                    j_global = j_col * p + k_col
                    k_global = k_col * n + i_row
                    T_true[k_global, i_global, j_global] = 1
                    
        if not np.allclose(T_recon, T_true):
            return False
        return True

    if verify_decomposition(U_28, V_28, W_28):
        best_U, best_V, best_W = U_28, V_28, W_28
        best_rank = 28

    # If we have time, we could try to reduce further, but 28 is the known upper bound.
    # Theoretical lower bound for 2x4x4 is 25.
    # Finding 25 or 26 is an open problem or requires complex search.
    # Given the time limit and the need for a reliable improvement over 32, 28 is a solid structural improvement.
    
    if time.time() > deadline:
        pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
