import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    # Decompose: A is 4x5, B is 5x5, C is 4x5
    # Split as: A = [A1 (3x5) ; A2 (1x5)], B = [B1 (5x4) B2 (5x1)], C = [C1 (4x4) C2 (4x1)]
    # Actually let's think differently:
    # A (4x5) = [A_top (3x5); A_bot (1x5)]
    # B (5x5) = [B_left (5x4) | B_right (5x1)]
    # C (4x5) = [C_main (4x4) | C_col (4x1)]
    # C_main = A_top * B_left + A_bot * B_left = (A_top + A_bot) * B_left  -- no, that's not right
    # Actually: C = A * B, so:
    # C[0:3, 0:4] = A[0:3, :] * B[:, 0:4]
    # C[3, 0:4] = A[3, :] * B[:, 0:4]
    # C[:, 4] = A[:, :] * B[:, 4]
    
    # Let's use: A = [A1 (3x5); A2 (1x5)], B = [B1 (5x4) B2 (5x1)]
    # C1 (3x4) = A1 * B1  -- use 3x3x3 Strassen on the 3x3x3 part, handle the 3x5x4 and 5x4 parts
    # Hmm, this is getting complicated. Let me think about a cleaner split.
    
    # Actually, the cleanest approach:
    # A (4x5), B (5x5), C (4x5)
    # Split A into A1 (3x5) and A2 (1x5)
    # Split B into B1 (5x4) and B2 (5x1)
    # C = A * B, so:
    # C[0:3, 0:4] = A1 * B1
    # C[3, 0:4] = A2 * B1
    # C[:, 4] = A * B2
    
    # For C[0:3, 0:4] = A1 * B1:
    # A1 is 3x5, B1 is 5x4
    # Split A1 = [A1a (3x3) A1b (3x2)], B1 = [B1a (5x3); B1b (5x1)]
    # C[0:3,0:3] = A1a * B1a + A1b * B1b  -- no, that's wrong
    # Actually C[0:3,0:4] = A1 * B1, where A1 is 3x5 and B1 is 5x4
    # This is a 3x5x4 multiplication
    
    # Let me use a different strategy. Use the known result that the rank of M_{4,5,5} 
    # can be bounded by combining smaller decompositions.
    
    # Strategy: 
    # C (4x5) = A (4x5) * B (5x5)
    # Write A = [A1 (3x5); A2 (1x5)]
    # Write B = [B1 (5x4) B2 (5x1)]
    # Then:
    # C[0:3, 0:4] = A1 * B1  (3x5 * 5x4 = 3x4)
    # C[3, 0:4] = A2 * B1    (1x5 * 5x4 = 1x4)
    # C[:, 4] = A * B2       (4x5 * 5x1 = 4x1)
    
    # For C[0:3, 0:4] = A1 * B1:
    # A1 (3x5) = [A11 (3x3) A12 (3x2)]
    # B1 (5x4) = [B11 (5x3); B12 (5x1)]
    # C[0:3, 0:3] = A11 * B11 + A12 * B12  -- WRONG, this isn't how matrix mult works
    # Actually: C[0:3, 0:4] = sum_k A1[:,k] * B1[k,:]
    # = sum_{k=0}^{2} A1[:,k] * B1[k, 0:4] + sum_{k=3}^{4} A1[:,k] * B1[k, 0:4]
    # = A1[:, 0:3] * B1[0:3, 0:4] + A1[:, 3:5] * B1[3:5, 0:4]
    # Let A1a = A1[:, 0:3] (3x3), A1b = A1[:, 3:5] (3x2)
    # Let B1a = B1[0:3, 0:4] (3x4), B1b = B1[3:5, 0:4] (2x4)
    # C[0:3, 0:4] = A1a * B1a + A1b * B1b
    
    # A1a * B1a is 3x3 * 3x4. Use 3x3x3 Strassen for the 3x3 part:
    # B1a = [B1a1 (3x3) B1a2 (3x1)]
    # A1a * B1a = A1a * B1a1 + A1a * B1a2
    # A1a * B1a1: 3x3 * 3x3, rank 23 via Strassen
    # A1a * B1a2: 3x3 * 3x1, rank 3 (or can be combined)
    # A1b * B1b: 3x2 * 2x4, rank 8 (or can be combined)
    
    # This is getting very complex. Let me just implement it carefully.
    
    rank_total = 0
    U = []
    V = []
    W = []
    
    def add_decomposition(u_list, v_list, w_list):
        nonlocal rank_total
        rank = len(u_list)
        for i in range(rank):
            U.append(u_list[i])
            V.append(v_list[i])
            W.append(w_list[i])
        rank_total += rank
    
    # Initialize U, V, W as lists of dicts (sparse)
    # U shape: (4*5, rank) = (20, rank)
    # V shape: (5*5, rank) = (25, rank)  
    # W shape: (5*4, rank) = (20, rank)  -- W[k*n+i, r] where k is col of C (0..4), i is row of C (0..3)
    
    U = [{} for _ in range(0)]  # will be filled
    V = [{} for _ in range(0)]
    W = [{} for _ in range(0)]
    
    def get_strassen_3x3_factors():
        # Returns (u, v, w) each of shape (9, 23) for 3x3x3 Strassen
        # M = M11 + M22
        # M12 = M11 - M22 + M1 + M2 - M3
        # M21 = M11 - M22 + M4 - M5 + M6
        # M22 = M22
        # Where:
        # M1 = (A11 - A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        
        u = np.zeros((9, 23))
        v = np.zeros((9, 23))
        w = np.zeros((9, 23))
        
        # M1 = (A11 - A22) * (B11 + B22)
        # A11 is at (0,0) -> index 0, A22 is at (2,2) -> index 8
        # B11 is at (0,0) -> index 0, B22 is at (2,2) -> index 8
        # Result: C11 += M1, C12 -= M1, C21 += M1, C22 -= M1
        # Wait, let me use the standard 3x3 Strassen:
        # M = M11 + M22
        # M12 = M11 - M22 + M1 + M2 - M3
        # M21 = M11 - M22 + M4 - M5 + M6
        # M22 = M22
        # Where:
        # M1 = (A11 - A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        
        # M11 = M1 + M2 + M3 + M4
        # M12 = M1 - M3 + M5 + M6
        # M21 = M2 - M4 + M5 + M7
        # M22 = M4 - M2 + M3 - M1 + M5 + M6 + M7
        
        # Let me use the well-known 3x3 Strassen with 23 multiplications:
        # M1 = (A11 - A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        # M8 = (A11 + A22) * B11
        # M9 = A12 * (B22 - B11)
        # M10 = A21 * (B11 - B22)
        # M11 = (A11 - A12) * B22
        # M12 = (A22 - A21) * B11
        # M13 = A11 * (B11 - B22)
        # M14 = A22 * (B22 - B11)
        # M15 = (A11 + A12) * (B22 - B11)
        # M16 = (A21 + A22) * (B11 - B22)
        # M17 = A12 * (B11 + B22)
        # M18 = A21 * (B22 + B11)
        # M19 = (A11 - A21) * (B22 - B12)
        # M20 = (A22 - A12) * (B11 - B21)
        # M21 = (A11 + A21) * (B22 - B11)
        # M22 = (A22 + A12) * (B11 - B22)
        # M23 = (A11 + A22 + A12 + A21) * (B11 + B22 - B12 - B21)
        
        # This is getting very messy. Let me just hardcode the 23-term 3x3 Strassen.
        # I'll use the formulation from the literature.
        
        # Actually, let me just implement it more carefully using the known decomposition.
        # For 3x3x3, the optimal is 23. Let me use a known decomposition.
        
        # I'll use the following approach: decompose 3x3x3 as:
        # (1) 2x2x2 Strassen for the top-left 2x2 block (7 terms)
        # (2) Handle the remaining parts
        
        # Actually, this is too complex to get right without referencing. Let me use a 
        # simpler approach: use 3x3x3 with rank 18 (which is known to exist) or just 
        # use the naive approach for the 3x3 part.
        
        # Let me reconsider. The naive 4x5x5 has rank 100.
        # The current approach has rank 85.
        # Let me try: 3x3x3 Strassen (23) + 3x3x1 (3) + 3x1x3 (3) + 3x1x1 (1) = 30 for the 3x3x3 block
        # + 1x5x5 block handling
        
        # Actually, let me just use a cleaner decomposition:
        # Split A (4x5) into A1 (3x5) and A2 (1x5)
        # Split B (5x5) into B1 (5x4) and B2 (5x1)
        # C[0:3, 0:4] = A1 * B1
        # C[3, 0:4] = A2 * B1
        # C[:, 4] = A * B2
        
        # For A1 * B1 (3x5 * 5x4):
        # = A1[:,0:3] * B1[0:3,:] + A1[:,3:5] * B1[3:5,:]
        # Let A1a = A1[:,0:3] (3x3), A1b = A1[:,3:5] (3x2)
        # Let B1a = B1[0:3,:] (3x4), B1b = B1[3:5,:] (2x4)
        # C[0:3,0:4] = A1a * B1a + A1b * B1b
        
        # A1a * B1a (3x3 * 3x4):
        # = A1a * B1a[:,0:3] + A1a * B1a[:,3]
        # A1a * B1a[:,0:3] (3x3 * 3x3): use 3x3 Strassen, rank 23
        # A1a * B1a[:,3] (3x3 * 3x1): rank 3 (or combine)
        
        # A1b * B1b (3x2 * 2x4): rank 8 (naive)
        
        # C[3, 0:4] = A2 * B1 (1x5 * 5x4): rank 4 (naive)
        # C[:, 4] = A * B2 (4x5 * 5x1): rank 5 (naive)
        
        # Total: 23 + 3 + 8 + 4 + 5 = 43
        
        # But we can do better by combining some terms.
        # Actually, let me just implement the 3x3 Strassen properly.
        
        return u, v, w
    
    # Let me just implement a clean version with the 3x3 Strassen hardcoded.
    # I'll use the decomposition from https://en.wikipedia.org/wiki/Strassen_algorithm#3x3_matrices
    
    # For 3x3x3, one known 23-term decomposition:
    # Let me use a simpler approach and just build it up.
    
    # Actually, let me try a different strategy entirely.
    # Use the fact that M_{4,5,5} can be decomposed as:
    # M_{4,5,5} = M_{3,5,5} + M_{1,5,5} (splitting rows of A)
    # M_{3,5,5} = M_{3,3,5} + M_{3,2,5} (splitting cols of B)
    # This is getting too complex.
    
    # Let me just carefully implement the 3x3 Strassen and combine.
    
    # 3x3 Strassen (23 multiplications):
    # M1  = (A11 - A22) * (B11 + B22)
    # M2  = (A21 + A22) * B11
    # M3  = A11 * (B12 - B22)
    # M4  = A22 * (B21 - B11)
    # M5  = (A11 + A12) * B22
    # M6  = (A21 - A11) * (B11 + B12)
    # M7  = (A12 - A22) * (B21 + B22)
    # M8  = (A11 + A22) * B11
    # M9  = A12 * (B22 - B11)
    # M10 = A21 * (B11 - B22)
    # M11 = (A11 - A12) * B22
    # M12 = (A22 - A21) * B11
    # M13 = A11 * (B11 - B22)
    # M14 = A22 * (B22 - B11)
    # M15 = (A11 + A12) * (B22 - B11)
    # M16 = (A21 + A22) * (B11 - B22)
    # M17 = A12 * (B11 + B22)
    # M18 = A21 * (B22 + B11)
    # M19 = (A11 - A21) * (B22 - B12)
    # M20 = (A22 - A12) * (B11 - B21)
    # M21 = (A11 + A21) * (B22 - B11)
    # M22 = (A22 + A12) * (B11 - B22)
    # M23 = (A11 + A22 + A12 + A21) * (B11 + B22 - B12 - B21)
    
    # C11 = M1 + M2 + M3 + M4 + M8 + M13
    # C12 = M1 - M3 + M5 + M6 + M9 + M15 + M17
    # C21 = M2 - M4 + M5 + M7 + M10 + M16 + M18
    # C22 = M4 - M2 + M3 - M1 + M5 + M6 + M7 + M8 + M14 - M9 - M10 - M11 - M12 - M13 - M14 - M15 - M16 - M17 - M18 - M19 - M20 - M21 - M22 - M23
    
    # Hmm, this doesn't look right. Let me use a verified decomposition.
    
    # I'll use a different approach: just implement the naive 3x3x3 (9 terms) and 
    # combine with the rest. The total will be:
    # 3x3x3 (9) + 3x3x1 (3) + 3x1x3 (3) + 3x1x1 (1) = 16 for the 3x3x3 sub-block
    # + 3x2x4 (8) for A1b * B1b
    # + 1x5x4 (4) for A2 * B1
    # + 4x5x1 (5) for A * B2
    # Total: 16 + 8 + 4 + 5 = 33
    
    # That's much better than 85! Let me implement this.
    
    U = {}
    V = {}
    W = {}
    
    def setU(i, j, r, val):
        U.setdefault((i, j), 0)
        U[(i, j), r] = val  # This won't work with dict
    
    # Let me use a different data structure
    U_dict = {}  # (row, col) -> list of (rank_idx, val)
    
    # Actually, let me just build arrays directly.
    
    # Plan:
    # 1. 3x3x3 naive: 9 terms for C[0:3,0:3] = A1a * B1a
    # 2. 3x3x1: 3 terms for C[0:3,3] = A1a * B1a[:,3]
    # 3. 3x2x4: 8 terms for C[0:3,0:4] += A1b * B1b
    # 4. 1x5x4: 4 terms for C[3,0:4] = A2 * B1
    # 5. 4x5x1: 5 terms for C[:,4] = A * B2
    # Total: 9 + 3 + 8 + 4 + 5 = 29
    
    # Wait, I need to be more careful about the indices.
    # A is 4x5, B is 5x5, C is 4x5
    # A[i,j] where i in 0..3, j in 0..4
    # B[j,k] where j in 0..4, k in 0..4
    # C[i,k] where i in 0..3, k in 0..4
    
    # Split:
    # A1 = A[0:3, :] (3x5), A2 = A[3, :] (1x5)
    # B1 = B[:, 0:4] (5x4), B2 = B[:, 4] (5x1)
    # C[0:3, 0:4] = A1 * B1
    # C[3, 0:4] = A2 * B1
    # C[:, 4] = A * B2
    
    # For A1 * B1 (3x5 * 5x4):
    # A1 = [A1a (3x3) A1b (3x2)] where A1a = A1[:, 0:3], A1b = A1[:, 3:5]
    # B1 = [B1a (5x3); B1b (5x1)] -- no, B1 is 5x4
    # B1 = [B1a (3x4); B1b (2x4)] where B1a = B1[0:3, :], B1b = B1[3:5, :]
    # C[0:3, 0:4] = A1a * B1a + A1b * B1b
    
    # A1a * B1a (3x3 * 3x4):
    # B1a = [B1a1 (3x3) B1a2 (3x1)] where B1a1 = B1a[:, 0:3], B1a2 = B1a[:, 3]
    # A1a * B1a = A1a * B1a1 + A1a * B1a2
    # A1a * B1a1 (3x3 * 3x3): 9 terms (naive)
    # A1a * B1a2 (3x3 * 3x1): 3 terms
    
    # A1b * B1b (3x2 * 2x4): 8 terms (naive)
    
    # C[3, 0:4] = A2 * B1 (1x5 * 5x4): 4 terms
    # C[:, 4] = A * B2 (4x5 * 5x1): 5 terms
    
    # Total: 9 + 3 + 8 + 4 + 5 = 29
    
    rank = 29
    U_arr = np.zeros((20, rank))
    V_arr = np.zeros((25, rank))
    W_arr = np.zeros((20, rank))
    
    col = 0
    
    # 1. A1a * B1a1 (3x3 * 3x3) -> C[0:3, 0:3]
    # A1a[i,j] = A[i, j] for i in 0..2, j in 0..2
    # B1a1[j,k] = B[j, k] for j in 0..2, k in 0..2
    # C[i,k] = sum_j A1a[i,j] * B1a1[j,k]
    for i in range(3):
        for k in range(3):
            for j in range(3):
                U_arr[i * 5 + j, col] = 1
                V_arr[j * 5 + k, col] = 1
                W_arr[k * 4 + i, col] = 1
                col += 1
    
    # 2. A1a * B1a2 (3x3 * 3x1) -> C[0:3, 3]
    # A1a[i,j] = A[i, j] for i in 0..2, j in 0..2
    # B1a2[j] = B[j, 3] for j in 0..2
    # C[i,3] = sum_j A1a[i,j] * B1a2[j]
    for i in range(3):
        for j in range(3):
            U_arr[i * 5 + j, col] = 1
            V_arr[j * 5 + 3, col] = 1
            W_arr[3 * 4 + i, col] = 1
            col += 1
    
    # 3. A1b * B1b (3x2 * 2x4) -> C[0:3, 0:4]
    # A1b[i,j] = A[i, 3+j] for i in 0..2, j in 0..1
    # B1b[j,k] = B[3+j, k] for j in 0..1, k in 0..3
    # C[i,k] = sum_j A1b[i,j] * B1b[j,k]
    for i in range(3):
        for k in range(4):
            for j in range(2):
                U_arr[i * 5 + (3 + j), col] = 1
                V_arr[(3 + j) * 5 + k, col] = 1
                W_arr[k * 4 + i, col] = 1
                col += 1
    
    # 4. A2 * B1 (1x5 * 5x4) -> C[3, 0:4]
    # A2[j] = A[3, j] for j in 0..4
    # B1[j,k] = B[j, k] for j in 0..4, k in 0..3
    # C[3,k] = sum_j A2[j] * B1[j,k]
    for k in range(4):
        for j in range(5):
            U_arr[3 * 5 + j, col] = 1
            V_arr[j * 5 + k, col] = 1
            W_arr[k * 4 + 3, col] = 1
            col += 1
    
    # 5. A * B2 (4x5 * 5x1) -> C[:, 4]
    # A[i,j] = A[i, j] for i in 0..3, j in 0..4
    # B2[j] = B[j, 4] for j in 0..4
    # C[i,4] = sum_j A[i,j] * B2[j]
    for i in range(4):
        for j in range(5):
            U_arr[i * 5 + j, col] = 1
            V_arr[j * 5 + 4, col] = 1
            W_arr[4 * 4 + i, col] = 1
            col += 1
    
    assert col == rank, f"col={col}, rank={rank}"
    
    return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}

