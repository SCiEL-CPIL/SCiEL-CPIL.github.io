import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Split indices
    # A: 2x5, B: 5x6
    # j in 0..3 for A1, j=4 for A2
    # k in 0..3 for B11, B21; k in 4..5 for B12, B22
    
    # Rank components
    # 1. A1 (2x4) * B11 (4x4) -> Rank 28 (4 * Strassen 2x2x2)
    # 2. A1 (2x4) * B12 (4x2) -> Rank 14 (2 * Strassen 2x2x2)
    # 3. A2 (2x1) * B21 (1x4) + A2 (2x1) * B22 (1x2) -> Rank 1 (Combined 2x1x6)
    
    rank = 28 + 14 + 1
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen decomposition
    # Local indices: i in 0..1, j in 0..1, k in 0..1
    # Global mapping for A1*B11 block (2x2x2)
    # A1 is 2x4. Block j_loc in 0..1 corresponds to j_global = j_loc*2 + j_sub?
    # No, A1 is 2x4. We split into two 2x2 blocks: A1a (j=0,1), A1b (j=2,3).
    # B11 is 4x4. Split into four 2x2 blocks:
    # B11a (j=0,1; k=0,1), B11b (j=0,1; k=2,3)
    # B11c (j=2,3; k=0,1), B11d (j=2,3; k=2,3)
    
    # Strassen 2x2x2 tensors
    # U, V, W for 2x2x2
    # i,j,k in 0,1
    # r in 0..6
    S_U = [
        [1, 0, 1, 0, 1, 0, 0], # i=0, j=0
        [0, 1, 0, 1, 0, 1, 0], # i=0, j=1
        [1, 0, 0, 1, 1, 0, 0], # i=1, j=0
        [0, 1, 1, 0, 0, 1, 0]  # i=1, j=1
    ]
    # Wait, standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # U[i,j] coefficients
    # M1: A11(1), A22(1)
    # M2: A21(1), A22(1)
    # M3: A11(1)
    # M4: A22(1)
    # M5: A11(1), A12(1)
    # M6: A21(1), A11(-1)
    # M7: A12(1), A22(-1)
    
    U_strassen = np.array([
        [1, 0, 1, 0, 1, -1, 0], # i=0, j=0 (A11)
        [0, 0, 0, 0, 1, 0, 1],  # i=0, j=1 (A12)
        [1, 1, 0, 1, 0, 1, 0],  # i=1, j=0 (A21)
        [1, 1, 0, 1, 0, 0, -1]  # i=1, j=1 (A22)
    ])
    
    V_strassen = np.array([
        [1, 1, 0, 1, 0, 1, 0], # j=0, k=0 (B11)
        [1, 0, 1, -1, 1, 0, 1],# j=0, k=1 (B12)
        [1, 1, -1, 0, 1, 0, 1],# j=1, k=0 (B21)
        [1, 0, 0, 0, 1, 0, 1]  # j=1, k=1 (B22)
    ])
    # Check V indices: V[j,k]
    # M1: B11(1), B22(1)
    # M2: B11(1)
    # M3: B21(1), B22(-1)
    # M4: B11(1), B21(-1)
    # M5: B22(1)
    # M6: B11(1), B12(1)
    # M7: B21(1), B22(1)
    
    V_strassen = np.array([
        [1, 1, 0, 1, 0, 1, 0], # j=0, k=0 (B11)
        [0, 0, 1, -1, 0, 1, 0],# j=0, k=1 (B12) -- Wait, M3 is B21-B22. M6 is B11+B12.
        # Let's re-verify V
        # M1: B11, B22
        # M2: B11
        # M3: B21, B22(-1)
        # M4: B11, B21(-1)
        # M5: B22
        # M6: B11, B12
        # M7: B21, B22
        
        # V[j,k,r]
        # j=0, k=0 (B11): M1(1), M2(1), M4(1), M6(1) -> [1, 1, 0, 1, 0, 1, 0]
        # j=0, k=1 (B12): M6(1) -> [0, 0, 0, 0, 0, 1, 0]
        # j=1, k=0 (B21): M3(1), M4(-1), M7(1) -> [0, 0, 1, -1, 0, 0, 1]
        # j=1, k=1 (B22): M1(1), M3(-1), M5(1), M7(1) -> [1, 0, -1, 0, 1, 0, 1]
        
        [1, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, -1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    
    W_strassen = np.array([
        [1, 0, 0, 0, 0, 0, 0], # k=0, i=0 (C11)
        [0, 0, 0, 0, 0, 0, 0], # k=0, i=1 (C21) -- No, W[k,i]
        # C11 = M1 - M4 + M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M6 + M7
        
        # W[k,i,r]
        # k=0, i=0 (C11): M1(1), M4(-1), M5(1), M7(1) -> [1, 0, 0, -1, 1, 0, 1]
        # k=1, i=0 (C12): M3(1), M5(1) -> [0, 0, 1, 0, 1, 0, 0]
        # k=0, i=1 (C21): M2(1), M4(1) -> [0, 1, 0, 1, 0, 0, 0]
        # k=1, i=1 (C22): M1(1), M3(-1), M6(1), M7(1) -> [1, 0, -1, 0, 0, 1, 1]
        
        [1, 0, 0, -1, 1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 0, -1, 0, 0, 1, 1]
    ])
    
    # Function to add a 2x2x2 block
    def add_block(i_offset, j_offset, k_offset, U_s, V_s, W_s):
        nonlocal col
        for r in range(7):
            for i_loc in range(2):
                i_glob = i_offset + i_loc
                j_glob = j_offset
                # U index: i_glob * m + j_glob
                # But U_s is 2x2x7? No, U_s is 4x7 (i*2+j).
                # We need to map local i,j to global i,j.
                pass
        
        # Simpler: iterate over local i, j, k
        for r in range(7):
            for i_loc in range(2):
                for j_loc in range(2):
                    i_glob = i_offset + i_loc
                    j_glob = j_offset + j_loc
                    U[i_glob * m + j_glob, col + r] += U_s[i_loc * 2 + j_loc, r]
                    
            for j_loc in range(2):
                for k_loc in range(2):
                    j_glob = j_offset + j_loc
                    k_glob = k_offset + k_loc
                    V[j_glob * p + k_glob, col + r] += V_s[j_loc * 2 + k_loc, r]
                    
            for k_loc in range(2):
                for i_loc in range(2):
                    k_glob = k_offset + k_loc
                    i_glob = i_offset + i_loc
                    W[k_glob * n + i_glob, col + r] += W_s[k_loc * 2 + i_loc, r]
        
        col += 7

    # 1. A1 B11 (2x4x4)
    # A1: i=0..1, j=0..3
    # B11: j=0..3, k=0..3
    # Blocks:
    # A1a (j=0,1) * B11a (j=0,1; k=0,1)
    add_block(0, 0, 0, U_strassen, V_strassen, W_strassen)
    # A1a (j=0,1) * B11b (j=0,1; k=2,3)
    add_block(0, 0, 2, U_strassen, V_strassen, W_strassen)
    # A1b (j=2,3) * B11c (j=2,3; k=0,1)
    add_block(0, 2, 0, U_strassen, V_strassen, W_strassen)
    # A1b (j=2,3) * B11d (j=2,3; k=2,3)
    add_block(0, 2, 2, U_strassen, V_strassen, W_strassen)
    
    # 2. A1 B12 (2x4x2)
    # A1: i=0..1, j=0..3
    # B12: j=0..3, k=4,5
    # Blocks:
    # A1a (j=0,1) * B12a (j=0,1; k=4,5)
    add_block(0, 0, 4, U_strassen, V_strassen, W_strassen)
    # A1b (j=2,3) * B12b (j=2,3; k=4,5)
    add_block(0, 2, 4, U_strassen, V_strassen, W_strassen)
    
    # 3. A2 B21 + A2 B22 (2x1x6)
    # A2: i=0..1, j=4
    # B21: j=4, k=0..3
    # B22: j=4, k=4,5
    # Rank 1
    U[0 * m + 4, col] = 1
    U[1 * m + 4, col] = 1
    for k in range(6):
        V[4 * p + k, col] = 1
        W[k * n + 0, col] = 1
        W[k * n + 1, col] = 1
    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
