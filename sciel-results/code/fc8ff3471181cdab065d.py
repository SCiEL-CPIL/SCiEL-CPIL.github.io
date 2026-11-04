"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # We use a hierarchical decomposition:
    # 1. Split m=4 into m1=2, m2=2.
    # 2. For each 3x2x7 subproblem:
    #    a. Split p=7 into p1=2, p2=2, p3=2, p4=1.
    #    b. For each 3x2x2 part:
    #       - Split n=3 into n1=2, n2=1.
    #       - Use Strassen (rank 7) for 2x2x2.
    #       - Use Schoolbook (rank 4) for 1x2x2.
    #       - Total rank 11.
    #    c. For the 3x2x1 part:
    #       - Use Schoolbook (rank 6).
    #    d. Total rank for 3x2x7 is 3*11 + 6 = 39.
    # 3. Total rank is 2 * 39 = 78.
    
    rank_total = 78
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a decomposition
    def add_decomp(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # 1. 2x2x2 Strassen Decomposition
    # Inputs: A (2x2), B (2x2). Output C (2x2).
    # Indices: A[0,0], A[0,1], A[1,0], A[1,1] -> 0,1,2,3
    # B[0,0], B[0,1], B[1,0], B[1,1] -> 0,1,2,3
    # C[0,0], C[0,1], C[1,0], C[1,1] -> 0,1,2,3
    # U indices for A: 0,1,2,3
    # V indices for B: 0,1,2,3
    # W indices for C: 0,1,2,3
    # Strassen M1..M7
    
    def add_strassen_2x2x2(base_u, base_v, base_w):
        nonlocal col
        # M1 = (A00 + A11) * (B00 + B11)
        U[base_u + 0][col] = 1
        U[base_u + 3][col] = 1
        V[base_v + 0][col] = 1
        V[base_v + 3][col] = 1
        W[base_w + 0][col] = 1
        W[base_w + 1][col] = -1
        W[base_w + 2][col] = -1
        W[base_w + 3][col] = 1
        col += 1
        
        # M2 = (A10 + A11) * B00
        U[base_u + 2][col] = 1
        U[base_u + 3][col] = 1
        V[base_v + 0][col] = 1
        W[base_w + 2][col] = 1
        W[base_w + 3][col] = 1
        col += 1
        
        # M3 = A00 * (B01 - B11)
        U[base_u + 0][col] = 1
        V[base_v + 1][col] = 1
        V[base_v + 3][col] = -1
        W[base_w + 1][col] = 1
        W[base_w + 3][col] = -1
        col += 1
        
        # M4 = A11 * (B10 - B00)
        U[base_u + 3][col] = 1
        V[base_v + 2][col] = 1
        V[base_v + 0][col] = -1
        W[base_w + 2][col] = -1
        W[base_w + 0][col] = 1
        col += 1
        
        # M5 = (A00 + A01) * B11
        U[base_u + 0][col] = 1
        U[base_u + 1][col] = 1
        V[base_v + 3][col] = 1
        W[base_w + 1][col] = 1
        W[base_w + 3][col] = 1
        col += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[base_u + 2][col] = 1
        U[base_u + 0][col] = -1
        V[base_v + 0][col] = 1
        V[base_v + 1][col] = 1
        W[base_w + 0][col] = 1
        W[base_w + 2][col] = -1
        col += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[base_u + 1][col] = 1
        U[base_u + 3][col] = -1
        V[base_v + 2][col] = 1
        V[base_v + 3][col] = 1
        W[base_w + 1][col] = -1
        W[base_w + 3][col] = 1
        col += 1

    # 2. 1x2x2 Schoolbook Decomposition
    # A (1x2), B (2x2). C (1x2).
    # A[0,0], A[0,1] -> 0,1
    # B[0,0], B[0,1], B[1,0], B[1,1] -> 0,1,2,3
    # C[0,0], C[0,1] -> 0,1
    
    def add_schoolbook_1x2x2(base_u, base_v, base_w):
        nonlocal col
        # C00 = A00 B00
        U[base_u + 0][col] = 1
        V[base_v + 0][col] = 1
        W[base_w + 0][col] = 1
        col += 1
        # C01 = A00 B01
        U[base_u + 0][col] = 1
        V[base_v + 1][col] = 1
        W[base_w + 1][col] = 1
        col += 1
        # C00 += A01 B10
        U[base_u + 1][col] = 1
        V[base_v + 2][col] = 1
        W[base_w + 0][col] = 1
        col += 1
        # C01 += A01 B11
        U[base_u + 1][col] = 1
        V[base_v + 3][col] = 1
        W[base_w + 1][col] = 1
        col += 1

    # 3. 3x2x1 Schoolbook Decomposition
    # A (3x2), B (2x1). C (3x1).
    # A[i,j], B[j,0]. C[i,0].
    
    def add_schoolbook_3x2x1(base_u, base_v, base_w):
        nonlocal col
        for i in range(3):
            for j in range(2):
                U[base_u + i*2 + j][col] = 1
                V[base_v + j][col] = 1
                W[base_w + i][col] = 1
                col += 1

    # Main Loop
    # Split m=4 into 2+2
    for block_m in range(2):
        # Base indices for U (A part) for this m-block
        # A is 3x4. Block m is 3x2.
        # U indices: i*m + j.
        # If block_m=0, j in 0,1.
        # If block_m=1, j in 2,3.
        # We map local j to global j = j_local + block_m*2.
        
        # Split p=7 into 2,2,2,1
        p_splits = [2, 2, 2, 1]
        
        for idx_p, p_size in enumerate(p_splits):
            # Base indices for V (B part)
            # B is 4x7. We are in m-block (2x7) and p-split.
            # V indices: j*p + k.
            # Local j in 0,1. Global j = j_local + block_m*2.
            # Local k in 0..p_size-1. Global k = k_local + offset_p.
            
            # Calculate offset for k
            offset_p = sum(p_splits[:idx_p])
            
            if p_size == 2:
                # 3x2x2 subproblem
                # Split n=3 into 2+1
                
                # Part 1: 2x2x2 (Strassen)
                # A rows 0,1. B cols 0,1.
                # U base: i*4 + j. i in 0,1. j in 0,1 (local).
                # V base: j*7 + k. j in 0,1 (local). k in 0,1 (local).
                # W base: k*3 + i. k in 0,1. i in 0,1.
                
                # We need to map local indices to global indices in the matrices U, V, W.
                
                # Strassen U indices (A part):
                # Local A: 2x2. Indices 0,1,2,3 correspond to (0,0),(0,1),(1,0),(1,1).
                # Global A indices: i*4 + (j_local + block_m*2).
                # i=0, j=0 -> 0*4 + block_m*2
                # i=0, j=1 -> 0*4 + (1 + block_m*2)
                # i=1, j=0 -> 1*4 + block_m*2
                # i=1, j=1 -> 1*4 + (1 + block_m*2)
                
                u_strassen_map = [
                    0*4 + block_m*2,      # (0,0)
                    0*4 + (1 + block_m*2), # (0,1)
                    1*4 + block_m*2,      # (1,0)
                    1*4 + (1 + block_m*2) # (1,1)
                ]
                
                # Strassen V indices (B part):
                # Local B: 2x2. Indices 0,1,2,3 correspond to (0,0),(0,1),(1,0),(1,1).
                # Global B indices: (j_local + block_m*2)*7 + (k_local + offset_p).
                # j=0, k=0 -> block_m*2*7 + offset_p
                # j=0, k=1 -> block_m*2*7 + offset_p + 1
                # j=1, k=0 -> (1 + block_m*2)*7 + offset_p
                # j=1, k=1 -> (1 + block_m*2)*7 + offset_p + 1
                
                v_strassen_map = [
                    (0 + block_m*2)*7 + offset_p,       # (0,0)
                    (0 + block_m*2)*7 + offset_p + 1,   # (0,1)
                    (1 + block_m*2)*7 + offset_p,       # (1,0)
                    (1 + block_m*2)*7 + offset_p + 1    # (1,1)
                ]
                
                # Strassen W indices (C part):
                # Local C: 2x2. Indices 0,1,2,3 correspond to (0,0),(0,1),(1,0),(1,1).
                # Global C indices: k*3 + i.
                # k=0, i=0 -> 0*3 + 0
                # k=0, i=1 -> 0*3 + 1
                # k=1, i=0 -> 1*3 + 0
                # k=1, i=1 -> 1*3 + 1
                # Note: k is local p index. Global k = k_local + offset_p.
                # i is local n index (0,1). Global i = i_local.
                
                w_strassen_map = [
                    offset_p*3 + 0,       # (0,0)
                    offset_p*3 + 1,       # (0,1)
                    (offset_p + 1)*3 + 0, # (1,0)
                    (offset_p + 1)*3 + 1  # (1,1)
                ]
                
                # Apply Strassen
                # We need to pass the mapped indices to the add function?
                # The current add_strassen_2x2x2 assumes contiguous indices 0..3.
                # We need to modify the helper to accept mapped indices or just inline it.
                
                # Let's inline for clarity and correctness of mapping.
                
                # M1
                U[u_strassen_map[0]][col] = 1
                U[u_strassen_map[3]][col] = 1
                V[v_strassen_map[0]][col] = 1
                V[v_strassen_map[3]][col] = 1
                W[w_strassen_map[0]][col] = 1
                W[w_strassen_map[1]][col] = -1
                W[w_strassen_map[2]][col] = -1
                W[w_strassen_map[3]][col] = 1
                col += 1
                
                # M2
                U[u_strassen_map[2]][col] = 1
                U[u_strassen_map[3]][col] = 1
                V[v_strassen_map[0]][col] = 1
                W[w_strassen_map[2]][col] = 1
                W[w_strassen_map[3]][col] = 1
                col += 1
                
                # M3
                U[u_strassen_map[0]][col] = 1
                V[v_strassen_map[1]][col] = 1
                V[v_strassen_map[3]][col] = -1
                W[w_strassen_map[1]][col] = 1
                W[w_strassen_map[3]][col] = -1
                col += 1
                
                # M4
                U[u_strassen_map[3]][col] = 1
                V[v_strassen_map[2]][col] = 1
                V[v_strassen_map[0]][col] = -1
                W[w_strassen_map[2]][col] = -1
                W[w_strassen_map[0]][col] = 1
                col += 1
                
                # M5
                U[u_strassen_map[0]][col] = 1
                U[u_strassen_map[1]][col] = 1
                V[v_strassen_map[3]][col] = 1
                W[w_strassen_map[1]][col] = 1
                W[w_strassen_map[3]][col] = 1
                col += 1
                
                # M6
                U[u_strassen_map[2]][col] = 1
                U[u_strassen_map[0]][col] = -1
                V[v_strassen_map[0]][col] = 1
                V[v_strassen_map[1]][col] = 1
                W[w_strassen_map[0]][col] = 1
                W[w_strassen_map[2]][col] = -1
                col += 1
                
                # M7
                U[u_strassen_map[1]][col] = 1
                U[u_strassen_map[3]][col] = -1
                V[v_strassen_map[2]][col] = 1
                V[v_strassen_map[3]][col] = 1
                W[w_strassen_map[1]][col] = -1
                W[w_strassen_map[3]][col] = 1
                col += 1
                
                # Part 2: 1x2x2 (Schoolbook) for row 2 of A
                # A row 2. j in 0,1.
                # U index: 2*4 + (j_local + block_m*2)
                
                # C rows 0,1. k in 0,1.
                
                # C00
                U[2*4 + block_m*2][col] = 1
                V[(0 + block_m*2)*7 + offset_p][col] = 1
                W[offset_p*3 + 2][col] = 1
                col += 1
                
                # C01
                U[2*4 + block_m*2][col] = 1
                V[(0 + block_m*2)*7 + offset_p + 1][col] = 1
                W[offset_p*3 + 2][col] = 1
                col += 1
                
                # C10 (Local C index 1,0 -> Global k=offset_p+1, i=2)
                # Wait, C is 3x2.
                # Local C indices 0,1 correspond to k=0,1.
                # W index: k*3 + i.
                # k=0, i=2 -> offset_p*3 + 2.
                # k=1, i=2 -> (offset_p+1)*3 + 2.
                
                # C10 (k=1, i=2) is not part of 1x2x2?
                # 1x2x2 produces C (1x2). So k=0,1. i=2.
                # So we need W[offset_p*3 + 2] and W[(offset_p+1)*3 + 2].
                
                # C10 (k=1, i=2)
                U[2*4 + (1 + block_m*2)][col] = 1
                V[(1 + block_m*2)*7 + offset_p][col] = 1
                W[(offset_p + 1)*3 + 2][col] = 1
                col += 1
                
                # C11 (k=1, i=2)
                U[2*4 + (1 + block_m*2)][col] = 1
                V[(1 + block_m*2)*7 + offset_p + 1][col] = 1
                W[(offset_p + 1)*3 + 2][col] = 1
                col += 1

            else: # p_size == 1
                # 3x2x1 Schoolbook
                # A (3x2), B (2x1). C (3x1).
                # k_local = 0. Global k = offset_p.
                
                for i in range(3):
                    for j in range(2):
                        U[i*4 + (j + block_m*2)][col] = 1
                        V[(j + block_m*2)*7 + offset_p][col] = 1
                        W[offset_p*3 + i][col] = 1
                        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
