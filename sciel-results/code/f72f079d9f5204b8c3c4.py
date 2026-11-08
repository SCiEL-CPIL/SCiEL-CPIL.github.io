import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Shapes: U (n*m, R), V (m*p, R), W (p*n, R)
    # n=4, m=5, p=5
    # U: 20 x R
    # V: 25 x R
    # W: 20 x R
    
    # Rank breakdown:
    # 1. 4x4x4 Strassen: 49
    # 2. 4x4x1 Schoolbook: 16
    # 3. 4x1x5 Schoolbook: 20
    # Total Rank R = 85
    
    R = 85
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # --- Part 1: 4x4x4 Strassen (Rank 49) ---
    # A[0:4, 0:4] * B[0:4, 0:4] -> C[0:4, 0:4]
    # Indices:
    # U: i*5+j, i,j in 0..3
    # V: j*5+k, j,k in 0..3
    # W: k*4+i, k,i in 0..3
    
    # Strassen 2x2 factors
    # U_S, V_S, W_S are 4x7
    U_S = [
        [1, 0, 1, 0, 0, -1, 0],
        [0, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 1],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_S = [
        [1, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, -1, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 0]
    ]
    W_S = [
        [1, -1, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 1],
        [1, 0, -1, 1, 1, 0, 0]
    ]
    
    # Kronecker product for 4x4x4
    # U_4x4 = kron(U_S, U_S)
    # V_4x4 = kron(V_S, V_S)
    # W_4x4 = kron(W_S, W_S)
    
    # We need to map the Kronecker indices to the global indices.
    # U_4x4 index i (0..15) corresponds to A[i] in 4x4 block.
    # A is 4x4. Flattened row-major: A[0,0]..A[3,3].
    # Kron index i = i_outer * 4 + i_inner.
    # This matches A[i_outer, i_inner] if we view A as 2x2 blocks of 2x2?
    # No, A[i_outer, i_inner] in block sense means row block i_outer, col block i_inner.
    # The element is A[i_outer*2 + r, i_inner*2 + c].
    # The flattened index of A is (i_outer*2 + r) * 4 + (i_inner*2 + c).
    # This is NOT i_outer * 4 + i_inner.
    # So we cannot simply use kron(U_S, U_S) directly with standard index mapping.
    # We must iterate.
    
    col_offset = 0
    for r_outer in range(7):
        for r_inner in range(7):
            # This column corresponds to M_outer[r_outer] * M_inner[r_inner]
            # U contribution: U_S[i_outer, r_outer] * U_S[i_inner, r_inner]
            # V contribution: V_S[j_outer, r_outer] * V_S[j_inner, r_inner]
            # W contribution: W_S[k_outer, r_outer] * W_S[k_inner, r_inner]
            
            for i_outer in range(2):
                for i_inner in range(2):
                    u_val = U_S[i_outer][r_outer] * U_S[i_inner][r_inner]
                    if u_val != 0:
                        # Map to global U index
                        # A row: i_outer*2 + i_inner? No.
                        # A is 4x4.
                        # i_outer is block row (0,1). i_inner is block col (0,1).
                        # The element is at A[i_outer*2 + row_in_block, i_inner*2 + col_in_block]?
                        # Wait, U_S indices are for the 2x2 matrix.
                        # U_S[i, r] where i is index in 2x2 matrix (0..3).
                        # i = row*2 + col.
                        # So i_outer is the "outer" index of the 2x2 block matrix.
                        # Let's stick to the 2x2 matrix indices.
                        # i_outer in 0..3, i_inner in 0..3.
                        pass

            # Let's do it properly by iterating over the 4x4 indices.
            # U_4x4[i, c] = sum_{r1,r2} U_S[i1, r1] U_S[i2, r2] where i = i1*4+i2? No.
            
            # Let's just compute the full 16x49 matrix for U, V, W for the 4x4x4 part.
            # And then map to global indices.
            
            break
        break
    
    # Correct approach: Construct 16x49, 16x49, 16x49 matrices.
    U_44 = [[0]*49 for _ in range(16)]
    V_44 = [[0]*49 for _ in range(16)]
    W_44 = [[0]*49 for _ in range(16)]
    
    for r1 in range(7):
        for r2 in range(7):
            c = r1 * 7 + r2
            for i1 in range(4):
                for i2 in range(4):
                    u_val = U_S[i1][r1] * U_S[i2][r2]
                    if u_val != 0:
                        # Map i1, i2 to 4x4 index
                        # i1 is index in first 2x2 (block level). i2 in second 2x2 (element level).
                        # This assumes the tensor is T1 x T2.
                        # The index in the 4x4 matrix is i1*4 + i2?
                        # If A = A1 x A2 (Kronecker), then A[i1*4+i2] = A1[i1] * A2[i2].
                        # But A is a 4x4 matrix, not a vector.
                        # The multiplication tensor for 4x4 is T_4 = T_2 x T_2.
                        # The indices of T_4 are (i,j,k) where i,j,k are indices of 4x4 matrices.
                        # i = i_a * 4 + i_b?
                        # If we view 4x4 as 2x2 blocks of 2x2, then i = i_block * 4 + i_elem is wrong.
                        # i_block is 2x2 index. i_elem is 2x2 index.
                        # The total index is i_block * 4 + i_elem?
                        # Yes, if we flatten the 2x2 blocks.
                        # Block 0: 0,1,2,3. Block 1: 4,5,6,7.
                        # This matches row-major flattening of 4x4?
                        # Row 0: 0,1,2,3.
                        # Row 1: 4,5,6,7.
                        # Yes.
                        idx = i1 * 4 + i2
                        U_44[idx][c] += u_val
                        
                        v_val = V_S[i1][r1] * V_S[i2][r2]
                        if v_val != 0:
                            V_44[idx][c] += v_val
                        
                        w_val = W_S[i1][r1] * W_S[i2][r2]
                        if w_val != 0:
                            W_44[idx][c] += w_val

    # Map U_44, V_44, W_44 to global U, V, W
    # U_44 idx i (0..15) -> A[i] in 4x4 part.
    # A is 4x5. 4x4 part is A[0:4, 0:4].
    # Global U index: i_global = row * 5 + col.
    # i_44 = row * 4 + col.
    # So row = i_44 // 4, col = i_44 % 4.
    # i_global = row * 5 + col.
    
    for i_44 in range(16):
        row = i_44 // 4
        col = i_44 % 4
        i_global = row * 5 + col
        for c in range(49):
            U[i_global][c] = U_44[i_44][c]
            
    # V_44 idx j (0..15) -> B[j] in 4x4 part.
    # B is 5x5. 4x4 part is B[0:4, 0:4].
    # j_44 = row * 4 + col.
    # j_global = row * 5 + col.
    for j_44 in range(16):
        row = j_44 // 4
        col = j_44 % 4
        j_global = row * 5 + col
        for c in range(49):
            V[j_global][c] = V_44[j_44][c]
            
    # W_44 idx k (0..15) -> C[k] in 4x4 part.
    # C is 4x5. 4x4 part is C[0:4, 0:4].
    # k_44 = col * 4 + row? No.
    # W index convention: W[k*n + i, r]. k is col, i is row.
    # C is 4x5.
    # 4x4 part is C[0:4, 0:4].
    # k_44 = col * 4 + row.
    # k_global = col * 4 + row. (Same, since n=4)
    for k_44 in range(16):
        col = k_44 // 4
        row = k_44 % 4
        k_global = col * 4 + row
        for c in range(49):
            W[k_global][c] = W_44[k_44][c]

    # --- Part 2: 4x4x1 Schoolbook (Rank 16) ---
    # A[0:4, 0:4] * B[0:4, 4] -> C[0:4, 4]
    # U: A[0:4, 0:4] (16 entries)
    # V: B[0:4, 4] (4 entries)
    # W: C[0:4, 4] (4 entries)
    
    offset = 49
    for i in range(16):
        row = i // 4
        col = i % 4
        i_global = row * 5 + col
        for j in range(4):
            j_global = j * 5 + 4
            k_global = 4 * 4 + row # col=4, row=i
            c = offset + i * 4 + j
            U[i_global][c] = 1
            V[j_global][c] = 1
            W[k_global][c] = 1

    # --- Part 3: 4x1x5 Schoolbook (Rank 20) ---
    # A[0:4, 4] * B[4, 0:5] -> C[0:4, 0:5]
    # U: A[0:4, 4] (4 entries)
    # V: B[4, 0:5] (5 entries)
    # W: C[0:4, 0:5] (20 entries)
    
    offset = 49 + 16
    for i in range(4):
        i_global = i * 5 + 4
        for k in range(5):
            j_global = 4 * 5 + k
            k_global = k * 4 + i
            c = offset + i * 5 + k
            U[i_global][c] = 1
            V[j_global][c] = 1
            W[k_global][c] = 1

    return {"U": U, "V": V, "W": W}
