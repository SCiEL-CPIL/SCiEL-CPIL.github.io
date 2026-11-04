import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Splits
    # n: 4 = 2 + 2
    # m: 5 = 2 + 2 + 1
    # p: 5 = 2 + 2 + 1
    
    n_splits = [2, 2]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 1]
    
    n_offsets = [0, 2]
    m_offsets = [0, 2, 4]
    p_offsets = [0, 2, 4]
    
    rank = 0
    U_rows = []
    V_rows = []
    W_rows = []
    
    # Helper to add a local decomposition to global
    # local_n, local_m, local_p are sizes
    # offset_i, offset_j, offset_k are global offsets
    def add_block(local_n, local_m, local_p, offset_i, offset_j, offset_k, local_U, local_V, local_W, local_rank):
        nonlocal U_rows, V_rows, W_rows
        # local_U shape: (local_n * local_m, local_rank)
        # local_V shape: (local_m * local_p, local_rank)
        # local_W shape: (local_p * local_n, local_rank)
        
        # We need to embed these into global matrices.
        # We will append rows to global U, V, W.
        # But the rank is global. We need to place these columns in the global rank space.
        # Since we are summing blocks, the total rank is the sum of local ranks.
        # We simply append the local columns to the global matrices, padded with zeros for other blocks.
        
        # However, the "rows" of U, V, W are fixed by the global dimensions.
        # U has shape (n*m, Total_Rank).
        # We need to create the full (n*m, Total_Rank) matrix.
        # It's easier to accumulate columns.
        
        pass

    # It is more efficient to build the full matrices directly.
    # Total Rank = Sum of local ranks.
    # Let's first calculate the total rank.
    
    def get_block_decomposition(l_n, l_m, l_p):
        if l_n == 2 and l_m == 2 and l_p == 2:
            # Strassen Rank 7
            # Indices: A(i,j), B(j,k), C(k,i)
            # U: (2*2, 7), V: (2*2, 7), W: (2*2, 7)
            U = np.zeros((4, 7))
            V = np.zeros((4, 7))
            W = np.zeros((4, 7))
            
            # M1 = (A11+A22)(B11+B22)
            # U: A11, A22
            U[0, 0] = 1 # A11
            U[3, 0] = 1 # A22
            # V: B11, B22
            V[0, 0] = 1 # B11
            V[3, 0] = 1 # B22
            # W: C11, C22
            W[0, 0] = 1 # C11
            W[3, 0] = 1 # C22
            
            # M2 = (A21+A22)B11
            U[2, 1] = 1 # A21
            U[3, 1] = 1 # A22
            V[0, 1] = 1 # B11
            W[2, 1] = 1 # C21
            W[3, 1] = 1 # C22 # Wait, C22 = M1 - M2 ...
            # Let's use the standard Strassen formulas carefully.
            
            # Reset
            U = np.zeros((4, 7))
            V = np.zeros((4, 7))
            W = np.zeros((4, 7))
            
            # Term 1: (A11+A22)(B11+B22) -> contributes to C11, C22
            # C11 += 1, C22 += 1
            U[0, 0] = 1; U[3, 0] = 1
            V[0, 0] = 1; V[3, 0] = 1
            W[0, 0] = 1; W[3, 0] = 1
            
            # Term 2: (A21+A22)B11 -> contributes to C21, C22
            # C21 += 1, C22 -= 1
            U[2, 1] = 1; U[3, 1] = 1
            V[0, 1] = 1
            W[2, 1] = 1; W[3, 1] = -1
            
            # Term 3: A11(B12-B22) -> contributes to C12
            # C12 += 1
            U[0, 2] = 1
            V[1, 2] = 1; V[3, 2] = -1
            W[1, 2] = 1
            
            # Term 4: A22(B21-B11) -> contributes to C21, C11
            # C21 += 1, C11 += 1
            U[3, 3] = 1
            V[2, 3] = 1; V[0, 3] = -1
            W[2, 3] = 1; W[0, 3] = 1
            
            # Term 5: (A11+A12)B22 -> contributes to C12, C11
            # C12 += 1, C11 -= 1
            U[0, 4] = 1; U[1, 4] = 1
            V[3, 4] = 1
            W[1, 4] = 1; W[0, 4] = -1
            
            # Term 6: (A21-A11)(B11+B12) -> contributes to C22
            # C22 += 1
            U[2, 5] = 1; U[0, 5] = -1
            V[0, 5] = 1; V[1, 5] = 1
            W[3, 5] = 1
            
            # Term 7: (A12-A22)(B21+B22) -> contributes to C11
            # C11 += 1
            U[1, 6] = 1; U[3, 6] = -1
            V[2, 6] = 1; V[3, 6] = 1
            W[0, 6] = 1
            
            return U, V, W, 7

        elif l_n == 2 and l_m == 2 and l_p == 1:
            # Rank 4
            U = np.zeros((4, 4))
            V = np.zeros((2, 4)) # l_m*l_p = 2*1=2
            W = np.zeros((2, 4)) # l_p*l_n = 1*2=2
            
            # C1 = A11 B11 + A12 B21
            # C2 = A21 B11 + A22 B21
            
            # Term 1: A11 B11 C1
            U[0, 0] = 1
            V[0, 0] = 1
            W[0, 0] = 1
            
            # Term 2: A12 B21 C1
            U[1, 1] = 1
            V[1, 1] = 1
            W[0, 1] = 1
            
            # Term 3: A21 B11 C2
            U[2, 2] = 1
            V[0, 2] = 1
            W[1, 2] = 1
            
            # Term 4: A22 B21 C2
            U[3, 3] = 1
            V[1, 3] = 1
            W[1, 3] = 1
            
            return U, V, W, 4

        elif l_n == 2 and l_m == 1 and l_p == 2:
            # Rank 4
            U = np.zeros((2, 4)) # 2*1
            V = np.zeros((2, 4)) # 1*2
            W = np.zeros((4, 4)) # 2*2
            
            # C11 = A11 B11
            # C12 = A11 B12
            # C21 = A21 B11
            # C22 = A21 B12
            
            U[0, 0] = 1
            V[0, 0] = 1
            W[0, 0] = 1
            
            U[0, 1] = 1
            V[1, 1] = 1
            W[1, 1] = 1
            
            U[1, 2] = 1
            V[0, 2] = 1
            W[2, 2] = 1
            
            U[1, 3] = 1
            V[1, 3] = 1
            W[3, 3] = 1
            
            return U, V, W, 4

        elif l_n == 2 and l_m == 1 and l_p == 1:
            # Rank 2
            U = np.zeros((2, 2))
            V = np.zeros((1, 2))
            W = np.zeros((2, 2))
            
            # C1 = A11 B11
            # C2 = A21 B11
            
            U[0, 0] = 1
            V[0, 0] = 1
            W[0, 0] = 1
            
            U[1, 1] = 1
            V[0, 1] = 1
            W[1, 1] = 1
            
            return U, V, W, 2

        else:
            # Fallback schoolbook
            r = l_n * l_m * l_p
            U = np.zeros((l_n * l_m, r))
            V = np.zeros((l_m * l_p, r))
            W = np.zeros((l_p * l_n, r))
            c = 0
            for i in range(l_n):
                for j in range(l_m):
                    for k in range(l_p):
                        U[i * l_m + j, c] = 1
                        V[j * l_p + k, c] = 1
                        W[k * l_n + i, c] = 1
                        c += 1
            return U, V, W, r

    # Calculate total rank
    total_rank = 0
    blocks = []
    for i, ln in enumerate(n_splits):
        for j, lm in enumerate(m_splits):
            for k, lp in enumerate(p_splits):
                U_loc, V_loc, W_loc, r_loc = get_block_decomposition(ln, lm, lp)
                blocks.append((i, j, k, ln, lm, lp, U_loc, V_loc, W_loc, r_loc))
                total_rank += r_loc

    U_global = np.zeros((n * m, total_rank))
    V_global = np.zeros((m * p, total_rank))
    W_global = np.zeros((p * n, total_rank))

    col_offset = 0
    for i, j, k, ln, lm, lp, U_loc, V_loc, W_loc, r_loc in blocks:
        row_offset_U = n_offsets[i] * m + m_offsets[j]
        # Note: U index is i*m + j.
        # Global i is in block i, local i' is 0..ln-1.
        # Global j is in block j, local j' is 0..lm-1.
        # Global index for U: (n_offsets[i] + i') * m + (m_offsets[j] + j')
        
        row_offset_V = m_offsets[j] * p + p_offsets[k]
        # Global index for V: (m_offsets[j] + j') * p + (p_offsets[k] + k')
        
        row_offset_W = p_offsets[k] * n + n_offsets[i]
        # Global index for W: (p_offsets[k] + k') * n + (n_offsets[i] + i')

        for r in range(r_loc):
            # Embed U_loc
            for local_row in range(ln * lm):
                val = U_loc[local_row, r]
                if val != 0:
                    # local_row = i' * lm + j'
                    i_prime = local_row // lm
                    j_prime = local_row % lm
                    global_row = (n_offsets[i] + i_prime) * m + (m_offsets[j] + j_prime)
                    U_global[global_row, col_offset + r] = val

            # Embed V_loc
            for local_row in range(lm * lp):
                val = V_loc[local_row, r]
                if val != 0:
                    j_prime = local_row // lp
                    k_prime = local_row % lp
                    global_row = (m_offsets[j] + j_prime) * p + (p_offsets[k] + k_prime)
                    V_global[global_row, col_offset + r] = val

            # Embed W_loc
            for local_row in range(lp * ln):
                val = W_loc[local_row, r]
                if val != 0:
                    k_prime = local_row // ln
                    i_prime = local_row % ln
                    global_row = (p_offsets[k] + k_prime) * n + (n_offsets[i] + i_prime)
                    W_global[global_row, col_offset + r] = val

        col_offset += r_loc

    # Convert to lists of lists for the output interface
    U_list = U_global.tolist()
    V_list = V_global.tolist()
    W_list = W_global.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
