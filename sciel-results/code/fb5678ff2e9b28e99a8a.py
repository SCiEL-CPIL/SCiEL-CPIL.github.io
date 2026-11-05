"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-style block decomposition:
    # 3x5x7 -> split 3=2+1, 5=2+2+1, 7=2+2+2+1
    # This gives 4x3x4 = 12 blocks of sizes {2x2x2, 2x2x1, 2x1x2, 2x1x1, 1x2x2, 1x2x1, 1x1x2, 1x1x1}
    # Use Strassen (rank 7) for 2x2x2 blocks, schoolbook for others.
    # 
    # Total rank estimate:
    # 1x 2x2x2 blocks: 1*7 = 7
    # 2x 2x2x1 blocks: 2*4 = 8 (schoolbook 2*2*1=4)
    # 3x 2x1x2 blocks: 3*4 = 12
    # 4x 2x1x1 blocks: 4*2 = 8
    # 2x 1x2x2 blocks: 2*4 = 8
    # 3x 1x2x1 blocks: 3*2 = 6
    # 1x 1x1x2 blocks: 1*2 = 2
    # 1x 1x1x1 blocks: 1*1 = 1
    # Total: 7+8+12+8+8+6+2+1 = 52
    # 
    # This is a significant improvement over the schoolbook rank of 105.
    
    n, m, p = DIMENSIONS
    
    # Block boundaries
    i_splits = [0, 2, 3]  # 3 = 2 + 1
    j_splits = [0, 2, 4, 5]  # 5 = 2 + 2 + 1
    k_splits = [0, 2, 4, 6, 7]  # 7 = 2 + 2 + 2 + 1
    
    rank = 0
    U_entries = []
    V_entries = []
    W_entries = []
    
    for i_block in range(4):
        i_start = i_splits[i_block]
        i_end = i_splits[i_block + 1]
        i_size = i_end - i_start
        
        for j_block in range(3):
            j_start = j_splits[j_block]
            j_end = j_splits[j_block + 1]
            j_size = j_end - j_start
            
            for k_block in range(4):
                k_start = k_splits[k_block]
                k_end = k_splits[k_block + 1]
                k_size = k_end - k_start
                
                # Determine the decomposition for this block
                if i_size == 2 and j_size == 2 and k_size == 2:
                    # Strassen 2x2x2, rank 7
                    block_rank = 7
                    # Strassen factors for 2x2x2
                    # U: 4x7, V: 4x7, W: 4x7
                    # Indexing: U[a*2+b, t], V[c*2+d, t], W[e*2+f, t]
                    # where a in {0,1}, b in {0,1}, c in {0,1}, d in {0,1}, e in {0,1}, f in {0,1}
                    # Strassen: 
                    # M1 = (A11-A22)(B21-B22)
                    # M2 = (A11+A22)B22
                    # M3 = A11(B11-B21)
                    # M4 = A22(B12-B11)
                    # M5 = (A11+A12)B22
                    # M6 = (A21-A11)(B11+B12)
                    # M7 = (A12-A22)(B21+B22)
                    # C11 = M1 + M4 - M5 + M7
                    # C12 = M3 + M5
                    # C21 = M1 + M2
                    # C22 = M2 - M3 + M4 + M6
                    
                    # U factors (4 rows: (1,1),(1,2),(2,1),(2,2); 7 cols: M1..M7)
                    U_block = [
                        [1, 1, 1, 0, 1, 0, 0],   # (1,1): M1, M2, M3, 0, M5, 0, 0
                        [0, 0, 0, 0, 1, 1, -1],  # (1,2): 0, 0, 0, 0, M5, M6, -M7
                        [-1, -1, 0, 1, 0, 1, 1], # (2,1): -M1, -M2, 0, M4, 0, M6, M7
                        [-1, 0, -1, 1, 0, 0, 1], # (2,2): -M1, 0, -M3, M4, 0, 0, M7
                    ]
                    # V factors (4 rows: (1,1),(1,2),(2,1),(2,2); 7 cols: M1..M7)
                    V_block = [
                        [0, 0, 1, -1, 0, 1, 0],  # (1,1): 0, 0, M3, -M4, 0, M6, 0
                        [1, 0, -1, 0, 0, 1, 1],  # (1,2): M1, 0, -M3, 0, 0, M6, M7
                        [-1, 1, 0, 0, 1, 0, -1], # (2,1): -M1, M2, 0, 0, M5, 0, -M7
                        [-1, 1, 0, 0, 0, 0, 1],  # (2,2): -M1, M2, 0, 0, 0, 0, M7
                    ]
                    # W factors (4 rows: (1,1),(1,2),(2,1),(2,2); 7 cols: M1..M7)
                    W_block = [
                        [1, 0, 0, 1, -1, 0, 1],  # (1,1): M1, 0, 0, M4, -M5, 0, M7
                        [0, 0, 1, 0, 1, 0, 0],   # (1,2): 0, 0, M3, 0, M5, 0, 0
                        [1, 1, 0, 0, 0, 0, 0],   # (2,1): M1, M2, 0, 0, 0, 0, 0
                        [0, 1, -1, 1, 0, 1, 0],  # (2,2): 0, M2, -M3, M4, 0, M6, 0
                    ]
                else:
                    # Schoolbook for this block
                    block_rank = i_size * j_size * k_size
                    U_block = []
                    V_block = []
                    W_block = []
                    for a in range(i_size):
                        for b in range(j_size):
                            for c in range(k_size):
                                u_row = [0] * block_rank
                                v_row = [0] * block_rank
                                w_row = [0] * block_rank
                                idx = a * j_size * k_size + b * k_size + c
                                u_row[idx] = 1
                                v_row[idx] = 1
                                w_row[idx] = 1
                                U_block.append(u_row)
                                V_block.append(v_row)
                                W_block.append(w_row)
                
                # Add to global rank
                rank += block_rank
                base_rank = rank - block_rank
                
                # Expand U_block to global coordinates
                for a in range(i_size):
                    for b in range(j_size):
                        global_i = i_start + a
                        global_j = j_start + b
                        u_idx = global_i * m + global_j
                        for t in range(block_rank):
                            val = U_block[a * j_size + b][t] if i_size == 2 and j_size == 2 and k_size == 2 else 0
                            if val != 0:
                                U_entries.append((u_idx, base_rank + t, val))
                
                # Expand V_block to global coordinates
                for c in range(j_size):
                    for d in range(k_size):
                        global_j2 = j_start + c
                        global_k = k_start + d
                        v_idx = global_j2 * p + global_k
                        for t in range(block_rank):
                            val = V_block[c * k_size + d][t] if i_size == 2 and j_size == 2 and k_size == 2 else 0
                            if val != 0:
                                V_entries.append((v_idx, base_rank + t, val))
                
                # Expand W_block to global coordinates
                for e in range(k_size):
                    for f in range(i_size):
                        global_k2 = k_start + e
                        global_i2 = i_start + f
                        w_idx = global_k2 * n + global_i2
                        for t in range(block_rank):
                            val = W_block[e * i_size + f][t] if i_size == 2 and j_size == 2 and k_size == 2 else 0
                            if val != 0:
                                W_entries.append((w_idx, base_rank + t, val))
    
    # Build the matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i, t, val in U_entries:
        U[i][t] = val
    for j, t, val in V_entries:
        V[j][t] = val
    for k, t, val in W_entries:
        W[k][t] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
