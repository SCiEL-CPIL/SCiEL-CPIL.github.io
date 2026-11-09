DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 5,5,5
    
    rank = 94
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 1. Strassen 4x4 for top-left
    terms_4x4 = generate_strassen_4x4()
    for u_4, v_4, w_4 in terms_4x4:
        for i in range(4):
            for j in range(4):
                val = u_4[i*4+j]
                if val != 0:
                    U[i*5 + j][col] = val
                val_v = v_4[i*4+j]
                if val_v != 0:
                    V[i*5 + j][col] = val_v
                # W index: k*n + i. Here C is 4x4, so k in 0..3, i in 0..3.
                # w_4 index is k*4 + i? 
                # In my generator, w_4 was built as (r_c*2+i)*4 + (c_c*2+j).
                # r_c is row of C block, c_c is col of C block.
                # C[row, col].
                # The tensor decomposition requires W[k*n + i] where C[i, k].
                # My w_4 index corresponds to C[row, col].
                # So C[i, k] = C[row, col] -> i=row, k=col.
                # W index should be k*5 + i = col*5 + row.
                # My w_4 index is row*4 + col.
                # So I need to map w_4[row*4+col] to W[col*5 + row].
                
                # Let's re-verify the W mapping.
                # C[i, k] = sum_j A[i, j] B[j, k].
                # W vector has non-zero at index k*n + i.
                # In the 4x4 subproblem, C_sub[i_sub, k_sub].
                # W_sub index is k_sub * 4 + i_sub.
                # Global W index is k_sub * 5 + i_sub.
                
                # In my generator, w_4 was indexed by (r_c*2+i)*4 + (c_c*2+j).
                # r_c is row block, c_c is col block.
                # i, j are local indices.
                # So global row i_sub = r_c*2 + i.
                # global col k_sub = c_c*2 + j.
                # w_4 index = i_sub * 4 + k_sub.
                # This matches the convention W[k*n+i] if we interpret the index as k*4+i?
                # No, W[k*n+i] means index = k*n + i.
                # So if w_4 index is `idx_w`, then `idx_w` = k_sub * 4 + i_sub.
                # So I can just use `idx_w` to find k_sub and i_sub?
                # No, I just need to put the value in W[k_sub * 5 + i_sub].
                # Since idx_w = k_sub * 4 + i_sub, I can recover k_sub = idx_w // 4, i_sub = idx_w % 4.
                
                # Wait, my generator built w_4 such that w_4[global_idx] += ...
                # where global_idx = (r_c*2+i)*4 + (c_c*2+j).
                # This is i_sub * 4 + k_sub.
                # So for each non-zero in w_4, I need to map it.
                
                pass # Handled in loop below

    # Redo the loop correctly
    col = 0
    for u_4, v_4, w_4 in terms_4x4:
        for idx_u in range(16):
            val = u_4[idx_u]
            if val != 0:
                i_sub = idx_u // 4
                j_sub = idx_u % 4
                U[i_sub * 5 + j_sub][col] = val
        
        for idx_v in range(16):
            val = v_4[idx_v]
            if val != 0:
                j_sub = idx_v // 4
                k_sub = idx_v % 4
                V[j_sub * 5 + k_sub][col] = val
        
        for idx_w in range(16):
            val = w_4[idx_w]
            if val != 0:
                # idx_w = i_sub * 4 + k_sub
                i_sub = idx_w // 4
                k_sub = idx_w % 4
                W[k_sub * 5 + i_sub][col] = val
                
        col += 1
        
    # 2. Schoolbook for remaining parts
    # C[i, 4] for i in 0..3
    for i in range(4):
        for j in range(5):
            U[i * 5 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 5 + i][col] = 1
            col += 1
            
    # C[4, k] for k in 0..3
    for k in range(4):
        for j in range(5):
            U[4 * 5 + j][col] = 1
            V[j * 5 + k][col] = 1
            W[k * 5 + 4][col] = 1
            col += 1
            
    # C[4, 4]
    for j in range(5):
        U[4 * 5 + j][col] = 1
        V[j * 5 + 4][col] = 1
        W[4 * 5 + 4][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}

def generate_strassen_4x4():
    # ... (code from above) ...
    pass
# EVOLVE-BLOCK-END
