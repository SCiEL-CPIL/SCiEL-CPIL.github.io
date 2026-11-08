import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    n, m, p = DIMENSIONS
    
    # Construct rank-97 decomposition
    rank = 97
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 4x4x4 (Rank 49)
    u_in = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    v_in = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    w_out = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    u_out = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    v_out = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    w_outer = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    col = 0
    for k in range(7):
        for l in range(7):
            if time.time() > deadline:
                break
            u_vec = [0] * 16
            for i_inner in range(2):
                for j_inner in range(2):
                    coeff_inner = u_in[l][i_inner * 2 + j_inner]
                    if coeff_inner == 0: continue
                    for b in range(4):
                        coeff_outer = u_out[k][b]
                        if coeff_outer == 0: continue
                        b_row = b // 2
                        b_col = b % 2
                        i_global = 2 * b_row + i_inner
                        j_global = 2 * b_col + j_inner
                        u_vec[i_global * 4 + j_global] += coeff_inner * coeff_outer
            
            v_vec = [0] * 28
            for i_inner in range(2):
                for j_inner in range(2):
                    coeff_inner = v_in[l][i_inner * 2 + j_inner]
                    if coeff_inner == 0: continue
                    for b in range(4):
                        coeff_outer = v_out[k][b]
                        if coeff_outer == 0: continue
                        b_row = b // 2
                        b_col = b % 2
                        j_global = 2 * b_row + i_inner
                        k_global = 2 * b_col + j_inner
                        v_vec[j_global * 7 + k_global] += coeff_inner * coeff_outer
            
            w_vec = [0] * 28
            for c_block in range(4):
                coeff_outer_w = w_outer[c_block][k]
                if coeff_outer_w == 0: continue
                for c_inner in range(4):
                    coeff_inner_w = w_out[c_inner][l]
                    if coeff_inner_w == 0: continue
                    total_coeff = coeff_outer_w * coeff_inner_w
                    if total_coeff == 0: continue
                    
                    c_block_row = c_block // 2
                    c_block_col = c_block % 2
                    c_inner_row = c_inner // 2
                    c_inner_col = c_inner % 2
                    
                    i_global = 2 * c_block_row + c_inner_row
                    k_global = 2 * c_block_col + c_inner_col
                    
                    w_vec[k_global * 4 + i_global] += total_coeff
            
            for r in range(16):
                U[r][col] = u_vec[r]
            for r in range(28):
                V[r][col] = v_vec[r]
            for r in range(28):
                W[r][col] = w_vec[r]
            
            col += 1
        if time.time() > deadline:
            break
    
    # Schoolbook 4x4x3 (Rank 48)
    for i in range(4):
        for j in range(4):
            for k_prime in range(3):
                if time.time() > deadline:
                    break
                k_global = 4 + k_prime
                U[i * 4 + j][col] = 1
                V[j * 7 + k_global][col] = 1
                W[k_global * 4 + i][col] = 1
                col += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Local search: try to remove terms
    # Check if the decomposition is valid
    def is_valid(U, V, W, n, m, p):
        # T_{(i,k), (i,j), (j,k)} = 1
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    sum_val = 0
                    for r in range(len(U[0])):
                        sum_val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if sum_val != 1:
                        return False
        return True
    
    # Try to remove each term
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    for r_to_remove in range(rank):
        if time.time() > deadline:
            break
        
        # Create new matrices without term r_to_remove
        new_rank = rank - 1
        new_U = [[0] * new_rank for _ in range(n * m)]
        new_V = [[0] * new_rank for _ in range(m * p)]
        new_W = [[0] * new_rank for _ in range(p * n)]
        
        for r in range(rank):
            if r == r_to_remove:
                continue
            new_r = r if r < r_to_remove else r - 1
            for i in range(n * m):
                new_U[i][new_r] = U[i][r]
            for j in range(m * p):
                new_V[j][new_r] = V[j][r]
            for k in range(p * n):
                new_W[k][new_r] = W[k][r]
        
        if is_valid(new_U, new_V, new_W, n, m, p):
            best_rank = new_rank
            best_U, best_V, best_W = new_U, new_V, new_W
            # If we found a valid decomposition, we can try to remove more?
            # But this is a single step.
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
