"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Rank 42 decomposition based on Strassen's algorithm applied to 2x2 blocks
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a Strassen 2x2x2 decomposition
    def add_strassen_block(a_idx, b_idx, w_idx, start_col):
        U[a_idx[0]][start_col + 0] += 1
        U[a_idx[3]][start_col + 0] += 1
        U[a_idx[2]][start_col + 1] += 1
        U[a_idx[3]][start_col + 1] += 1
        U[a_idx[0]][start_col + 2] += 1
        U[a_idx[3]][start_col + 3] += 1
        U[a_idx[0]][start_col + 4] += 1
        U[a_idx[1]][start_col + 4] += 1
        U[a_idx[2]][start_col + 5] += 1
        U[a_idx[0]][start_col + 5] -= 1
        U[a_idx[1]][start_col + 6] += 1
        U[a_idx[3]][start_col + 6] -= 1
        
        V[b_idx[0]][start_col + 0] += 1
        V[b_idx[3]][start_col + 0] += 1
        V[b_idx[0]][start_col + 1] += 1
        V[b_idx[1]][start_col + 2] += 1
        V[b_idx[3]][start_col + 2] -= 1
        V[b_idx[2]][start_col + 3] += 1
        V[b_idx[0]][start_col + 3] -= 1
        V[b_idx[3]][start_col + 4] += 1
        V[b_idx[0]][start_col + 5] += 1
        V[b_idx[1]][start_col + 5] += 1
        V[b_idx[2]][start_col + 6] += 1
        V[b_idx[3]][start_col + 6] += 1
        
        W[w_idx[0]][start_col + 0] += 1
        W[w_idx[0]][start_col + 3] += 1
        W[w_idx[0]][start_col + 4] -= 1
        W[w_idx[0]][start_col + 6] += 1
        
        W[w_idx[1]][start_col + 2] += 1
        W[w_idx[1]][start_col + 4] += 1
        
        W[w_idx[2]][start_col + 1] += 1
        W[w_idx[2]][start_col + 3] += 1
        
        W[w_idx[3]][start_col + 0] += 1
        W[w_idx[3]][start_col + 1] -= 1
        W[w_idx[3]][start_col + 2] += 1
        W[w_idx[3]][start_col + 5] += 1

    a_L = [0, 1, 4, 5]
    a_R = [2, 3, 6, 7]
    
    b_1_T = [0, 1, 6, 7]
    b_1_B = [12, 13, 18, 19]
    w_1 = [0, 2, 1, 3]
    
    b_2_T = [2, 3, 8, 9]
    b_2_B = [14, 15, 20, 21]
    w_2 = [4, 6, 5, 7]
    
    b_3_T = [4, 5, 10, 11]
    b_3_B = [16, 17, 22, 23]
    w_3 = [8, 10, 9, 11]
    
    add_strassen_block(a_L, b_1_T, w_1, 0)
    add_strassen_block(a_R, b_1_B, w_1, 7)
    
    add_strassen_block(a_L, b_2_T, w_2, 14)
    add_strassen_block(a_R, b_2_B, w_2, 21)
    
    add_strassen_block(a_L, b_3_T, w_3, 28)
    add_strassen_block(a_R, b_3_B, w_3, 35)
    
    # Verify the decomposition
    def verify(U_mat, V_mat, W_mat, n, m, p):
        rank = len(U_mat[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U_mat[i * m + j][r] * V_mat[j * p + k][r] * W_mat[k * n + i][r]
                    if total != 1:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank by checking if any factor column is redundant
    # A factor r is redundant if the tensor equation still holds when we set
    # U[:,r]=0, V[:,r]=0, W[:,r]=0 and the contributions from other factors
    # can compensate. More precisely, we check if removing factor r and
    # adjusting others can maintain the decomposition.
    
    # Simpler approach: check if any factor has all zeros in one of the matrices
    # (meaning it doesn't contribute) - this won't help but is a quick check
    
    # Better approach: try to find a linear dependency. For each factor r,
    # check if the tensor contribution of factor r can be expressed as a linear
    # combination of other factors. This is complex, so let's try a heuristic:
    # check if any factor column is a scalar multiple of another.
    
    # Convert to numpy for linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check for duplicate or proportional factor columns
    # Factor r is proportional to factor s if U[:,r] = a*U[:,s], V[:,r] = b*V[:,s], W[:,r] = c*W[:,s]
    # and a*b*c = 1 (to maintain the product)
    
    def try_remove_factor(U_list, V_list, W_list, r_to_remove, n, m, p):
        """Try removing factor r_to_remove and see if the decomposition still works."""
        rank = len(U_list[0])
        new_rank = rank - 1
        U_new = [row[:r_to_remove] + row[r_to_remove+1:] for row in U_list]
        V_new = [row[:r_to_remove] + row[r_to_remove+1:] for row in V_list]
        W_new = [row[:r_to_remove] + row[r_to_remove+1:] for row in W_list]
        
        # Verify
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(new_rank):
                        total += U_new[i * m + j][r] * V_new[j * p + k][r] * W_new[k * n + i][r]
                    if total != 1:
                        return None
        return {"U": U_new, "V": V_new, "W": W_new}
    
    # Try removing each factor one at a time
    best_result = None
    for r in range(rank):
        if time.time() > deadline:
            break
        result = try_remove_factor(U, V, W, r, n, m, p)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        return best_result
    
    # If no single factor can be removed, try pairs
    # This is more expensive, so only do it if we have time
    if time.time() < deadline - 20:
        for r1 in range(rank):
            for r2 in range(r1 + 1, rank):
                if time.time() > deadline:
                    break
                # Skip if r1 or r2 was already removed in a previous iteration
                # (we're trying from scratch each time)
                rank_curr = rank
                r1_adj = r1
                r2_adj = r2 if r2 > r1 else r2 + 1  # Adjust index after removing r1
                
                U_curr = [row[:] for row in U]
                V_curr = [row[:] for row in V]
                W_curr = [row[:] for row in W]
                
                # Remove r1
                U_curr = [row[:r1_adj] + row[r1_adj+1:] for row in U_curr]
                V_curr = [row[:r1_adj] + row[r1_adj+1:] for row in V_curr]
                W_curr = [row[:r1_adj] + row[r1_adj+1:] for row in W_curr]
                
                # Remove r2 (adjusted)
                r2_final = r2 if r2 > r1 else r2 + 1
                r2_final = r2_final - 1 if r2_final > r1_adj else r2_final
                # Actually, let's just do it more carefully
                U_curr = [row[:] for row in U]
                V_curr = [row[:] for row in V]
                W_curr = [row[:] for row in W]
                
                # Remove both r1 and r2
                indices_to_keep = [i for i in range(rank) if i != r1 and i != r2]
                U_curr = [row[i] for i in indices_to_keep] if False else [ [row[i] for i in indices_to_keep] for row in U ]
                V_curr = [ [row[i] for i in indices_to_keep] for row in V ]
                W_curr = [ [row[i] for i in indices_to_keep] for row in W ]
                
                # Verify
                valid = True
                new_rank = len(indices_to_keep)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            total = 0
                            for r in range(new_rank):
                                total += U_curr[i * m + j][r] * V_curr[j * p + k][r] * W_curr[k * n + i][r]
                            if total != 1:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    return {"U": U_curr, "V": V_curr, "W": W_curr}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
