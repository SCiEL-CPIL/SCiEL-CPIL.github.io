"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import random

def construct():
    """Return a rank-minimal exact tensor decomposition for 3x5 by 5x5 matrix multiplication."""
    n, m, p = DIMENSIONS  # 3, 5, 5
    deadline = time.time() + 120  # 120s wall-clock limit
    
    # Schoolbook decomposition: rank = n * m * p = 75
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try to reduce rank by finding linear dependencies
    # Use a simple local search: try removing columns and checking if the decomposition still works
    # This is heuristic and may not find the optimal rank, but can improve on schoolbook
    
    def validate(U_mat, V_mat, W_mat):
        """Check if the decomposition is exact."""
        n, m, p = 3, 5, 5
        rank = len(U_mat[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U_mat[i*m+j][r] * V_mat[j*p+k][r] * W_mat[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    def try_reduce_rank(U_mat, V_mat, W_mat, target_rank):
        """Try to reduce rank to target_rank by zeroing out columns."""
        if time.time() > deadline - 5:
            return None, None, None, False
        
        n, m, p = 3, 5, 5
        current_rank = len(U_mat[0])
        if current_rank <= target_rank:
            return U_mat, V_mat, W_mat, True
        
        # Try to remove (current_rank - target_rank) columns
        # This is a combinatorial problem, so use a greedy approach
        U_new = [row[:] for row in U_mat]
        V_new = [row[:] for row in V_mat]
        W_new = [row[:] for row in W_mat]
        
        # Greedily remove columns that have the smallest impact
        # For simplicity, try removing columns one at a time
        cols_to_remove = []
        for r in range(current_rank - 1, target_rank - 1, -1):
            # Check if removing column r still gives a valid decomposition
            U_test = [row[:] for row in U_new]
            V_test = [row[:] for row in V_new]
            W_test = [row[:] for row in W_new]
            
            # Zero out column r
            for i in range(n*m):
                U_test[i][r] = 0
            for j in range(m*p):
                V_test[j][r] = 0
            for k in range(p*n):
                W_test[k][r] = 0
            
            if validate(U_test, V_test, W_test):
                U_new = U_test
                V_new = V_test
                W_new = W_test
                cols_to_remove.append(r)
        
        if cols_to_remove:
            # Re-index to remove the zeroed columns
            keep_cols = [r for r in range(current_rank) if r not in cols_to_remove]
            U_final = [[U_new[i][r] for r in keep_cols] for i in range(n*m)]
            V_final = [[V_new[j][r] for r in keep_cols] for j in range(m*p)]
            W_final = [[W_new[k][r] for r in keep_cols] for k in range(p*n)]
            return U_final, V_final, W_final, True
        
        return U_mat, V_mat, W_mat, False
    
    # Try to reduce rank from 75 down to 25 (the theoretical lower bound)
    for target_rank in range(rank - 1, 24, -1):
        if time.time() > deadline - 10:
            break
        
        U_new, V_new, W_new, success = try_reduce_rank(best_U, best_V, best_W, target_rank)
        if success:
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = target_rank
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
