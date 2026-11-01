import time
import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Target rank is 14
    target_rank = 14
    
    # Dimensions of U, V, W
    dim_u = n * m  # 6
    dim_v = m * p  # 12
    dim_w = p * n  # 8
    
    # Fast error computation in scaled integer space
    # We search for integer matrices U_int, V_int, W_int
    # Such that sum_c U_int[i,j,c] * V_int[j,k,c] * W_int[k,i,c] = 8
    # The actual solution is U = U_int / 2, etc.
    
    def compute_error_int(U_int, V_int, W_int):
        """
        Compute sum of squared errors for the scaled integer problem.
        Target value is 8.
        """
        r = U_int.shape[1]
        U_mat = U_int.reshape(n, m, r)
        V_mat = V_int.reshape(m, p, r)
        W_mat = W_int.reshape(p, n, r)
        
        # Vectorized over c for each (i,j,k)
        # n*m*p is 24, which is small.
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = U_mat[i, j, :]
                    v = V_mat[j, k, :]
                    w = W_mat[k, i, :]
                    val = np.dot(u, v * w)
                    error += (val - 8.0) ** 2
        return error

    def is_valid_int(U_int, V_int, W_int):
        return compute_error_int(U_int, V_int, W_int) == 0.0

    # Fallback schoolbook solution (Rank 24)
    # In integer space, we need sum = 8.
    # Schoolbook: U=1, V=1, W=1 -> sum = 1.
    # We need sum = 8. So we can scale one of them by 8?
    # If U=8, V=1, W=1 -> sum = 8.
    # Or U=2, V=2, W=2 -> sum = 8.
    # Let's use U=2, V=2, W=2 for the schoolbook basis.
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school), dtype=int)
    V_school = np.zeros((m * p, rank_school), dtype=int)
    W_school = np.zeros((p * n, rank_school), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 2
                V_school[j * p + k, col] = 2
                W_school[k * n + i, col] = 2
                col += 1
    # Verify: sum_c 2*2*2 = 8. Correct.
    
    best_solution_int = (U_school, V_school, W_school)
    best_rank = rank_school
    
    # Search for rank 14
    if time.time() < deadline:
        np.random.seed(42)
        
        max_restarts = 2000 # Increased due to fast error calc
        max_iterations = 100
        
        for attempt in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Random initialization with small integers
            # Values in {-2, -1, 0, 1, 2}
            U_int = np.random.randint(-2, 3, size=(dim_u, target_rank))
            V_int = np.random.randint(-2, 3, size=(dim_v, target_rank))
            W_int = np.random.randint(-2, 3, size=(dim_w, target_rank))
            
            current_error = compute_error_int(U_int, V_int, W_int)
            
            # Local search
            improved = True
            iter_count = 0
            while improved and time.time() < deadline and iter_count < max_iterations:
                improved = False
                iter_count += 1
                
                # Pick a random matrix and element
                mat_choice = np.random.randint(0, 3)
                if mat_choice == 0:
                    row = np.random.randint(0, dim_u)
                    col_idx = np.random.randint(0, target_rank)
                    old_val = U_int[row, col_idx]
                    # Try deltas in {-2, -1, 1, 2}
                    for delta in [-2, -1, 1, 2]:
                        new_val = old_val + delta
                        # Keep values small to avoid overflow or divergence
                        if abs(new_val) > 4: continue
                        U_int[row, col_idx] = new_val
                        new_error = compute_error_int(U_int, V_int, W_int)
                        if new_error < current_error:
                            current_error = new_error
                            improved = True
                            break
                        else:
                            U_int[row, col_idx] = old_val
                            
                elif mat_choice == 1:
                    row = np.random.randint(0, dim_v)
                    col_idx = np.random.randint(0, target_rank)
                    old_val = V_int[row, col_idx]
                    for delta in [-2, -1, 1, 2]:
                        new_val = old_val + delta
                        if abs(new_val) > 4: continue
                        V_int[row, col_idx] = new_val
                        new_error = compute_error_int(U_int, V_int, W_int)
                        if new_error < current_error:
                            current_error = new_error
                            improved = True
                            break
                        else:
                            V_int[row, col_idx] = old_val
                            
                else:
                    row = np.random.randint(0, dim_w)
                    col_idx = np.random.randint(0, target_rank)
                    old_val = W_int[row, col_idx]
                    for delta in [-2, -1, 1, 2]:
                        new_val = old_val + delta
                        if abs(new_val) > 4: continue
                        W_int[row, col_idx] = new_val
                        new_error = compute_error_int(U_int, V_int, W_int)
                        if new_error < current_error:
                            current_error = new_error
                            improved = True
                            break
                        else:
                            W_int[row, col_idx] = old_val
                
                if current_error == 0.0:
                    # Found a solution!
                    best_rank = target_rank
                    best_solution_int = (U_int, V_int, W_int)
                    break
            
            if best_rank == target_rank:
                break
        
    U_int, V_int, W_int = best_solution_int
    
    # Convert back to float (half-integers)
    U = U_int / 2.0
    V = V_int / 2.0
    W = W_int / 2.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
