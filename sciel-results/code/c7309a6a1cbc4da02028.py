import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    target_rank = 26
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # Schoolbook fallback
    def schoolbook():
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

    # Try to find rank 26
    # Variables: U (9x26), V (12x26), W (12x26)
    # We use integers/half-integers.
    # Let's use a local search on a relaxed continuous space first? 
    # Or discrete search.
    
    # Discrete search is safer for exactness.
    # We'll try to find a solution where the sum of products is 1 for all 36 indices.
    
    def calculate_error(U, V, W):
        # U, V, W are numpy arrays
        # Compute sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # We can do this efficiently with einsum or loops
        # Result shape (n, m, p)
        # Res = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        # But indices in W are k*n+i, so W[k,i,c]
        # Let's reshape
        # U: (n, m, rank)
        # V: (m, p, rank)
        # W: (p, n, rank) -> W[k, i, c]
        
        U_reshaped = U.reshape(n, m, -1)
        V_reshaped = V.reshape(m, p, -1)
        W_reshaped = W.reshape(p, n, -1)
        
        # Sum over rank
        # Term[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Note: W index is (k, i).
        
        # Use einsum
        # U: i j c
        # V: j k c
        # W: k i c
        # Result: i j k
        try:
            # This might be slow if done repeatedly, but 26 is small.
            # Let's optimize by precomputing pairs?
            # For now, simple loop over c might be faster in Python if vectorized over i,j,k?
            # No, einsum is best.
            res = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
            target = np.ones((n, m, p))
            diff = res - target
            return np.sum(diff * diff)
        except Exception:
            return 1e9

    def is_valid_solution(U, V, W):
        # Check if entries are integer or half-integer
        # And if error is 0
        if calculate_error(U, V, W) > 1e-9:
            return False
        for mat in [U, V, W]:
            for val in mat:
                # Check if val is integer or half-integer
                # val * 2 should be integer
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
        return True

    # Local Search
    # Initialize with random integers in {-1, 0, 1}
    best_solution = None
    best_error = float('inf')
    
    max_iterations = 5000
    step_size = 0.5 # We only care about half-integers, so step 0.5
    
    for restart in range(20):
        if time.time() > deadline:
            break
            
        U = np.random.randint(-1, 2, (n*m, target_rank))
        V = np.random.randint(-1, 2, (m*p, target_rank))
        W = np.random.randint(-1, 2, (p*n, target_rank))
        
        current_error = calculate_error(U, V, W)
        
        for it in range(max_iterations):
            if time.time() > deadline:
                break
                
            if current_error == 0:
                best_solution = (U.copy(), V.copy(), W.copy())
                break
                
            # Pick a random matrix and a random element
            mat_idx = np.random.randint(0, 3)
            if mat_idx == 0:
                row, col = np.random.randint(0, n*m), np.random.randint(0, target_rank)
                old_val = U[row, col]
                # Try changing by +/- 0.5 or +/- 1
                # Since we want half-integers, we can just try new values from a small set
                # Or just step by 0.5
                new_val = old_val + np.random.choice([-1, 1]) * 0.5
                U[row, col] = new_val
                new_error = calculate_error(U, V, W)
                if new_error < current_error:
                    current_error = new_error
                    if current_error < best_error:
                        best_error = current_error
                        # Save best
                        if best_solution is None or current_error < 1e-9:
                            pass # Just track error
                else:
                    U[row, col] = old_val
            elif mat_idx == 1:
                row, col = np.random.randint(0, m*p), np.random.randint(0, target_rank)
                old_val = V[row, col]
                new_val = old_val + np.random.choice([-1, 1]) * 0.5
                V[row, col] = new_val
                new_error = calculate_error(U, V, W)
                if new_error < current_error:
                    current_error = new_error
                else:
                    V[row, col] = old_val
            else:
                row, col = np.random.randint(0, p*n), np.random.randint(0, target_rank)
                old_val = W[row, col]
                new_val = old_val + np.random.choice([-1, 1]) * 0.5
                W[row, col] = new_val
                new_error = calculate_error(U, V, W)
                if new_error < current_error:
                    current_error = new_error
                else:
                    W[row, col] = old_val
            
            if current_error < 1e-9:
                best_solution = (U.copy(), V.copy(), W.copy())
                break
                
        if best_solution is not None:
            break
            
    if best_solution is not None:
        U, V, W = best_solution
        # Convert to lists of lists
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return schoolbook()
