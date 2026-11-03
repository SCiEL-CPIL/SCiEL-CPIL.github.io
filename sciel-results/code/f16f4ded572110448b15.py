import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Target dimensions for U, V, W
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Precompute indices for the tensor
    # T[i, j, k] = 1
    # We want sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # We will try ranks from rank_schoolbook - 1 down to some limit
    # We use a time budget
    start_time = time.time()
    time_limit = 140.0  # Leave some margin
    
    best_rank = rank_schoolbook
    best_solution = None
    
    # Function to check if a solution is exact
    def check_solution(U, V, W):
        # U: (dim_U, r), V: (dim_V, r), W: (dim_W, r)
        # Compute sum_c U_ij,c V_jk,c W_ki,c
        # We can do this efficiently by reshaping and contracting
        # But for small sizes, a loop or einsum is fine
        # Let's use einsum
        # U: (i, j, c) -> reshape to (n, m, r)
        # V: (j, k, c) -> reshape to (m, p, r)
        # W: (k, i, c) -> reshape to (p, n, r)
        
        U_reshaped = U.reshape(n, m, -1)
        V_reshaped = V.reshape(m, p, -1)
        W_reshaped = W.reshape(p, n, -1)
        
        # Contract over c
        # Result shape (n, m, p, n) -> we want (n, m, p) to be 1
        # Actually, the sum is over c. The result is a tensor of shape (n, m, p, n)?
        # No, U depends on i,j. V on j,k. W on k,i.
        # The product U[i,j,c] V[j,k,c] W[k,i,c] sums to a value for each i,j,k.
        # Let's use np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        try:
            result = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        except Exception:
            return False, 1e10
            
        # Check if result is all 1s
        # Allow small floating point error if we are checking continuous solutions
        # But for final answer, we need exact integers/half-integers
        # Here we check for exactness with integers
        if np.allclose(result, 1.0, atol=1e-9):
            return True, 0.0
        else:
            error = np.sum((result - 1.0)**2)
            return False, error

    # Function to run local search for a given rank
    def search_for_rank(r):
        # Initialize with random small integers
        # We can start with a random sparse matrix or just random
        U = np.random.randint(-2, 3, size=(dim_U, r))
        V = np.random.randint(-2, 3, size=(dim_V, r))
        W = np.random.randint(-2, 3, size=(dim_W, r))
        
        # Use a simple hill climbing / local search
        # We will try to reduce the error by flipping signs or adding 1/-1
        
        current_error = 1e10
        is_exact, current_error = check_solution(U, V, W)
        if is_exact:
            return U, V, W, True
            
        max_iters = 2000
        for _ in range(max_iters):
            # Pick a random matrix and element
            mat_idx = np.random.randint(0, 3)
            if mat_idx == 0:
                mat = U
            elif mat_idx == 1:
                mat = V
            else:
                mat = W
                
            row = np.random.randint(0, mat.shape[0])
            col = np.random.randint(0, mat.shape[1])
            
            # Try changing value by -1, 0, 1
            original_val = mat[row, col]
            best_val = original_val
            best_err = current_error
            
            for delta in [-2, -1, 1, 2]:
                new_val = original_val + delta
                # Keep values small
                if abs(new_val) > 5:
                    continue
                mat[row, col] = new_val
                is_exact, err = check_solution(U, V, W)
                if err < best_err:
                    best_err = err
                    best_val = new_val
                    if is_exact:
                        mat[row, col] = new_val
                        return U, V, W, True
            mat[row, col] = best_val
            current_error = best_err
            
            if current_error < 1e-9:
                is_exact, _ = check_solution(U, V, W)
                if is_exact:
                    return U, V, W, True
                    
        return U, V, W, False

    # Try ranks
    for r in range(rank_schoolbook - 1, 30, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Run a few restarts for this rank
        found = False
        for restart in range(5):
            if time.time() - start_time > time_limit:
                break
            U, V, W, is_exact = search_for_rank(r)
            if is_exact:
                # Convert to lists
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    # If no better solution found, return schoolbook
    U = np.zeros((dim_U, rank_schoolbook), dtype=int)
    V = np.zeros((dim_V, rank_schoolbook), dtype=int)
    W = np.zeros((dim_W, rank_schoolbook), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
