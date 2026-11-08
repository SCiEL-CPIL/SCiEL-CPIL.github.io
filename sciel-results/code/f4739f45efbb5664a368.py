import numpy as np
import time
import random

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Start with schoolbook rank
    rank_schoolbook = n * m * p
    
    # Try to find a lower rank
    # We will search for ranks from 15 to rank_schoolbook
    # The prompt implies integer/half-integer entries, which suggests a discrete search
    # might be effective or that the optimal solution has such entries.
    
    # We'll use a randomized local search on a grid of half-integers.
    # To keep it fast, we'll limit the search time.
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_U, best_V, best_W = None, None, None
    best_rank = rank_schoolbook
    best_error = float('inf')
    
    # Initialize with schoolbook
    U0 = np.zeros((n*m, rank_schoolbook))
    V0 = np.zeros((m*p, rank_schoolbook))
    W0 = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
                
    # Function to compute error
    def compute_error(U, V, W, r):
        # Reshape U, V, W to 3D tensors for faster dot product?
        # U: (n, m, r), V: (m, p, r), W: (p, n, r)
        # Sum over c of U[i,j,c] * V[j,k,c] * W[k,i,c]
        # We want this sum to be 1 for all i,j,k.
        # Error = sum( (sum_c ... - 1)^2 )
        
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        
        # Compute tensor product sum
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # This is a bit tricky to vectorize simply without einsum
        # np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        
        # Target is all 1s
        target = np.ones((n, m, p))
        error = np.sum((T - target) ** 2)
        return error

    # Check schoolbook
    err0 = compute_error(U0, V0, W0, rank_schoolbook)
    if err0 < 1e-9:
        best_error = err0
        best_U, best_V, best_W = U0, V0, W0
        best_rank = rank_schoolbook

    # Try lower ranks
    # We'll try ranks 15, 16, 17
    for r in range(15, rank_schoolbook + 1):
        if time.time() > deadline - 5:
            break
            
        # Try a few random restarts for this rank
        max_restarts = 10
        for restart in range(max_restarts):
            if time.time() > deadline - 2:
                break
                
            # Initialize U, V, W with small random integers/half-integers
            # Scale to avoid overflow/underflow
            # Start with random values in [-2, 2] step 0.5
            U = (np.random.randint(-4, 5, size=(n*m, r)) * 0.5)
            V = (np.random.randint(-4, 5, size=(m*p, r)) * 0.5)
            W = (np.random.randint(-4, 5, size=(p*n, r)) * 0.5)
            
            err = compute_error(U, V, W, r)
            
            # Local search
            for step in range(1000):
                if time.time() > deadline - 1:
                    break
                
                # Pick a random matrix and index
                mat_idx = np.random.randint(0, 3)
                if mat_idx == 0:
                    row, col = np.random.randint(0, n*m), np.random.randint(0, r)
                    # Perturb U
                    old_val = U[row, col]
                    # Try +/- 0.5
                    delta = np.random.choice([-0.5, 0.5])
                    U[row, col] = old_val + delta
                    new_err = compute_error(U, V, W, r)
                    if new_err < err:
                        err = new_err
                    else:
                        U[row, col] = old_val
                elif mat_idx == 1:
                    row, col = np.random.randint(0, m*p), np.random.randint(0, r)
                    old_val = V[row, col]
                    delta = np.random.choice([-0.5, 0.5])
                    V[row, col] = old_val + delta
                    new_err = compute_error(U, V, W, r)
                    if new_err < err:
                        err = new_err
                    else:
                        V[row, col] = old_val
                else:
                    row, col = np.random.randint(0, p*n), np.random.randint(0, r)
                    old_val = W[row, col]
                    delta = np.random.choice([-0.5, 0.5])
                    W[row, col] = old_val + delta
                    new_err = compute_error(U, V, W, r)
                    if new_err < err:
                        err = new_err
                    else:
                        W[row, col] = old_val
                
                if err < 1e-9:
                    break
            
            if err < best_error:
                best_error = err
                best_U, best_V, best_W = U, V, W
                best_rank = r
                
                if best_error < 1e-9:
                    # Found exact solution
                    break
        
        if best_error < 1e-9:
            break

    # Return the best found
    # Convert to lists of lists as required
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    # Ensure entries are exactly half-integers if they are close
    # The problem asks for integer/half-integer entries
    def clean_list(lst):
        cleaned = []
        for row in lst:
            new_row = []
            for val in row:
                # Round to nearest 0.5
                nearest = round(val * 2) / 2
                new_row.append(nearest)
            cleaned.append(new_row)
        return cleaned

    U_list = clean_list(U_list)
    V_list = clean_list(V_list)
    W_list = clean_list(W_list)

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
