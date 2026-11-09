import numpy as np
import random
import time
import math

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Target tensor indices
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target_indices.append((u, v, w))
                
    num_target = len(target_indices)
    
    # Precompute target tensor as a sparse dict or dense array
    T_target = np.zeros((dim_u, dim_v, dim_w), dtype=int)
    for u, v, w in target_indices:
        T_target[u, v, w] = 1

    def get_error(U, V, W):
        """Compute the number of mismatched entries in the reconstructed tensor."""
        r = U.shape[1]
        T_recon = np.zeros((dim_u, dim_v, dim_w), dtype=int)
        for c in range(r):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            T_recon += uc[:, None, None] * vc[None, :, None] * wc[None, None, :]
        
        # Error is the number of positions where T_recon != T_target
        return np.sum(T_recon != T_target)

    def is_valid_solution(U, V, W):
        r = U.shape[1]
        T_recon = np.zeros((dim_u, dim_v, dim_w), dtype=int)
        for c in range(r):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            T_recon += uc[:, None, None] * vc[None, :, None] * wc[None, None, :]
        return np.all(T_recon == T_target)

    start_time = time.time()
    max_time = 110  # Leave 10s buffer
    
    # We will try to find a solution with rank r < 24.
    # Start from r=20 and go up to 23.
    # Use Simulated Annealing for each rank.
    
    best_solution = None
    best_rank = None
    
    for r in range(20, 24):
        if time.time() - start_time > max_time:
            break
            
        # Run several SA restarts for this rank
        for restart in range(20):
            if time.time() - start_time > max_time:
                break
            
            # Initialize with random small integers
            # Sparse initialization helps
            U = np.zeros((dim_u, r), dtype=int)
            V = np.zeros((dim_v, r), dtype=int)
            W = np.zeros((dim_w, r), dtype=int)
            
            # Fill with random -1, 0, 1
            density = 0.3
            U[np.random.rand(dim_u, r) < density] = np.random.choice([-1, 1], size=np.sum(np.random.rand(dim_u, r) < density))
            V[np.random.rand(dim_v, r) < density] = np.random.choice([-1, 1], size=np.sum(np.random.rand(dim_v, r) < density))
            W[np.random.rand(dim_w, r) < density] = np.random.choice([-1, 1], size=np.sum(np.random.rand(dim_w, r) < density))
            
            current_err = get_error(U, V, W)
            best_err = current_err
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            
            # Simulated Annealing
            T = 50.0
            cooling = 0.95
            min_T = 0.1
            steps = 2000
            
            for step in range(steps):
                if time.time() - start_time > max_time:
                    break
                if T < min_T:
                    break
                    
                # Random Move: Change one element in one matrix
                mat_idx = np.random.randint(3)
                if mat_idx == 0:
                    mat = U
                    dims = (dim_u, r)
                elif mat_idx == 1:
                    mat = V
                    dims = (dim_v, r)
                else:
                    mat = W
                    dims = (dim_w, r)
                    
                row = np.random.randint(dims[0])
                col = np.random.randint(dims[1])
                
                old_val = mat[row, col]
                # Try a new random value from a small set
                new_val = np.random.choice([-2, -1, 0, 1, 2])
                
                if new_val == old_val:
                    continue
                    
                mat[row, col] = new_val
                new_err = get_error(U, V, W)
                
                delta = new_err - current_err
                if delta <= 0 or np.random.random() < math.exp(-delta / T):
                    current_err = new_err
                    if current_err < best_err:
                        best_err = current_err
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                        if best_err == 0:
                            # Found a valid solution
                            best_rank = r
                            best_solution = (best_U, best_V, best_W)
                            break
                else:
                    mat[row, col] = old_val
                
                T *= cooling
                
            if best_solution is not None:
                break
                
        if best_solution is not None:
            break
    
    if best_solution is not None:
        U, V, W = best_solution
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fall back to schoolbook
        rank = n * m * p
        U = np.zeros((dim_u, rank), dtype=int)
        V = np.zeros((dim_v, rank), dtype=int)
        W = np.zeros((dim_w, rank), dtype=int)
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    U[u, column] = 1
                    V[v, column] = 1
                    W[w, column] = 1
                    column += 1
        
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
# EVOLVE-BLOCK-END
