"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Try to find a low-rank decomposition using discrete local search
    # We assume entries are in {-1, 0, 1}
    for rank in range(80, 90):
        for seed in range(10):
            result = _local_search_decompose(n, m, p, rank, seed)
            if result is not None:
                return result
                
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _local_search_decompose(n, m, p, rank, seed):
    np.random.seed(seed)
    nm, mp, pn = n * m, m * p, p * n
    
    # Initialize U, V, W with random values in {-1, 0, 1}
    U = np.random.choice([-1, 0, 1], size=(nm, rank))
    V = np.random.choice([-1, 0, 1], size=(mp, rank))
    W = np.random.choice([-1, 0, 1], size=(pn, rank))
    
    # Precompute S[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    S = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                S[i, j, k] = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                
    current_error = np.sum((1.0 - S)**2)
    
    if current_error < 1e-9:
        return _to_list(U, V, W)
        
    max_iter = 50000
    no_improve_count = 0
    max_no_improve = 2000
    
    for it in range(max_iter):
        improved = False
        
        # Try to improve by changing a single entry
        # We pick a random variable type (U, V, or W) and index
        var_type = np.random.choice(['U', 'V', 'W'])
        
        if var_type == 'U':
            idx_i = np.random.randint(n)
            idx_j = np.random.randint(m)
            idx_r = np.random.randint(rank)
            flat_idx = idx_i * m + idx_j
            
            # Try changing U[flat_idx, idx_r] to other values in {-1, 0, 1}
            current_val = U[flat_idx, idx_r]
            best_val = current_val
            best_err = current_error
            
            for new_val in [-1, 0, 1]:
                if new_val == current_val:
                    continue
                delta = new_val - current_val
                # Update S[i, j, :]
                # S[i, j, k] += delta * V[j, k, idx_r] * W[k, i, idx_r]
                # Note: V index is j*p+k, W index is k*n+i
                # We need to compute the change in error for all k
                old_S_vals = S[idx_i, idx_j, :]
                # Vectorized update for S
                # V_slice = V[idx_j*p : idx_j*p+p, idx_r]
                # W_slice = W[idx_r*n? No, W is (pn, rank). W[k*n+i, idx_r]
                # Indices for W: [0*n+i, 1*n+i, ..., (p-1)*n+i]
                w_indices = np.arange(p) * n + idx_i
                v_slice = V[idx_j*p : idx_j*p+p, idx_r]
                w_slice = W[w_indices, idx_r]
                
                new_S_vals = old_S_vals + delta * v_slice * w_slice
                
                old_err_part = np.sum((1.0 - old_S_vals)**2)
                new_err_part = np.sum((1.0 - new_S_vals)**2)
                new_err = current_error - old_err_part + new_err_part
                
                if new_err < best_err - 1e-9:
                    best_err = new_err
                    best_val = new_val
                    best_new_S_vals = new_S_vals
            
            if best_val != current_val:
                U[flat_idx, idx_r] = best_val
                S[idx_i, idx_j, :] = best_new_S_vals
                current_error = best_err
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1

        elif var_type == 'V':
            idx_j = np.random.randint(m)
            idx_k = np.random.randint(p)
            idx_r = np.random.randint(rank)
            flat_idx = idx_j * p + idx_k
            
            current_val = V[flat_idx, idx_r]
            best_val = current_val
            best_err = current_error
            
            for new_val in [-1, 0, 1]:
                if new_val == current_val:
                    continue
                delta = new_val - current_val
                # S[i, j, k] += delta * U[i, j, idx_r] * W[k, i, idx_r]
                # Affected i in [0, n)
                old_S_vals = S[:, idx_j, idx_k]
                u_indices = np.arange(n) * m + idx_j
                u_slice = U[u_indices, idx_r]
                w_indices = np.arange(p) * n + np.arange(n) # This is wrong, W index depends on i and k
                # W index for (k, i) is k*n + i.
                # We fixed j, k, r. We vary i.
                # W[k*n + i, idx_r]
                # So for each i, W_idx = idx_k * n + i
                w_slice = W[idx_k * n : idx_k * n + n, idx_r]
                
                new_S_vals = old_S_vals + delta * u_slice * w_slice
                
                old_err_part = np.sum((1.0 - old_S_vals)**2)
                new_err_part = np.sum((1.0 - new_S_vals)**2)
                new_err = current_error - old_err_part + new_err_part
                
                if new_err < best_err - 1e-9:
                    best_err = new_err
                    best_val = new_val
                    best_new_S_vals = new_S_vals
            
            if best_val != current_val:
                V[flat_idx, idx_r] = best_val
                S[:, idx_j, idx_k] = best_new_S_vals
                current_error = best_err
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1

        else: # W
            idx_k = np.random.randint(p)
            idx_i = np.random.randint(n)
            idx_r = np.random.randint(rank)
            flat_idx = idx_k * n + idx_i
            
            current_val = W[flat_idx, idx_r]
            best_val = current_val
            best_err = current_error
            
            for new_val in [-1, 0, 1]:
                if new_val == current_val:
                    continue
                delta = new_val - current_val
                # S[i, j, k] += delta * U[i, j, idx_r] * V[j, k, idx_r]
                # Affected j in [0, m)
                old_S_vals = S[idx_i, :, idx_k]
                u_indices = np.arange(m) # U index is i*m + j
                u_slice = U[idx_i * m : idx_i * m + m, idx_r]
                v_indices = np.arange(m) * p + idx_k
                v_slice = V[v_indices, idx_r]
                
                new_S_vals = old_S_vals + delta * u_slice * v_slice
                
                old_err_part = np.sum((1.0 - old_S_vals)**2)
                new_err_part = np.sum((1.0 - new_S_vals)**2)
                new_err = current_error - old_err_part + new_err_part
                
                if new_err < best_err - 1e-9:
                    best_err = new_err
                    best_val = new_val
                    best_new_S_vals = new_S_vals
            
            if best_val != current_val:
                W[flat_idx, idx_r] = best_val
                S[idx_i, :, idx_k] = best_new_S_vals
                current_error = best_err
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1

        if current_error < 1e-9:
            return _to_list(U, V, W)
            
        if no_improve_count > max_no_improve:
            break
            
    return None

def _to_list(U, V, W):
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
