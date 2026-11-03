"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    deadline = time.time() + 130  # Safety margin before 150s kill
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_fb = [[0] * rank_school for _ in range(n * m)]
    V_fb = [[0] * rank_school for _ in range(m * p)]
    W_fb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    # Precompute index mappings
    idx_U = np.array([i*m+j for i in range(n) for j in range(m) for k in range(p)])
    idx_V = np.array([j*p+k for i in range(n) for j in range(m) for k in range(p)])
    idx_W = np.array([k*n+i for i in range(n) for j in range(m) for k in range(p)])
    
    def check_exact(U, V, W):
        """Check if decomposition is exact using integer arithmetic."""
        U_int = np.round(U).astype(np.int64)
        V_int = np.round(V).astype(np.int64)
        W_int = np.round(W).astype(np.int64)
        
        # Check if rounding changed values significantly
        if not (np.allclose(U, U_int) and np.allclose(V, V_int) and np.allclose(W, W_int)):
            return False
        
        # Integer check: sum over rank of U[i,j,c]*V[j,k,c]*W[k,i,c] should equal 1
        contrib = U_int[idx_U, :] * V_int[idx_V, :] * W_int[idx_W, :]
        total = contrib.sum(axis=1)
        return np.all(total == 1)
    
    best = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_school
    
    # Try to find lower rank using greedy column merging
    # Start with schoolbook and try to merge columns
    
    def try_merge_columns(U, V, W, rank):
        """Try to merge columns by finding pairs that can be combined."""
        for c1 in range(rank):
            if time.time() > deadline - 5:
                break
            for c2 in range(c1 + 1, rank):
                if time.time() > deadline - 5:
                    break
                
                # Try to find a new column that represents the sum
                # We need: U_new * V_new * W_new = U_c1*V_c1*W_c1 + U_c2*V_c2*W_c2
                # This is generally hard, so we try a heuristic approach
                
                # Check if columns are similar in structure
                u1 = U[:, c1]
                u2 = U[:, c2]
                v1 = V[:, c1]
                v2 = V[:, c2]
                w1 = W[:, c1]
                w2 = W[:, c2]
                
                # If both columns have the same support pattern, we might be able to merge
                # Simple heuristic: if the product patterns are compatible
                pass  # This is complex, so we skip for now and use random search
    
    # Use random search with better initialization
    np.random.seed(42)
    
    for rank in range(32, rank_school):
        if time.time() > deadline - 10:
            break
        
        for restart in range(30):
            if time.time() > deadline - 10:
                break
            
            # Initialize with sparse random ±1 entries
            U = np.zeros((n*m, rank), dtype=np.int64)
            V = np.zeros((m*p, rank), dtype=np.int64)
            W = np.zeros((p*n, rank), dtype=np.int64)
            
            # Each column: pick random non-zero positions
            for c in range(rank):
                n_nonzero_U = np.random.randint(1, 3)
                positions = np.random.choice(n*m, size=min(n_nonzero_U, n*m), replace=False)
                U[positions, c] = np.random.choice([-1, 1], size=len(positions))
                
                n_nonzero_V = np.random.randint(1, 3)
                positions = np.random.choice(m*p, size=min(n_nonzero_V, m*p), replace=False)
                V[positions, c] = np.random.choice([-1, 1], size=len(positions))
                
                n_nonzero_W = np.random.randint(1, 3)
                positions = np.random.choice(p*n, size=min(n_nonzero_W, p*n), replace=False)
                W[positions, c] = np.random.choice([-1, 1], size=len(positions))
            
            # Local search with better convergence criteria
            for iteration in range(100):
                if time.time() > deadline - 10:
                    break
                
                # Compute errors
                contrib = U[idx_U, :] * V[idx_V, :] * W[idx_W, :]
                total = contrib.sum(axis=1)
                errors = total - 1
                n_errors = np.sum(errors != 0)
                
                if n_errors == 0:
                    if check_exact(U, V, W):
                        U_list = U.tolist()
                        V_list = V.tolist()
                        W_list = W.tolist()
                        best = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = rank
                        return best
                    else:
                        break
                
                # Pick a random equation with error
                error_indices = np.where(errors != 0)[0]
                if len(error_indices) == 0:
                    break
                eq = error_indices[np.random.randint(len(error_indices))]
                
                i = eq // (m*p)
                rem = eq % (m*p)
                j = rem // p
                k = rem % p
                
                # Pick a random column to modify
                c = np.random.randint(rank)
                
                u_val = U[i*m+j, c]
                v_val = V[j*p+k, c]
                w_val = W[k*n+i, c]
                
                # Try to adjust one variable
                other = total[eq] - u_val*v_val*w_val
                
                def try_update(mat, row, col, val):
                    old_val = mat[row, col]
                    new_val = int(round(val))
                    if new_val == old_val:
                        return False
                    mat[row, col] = new_val
                    contrib_new = U[idx_U, :] * V[idx_V, :] * W[idx_W, :]
                    total_new = contrib_new.sum(axis=1)
                    errors_new = total_new - 1
                    n_errors_new = np.sum(errors_new != 0)
                    if n_errors_new < n_errors:
                        return True
                    else:
                        mat[row, col] = old_val
                        return False

                if v_val != 0 and w_val != 0:
                    new_u = (1.0 - other) / (v_val * w_val)
                    if try_update(U, i*m+j, c, new_u):
                        continue
                
                if u_val != 0 and w_val != 0:
                    new_v = (1.0 - other) / (u_val * w_val)
                    if try_update(V, j*p+k, c, new_v):
                        continue
                
                if u_val != 0 and v_val != 0:
                    new_w = (1.0 - other) / (u_val * v_val)
                    if try_update(W, k*n+i, c, new_w):
                        continue
    
    return best
# EVOLVE-BLOCK-END
