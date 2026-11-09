import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # Safety margin before 150s kill
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb
    
    def verify_fast(U, V, W):
        """Vectorized check if U, V, W form a valid decomposition."""
        r = U.shape[1]
        # Build index arrays for all (i,j,k) triples
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        
        # Index into U, V, W
        u_idx = i_idx * m + j_idx
        v_idx = j_idx * p + k_idx
        w_idx = k_idx * n + i_idx
        
        # Compute sum: sum_c U[u_idx,c] * V[v_idx,c] * W[w_idx,c]
        # Use einsum for efficiency
        prod = U[u_idx, :] * V[v_idx, :] * W[w_idx, :]
        sums = prod.sum(axis=1)
        
        if np.any(np.abs(sums - 1.0) > 1e-9):
            return False
        return True
    
    def verify(U, V, W):
        """Check if U, V, W form a valid decomposition."""
        return verify_fast(U, V, W)
    
    def to_half_int(x):
        """Round to nearest half-integer."""
        return np.round(x * 2) / 2
    
    def random_decomp(rank, scale=2.0):
        """Generate a random decomposition with integer/half-integer entries."""
        U = np.round(np.random.uniform(-scale, scale, (n * m, rank)) * 2) / 2
        V = np.round(np.random.uniform(-scale, scale, (m * p, rank)) * 2) / 2
        W = np.round(np.random.uniform(-scale, scale, (p * n, rank)) * 2) / 2
        return U, V, W
    
    def local_search(U, V, W, max_iters=500):
        """Local search: try to reduce rank by merging terms."""
        r = U.shape[1]
        U_cur = U.copy()
        V_cur = V.copy()
        W_cur = W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            if r <= 30:  # Lower bound
                break
            
            # Try to remove one column (term)
            best_col = -1
            best_err = float('inf')
            
            for col in range(r):
                if time.time() > deadline - 5:
                    break
                # Temporarily zero out this column
                U_temp = U_cur.copy()
                V_temp = V_cur.copy()
                W_temp = W_cur.copy()
                U_temp[:, col] = 0
                V_temp[:, col] = 0
                W_temp[:, col] = 0
                
                if verify_fast(U_temp, V_temp, W_temp):
                    # This column can be removed!
                    U_cur = np.delete(U_cur, col, axis=1)
                    V_cur = np.delete(V_cur, col, axis=1)
                    W_cur = np.delete(W_cur, col, axis=1)
                    r -= 1
                    best_col = col
                    break
            
            if best_col >= 0:
                continue
            
            # No column could be removed directly. Try perturbing.
            # Pick a random column and try adjusting it
            col = np.random.randint(0, r)
            
            # Try small perturbations on this column
            found_improvement = False
            for delta_u in [-0.5, -0.25, 0.25, 0.5]:
                for delta_v in [-0.5, -0.25, 0.25, 0.5]:
                    for delta_w in [-0.5, -0.25, 0.25, 0.5]:
                        if time.time() > deadline - 5:
                            break
                        U_temp = U_cur.copy()
                        V_temp = V_cur.copy()
                        W_temp = W_cur.copy()
                        U_temp[:, col] += delta_u
                        V_temp[:, col] += delta_v
                        W_temp[:, col] += delta_w
                        
                        # Round to half-integers
                        U_temp = to_half_int(U_temp)
                        V_temp = to_half_int(V_temp)
                        W_temp = to_half_int(W_temp)
                        
                        if verify_fast(U_temp, V_temp, W_temp):
                            U_cur = U_temp
                            V_cur = V_temp
                            W_cur = W_temp
                            found_improvement = True
                            break
                    if found_improvement:
                        break
                if found_improvement:
                    break
            if found_improvement:
                continue
            
            # If no improvement, break
            break
        
        return U_cur, V_cur, W_cur, U_cur.shape[1]
    
    # Strategy 1: Try random search for low ranks
    for target_rank in range(30, min(best_rank, 60)):
        if time.time() > deadline - 15:
            break
        
        # Try random search with multiple attempts
        for attempt in range(200):
            if time.time() > deadline - 10:
                break
            
            U, V, W = random_decomp(target_rank, scale=2.0)
            if verify_fast(U, V, W):
                best_U, best_V, best_W, best_rank = U, V, W, target_rank
                break
        
        if best_rank <= target_rank:
            break
    
    # Strategy 2: Local search from schoolbook
    if time.time() < deadline - 20:
        U_ls, V_ls, W_ls, rank_ls = local_search(U_sb, V_sb, W_sb, max_iters=200)
        if rank_ls < best_rank:
            best_U, best_V, best_W, best_rank = U_ls, V_ls, W_ls, rank_ls
    
    # Strategy 3: Try structured decompositions
    # Use the fact that we can set up a system of equations
    if time.time() < deadline - 10:
        for target_rank in range(30, min(best_rank, 50)):
            if time.time() > deadline - 5:
                break
            
            # Try to solve for W given random U, V
            for attempt in range(100):
                if time.time() > deadline - 5:
                    break
                
                U_rand = np.round(np.random.uniform(-2, 2, (n * m, target_rank)) * 2) / 2
                V_rand = np.round(np.random.uniform(-2, 2, (m * p, target_rank)) * 2) / 2
                
                # For each (i,j,k), we need: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
                # This is linear in W for fixed U,V
                # Set up: for each (i,j,k): sum_c (U[i*m+j,c]*V[j*p+k,c]) * W[k*n+i,c] = 1
                # This is a system of n*m*p equations in p*n*target_rank unknowns
                # But W has structure: W[k*n+i, c] for c in range(target_rank)
                
                # Build the system
                num_eq = n * m * p
                num_unk = p * n * target_rank
                
                A_mat = np.zeros((num_eq, num_unk))
                b_vec = np.ones(num_eq)
                
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for c in range(target_rank):
                                coeff = U_rand[i * m + j, c] * V_rand[j * p + k, c]
                                # W[k*n+i, c] is at row k*n+i, col c
                                col_idx = (k * n + i) * target_rank + c
                                A_mat[eq_idx, col_idx] = coeff
                            eq_idx += 1
                
                # Solve least squares
                try:
                    W_flat, residuals, rank_sys, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    W_sol = W_flat.reshape(p * n, target_rank)
                    W_rounded = to_half_int(W_sol)
                    
                    if verify_fast(U_rand, V_rand, W_rounded):
                        best_U, best_V, best_W, best_rank = U_rand, V_rand, W_rounded, target_rank
                        break
                except:
                    pass
            
            if best_rank <= target_rank:
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
