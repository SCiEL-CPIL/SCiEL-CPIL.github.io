"""Strategy: Combinatorial random search with local repair in the half-integer domain."""
import time
import numpy as np

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135  # Safety margin for 150s limit

    # 1. Schoolbook fallback (Rank 28)
    def schoolbook():
        r = n * m * p
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k, col] = 1
                    V[k * p + j, col] = 1
                    W[j * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, r

    best_result, best_rank = schoolbook()

    # 2. Verification function
    def verify(U, V, W):
        """
        U: (n*m, r)
        V: (m*p, r)
        W: (p*n, r)
        Check if sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] == 1
        """
        r = U.shape[1]
        u_r = U.T.reshape(n, m, r)
        v_r = V.T.reshape(m, p, r)
        w_r = W.T.reshape(p, n, r)
        T = np.einsum('ikr,kjr,jir->ijk', u_r, v_r, w_r)
        return np.allclose(T, 1.0, atol=1e-9)

    # 3. Error function
    def error(U, V, W):
        r = U.shape[1]
        u_r = U.T.reshape(n, m, r)
        v_r = V.T.reshape(m, p, r)
        w_r = W.T.reshape(p, n, r)
        T = np.einsum('ikr,kjr,jir->ijk', u_r, v_r, w_r)
        return np.sum((T - 1.0)**2)

    # 4. Local improvement step (Hill Climbing on half-integers)
    def improve(U, V, W, max_steps=500):
        """
        Try to reduce error by flipping individual half-integer values.
        """
        r = U.shape[1]
        U = U.copy()
        V = V.copy()
        W = W.copy()
        curr_err = error(U, V, W)
        
        # Pre-compute slices for fast update
        # T[i,j,k] = sum_c U[i,k,c] V[k,j,c] W[j,i,c]
        # Changing U[i0, k0, c0] affects all T[i0, j, k0] for all j.
        # Change in T[i0, j, k0] = delta * V[k0, j, c0] * W[j, i0, c0]
        
        # We will iterate through random positions and try delta = +/- 0.5
        
        for _ in range(max_steps):
            if time.time() > deadline - 10: break
            
            # Pick a random matrix and position
            mat_choice = np.random.randint(3)
            if mat_choice == 0:
                idx_i, idx_k, idx_c = np.random.randint(n), np.random.randint(m), np.random.randint(r)
                flat_idx = idx_i * m + idx_k
                old_val = U[flat_idx, idx_c]
                # Try changing U[flat_idx, idx_c]
                # Affected T: [idx_i, :, idx_k]
                # Delta T[j] = delta * V[idx_k, j, idx_c] * W[j, idx_i, idx_c]
                vec_V = V[idx_k * p : (idx_k+1)*p, idx_c] # shape (p,)
                vec_W = np.array([W[j*n + idx_i, idx_c] for j in range(p)]) # shape (p,)
                delta_T = vec_V * vec_W
                
                # Current T slice
                T_slice = np.einsum('kr,jr->kj', U[idx_i*m:idx_i*m+m, idx_c], V[idx_k*p:(idx_k+1)*p, idx_c]) 
                # Wait, this is getting complex. Let's just recompute error for candidate if cheap?
                # No, recompute is O(n*p*m). 
                # Let's do a simpler local search: pick a random entry, flip it by 0.5, check error.
                # Since dimensions are small, recompute is fast enough for a few hundred steps.
                
                U_new = U.copy()
                U_new[flat_idx, idx_c] = old_val + 0.5
                err_new = error(U_new, V, W)
                if err_new < curr_err:
                    U = U_new
                    curr_err = err_new
                    continue
                
                U_new[flat_idx, idx_c] = old_val - 0.5
                err_new = error(U_new, V, W)
                if err_new < curr_err:
                    U = U_new
                    curr_err = err_new
                    continue
                    
            elif mat_choice == 1:
                idx_k, idx_j, idx_c = np.random.randint(m), np.random.randint(p), np.random.randint(r)
                flat_idx = idx_k * p + idx_j
                old_val = V[flat_idx, idx_c]
                
                V_new = V.copy()
                V_new[flat_idx, idx_c] = old_val + 0.5
                err_new = error(U, V_new, W)
                if err_new < curr_err:
                    V = V_new
                    curr_err = err_new
                    continue
                
                V_new[flat_idx, idx_c] = old_val - 0.5
                err_new = error(U, V_new, W)
                if err_new < curr_err:
                    V = V_new
                    curr_err = err_new
                    continue
                    
            else:
                idx_j, idx_i, idx_c = np.random.randint(p), np.random.randint(n), np.random.randint(r)
                flat_idx = idx_j * n + idx_i
                old_val = W[flat_idx, idx_c]
                
                W_new = W.copy()
                W_new[flat_idx, idx_c] = old_val + 0.5
                err_new = error(U, V, W_new)
                if err_new < curr_err:
                    W = W_new
                    curr_err = err_new
                    continue
                
                W_new[flat_idx, idx_c] = old_val - 0.5
                err_new = error(U, V, W_new)
                if err_new < curr_err:
                    W = W_new
                    curr_err = err_new
                    continue
                    
        return U, V, W, curr_err

    # 5. Search for lower rank
    # Known theoretical bounds: M(2,2,7) is likely 25 or 26.
    # Let's try 25 first, then 26.
    
    target_ranks = [25, 26]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 30: break
        
        num_restarts = 20
        for restart in range(num_restarts):
            if time.time() > deadline - 15: break
            
            np.random.seed(1000 + restart)
            
            # Initialize with small half-integers
            # Using a mix of 0, 0.5, 1.0, 1.5
            # Start with a "sparse" initialization to mimic sparsity of schoolbook
            U = np.random.choice([0, 0.5, 1.0, -0.5, -1.0], size=(n*m, target_rank))
            V = np.random.choice([0, 0.5, 1.0, -0.5, -1.0], size=(m*p, target_rank))
            W = np.random.choice([0, 0.5, 1.0, -0.5, -1.0], size=(p*n, target_rank))
            
            # Scale down to avoid huge errors initially? 
            # Actually, random integers often lead to huge errors. 
            # Let's try a "structured" random init:
            # Pick random subsets of rows/cols to be non-zero.
            
            # Alternative Init: Start from a random permutation of a known valid decomposition?
            # We don't have one.
            
            # Let's try "Greedy Construction" init:
            # Start with zeros.
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Add random "features"
            for c in range(target_rank):
                # Pick a random pattern
                # Maybe just random half-integers is better for exploration
                pass
            
            # Let's stick to random half-integers but with a bias towards small values
            U = np.round(np.random.randn(n*m, target_rank) * 0.5) / 2.0
            V = np.round(np.random.randn(m*p, target_rank) * 0.5) / 2.0
            W = np.round(np.random.randn(p*n, target_rank) * 0.5) / 2.0
            
            # Improve
            U, V, W, err = improve(U, V, W, max_steps=200)
            
            if err < 1e-9:
                if verify(U, V, W):
                    best_result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    best_rank = target_rank
                    break
        
        if best_rank < 28:
            break

    return best_result
# EVOLVE-BLOCK-END
