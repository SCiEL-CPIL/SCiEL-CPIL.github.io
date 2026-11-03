"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    # Schoolbook rank = n*m*p = 54
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None

    # Verify a decomposition
    def verify(U, V, W, rank):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # Tensor T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We want T[i, j, k] = 1 for all i, j, k.
        
        # Efficient verification using einsum or loops if small
        # Since n, m, p are small (3, 3, 6), even loops are okay for final check,
        # but let's use vectorization for speed if possible.
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_t = U_arr.reshape(n, m, rank)
        V_t = V_arr.reshape(m, p, rank)
        W_t = W_arr.reshape(p, n, rank)
        
        # Compute T[i, j, k]
        # T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
        # This is a bit tricky to fully vectorize without large intermediate arrays.
        # Let's stick to a simple loop for verification as it's only called a few times.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(val - 1.0) > 1e-8:
                        return False
        return True

    # Schoolbook decomposition
    def schoolbook_decomp():
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

    # Get initial schoolbook solution
    best_U, best_V, best_W, best_rank = schoolbook_decomp()

    # Vectorized Gradient Descent
    def optimize(U, V, W, max_iters, lr):
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        
        # Precompute indices for vectorized gradient calculation
        # U_idx[i, j, r] = i*m + j
        # V_idx[j, k, r] = j*p + k
        # W_idx[k, i, r] = k*n + i
        
        # We compute Error E[i, j, k] = 1 - sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # Gradient dU[i*m+j, r] = sum_k E[i, j, k] V[j*p+k, r] W[k*n+i, r]
        # Gradient dV[j*p+k, r] = sum_i E[i, j, k] U[i*m+j, r] W[k*n+i, r]
        # Gradient dW[k*n+i, r] = sum_j E[i, j, k] U[i*m+j, r] V[j*p+k, r]
        
        # To vectorize, we can use einsum.
        # Let U_t be U reshaped to (n, m, rank)
        # Let V_t be V reshaped to (m, p, rank)
        # Let W_t be W reshaped to (p, n, rank)
        
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        
        # Compute T[i, j, k] = sum_r U_t[i, j, r] * V_t[j, k, r] * W_t[k, i, r]
        # This contraction is not a standard simple einsum because indices are permuted.
        # T[i, j, k] = einsum('ijr,jkr,ki r->ijk', U_t, V_t, W_t) ?
        # Let's check indices:
        # U_t[i, j, r]
        # V_t[j, k, r]
        # W_t[k, i, r] -> Note W is indexed by (k, n, i) in tensor logic?
        # W row is k*n + i. So W_t[k, i, r].
        # Yes.
        
        try:
            T = np.einsum('ijr,jkr,ki r->ijk', U_t, V_t, W_t)
        except Exception:
            # Fallback to loops if einsum fails or is too slow to setup (unlikely)
            T = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] = np.sum(U_t[i, j, :] * V_t[j, k, :] * W_t[k, i, :])
                        
        E = 1.0 - T
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            if np.max(np.abs(E)) < 1e-9:
                break
                
            # Compute gradients
            # dU[i, j, r] = sum_k E[i, j, k] * V_t[j, k, r] * W_t[k, i, r]
            # dV[j, k, r] = sum_i E[i, j, k] * U_t[i, j, r] * W_t[k, i, r]
            # dW[k, i, r] = sum_j E[i, j, k] * U_t[i, j, r] * V_t[j, k, r]
            
            dU_t = np.einsum('ijk,jkr,ki r->ijr', E, V_t, W_t)
            dV_t = np.einsum('ijk,ijr,ki r->jkr', E, U_t, W_t)
            dW_t = np.einsum('ijk,ijr,jkr->ki r', E, U_t, V_t)
            
            # Update
            U_t -= lr * dU_t
            V_t -= lr * dV_t
            W_t -= lr * dW_t
            
            # Recompute T and E
            T = np.einsum('ijr,jkr,ki r->ijk', U_t, V_t, W_t)
            E = 1.0 - T
            
        return U_t.reshape(n*m, rank), V_t.reshape(m*p, rank), W_t.reshape(p*n, rank)

    # Search for lower ranks
    # We start from the schoolbook solution and try to reduce rank.
    # A common strategy is to try rank R-1, initialize with the first R-1 columns of the R solution,
    # and optimize. If it converges, we have a solution for R-1.
    
    current_U, current_V, current_W = best_U, best_V, best_W
    current_rank = best_rank
    
    # Try reducing rank from 53 down to 30
    for target_rank in range(current_rank - 1, 29, -1):
        if time.time() > deadline:
            break
            
        # Initialize with the previous solution, truncating the last column
        # Or better: Random restarts might be better if stuck in local minima.
        # Let's try a few initializations.
        
        found = False
        for init_type in range(3):
            if time.time() > deadline:
                break
                
            if init_type == 0:
                # Truncate previous solution
                U_init = current_U[:, :target_rank].copy()
                V_init = current_V[:, :target_rank].copy()
                W_init = current_W[:, :target_rank].copy()
            elif init_type == 1:
                # Random small perturbation around truncated
                U_init = current_U[:, :target_rank].copy()
                V_init = current_V[:, :target_rank].copy()
                W_init = current_W[:, :target_rank].copy()
                U_init += np.random.randn(*U_init.shape) * 0.1
                V_init += np.random.randn(*V_init.shape) * 0.1
                W_init += np.random.randn(*W_init.shape) * 0.1
            else:
                # Pure random
                U_init = np.random.randn(n*m, target_rank) * 0.5
                V_init = np.random.randn(m*p, target_rank) * 0.5
                W_init = np.random.randn(p*n, target_rank) * 0.5
                
            # Optimize
            max_iters = 1000
            lr = 0.05
            U_opt, V_opt, W_opt = optimize(U_init, V_init, W_opt if False else W_init, max_iters, lr)
            
            # Verify
            if verify(U_opt, V_opt, W_opt, target_rank):
                best_rank = target_rank
                best_U = U_opt
                best_V = V_opt
                best_W = W_opt
                current_U, current_V, current_W = best_U, best_V, best_W
                found = True
                break # Move to next lower rank
        
        if not found:
            # If we couldn't reduce rank, we might still try to improve the current one
            # but for now, we stop reducing if we fail for a rank?
            # It's possible we missed a solution. Let's try one more random init for the failed rank.
            U_init = np.random.randn(n*m, target_rank) * 0.5
            V_init = np.random.randn(m*p, target_rank) * 0.5
            W_init = np.random.randn(p*n, target_rank) * 0.5
            U_opt, V_opt, W_opt = optimize(U_init, V_init, W_init, 2000, 0.05)
            if verify(U_opt, V_opt, W_opt, target_rank):
                best_rank = target_rank
                best_U = U_opt
                best_V = V_opt
                best_W = W_opt
                current_U, current_V, current_W = best_U, best_V, best_W
            else:
                # If we fail to find a solution for this rank, we can't safely assume 
                # lower ranks are impossible, but we can't use the current U,V,W 
                # (which are for a higher rank) to initialize a lower rank effectively 
                # unless we have a valid solution for the higher rank that we can truncate.
                # Since we didn't find a solution for target_rank, we should probably 
                # stop trying lower ranks based on this specific path, or restart 
                # the search for lower ranks from scratch if we had a valid 
                # solution for a higher rank. 
                # Given the complexity, we will break out of the reduction loop 
                # if we fail to reduce the rank, as truncating an invalid 
                # decomposition is not useful.
                break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
