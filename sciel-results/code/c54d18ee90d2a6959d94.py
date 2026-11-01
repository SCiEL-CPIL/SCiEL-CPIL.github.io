import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    def verify_decomposition(U, V, W, rank):
        """Verify that U, V, W form a valid rank-r decomposition.
        U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        Check: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
        """
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        # T_calc[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
        # Use einsum for efficiency
        T_calc = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        # Target is all ones
        max_err = np.max(np.abs(T_calc - 1.0))
        return max_err < 1e-9, max_err

    def try_random_search(target_rank, num_trials=50, time_budget=30.0):
        """Attempt to find a rank-target_rank decomposition via random init + gradient-free local search."""
        elapsed = time.time() - start_time
        if elapsed > deadline - time_budget:
            return None
        
        best_U, best_V, best_W = None, None, None
        best_err = float('inf')
        
        for trial in range(num_trials):
            if time.time() > deadline - 5.0:
                break
            
            # Random initialization
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Simple local refinement: coordinate descent on each factor
            for iteration in range(50):
                if time.time() > deadline - 5.0:
                    break
                
                # Update U: for each r, minimize ||target - sum_{r'} U_r'⊗V_r'⊗W_r'||
                # This is equivalent to: for fixed V,W, the optimal U[:,r] satisfies
                # a linear system. But simpler: use stochastic gradient-like updates.
                
                # Compute current error
                U_r = U.reshape(n, m, target_rank)
                V_r = V.reshape(m, p, target_rank)
                W_r = W.reshape(p, n, target_rank)
                T_calc = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                err = np.max(np.abs(T_calc - 1.0))
                
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if err < 1e-9:
                    return U.copy(), V.copy(), W.copy()
                
                # Simple perturbation-based local search
                # Pick a random factor and a random entry, perturb it
                factor_idx = np.random.randint(3)
                if factor_idx == 0:
                    idx = np.random.randint(n * m * target_rank)
                    old_val = U.flat[idx]
                    U.flat[idx] += np.random.randn() * 0.1
                    # Check if error decreased
                    U_r = U.reshape(n, m, target_rank)
                    T_calc = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                    new_err = np.max(np.abs(T_calc - 1.0))
                    if new_err >= err:
                        U.flat[idx] = old_val
                elif factor_idx == 1:
                    idx = np.random.randint(m * p * target_rank)
                    old_val = V.flat[idx]
                    V.flat[idx] += np.random.randn() * 0.1
                    V_r = V.reshape(m, p, target_rank)
                    T_calc = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                    new_err = np.max(np.abs(T_calc - 1.0))
                    if new_err >= err:
                        V.flat[idx] = old_val
                else:
                    idx = np.random.randint(p * n * target_rank)
                    old_val = W.flat[idx]
                    W.flat[idx] += np.random.randn() * 0.1
                    W_r = W.reshape(p, n, target_rank)
                    T_calc = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                    new_err = np.max(np.abs(T_calc - 1.0))
                    if new_err >= err:
                        W.flat[idx] = old_val
        
        if best_err < 1e-9 and best_U is not None:
            return best_U, best_V, best_W
        return None
    
    # Try to find a decomposition with rank < 24
    # Start from rank 23 and go down (but realistically, 24 is likely minimal)
    # For efficiency, just try rank 23 first
    found = None
    for target_rank in [23]:
        result = try_random_search(target_rank, num_trials=30, time_budget=40.0)
        if result is not None:
            found = (target_rank, result)
            break
    
    if found is not None:
        rank, (U, V, W) = found
        # Final verification
        valid, err = verify_decomposition(U, V, W, rank)
        if valid:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook decomposition with rank = n*m*p = 24
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
    
    # Verify the schoolbook solution
    valid, err = verify_decomposition(U, V, W, rank)
    if not valid:
        raise RuntimeError(f"Schoolbook decomposition failed verification: err={err}")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
