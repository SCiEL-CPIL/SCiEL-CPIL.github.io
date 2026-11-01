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

    def gradient_descent_search(target_rank, max_iters=200, lr=0.01, time_budget=40.0):
        """
        Attempt to find a rank-target_rank decomposition via gradient descent.
        """
        elapsed = time.time() - start_time
        if elapsed > deadline - time_budget:
            return None
            
        # Target tensor is all ones
        T_target = np.ones((n, m, p))
        
        best_U, best_V, best_W = None, None, None
        best_err = float('inf')
        
        # Try a few random initializations
        for trial in range(5):
            if time.time() > deadline - 10.0:
                break
                
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Scale to avoid huge gradients
            U /= np.sqrt(target_rank)
            V /= np.sqrt(target_rank)
            W /= np.sqrt(target_rank)
            
            for it in range(max_iters):
                if time.time() > deadline - 10.0:
                    break
                    
                U_r = U.reshape(n, m, target_rank)
                V_r = V.reshape(m, p, target_rank)
                W_r = W.reshape(p, n, target_rank)
                
                T_calc = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                E = T_calc - T_target
                err = np.max(np.abs(E))
                
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    
                if err < 1e-9:
                    return U.copy(), V.copy(), W.copy()
                
                # Compute gradients
                # dJ/dU[i,j,r] = 2 * sum_{k} E[i,j,k] * V[j,k,r] * W[k,i,r]
                # Note: V is (m, p, R), W is (p, n, R)
                # V[j,k,r] is V_r[j,k,r]
                # W[k,i,r] is W_r[k,i,r]
                
                # grad_U[i,j,r] = 2 * sum_k E[i,j,k] V_r[j,k,r] W_r[k,i,r]
                grad_U = 2.0 * np.einsum('ijk,jkr,kir->ijr', E, V_r, W_r)
                # grad_V[j,k,r] = 2 * sum_i E[i,j,k] U_r[i,j,r] W_r[k,i,r]
                grad_V = 2.0 * np.einsum('ijk,ijr,kir->jkr', E, U_r, W_r)
                # grad_W[k,i,r] = 2 * sum_j E[i,j,k] U_r[i,j,r] V_r[j,k,r]
                grad_W = 2.0 * np.einsum('ijk,ijr,jkr->kir', E, U_r, V_r)
                
                # Reshape gradients to match U, V, W shapes
                grad_U_flat = grad_U.reshape(n * m, target_rank)
                grad_V_flat = grad_V.reshape(m * p, target_rank)
                grad_W_flat = grad_W.reshape(p * n, target_rank)
                
                # Update
                U -= lr * grad_U_flat
                V -= lr * grad_V_flat
                W -= lr * grad_W_flat
                
                # Optional: Simple adaptive learning rate or momentum could help, but keep it simple
                # If error increases significantly, reduce lr
                new_err = np.max(np.abs(np.einsum('ijr,jkr,kir->ijk', U.reshape(n, m, target_rank), V.reshape(m, p, target_rank), W.reshape(p, n, target_rank)) - T_target))
                if new_err > err * 1.5:
                    lr *= 0.5
                    # Revert
                    U += lr * 2.0 * grad_U_flat
                    V += lr * 2.0 * grad_V_flat
                    W += lr * 2.0 * grad_W_flat
                    # This revert logic is approximate, better to just save previous state
                    # For simplicity, we just reduce lr and continue, hoping it stabilizes
                
        if best_err < 1e-9 and best_U is not None:
            return best_U, best_V, best_W
        return None

    found = None
    # Try ranks 19, 20, 21, 22, 23
    for target_rank in [19, 20, 21, 22, 23]:
        result = gradient_descent_search(target_rank, max_iters=100, lr=0.05, time_budget=20.0)
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
