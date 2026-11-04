"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin before 120s limit

    def verify_fast(U, V, W, rank):
        """Verify that the decomposition is exact using NumPy."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Reshape for broadcasting
        # U: (n*m, rank) -> (n, m, rank)
        U_t = U_arr.reshape(n, m, rank)
        # V: (m*p, rank) -> (m, p, rank)
        V_t = V_arr.reshape(m, p, rank)
        # W: (p*n, rank) -> (p, n, rank)
        W_t = W_arr.reshape(p, n, rank)
        
        # We want to compute S[i,j,k] = sum_c U_t[i,j,c] * V_t[j,k,c] * W_t[k,i,c]
        # This is a contraction over c, with indices i,j,k.
        # Using einsum: 'ijc,jkc,kic->ijk'
        
        try:
            S = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
        except Exception:
            return False
        
        # Check if all entries are close to 1
        return np.allclose(S, 1.0, atol=1e-9)
    
    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank

    # Start with schoolbook
    U, V, W, rank = schoolbook()
    
    def try_rank(target_rank):
        if time.time() > deadline:
            return None
            
        # We will try to find a solution using gradient descent on floats,
        # then snap to half-integers.
        # To increase chances, we can try different initializations.
        
        for attempt in range(10):
            if time.time() > deadline:
                break
                
            # Random initialization
            U_init = np.random.uniform(-1, 1, (n*m, target_rank))
            V_init = np.random.uniform(-1, 1, (m*p, target_rank))
            W_init = np.random.uniform(-1, 1, (p*n, target_rank))
            
            # Gradient Descent
            U_curr = U_init.copy()
            V_curr = V_init.copy()
            W_curr = W_init.copy()
            
            lr = 0.01
            # Adaptive learning rate or just fixed?
            # Let's use a few steps with decreasing LR or just fixed.
            # Given the small size, 200 steps should be fast.
            
            for step in range(200):
                if time.time() > deadline:
                    break
                
                U_t = U_curr.reshape(n, m, target_rank)
                V_t = V_curr.reshape(m, p, target_rank)
                W_t = W_curr.reshape(p, n, target_rank)
                
                # Compute S
                S = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
                E = S - 1.0
                
                # Compute gradients
                # dU[i,j,c] = sum_k E[i,j,k] * V[j,k,c] * W[k,i,c]
                # dV[j,k,c] = sum_i E[i,j,k] * U[i,j,c] * W[k,i,c]
                # dW[k,i,c] = sum_j E[i,j,k] * U[i,j,c] * V[j,k,c]
                
                dU_t = np.einsum('ijk,jkc,kic->ijc', E, V_t, W_t)
                dV_t = np.einsum('ijk,ijc,kic->jkc', E, U_t, W_t)
                dW_t = np.einsum('ijk,ijc,jkc->kic', E, U_t, V_t)
                
                # Update
                U_curr -= lr * dU_t.reshape(n*m, target_rank)
                V_curr -= lr * dV_t.reshape(m*p, target_rank)
                W_curr -= lr * dW_t.reshape(p*n, target_rank)
                
                # Check convergence
                if np.max(np.abs(E)) < 1e-10:
                    break
            
            # Check if valid float solution
            if verify_fast(U_curr, V_curr, W_curr, target_rank):
                # Snap to half-integers
                U_snapped = np.round(U_curr * 2) / 2
                V_snapped = np.round(V_curr * 2) / 2
                W_snapped = np.round(W_curr * 2) / 2
                
                # Verify snapped version
                if verify_fast(U_snapped, V_snapped, W_snapped, target_rank):
                    return U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist(), target_rank
                else:
                    # If snapping fails, maybe try local search around snapped values?
                    # Or just accept if we can find another float solution that snaps better.
                    # For now, we discard this attempt and try another initialization.
                    pass
                    
        return None

    # Try lower ranks
    # We know rank 90 is valid.
    # Try 89, 88, ...
    # If we find one, return it.
    # If we run out of time, return the best found (which might be 90).
    
    best_result = (U, V, W, rank)
    
    # It is unlikely we can go much lower than 90 quickly without specific knowledge.
    # However, for small dimensions, sometimes rank reduction is possible.
    # Let's try a few ranks below 90.
    
    for target_rank in range(89, 80, -1):
        if time.time() > deadline:
            break
        result = try_rank(target_rank)
        if result is not None:
            best_result = result
            # We could continue to try lower, but let's see if we have time.
            # If we found 89, try 88?
            # The loop does this automatically.
            continue
            
    # If no better rank found, return schoolbook
    U, V, W, rank = best_result
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
