import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Try to find rank 34 decomposition
    target_rank = 34
    start_time = time.time()
    deadline = start_time + 120 # Safety margin
    
    best_result = None
    best_score = float('inf')
    
    # We want to solve sum_c U_ij,c V_jk,c W_ik,c = 1 for all i,j,k
    # Variables: U (nm x r), V (mp x r), W (pn x r)
    # Total vars: r * (nm + mp + pn)
    
    # Initialize with random values scaled appropriately
    # A good initialization might be random normal
    # We will try a few random restarts
    
    for restart in range(50):
        if time.time() > deadline - 20:
            break
            
        # Initialize U, V, W
        # Scale: if U,V,W are ~1, product is ~1. Sum of r terms is ~r.
        # We want sum to be 1. So entries should be ~1/sqrt(r) or similar?
        # Actually, if we use random +/- 1, the sum might be large.
        # Let's try random values in [-1, 1]
        
        U = np.random.uniform(-1, 1, (n * m, target_rank))
        V = np.random.uniform(-1, 1, (m * p, target_rank))
        W = np.random.uniform(-1, 1, (p * n, target_rank))
        
        # Flatten to vector for optimization?
        # Or just use gradient descent on the matrices?
        # Simple gradient descent might be slow.
        # Let's use a simple local search or just check if random works (unlikely).
        
        # Better: Use scipy.optimize.minimize if available?
        # The prompt says NumPy and SciPy are available.
        # But calling scipy inside a tight loop might be slow if not careful.
        
        # Let's try a simple iterative refinement:
        # Fix two, solve for the third?
        # The equation is linear in W if U, V are fixed.
        # For each i, k:
        # W_ik,c are variables.
        # Sum_c (U_ij,c V_jk,c) W_ik,c = 1
        # This is a linear system for W_ik,c?
        # No, W_ik,c is a vector of length r for each i,k.
        # The equation couples all c.
        
        # Let's define the tensor T.
        # T_ijk = 1.
        # We want U, V, W such that T = sum_c u_c v_c w_c.
        
        # We can use a simple gradient step on the loss function.
        # Loss = sum_ijk (sum_c U_ij,c V_jk,c W_ik,c - 1)^2
        
        # Compute gradients
        # dL/dW_ik,c = 2 * (sum_c' U_ij,c' V_jk,c' W_ik,c' - 1) * U_ij,c V_jk,c
        # Summed over j?
        # Wait, the equation is for each i, j, k.
        # L = sum_i sum_j sum_k (S_ijk - 1)^2
        # S_ijk = sum_c U_ij,c V_jk,c W_ik,c
        
        # dL/dW_ik,c = sum_j 2 * (S_ijk - 1) * U_ij,c V_jk,c
        # dL/dU_ij,c = sum_k 2 * (S_ijk - 1) * V_jk,c W_ik,c
        # dL/dV_jk,c = sum_i 2 * (S_ijk - 1) * U_ij,c W_ik,c
        
        # We can implement this gradient descent.
        
        U_curr = U.copy()
        V_curr = V.copy()
        W_curr = W.copy()
        
        lr = 0.01
        steps = 1000
        
        for step in range(steps):
            if time.time() > deadline - 10:
                break
                
            # Compute S_ijk
            # S is shape (n, m, p)
            # S_ijk = sum_c U_ij,c V_jk,c W_ik,c
            # Efficient computation:
            # U is (nm, r) -> reshape to (n, m, r)
            # V is (mp, r) -> reshape to (m, p, r)
            # W is (pn, r) -> reshape to (p, n, r)
            
            U_r = U_curr.reshape(n, m, target_rank)
            V_r = V_curr.reshape(m, p, target_rank)
            W_r = W_curr.reshape(p, n, target_rank)
            
            # S_ijk = sum_c U_r[i,j,c] V_r[j,k,c] W_r[k,i,c]
            # Note: W index is (k, i) because W is (p*n, r) -> (p, n, r)
            # So W_r[k, i, c]
            
            # We can compute this using einsum
            # S_ijk = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
            S = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
            
            error = S - 1.0
            loss = np.sum(error**2)
            
            if loss < 1e-8:
                # Found a solution?
                # Check exactness with rounding
                U_round = np.round(U_curr * 2) / 2
                V_round = np.round(V_curr * 2) / 2
                W_round = np.round(W_curr * 2) / 2
                
                # Verify exactness
                U_r_r = U_round.reshape(n, m, target_rank)
                V_r_r = V_round.reshape(m, p, target_rank)
                W_r_r = W_round.reshape(p, n, target_rank)
                
                S_check = np.einsum('ijc,jkc,kic->ijk', U_r_r, V_r_r, W_r_r)
                
                if np.allclose(S_check, 1.0, atol=1e-9):
                    # Return this solution
                    # Convert to lists
                    return {
                        "U": U_round.tolist(),
                        "V": V_round.tolist(),
                        "W": W_round.tolist()
                    }
                
            # Compute gradients
            # dL/dW_ik,c = sum_j 2 * error_ijk * U_r[i,j,c] V_r[j,k,c]
            # W_grad shape (p, n, r)
            # Indices: k, i, c
            # error_ijk is (n, m, p) -> (i, j, k)
            # We need to sum over j.
            # dW_ikc = 2 * sum_j error_ijk U_ijc V_jkc
            
            # Using einsum:
            # error_ijk: i, j, k
            # U_r: i, j, c
            # V_r: j, k, c
            # Result: k, i, c
            W_grad = 2.0 * np.einsum('ijk,ijc,jkc->kic', error, U_r, V_r)
            
            # dL/dU_ij,c = sum_k 2 * error_ijk V_r[j,k,c] W_r[k,i,c]
            # Result: i, j, c
            U_grad = 2.0 * np.einsum('ijk,jkc,kic->ijc', error, V_r, W_r)
            
            # dL/dV_jk,c = sum_i 2 * error_ijk U_r[i,j,c] W_r[k,i,c]
            # Result: j, k, c
            V_grad = 2.0 * np.einsum('ijk,ijc,kic->jkc', error, U_r, W_r)
            
            # Update
            U_curr -= lr * U_grad
            V_curr -= lr * V_grad
            W_curr -= lr * W_grad
            
            # Optional: normalize or clip to prevent explosion
            # U_curr = np.clip(U_curr, -10, 10)
            
        # End steps
        
        # Check final state
        U_r = U_curr.reshape(n, m, target_rank)
        V_r = V_curr.reshape(m, p, target_rank)
        W_r = W_curr.reshape(p, n, target_rank)
        S = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        loss = np.sum((S - 1.0)**2)
        
        if loss < best_score:
            best_score = loss
            if loss < 1e-6:
                U_round = np.round(U_curr * 2) / 2
                V_round = np.round(V_curr * 2) / 2
                W_round = np.round(W_curr * 2) / 2
                
                U_r_r = U_round.reshape(n, m, target_rank)
                V_r_r = V_round.reshape(m, p, target_rank)
                W_r_r = W_round.reshape(p, n, target_rank)
                S_check = np.einsum('ijc,jkc,kic->ijk', U_r_r, V_r_r, W_r_r)
                
                if np.allclose(S_check, 1.0, atol=1e-9):
                    return {
                        "U": U_round.tolist(),
                        "V": V_round.tolist(),
                        "W": W_round.tolist()
                    }

    # If no better solution found, return schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
# EVOLVE-BLOCK-END
