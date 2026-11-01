import time
import random
import itertools
import numpy as np

DIMENSIONS = [2, 2, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # First, try to find a rank-20 decomposition using border rank approach
    # Start from the rank-21 Strassen decomposition and try to eliminate one term
    
    strassen = [
        {'U': [1, 0, 0, 1], 'V': [1, 0, 0, 1], 'W': [1, 0, 0, 1]}, # P1
        {'U': [0, 0, 1, 1], 'V': [1, 0, 0, 0], 'W': [0, 1, 0, -1]}, # P2
        {'U': [1, 0, 0, 0], 'V': [0, 1, 0, -1], 'W': [0, 0, 1, 1]}, # P3
        {'U': [0, 0, 0, 1], 'V': [-1, 0, 1, 0], 'W': [1, 1, 0, 0]}, # P4
        {'U': [1, 1, 0, 0], 'V': [0, 0, 0, 1], 'W': [-1, 0, 1, 0]}, # P5
        {'U': [-1, 0, 1, 0], 'V': [1, 1, 0, 0], 'W': [0, 0, 0, 1]}, # P6
        {'U': [0, 1, 0, -1], 'V': [0, 0, 1, 1], 'W': [1, 0, 0, 0]}, # P7
    ]
    
    # Build the rank-21 decomposition
    rank21_U = np.zeros((n*m, 21))
    rank21_V = np.zeros((m*p, 21))
    rank21_W = np.zeros((p*n, 21))
    
    for b in range(3):
        for s in range(7):
            col = b * 7 + s
            sc = strassen[s]
            
            for idx in range(4):
                rank21_U[idx, col] = sc['U'][idx]
            
            v_idx = [2*b, 2*b+1, 6+2*b, 6+2*b+1]
            for idx in range(4):
                rank21_V[v_idx[idx], col] = sc['V'][idx]
            
            w_idx = [4*b, 4*b+1, 4*b+2, 4*b+3]
            for idx in range(4):
                rank21_W[w_idx[idx], col] = sc['W'][idx]
    
    # Try to find a rank-20 decomposition
    # Strategy: Remove one rank term and adjust the remaining 20 to compensate
    # Use the tensor identity: sum_r U[:,r] * V[:,r] * W[:,r] = T (the multiplication tensor)
    
    # The tensor T has entries T[i,j,k] = 1 if i==k (for the diagonal in the output)
    # More precisely, T[i,j,k] = delta_{i,k} * delta_{j,j} = delta_{i,k}
    # Wait, let me think about this more carefully.
    
    # For matrix multiplication C = A * B where A is 2x2 and B is 2x6:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all valid i,j,k (it's the identity in the "multiplication" sense)
    
    # Actually, the tensor is: T[i,j,k] = 1 if the (i,k) entry of C depends on A[i,j]*B[j,k]
    # For standard matrix multiplication, T[i,j,k] = 1 for all i,j,k in range
    
    # Let's define the target tensor
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1  # All entries are 1 for standard matrix mult
    
    # Flatten T to match the decomposition format
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition is: sum_r U[:,r] * V[:,r] * W[:,r] = T_flattened
    
    # Let's try to find a rank-20 decomposition by starting from rank-21 and removing one term
    # Then optimize the remaining 20 terms
    
    best_rank = 21
    best_U = None
    best_V = None
    best_W = None
    
    # Try removing each of the 21 terms and see if we can compensate
    for remove_col in range(21):
        if time.time() > deadline - 5:
            break
            
        # Start with all terms except the removed one
        U_init = np.delete(rank21_U, remove_col, axis=1)
        V_init = np.delete(rank21_V, remove_col, axis=1)
        W_init = np.delete(rank21_W, remove_col, axis=1)
        
        # Now try to optimize these 20 terms to satisfy the tensor identity
        # Use gradient descent or a simple optimization
        
        # Flatten the parameters
        def flatten_params(U, V, W):
            return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def unflatten_params(params):
            U = params[:n*m*20].reshape(n*m, 20)
            V = params[n*m*20:n*m*20 + m*p*20].reshape(m*p, 20)
            W = params[n*m*20 + m*p*20:].reshape(p*n, 20)
            return U, V, W
        
        # Compute the current tensor from the decomposition
        def compute_tensor(U, V, W):
            # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
            # But we need to map indices correctly
            # U index: i (0 to n-1), but U has n*m rows, so row index is i*m + ? 
            # Wait, let me reconsider the indexing.
            
            # U has shape (n*m, rank): row index = i * m + j? No...
            # Actually, for matrix multiplication A (n x m) * B (m x p) = C (n x p):
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # The tensor T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[i,k]
            # So T[i,j,k] = 1 for all i,j,k
            
            # In the decomposition:
            # U has shape (n*m, rank): the row index corresponds to (i, j) pairs
            # V has shape (m*p, rank): the row index corresponds to (j, k) pairs  
            # W has shape (p*n, rank): the row index corresponds to (k, i) pairs
            
            # So for a given rank term r:
            # U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] should sum to T[i,j,k] = 1
            
            # Let's compute the tensor
            tensor = np.zeros((n, m, p))
            for r in range(20):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_val = U[i*m + j, r]
                            v_val = V[j*p + k, r]
                            w_val = W[k*n + i, r]
                            tensor[i, j, k] += u_val * v_val * w_val
            return tensor
        
        # Objective: minimize ||computed_tensor - T||^2
        def objective(params):
            U, V, W = unflatten_params(params)
            tensor = compute_tensor(U, V, W)
            diff = tensor - T
            return np.sum(diff**2)
        
        # Try a few optimization steps using simple gradient descent
        params = flatten_params(U_init, V_init, W_init)
        
        # Use a few random restarts with small perturbations
        best_obj = objective(params)
        best_params = params.copy()
        
        for restart in range(5):
            if time.time() > deadline - 3:
                break
            
            # Add small random perturbation
            params_perturbed = params + np.random.normal(0, 0.1, params.shape)
            
            # Simple gradient descent
            lr = 0.01
            for step in range(50):
                if time.time() > deadline - 3:
                    break
                
                # Compute gradient numerically
                grad = np.zeros_like(params_perturbed)
                eps = 1e-5
                for idx in range(0, len(params_perturbed), 5):  # Sample every 5th parameter for speed
                    params_plus = params_perturbed.copy()
                    params_plus[idx] += eps
                    obj_plus = objective(params_plus)
                    params_perturbed[idx] -= eps * 2
                    obj_minus = objective(params_perturbed)
                    params_perturbed[idx] += eps
                    
                    grad[idx] = (obj_plus - obj_minus) / (2 * eps)
                
                params_perturbed -= lr * grad
                
                obj = objective(params_perturbed)
                if obj < best_obj:
                    best_obj = obj
                    best_params = params_perturbed.copy()
            
            if best_obj < 1e-6:
                break
        
        # Check if we found a valid rank-20 decomposition
        if best_obj < 1e-4:
            U_opt, V_opt, W_opt = unflatten_params(best_params)
            
            # Verify
            tensor_check = compute_tensor(U_opt, V_opt, W_opt)
            if np.allclose(tensor_check, T, atol=1e-3):
                # Found a rank-20 decomposition!
                # Convert to integer/half-integer if possible
                # For now, just return the floating point values
                best_rank = 20
                best_U = U_opt.tolist()
                best_V = V_opt.tolist()
                best_W = W_opt.tolist()
                break
    
    # If we found a rank-20 decomposition, return it
    if best_rank == 20:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Otherwise, return the rank-21 decomposition
    U = rank21_U.tolist()
    V = rank21_V.tolist()
    W = rank21_W.tolist()
    
    return {"U": U, "V": V, "W": W}
