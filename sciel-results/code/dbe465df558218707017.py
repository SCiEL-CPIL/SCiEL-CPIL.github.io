import numpy as np
import time
import random

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    target_rank = 60
    best_U, best_V, best_W = None, None, None
    
    # Time limit guard
    start_time = time.time()
    time_limit = 140  # 140s to be safe
    
    # Try ranks from 50 to 59
    # Start with 59, go down if time permits? 
    # Or just try 59 with a few restarts.
    # If 59 fails, return 60.
    
    # The tensor T has indices i, j, k
    # T[i, j, k] = 1 if j == k? No.
    # The decomposition is sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # We want this sum to be 1 if the indices match the "identity" of the multiplication.
    # The "identity" is: for each (i,j,k), the term is 1.
    # Wait, the schoolbook code sets:
    # U[i*m+j][c] = 1
    # V[j*p+k][c] = 1
    # W[k*n+i][c] = 1
    # This means for a specific c, U is non-zero at (i,j), V at (j,k), W at (k,i).
    # The product U*V*W is non-zero only at (i,j), (j,k), (k,i).
    # The sum over c gives 1 at these positions.
    # So the target tensor T has T[i,j,k] = 1?
    # No, the tensor is in the space of (i,j), (j,k), (k,i).
    # The entries are indexed by (i,j), (j,k), (k,i).
    # Let's flatten the indices.
    # idx_U = i*m + j
    # idx_V = j*p + k
    # idx_W = k*n + i
    # The target value for the triple (idx_U, idx_V, idx_W) is 1 if the indices are consistent (i.e., j in U matches j in V, k in V matches k in W, i in W matches i in U).
    # In the schoolbook, for each (i,j,k), we have a term.
    # So T[idx_U, idx_V, idx_W] = 1 if idx_U=i*m+j, idx_V=j*p+k, idx_W=k*n+i.
    # Otherwise 0.
    
    # Let's create the target tensor T
    # Shape: (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
                
    def try_rank(R):
        if time.time() - start_time > time_limit - 10:
            return None
        
        # Initialize U, V, W with small random values
        # To help convergence, we can scale them
        U = np.random.randn(n*m, R) * 0.1
        V = np.random.randn(m*p, R) * 0.1
        W = np.random.randn(p*n, R) * 0.1
        
        # We want to minimize || T - U V W ||^2
        # This is a non-convex problem.
        # We can use a simple gradient descent or use scipy.
        # Given the size (6x30x20), tensor is 3600 elements.
        # R=59.
        # Number of variables: 59*(6+30+20) = 59*56 = 3304.
        
        # Let's try a few iterations of gradient descent.
        # Error = T - np.einsum('ir,jr,kr->ijk', U, V, W)
        # Grad U = Error * (V W)^T ?
        
        # Actually, let's just use a randomized local search with a fixed number of steps.
        # If we don't converge, we return None.
        
        # To make it faster, we can try to fix one matrix?
        # No, let's just run a simple optimizer.
        
        # We'll use a manual gradient step to avoid overhead of scipy if it's too slow.
        # Or use scipy.optimize.minimize with method='L-BFGS-B' or similar.
        # L-BFGS-B is good for large sparse-ish problems.
        
        # Flatten U, V, W into a single vector x
        def unpack(x):
            u_size = n*m * R
            v_size = m*p * R
            w_size = p*n * R
            U_flat = x[:u_size]
            V_flat = x[u_size:u_size+v_size]
            W_flat = x[u_size+v_size:u_size+v_size+w_size]
            return U_flat.reshape(n*m, R), V_flat.reshape(m*p, R), W_flat.reshape(p*n, R)
            
        def objective(x):
            U, V, W = unpack(x)
            # Compute U V W
            # einsum: 'ir,jr,kr->ijk'
            # This is O(n*m*R * m*p*R * p*n*R)? No, it's a tensor contraction.
            # It's effectively a matrix multiplication chain.
            # (U^T V) is (n*m, m*p) ? No.
            # U is (n*m, R), V is (m*p, R).
            # We want T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r].
            # This is not a standard matrix multiplication.
            # It is a rank-R decomposition of the 3rd order tensor.
            # We can compute it as:
            # Temp[i, j, r] = U[i, r] * V[j, r]
            # Then T[i, j, k] = sum_r Temp[i, j, r] * W[k, r]
            
            # Vectorized:
            # U[:, :, None] * V[None, :, :] -> (n*m, m*p, R)
            # Then * W[None, None, :] -> (n*m, m*p, p*n)
            # This creates a large intermediate array (6 * 30 * 59) ~ 10k elements.
            # Final array 6 * 30 * 20 = 3600.
            # This is fast.
            
            UV = U[:, :, None] * V[None, :, :] # (n*m, m*p, R)
            UVW = UV * W[None, None, :] # (n*m, m*p, p*n)
            
            diff = T - UVW
            return np.sum(diff * diff)
            
        # Initial guess
        x0 = np.random.randn((n*m + m*p + p*n) * R) * 0.1
        
        # Use scipy
        try:
            from scipy.optimize import minimize
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-8})
            if res.fun < 1e-6:
                U, V, W = unpack(res.x)
                return U, V, W
        except Exception:
            pass
            
        return None

    # Try rank 59
    res = try_rank(59)
    if res:
        U, V, W = res
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # If 59 fails, return schoolbook
    U = np.zeros((n*m, 60))
    V = np.zeros((m*p, 60))
    W = np.zeros((p*n, 60))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
