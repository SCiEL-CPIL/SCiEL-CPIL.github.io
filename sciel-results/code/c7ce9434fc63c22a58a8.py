import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Target tensor indices
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # T[a, b, c] = 0 otherwise
    
    # Precompute the list of non-zero indices and their values (1)
    # And the list of zero indices
    nz_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nz_indices.append((a, b, c))
                
    # For verification, we can just iterate over all a,b,c
    # But for optimization, we only need to minimize error on nz_indices to 1
    # and on others to 0.
    
    def compute_error(U, V, W):
        err = 0.0
        # Non-zero entries must be 1
        for a, b, c in nz_indices:
            val = np.sum(U[a] * V[b] * W[c])
            err += (val - 1.0) ** 2
        
        # Zero entries must be 0
        # This is expensive if done for all 729 every step.
        # We can sample or do it every few steps.
        # For now, just check a subset of zeros or all if rank is small.
        # Let's check all zeros for final verification.
        # For optimization, we might just rely on the non-zero constraints 
        # and hope the structure enforces zeros? 
        # No, we must enforce zeros.
        
        # Efficient zero check:
        # Iterate over all a,b,c not in nz_indices
        # This is 729 - 27 = 702 entries.
        # 702 * 23 multiplications is ~16k ops. Very fast.
        
        nz_set = set(nz_indices)
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    if (a, b, c) not in nz_set:
                        val = np.sum(U[a] * V[b] * W[c])
                        err += val ** 2
        return err

    def get_fallback():
        rank = 27
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for idx, (a, b, c) in enumerate(nz_indices):
            U[a, idx] = 1.0
            V[b, idx] = 1.0
            W[c, idx] = 1.0
        return U, V, W

    def snap_and_verify(U, V, W, rank):
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        # Check exactness on all entries
        nz_set = set(nz_indices)
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    val = np.sum(U_snap[a] * V_snap[b] * W_snap[c])
                    if (a, b, c) in nz_set:
                        if abs(val - 1.0) > 1e-9:
                            return False
                    else:
                        if abs(val) > 1e-9:
                            return False
        
        # Prune zero columns
        mask = np.any(U_snap != 0, axis=0) | np.any(V_snap != 0, axis=0) | np.any(W_snap != 0, axis=0)
        U_final = U_snap[:, mask]
        V_final = V_snap[:, mask]
        W_final = W_snap[:, mask]
        
        return {
            "U": U_final.tolist(),
            "V": V_final.tolist(),
            "W": W_final.tolist()
        }

    # Try to find a lower rank solution
    # Start with rank 23
    best_result = None
    
    for rank in [23, 24, 25]:
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Initialize with random small values
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            W = np.random.randn(p*n, rank) * 0.1
            
            # Optimize
            # Use a simple gradient descent with Adam-like momentum
            # or just use scipy minimize on flattened vector
            
            def objective(params):
                U_flat = params[:n*m*rank]
                V_flat = params[n*m*rank : n*m*rank + m*p*rank]
                W_flat = params[n*m*rank + m*p*rank:]
                
                U_reshaped = U_flat.reshape(n*m, rank)
                V_reshaped = V_flat.reshape(m*p, rank)
                W_reshaped = W_flat.reshape(p*n, rank)
                
                return compute_error(U_reshaped, V_reshaped, W_reshaped)
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Use Nelder-Mead as it doesn't require gradients and is robust for small dims
            # Dim is 3 * 9 * 23 = 621. Nelder-Mead might be slow.
            # Let's stick to gradient descent with better hyperparameters.
            
            lr = 0.01
            steps = 200
            
            for step in range(steps):
                if time.time() > deadline:
                    break
                    
                # Compute gradients
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                # Non-zero terms
                for a, b, c in nz_indices:
                    val = np.sum(U[a] * V[b] * W[c])
                    err = val - 1.0
                    grad_U[a] += 2 * err * V[b] * W[c]
                    grad_V[b] += 2 * err * U[a] * W[c]
                    grad_W[c] += 2 * err * U[a] * V[b]
                
                # Zero terms (sample or all)
                # To save time, we can skip zero gradients in early steps?
                # No, they are crucial for sparsity.
                # Let's do all zero terms.
                nz_set = set(nz_indices)
                for a in range(n*m):
                    for b in range(m*p):
                        for c in range(p*n):
                            if (a, b, c) not in nz_set:
                                val = np.sum(U[a] * V[b] * W[c])
                                grad_U[a] += 2 * val * V[b] * W[c]
                                grad_V[b] += 2 * val * U[a] * W[c]
                                grad_W[c] += 2 * val * U[a] * V[b]
                
                # L1 penalty
                lambda_l1 = 0.05
                grad_U += lambda_l1 * np.sign(U)
                grad_V += lambda_l1 * np.sign(V)
                grad_W += lambda_l1 * np.sign(W)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
            # Check error
            err = compute_error(U, V, W)
            if err < 1e-4:
                result = snap_and_verify(U, V, W, rank)
                if result:
                    return result

    # Fallback
    U_f, V_f, W_f = get_fallback()
    return {
        "U": U_f.tolist(),
        "V": V_f.tolist(),
        "W": W_f.tolist()
    }
