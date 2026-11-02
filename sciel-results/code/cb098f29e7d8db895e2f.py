import time
import numpy as np
from numpy.linalg import norm

DIMENSIONS = [2, 4, 4]

def _strassen_decomposition():
    """Fallback: block Strassen decomposition giving rank 28."""
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_steps = [
        {'u': {0: 1, 3: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}},
        {'u': {2: 1, 3: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}},
        {'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}},
        {'u': {3: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}},
        {'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}},
        {'u': {2: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}},
        {'u': {1: 1, 3: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}}
    ]
    
    def add_block(a_col_start, b_row_start, b_col_start, c_col_start, col_offset):
        for s in range(7):
            step = strassen_steps[s]
            col = col_offset + s
            for local_idx, coeff in step['u'].items():
                l_row = local_idx // 2
                l_col = local_idx % 2
                global_i = l_row
                global_j = a_col_start + l_col
                global_idx = global_i * m + global_j
                U[global_idx][col] += coeff
            for local_idx, coeff in step['v'].items():
                l_row = local_idx // 2
                l_col = local_idx % 2
                global_i = b_row_start + l_row
                global_j = b_col_start + l_col
                global_idx = global_i * p + global_j
                V[global_idx][col] += coeff
            for local_idx, coeff in step['w'].items():
                l_row = local_idx // 2
                l_col = local_idx % 2
                global_k = c_col_start + l_col
                global_i = l_row
                global_idx = global_k * n + global_i
                W[global_idx][col] += coeff

    add_block(0, 0, 0, 0, 0)
    add_block(0, 0, 2, 2, 7)
    add_block(2, 2, 0, 0, 14)
    add_block(2, 2, 2, 2, 21)
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Reshape to 3D tensors
    # U: (n*m, rank) -> (n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)
    # W: (p*n, rank) -> (p, n, rank) with W[k*n+i, r]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # The tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where i in [0,n), j in [0,m), k in [0,p)
    # Check: for each (i,j,k), sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] should equal 1 if i,j,k form a valid multiplication path, 0 otherwise
    
    # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k) since
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor should be all ones for the multiplication tensor
    
    # Let's compute the reconstructed tensor
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
                expected = 1.0  # Matrix multiplication tensor is all 1s
                err = abs(val - expected)
                if err > max_err:
                    max_err = err
    
    return max_err < tol, max_err

def _als_minimize(rank, n, m, p, max_iters=500, tol=1e-12, seed=42, sparsity_lambda=0.01):
    """ALS with sparse regularization to find low-rank decomposition."""
    np.random.seed(seed)
    
    # Target tensor: all ones (matrix multiplication tensor)
    # T[i,j,k] = 1 for all i,j,k
    
    # Initialize factors
    U = np.random.randn(n, m, rank) * 0.5
    V = np.random.randn(m, p, rank) * 0.5
    W = np.random.randn(p, n, rank) * 0.5
    
    best_err = float('inf')
    best_factors = (U.copy(), V.copy(), W.copy())
    
    for it in range(max_iters):
        # Update U: for each (i,j), U[i,j,:] = argmin sum_k (T[i,j,k] - sum_r U[i,j,r]*V[j,k,r]*W[k,i,r])^2
        # Linear in U[i,j,:]
        for i in range(n):
            for j in range(m):
                # Build the linear system
                # residual_k = T[i,j,k] - sum_r' U[i,j,r']*V[j,k,r']*W[k,i,r']
                # We want to minimize sum_k residual_k^2
                # This is linear in U[i,j,:]
                
                # A[r1,r2] = sum_k V[j,k,r1] * W[k,i,r2]
                # b[r] = sum_k T[i,j,k] * V[j,k,r] * W[k,i,r]  (but T is all 1s)
                # Actually: minimize ||t - A^T u||^2 where t[k] = sum_r u_r * V[j,k,r] * W[k,i,r]
                
                # Let's build the normal equations
                # For each k: pred_k = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                # err_k = 1 - pred_k
                # d(err^2)/dU[i,j,s] = -2 * sum_k V[j,k,s] * W[k,i,s] * err_k
                
                # Set up: minimize ||b - M u||^2 where M[k,s] = V[j,k,s]*W[k,i,s], b[k] = 1
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j, k, :] * W[k, i, :]
                b = np.ones(p)
                
                # Solve M^T M u = M^T b
                MTM = M.T @ M
                MTb = M.T @ b
                
                # Add regularization
                reg = sparsity_lambda * np.eye(rank)
                MTM += reg
                
                try:
                    U[i, j, :] = np.linalg.solve(MTM, MTb)
                except:
                    U[i, j, :] = np.linalg.lstsq(MTM, MTb, rcond=None)[0]
        
        # Update V: for each (j,k), V[j,k,:] = argmin
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = U[i, j, :] * W[k, i, :]
                b = np.ones(n)
                
                MTM = M.T @ M
                MTb = M.T @ b
                MTM += sparsity_lambda * np.eye(rank)
                
                try:
                    V[j, k, :] = np.linalg.solve(MTM, MTb)
                except:
                    V[j, k, :] = np.linalg.lstsq(MTM, MTb, rcond=None)[0]
        
        # Update W: for each (k,i), W[k,i,:] = argmin
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U[i, j, :] * V[j, k, :]
                b = np.ones(m)
                
                MTM = M.T @ M
                MTb = M.T @ b
                MTM += sparsity_lambda * np.eye(rank)
                
                try:
                    W[k, i, :] = np.linalg.solve(MTM, MTb)
                except:
                    W[k, i, :] = np.linalg.lstsq(MTM, MTb, rcond=None)[0]
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i,j,:] * V[j,k,:] * W[k,i,:])
                    err += (1.0 - val) ** 2
        err = np.sqrt(err / (n * m * p))
        
        if err < best_err:
            best_err = err
            best_factors = (U.copy(), V.copy(), W.copy())
        
        if err < tol:
            break
    
    return best_err, best_factors

def _try_rank(rank, n, m, p, time_budget, start_time):
    """Try to find a valid decomposition of given rank."""
    if time.time() - start_time > time_budget - 5:
        return None
    
    # Try multiple seeds
    for seed in range(20):
        if time.time() - start_time > time_budget - 5:
            break
        
        err, (U3, V3, W3) = _als_minimize(
            rank, n, m, p, 
            max_iters=200, 
            tol=1e-12, 
            seed=seed,
            sparsity_lambda=0.01
        )
        
        if err < 1e-10:
            # Convert to flat format
            U_flat = U3.reshape(n*m, rank).tolist()
            V_flat = V3.reshape(m*p, rank).tolist()
            W_flat = W3.reshape(p*n, rank).tolist()
            
            # Verify
            valid, max_err = _verify_decomposition(U_flat, V_flat, W_flat, n, m, p)
            if valid:
                return {"U": U_flat, "V": V_flat, "W": W_flat}
    
    return None

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_budget = 120  # seconds
    
    # First, get the baseline Strassen solution (rank 28)
    baseline = _strassen_decomposition()
    baseline_rank = 28
    
    # Try to find a lower rank solution
    # Try rank 27, 26, 25, ...
    for target_rank in range(27, 20, -1):
        if time.time() - start_time > time_budget - 30:
            break
        
        result = _try_rank(target_rank, n, m, p, time_budget, start_time)
        if result is not None:
            return result
    
    # Return the baseline if no better solution found
    return baseline
# EVOLVE-BLOCK-END
