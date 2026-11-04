import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s limit
    
    # Seed solution: rank 40 (schoolbook)
    rank_seed = n * m * p
    U_seed = [[0] * rank_seed for _ in range(n * m)]
    V_seed = [[0] * rank_seed for _ in range(m * p)]
    W_seed = [[0] * rank_seed for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_seed[i * m + j][col] = 1
                V_seed[j * p + k][col] = 1
                W_seed[k * n + i][col] = 1
                col += 1
    best_U, best_V, best_W = U_seed, V_seed, W_seed
    best_rank = rank_seed
    
    # Build the matrix multiplication tensor T[i,j,k] = 1 if this is a valid term
    # The tensor is T_{i,j,k} for A_{i,j} * B_{j,k} = C_{i,k}
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,j,k} for the multiplication
    # Actually, the correct formulation:
    # C_{i,k} = sum_j A_{i,j} * B_{j,k}
    # So the tensor T has T[i,j,k] = 1 for all i,j,k (it's the full multiplication tensor)
    # And we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Wait, let me reconsider. The standard formulation:
    # We want to compute C = A @ B where A is n×m, B is m×p, C is n×p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # The tensor decomposition approach:
    # We have three arrays U (n*m × r), V (m*p × r), W (n*p × r)
    # The constraint is: for each (i,j,k) triple, we need the product to contribute correctly
    # 
    # Actually, the correct tensor for matrix multiplication M_{n,m,p} is:
    # T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # And the decomposition is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    # 
    # But wait, that's not quite right either. Let me think more carefully.
    # 
    # The bilinear complexity approach:
    # We want to compute C_{i,k} = sum_j A_{i,j} B_{j,k}
    # We can write: C_{i,k} = sum_r w_{i,k,r} * (sum_j u_{i,j,r} A_{i,j}) * (sum_j v_{j,k,r} B_{j,k})
    # No, that's not right either.
    # 
    # The standard tensor decomposition for matrix multiplication:
    # The multiplication tensor T ∈ R^{n×m} ⊗ R^{m×p} ⊗ R^{n×p}
    # T = sum_r U_r ⊗ V_r ⊗ W_r
    # where U_r ∈ R^{n×m}, V_r ∈ R^{m×p}, W_r ∈ R^{n×p}
    # and T[A,B,C] = sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} C_{i,k}
    # For matrix multiplication, T_{i,j,k} = 1 for all i,j,k
    # 
    # So we need: 1 = sum_r U_r[i,j] * V_r[j,k] * W_r[i,k] for all i,j,k
    # 
    # In vectorized form with the given indexing:
    # U[i*m+j, r] = U_r[i,j]
    # V[j*p+k, r] = V_r[j,k]  
    # W[k*n+i, r] = W_r[i,k]  (note: the problem says W[k*n+i,r])
    # 
    # So the constraint is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    def tensor_residual(U, V, W):
        """Compute the residual: how far are we from satisfying the tensor equation"""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        r = U.shape[1]
        # For each (i,j,k), compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        residuals = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r_idx in range(r):
                        val += U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx]
                    residuals.append(val - 1.0)
        return residuals
    
    def objective(params, r):
        """Objective function: sum of squared residuals"""
        total_size = n*m + m*p + n*p
        U_size = n * m * r
        V_size = m * p * r
        W_size = n * p * r
        
        U = params[:U_size].reshape(n*m, r)
        V = params[U_size:U_size+V_size].reshape(m*p, r)
        W = params[U_size+V_size:U_size+V_size+W_size].reshape(n*p, r)
        
        # Compute residuals
        res = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    res[idx] = val - 1.0
                    idx += 1
        return np.sum(res**2)
    
    def try_rank(r, max_restarts=5, max_iter=1000):
        """Try to find a rank-r decomposition"""
        if r >= best_rank:
            return None
        
        U_size = n * m * r
        V_size = m * p * r
        W_size = n * p * r
        total_size = U_size + V_size + W_size
        
        best_obj = float('inf')
        best_params = None
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization
            np.random.seed(42 + restart * 1000)
            scale = 1.0 / np.sqrt(r)
            params = np.random.randn(total_size) * scale
            
            # Try to optimize
            try:
                result = minimize(
                    objective,
                    params,
                    args=(r,),
                    method='L-BFGS-B',
                    options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_params = result.x
            except Exception:
                continue
        
        if best_params is not None and best_obj < 1e-8:
            U = best_params[:U_size].reshape(n*m, r)
            V = best_params[U_size:U_size+V_size].reshape(m*p, r)
            W = best_params[U_size+V_size:U_size+V_size+W_size].reshape(n*p, r)
            
            # Verify
            res = tensor_residual(U, V, W)
            if max(abs(x) for x in res) < 1e-6:
                # Convert to list of lists with rounded values
                U_list = [[round(float(x), 8) for x in row] for row in U]
                V_list = [[round(float(x), 8) for x in row] for row in V]
                W_list = [[round(float(x), 8) for x in row] for row in W]
                return U_list, V_list, W_list
        
        return None
    
    # Try ranks from 39 down to some minimum
    # Known lower bound for 2x4x5 is around 30-35
    for r in range(min(rank_seed - 1, 39), max(20, 0), -1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(r, max_restarts=3, max_iter=500)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = r
            break  # Found a valid solution at this rank
    
    return {"U": best_U, "V": best_V, "W": best_W}
