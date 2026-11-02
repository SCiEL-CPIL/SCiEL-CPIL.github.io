import time
import itertools
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s to stay safely under 150s
    
    n, m, p = DIMENSIONS
    # Schoolbook decomposition of rank n*m*p
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    best_rank = schoolbook_rank
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # Strategy: Find rank-11 decomposition for 2x3x3, then use 2 copies for 2x3x6
    # This gives rank 22 vs schoolbook 36
    
    def find_2x3x3_decomposition(rank=11, max_iter=20, time_budget=120):
        """Find a rank-r decomposition for 2x3x3 using scipy minimize."""
        n2, m2, p2 = 2, 3, 3
        start = time.time()
        
        # Dimensions
        dim_U = n2 * m2
        dim_V = m2 * p2
        dim_W = p2 * n2
        
        # Target tensor T[i,j,k] = 1
        # We minimize sum (sum_r U[i,j,r] V[j,k,r] W[k,i,r] - 1)^2
        
        def objective(params):
            if time.time() - start > time_budget:
                return float('inf')
            U = params[:dim_U*rank].reshape(dim_U, rank)
            V = params[dim_U*rank:dim_U*rank + dim_V*rank].reshape(dim_V, rank)
            W = params[dim_U*rank + dim_V*rank:].reshape(dim_W, rank)
            
            loss = 0.0
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        # Compute sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                        val = np.sum(U[i*m2+j, :] * V[j*p2+k, :] * W[k*n2+i, :])
                        loss += (val - 1.0)**2
            return loss

        def objective_grad(params):
            if time.time() - start > time_budget:
                return np.zeros_like(params)
            U = params[:dim_U*rank].reshape(dim_U, rank)
            V = params[dim_U*rank:dim_U*rank + dim_V*rank].reshape(dim_V, rank)
            W = params[dim_U*rank + dim_V*rank:].reshape(dim_W, rank)
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        val = np.sum(U[i*m2+j, :] * V[j*p2+k, :] * W[k*n2+i, :])
                        res = val - 1.0
                        # Gradient w.r.t U[i,j,r] is res * V[j,k,r] * W[k,i,r]
                        dU[i*m2+j, :] += res * V[j*p2+k, :] * W[k*n2+i, :]
                        dV[j*p2+k, :] += res * U[i*m2+j, :] * W[k*n2+i, :]
                        dW[k*n2+i, :] += res * U[i*m2+j, :] * V[j*p2+k, :]
                        
            grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
            return grad

        best_loss = float('inf')
        best_params = None
        
        for attempt in range(max_iter):
            if time.time() - start > time_budget:
                break
            
            # Random initialization
            # Scale matters. If U,V,W are small, products are tiny.
            # We want products to sum to 1.
            # Initialize with small random values.
            init_scale = 1.0 / np.sqrt(rank)
            params = np.random.randn(dim_U*rank + dim_V*rank + dim_W*rank) * init_scale
            
            # Try to optimize
            try:
                res = minimize(objective, params, jac=objective_grad, method='L-BFGS-B', 
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if res.success or res.fun < best_loss:
                    best_loss = res.fun
                    best_params = res.x
            except Exception:
                continue
            
            if best_loss < 1e-6:
                break
        
        if best_params is None:
            return None
            
        # Check finiteness
        if not np.all(np.isfinite(best_params)):
            return None
            
        # Extract U, V, W
        U = best_params[:dim_U*rank].reshape(dim_U, rank)
        V = best_params[dim_U*rank:dim_U*rank + dim_V*rank].reshape(dim_V, rank)
        W = best_params[dim_U*rank + dim_V*rank:].reshape(dim_W, rank)
        
        # Final verification
        max_res = 0
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    val = np.sum(U[i*m2+j, :] * V[j*p2+k, :] * W[k*n2+i, :])
                    max_res = max(max_res, abs(val - 1.0))
                    
        if max_res < 1e-5:
            return U, V, W
            
        return None

    # Try to find rank-11 decomposition for 2x3x3
    if time.time() < deadline - 10:
        result = find_2x3x3_decomposition(rank=11, max_iter=10, time_budget=120)
        if result is not None:
            U233, V233, W233 = result
            # Use this decomposition 2 times for 2x3x6
            rank = 22
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            for block in range(2):
                base = block * 11
                for i in range(n):
                    for j in range(m):
                        for r in range(11):
                            U[i*m+j][base+r] = U233[i*m+j, r]
                for j in range(m):
                    for k in range(3):
                        for r in range(11):
                            V[j*p + block*3 + k][base+r] = V233[j*3+k, r]
                for k in range(3):
                    for i in range(n):
                        for r in range(11):
                            W[(block*3+k)*n + i][base+r] = W233[k*2+i, r]
            
            best_rank = rank
            best_U = U
            best_V = V
            best_W = W
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
