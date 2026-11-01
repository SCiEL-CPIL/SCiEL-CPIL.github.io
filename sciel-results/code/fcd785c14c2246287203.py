import time
import math
import random
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Construct an exact tensor decomposition for 3x3 * 3x6 matrix multiplication.
    Uses gradient-based optimization (L-BFGS-B) to find a lower-rank decomposition
    by minimizing the squared error of the tensor reconstruction.
    Falls back to schoolbook rank 54.
    """
    n, m, p = 3, 3, 6
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    # Schoolbook decomposition: rank = n*m*p = 54
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    def verify(U, V, W, n, m, p):
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    def verify_np(U, V, W, n, m, p, tol=1e-7):
        """Vectorized verification using numpy."""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    if abs(s - 1) > tol:
                        return False
        return True
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Precompute index mappings for the tensor
    # For each (i,j,k), we need U[i*m+j], V[j*p+k], W[k*n+i]
    tensor_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor_indices.append((i*m+j, j*p+k, k*n+i))
    
    def objective_and_grad(x, r, n, m, p):
        """
        Compute objective (sum of squared errors) and gradient.
        x is a flat vector of length (n*m + m*p + p*n) * r
        """
        U = x[:n*m*r].reshape(n*m, r)
        V = x[n*m*r:(n*m+m*p)*r].reshape(m*p, r)
        W = x[(n*m+m*p)*r:].reshape(p*n, r)
        
        # Compute residuals for each (i,j,k)
        residuals = np.zeros(n*m*p)
        for idx, (ui, vi, wi) in enumerate(tensor_indices):
            residuals[idx] = np.sum(U[ui] * V[vi] * W[wi]) - 1.0
        
        loss = np.sum(residuals ** 2)
        
        # Gradient computation
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for idx, (ui, vi, wi) in enumerate(tensor_indices):
            res = residuals[idx]
            dU[ui] += 2 * res * V[vi] * W[wi]
            dV[vi] += 2 * res * U[ui] * W[wi]
            dW[wi] += 2 * res * U[ui] * V[vi]
        
        grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
        return loss, grad
    
    def try_rank_gradient(r, n, m, p, time_limit=30, n_restarts=5):
        """Try to find a rank-r decomposition using gradient optimization."""
        t0 = time.time()
        
        for attempt in range(n_restarts):
            if time.time() - t0 > time_limit:
                break
            
            # Initialize with random sparse values
            np.random.seed(attempt * 1000 + r)
            U_init = np.random.randn(n*m, r) * 0.5
            V_init = np.random.randn(m*p, r) * 0.5
            W_init = np.random.randn(p*n, r) * 0.5
            
            # Make initialization sparse to help convergence
            for mat in [U_init, V_init, W_init]:
                mask = np.random.rand(*mat.shape) < 0.3
                mat[~mask] = 0
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Run L-BFGS-B
            try:
                result = minimize(
                    objective_and_grad,
                    x0,
                    args=(r, n, m, p),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 2000}
                )
            except Exception:
                continue
            
            if not result.success:
                continue
            
            # Extract solution
            U_opt = result.x[:n*m*r].reshape(n*m, r)
            V_opt = result.x[n*m*r:(n*m+m*p)*r].reshape(m*p, r)
            W_opt = result.x[(n*m+m*p)*r:].reshape(p*n, r)
            
            # Check if solution is valid
            if verify_np(U_opt, V_opt, W_opt, n, m, p, tol=1e-6):
                # Try to round to half-integers
                U_rounded = np.round(U_opt * 2) / 2
                V_rounded = np.round(V_opt * 2) / 2
                W_rounded = np.round(W_opt * 2) / 2
                
                if verify_np(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-6):
                    U_list = U_rounded.tolist()
                    V_list = V_rounded.tolist()
                    W_list = W_rounded.tolist()
                    if verify(U_list, V_list, W_list, n, m, p):
                        return U_list, V_list, W_list
                
                # If rounding doesn't work, try to find nearby rational values
                # For now, return the float solution if it's very close
                if verify_np(U_opt, V_opt, W_opt, n, m, p, tol=1e-8):
                    U_list = U_opt.tolist()
                    V_list = V_opt.tolist()
                    W_list = W_opt.tolist()
                    if verify(U_list, V_list, W_list, n, m, p):
                        return U_list, V_list, W_list
        
        return None
    
    # Try various ranks from lower bound to schoolbook
    # Lower bound for 3x3*3x6 is at least 18 (max of 9, 18, 18)
    # Try from 18 up to 53
    for r in range(18, min(rank_school, 54)):
        if time.time() > deadline - 15:
            break
        
        time_per_rank = min(20, (deadline - 15 - time.time()) / max(1, (rank_school - r)))
        result = try_rank_gradient(r, n, m, p, time_limit=time_per_rank, n_restarts=3)
        
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            break
    
    return best_result
