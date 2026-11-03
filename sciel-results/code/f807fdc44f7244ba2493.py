import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def matmul_residual(params, rank):
        """Compute residual for a given rank decomposition."""
        U = params[:n*m*rank].reshape(n*m, rank)
        V = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
        recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.sum((T - recon)**2)

    def matmul_residual_and_grad(params, rank):
        """Compute residual and gradient."""
        U = params[:n*m*rank].reshape(n*m, rank)
        V = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
        recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        residual = T - recon
        
        # Gradient w.r.t U, V, W
        grad_U = -2 * np.einsum('ijk,jr,kr->ir', residual, V, W)
        grad_V = -2 * np.einsum('ijk,ir,kr->jr', residual, U, W)
        grad_W = -2 * np.einsum('ijk,ir,jr->kr', residual, U, V)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return np.sum(residual**2), grad

    best_result = None
    best_rank = n * m * p  # schoolbook 45

    # Try ranks from 44 down to 30
    # We prioritize finding ANY rank < 45
    for rank in range(44, 29, -1):
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        for trial in range(5):
            if time.time() > deadline - 5:
                break
                
            # Random initialization, scaled
            U_init = np.random.randn(n*m, rank) * 0.1
            V_init = np.random.randn(m*p, rank) * 0.1
            W_init = np.random.randn(p*n, rank) * 0.1
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Use BFGS for optimization
            try:
                res = minimize(
                    matmul_residual_and_grad,
                    params_init,
                    args=(rank,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 100, 'ftol': 1e-12}
                )
            except Exception:
                continue
                
            if res.success and res.fun < 1e-8:
                # Check if it can be rounded to integers/half-integers
                U = res.x[:n*m*rank].reshape(n*m, rank)
                V = res.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                W = res.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                
                # Try rounding to nearest integer
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                
                recon_int = np.einsum('ir,jr,kr->ijk', U_int, V_int, W_int)
                err_int = np.linalg.norm(T - recon_int)
                
                if err_int < 1e-6:
                    best_result = {
                        'U': U_int.tolist(),
                        'V': V_int.tolist(),
                        'W': W_int.tolist()
                    }
                    best_rank = rank
                    break
        
        if best_result is not None:
            break

    if best_result is None:
        # Return schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
                    
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
