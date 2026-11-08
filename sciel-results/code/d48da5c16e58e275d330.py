"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 5]

def verify_decomposition(U, V, W, n=2, m=3, p=5, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    
    # Check: T[i,j,k,l] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+l,c] = delta_{i,l}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for l in range(n):
                    val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+l, c] for c in range(rank))
                    expected = 1.0 if i == l else 0.0
                    if abs(val - expected) > tol:
                        return False
    return True

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x5 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    # Start with schoolbook rank 30
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook decomposition
    U = np.zeros((n*m, best_rank))
    V = np.zeros((m*p, best_rank))
    W = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    if verify_decomposition(U.tolist(), V.tolist(), W.tolist()):
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to find lower rank decompositions using optimization
    # We'll try ranks from 15 upward (theoretical lower bound considerations)
    
    # For a 2x3 * 3x5 multiplication, let's try various ranks
    # The bilinear complexity is at least max(n*m, m*p, p*n) = max(6, 15, 10) = 15
    # But we also need to satisfy the tensor structure
    
    for target_rank in range(15, best_rank):
        if time.time() > deadline:
            break
        
        # Initialize with random values
        np.random.seed(42 + target_rank)
        U_init = np.random.randn(n*m, target_rank)
        V_init = np.random.randn(m*p, target_rank)
        W_init = np.random.randn(p*n, target_rank)
        
        # Define the objective: sum of squared errors
        def objective(params):
            if time.time() > deadline:
                return float('inf')
            U_flat, V_flat, W_flat = params[:n*m*target_rank], params[n*m*target_rank:n*m*target_rank+m*p*target_rank], params[n*m*target_rank+m*p*target_rank:]
            U_mat = U_flat.reshape(n*m, target_rank)
            V_mat = V_flat.reshape(m*p, target_rank)
            W_mat = W_flat.reshape(p*n, target_rank)
            
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for l in range(n):
                            val = np.dot(U_mat[i*m+j], V_mat[j*p+k] * W_mat[k*n+l])
                            expected = 1.0 if i == l else 0.0
                            error += (val - expected) ** 2
            return error
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-9})
            
            if result.success or result.fun < 1e-10:
                U_opt = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_opt = result.x[n*m*target_rank:n*m*target_rank+m*p*target_rank].reshape(m*p, target_rank)
                W_opt = result.x[n*m*target_rank+m*p*target_rank:].reshape(p*n, target_rank)
                
                if verify_decomposition(U_opt.tolist(), V_opt.tolist(), W_opt.tolist()):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_opt.copy(), V_opt.copy(), W_opt.copy()
                    break  # Found minimal rank, stop
        except Exception:
            continue
    
    # Convert to lists for return
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
