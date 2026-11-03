"""Matrix multiplication 3x3x6 - rank-minimal tensor decomposition."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    # Schoolbook rank = n*m*p = 54
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None

    # Verify a decomposition
    def verify(U, V, W, rank):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Schoolbook decomposition
    def schoolbook_decomp():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, rank

    # Compute tensor using vectorized operations
    def compute_tensor(U, V, W, rank):
        U_reshaped = U.reshape(n, m, rank)
        V_reshaped = V.reshape(m, p, rank)
        W_reshaped = W.reshape(p, n, rank)
        tensor = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        return tensor

    # Compute objective: sum of squared errors
    def objective(x):
        rank = x.size // (n*m + m*p + p*n)
        U = x[:n*m*rank].reshape(n*m, rank)
        V = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
        tensor = compute_tensor(U, V, W, rank)
        error = 1.0 - tensor
        return np.sum(error**2)

    # Compute gradients
    def compute_gradients(U, V, W, rank, error):
        U_reshaped = U.reshape(n, m, rank)
        V_reshaped = V.reshape(m, p, rank)
        W_reshaped = W.reshape(p, n, rank)
        error_reshaped = error.reshape(n, m, p)
        
        dU_reshaped = np.einsum('ijk,jkr,kir->ijr', error_reshaped, V_reshaped, W_reshaped)
        dV_reshaped = np.einsum('ijk,ijr,kir->jkr', error_reshaped, U_reshaped, W_reshaped)
        dW_reshaped = np.einsum('ijk,ijr,jkr->kir', error_reshaped, U_reshaped, V_reshaped)
        
        dU = dU_reshaped.reshape(n*m, rank)
        dV = dV_reshaped.reshape(m*p, rank)
        dW = dW_reshaped.reshape(p*n, rank)
        
        return dU, dV, dW

    # Try to find a decomposition at a given rank using gradient-based optimization
    def try_rank(target_rank, n_init=5, max_iter=2000):
        nonlocal best_rank, best_U, best_V, best_W
        
        for init_idx in range(n_init):
            if time.time() > deadline:
                return False
            
            np.random.seed(42 + target_rank * 100 + init_idx)
            
            # Initialize with structure: try block-diagonal for even ranks
            if target_rank % 2 == 0:
                r3 = target_rank // 2
                U = np.zeros((n*m, target_rank))
                V = np.zeros((m*p, target_rank))
                W = np.zeros((p*n, target_rank))
                
                for block in range(2):
                    offset = block * r3
                    U[:, offset:offset+r3] = np.random.randn(n*m, r3) * 0.5
                    V[:, offset:offset+r3] = np.random.randn(m*p, r3) * 0.5
                    W[:, offset:offset+r3] = np.random.randn(p*n, r3) * 0.5
                    
                    if block == 0:
                        V[9:18, offset:offset+r3] = 0
                    else:
                        V[0:9, offset:offset+r3] = 0
            else:
                U = np.random.randn(n*m, target_rank) * 0.5
                V = np.random.randn(m*p, target_rank) * 0.5
                W = np.random.randn(p*n, target_rank) * 0.5
            
            # Gradient descent with adaptive step size
            eta = 0.01
            best_local_error = np.inf
            best_local_U = U.copy()
            best_local_V = V.copy()
            best_local_W = W.copy()
            
            for iteration in range(max_iter):
                if time.time() > deadline:
                    break
                
                tensor = compute_tensor(U, V, W, target_rank)
                error = 1.0 - tensor
                current_error = np.max(np.abs(error))
                
                if current_error < 1e-10:
                    best_rank = target_rank
                    best_U = U.tolist()
                    best_V = V.tolist()
                    best_W = W.tolist()
                    return True
                
                if current_error < best_local_error:
                    best_local_error = current_error
                    best_local_U = U.copy()
                    best_local_V = V.copy()
                    best_local_W = W.copy()
                
                # Compute gradients
                dU, dV, dW = compute_gradients(U, V, W, target_rank, error)
                
                # Adaptive step size
                grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
                if grad_norm > 1e-15:
                    step = eta * (current_error / grad_norm)
                    step = min(step, 0.1)
                else:
                    step = eta
                
                U -= step * dU
                V -= step * dV
                W -= step * dW
                
                # Decay learning rate
                if iteration % 50 == 0:
                    eta *= 0.95
                    if eta < 0.001:
                        eta = 0.001
            
            # Check if we found a solution
            if best_rank <= target_rank:
                return True
            
            # If close, try to polish
            if best_local_error < 1e-6:
                U, V, W = best_local_U, best_local_V, best_local_W
                eta = 0.001
                for iteration in range(500):
                    if time.time() > deadline:
                        break
                    
                    tensor = compute_tensor(U, V, W, target_rank)
                    error = 1.0 - tensor
                    current_error = np.max(np.abs(error))
                    
                    if current_error < 1e-10:
                        best_rank = target_rank
                        best_U = U.tolist()
                        best_V = V.tolist()
                        best_W = W.tolist()
                        return True
                    
                    dU, dV, dW = compute_gradients(U, V, W, target_rank, error)
                    U -= eta * dU
                    V -= eta * dV
                    W -= eta * dW
        
        return False

    # Try ranks from 46 down to 40
    candidate_ranks = [46, 45, 44, 43, 42, 41, 40, 39, 38]
    
    for target_rank in candidate_ranks:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
        
        success = try_rank(target_rank, n_init=3, max_iter=1500)
        if success:
            break
    
    # If we haven't found a better rank, use schoolbook
    if best_U is None:
        best_U, best_V, best_W, best_rank = schoolbook_decomp()
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
