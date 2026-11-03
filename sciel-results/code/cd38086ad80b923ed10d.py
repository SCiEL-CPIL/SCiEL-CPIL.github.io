import numpy as np
import time
import itertools
from scipy.optimize import minimize, LinearConstraint

DIMENSIONS = [3, 4, 8]
n, m, p = DIMENSIONS
nm, mp, pn = n * m, m * p, p * n

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matmul tensor with half-integer entries."""
    # Check entries are half-integers (within floating point tolerance)
    for M in (U, V, W):
        M2 = np.array(M, dtype=np.float64) * 2.0
        if not np.allclose(M2, np.round(M2), atol=1e-7):
            return False
    
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _snap_to_half_integer(M):
    """Snap matrix entries to nearest 0.5."""
    return np.round(np.array(M, dtype=np.float64) * 2.0) / 2.0

def _construct_schoolbook(n, m, p):
    """Construct the trivial rank n*m*p decomposition."""
    R = n * m * p
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def _residual_tensor(U, V, W, n, m, p):
    """Compute the residual tensor: T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] - 1"""
    R = U.shape[1]
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i,j,k] = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i]) - 1.0
    return T

def _residual_norm(U, V, W, n, m, p):
    """Compute the max absolute residual."""
    T = _residual_tensor(U, V, W, n, m, p)
    return np.max(np.abs(T))

def _try_rank_with_heuristics(R, n, m, p, time_limit, seed_base=0):
    """
    Try to find a rank-R decomposition using structured initialization and refinement.
    Returns (U, V, W) if found, else None.
    """
    start_time = time.time()
    max_attempts = 50
    
    for attempt in range(max_attempts):
        if time.time() - start_time > time_limit:
            break
            
        seed = seed_base + attempt * 137
        np.random.seed(seed)
        
        # Initialize with structured random values
        # Use a mix of 0, 0.5, -0.5, 1, -1 patterns
        U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, R), p=[0.1, 0.2, 0.2, 0.2, 0.1])
        V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, R), p=[0.1, 0.2, 0.2, 0.2, 0.1])
        W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, R), p=[0.1, 0.2, 0.2, 0.2, 0.1])
        
        # Ensure no column is entirely zero
        for r in range(R):
            if np.all(U[:, r] == 0):
                U[np.random.randint(n*m), r] = 1.0
            if np.all(V[:, r] == 0):
                V[np.random.randint(m*p), r] = 1.0
            if np.all(W[:, r] == 0):
                W[np.random.randint(p*n), r] = 1.0
        
        # Iterative refinement: alternating minimization with gradient descent
        lr = 0.05
        for step in range(200):
            if time.time() - start_time > time_limit:
                break
            
            # Compute residuals
            T = _residual_tensor(U, V, W, n, m, p)
            max_res = np.max(np.abs(T))
            
            if max_res < 1e-10:
                break
            
            # Gradient for U
            grad_U = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        grad_U[i*m+j] += -T[i,j,k] * V[j*p+k] * W[k*n+i]
            
            # Gradient for V
            grad_V = np.zeros_like(V)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        grad_V[j*p+k] += -T[i,j,k] * U[i*m+j] * W[k*n+i]
            
            # Gradient for W
            grad_W = np.zeros_like(W)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        grad_W[k*n+i] += -T[i,j,k] * U[i*m+j] * V[j*p+k]
            
            # Adaptive learning rate
            grad_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
            if grad_norm > 0:
                effective_lr = lr * max_res / max(grad_norm, 1e-10)
                effective_lr = min(effective_lr, 0.5)
            else:
                effective_lr = 0.0
            
            U += effective_lr * grad_U
            V += effective_lr * grad_V
            W += effective_lr * grad_W
        
        # Snap to half-integers
        U_snap = _snap_to_half_integer(U)
        V_snap = _snap_to_half_integer(V)
        W_snap = _snap_to_half_integer(W)
        
        if _verify(U_snap, V_snap, W_snap, n, m, p):
            return U_snap, V_snap, W_snap
        
        # Try local search: flip individual entries
        for local_iter in range(100):
            if time.time() - start_time > time_limit:
                break
            
            # Find the worst residual
            T = _residual_tensor(U_snap, V_snap, W_snap, n, m, p)
            max_res = np.max(np.abs(T))
            
            if max_res < 1e-10:
                break
            
            # Try flipping entries to reduce residual
            improved = False
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if abs(T[i,j,k]) < 1e-10:
                            continue
                        
                        # Try flipping U[i*m+j, r] for each r
                        for r in range(R):
                            old_val = U_snap[i*m+j, r]
                            for new_val in [-1, -0.5, 0, 0.5, 1]:
                                if new_val == old_val:
                                    continue
                                U_snap[i*m+j, r] = new_val
                                T_new = _residual_tensor(U_snap, V_snap, W_snap, n, m, p)
                                max_new = np.max(np.abs(T_new))
                                if max_new < max_res:
                                    improved = True
                                    max_res = max_new
                                    break
                                else:
                                    U_snap[i*m+j, r] = old_val
                        
                        # Try flipping V[j*p+k, r] for each r
                        for r in range(R):
                            old_val = V_snap[j*p+k, r]
                            for new_val in [-1, -0.5, 0, 0.5, 1]:
                                if new_val == old_val:
                                    continue
                                V_snap[j*p+k, r] = new_val
                                T_new = _residual_tensor(U_snap, V_snap, W_snap, n, m, p)
                                max_new = np.max(np.abs(T_new))
                                if max_new < max_res:
                                    improved = True
                                    max_res = max_new
                                    break
                                else:
                                    V_snap[j*p+k, r] = old_val
                        
                        # Try flipping W[k*n+i, r] for each r
                        for r in range(R):
                            old_val = W_snap[k*n+i, r]
                            for new_val in [-1, -0.5, 0, 0.5, 1]:
                                if new_val == old_val:
                                    continue
                                W_snap[k*n+i, r] = new_val
                                T_new = _residual_tensor(U_snap, V_snap, W_snap, n, m, p)
                                max_new = np.max(np.abs(T_new))
                                if max_new < max_res:
                                    improved = True
                                    max_res = max_new
                                    break
                                else:
                                    W_snap[k*n+i, r] = old_val
            
            if not improved:
                break
        
        if _verify(U_snap, V_snap, W_snap, n, m, p):
            return U_snap, V_snap, W_snap
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # 10s safety margin before 150s limit
    
    # Schoolbook fallback
    U_sb, V_sb, W_sb = _construct_schoolbook(n, m, p)
    best_result = {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
    best_rank = n * m * p
    
    # Search for lower rank
    # Theoretical lower bound: max(n*m, m*p, n*p) = max(12, 32, 24) = 32
    min_rank = max(n*m, m*p, n*p)
    
    for R in range(min_rank, best_rank + 1):
        if time.time() > deadline:
            break
            
        time_left = deadline - time.time()
        if time_left < 1:
            break
            
        # Give more time to lower ranks (harder problems)
        time_per_rank = min(10.0, time_left / max(1, best_rank - R + 1))
        
        result = _try_rank_with_heuristics(R, n, m, p, time_limit=time_per_rank, seed_base=R)
        
        if result is not None:
            U_sol, V_sol, W_sol = result
            best_result = {
                "U": U_sol.tolist(),
                "V": V_sol.tolist(),
                "W": W_sol.tolist()
            }
            best_rank = R
            # Continue searching for even lower rank
            # We found a valid decomposition at rank R, but there might be one at lower rank
            # However, we're iterating from min_rank upward, so the first valid one is minimal
            return best_result
    
    return best_result
