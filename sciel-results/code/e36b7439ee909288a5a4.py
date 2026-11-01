"""Matrix multiplication 2x4x8 - rank-minimal decomposition via random search with rank reduction."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 4, 8]
DEADLINE = 140  # seconds, safety margin before 150s kill

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    # W index: k*n + i, U index: i*m + j, V index: j*p + k
    # Tensor: T[i*m+j, j*p+k, k*n+i] = 1
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def build_matrices(rank):
        """Build U, V, W matrices as numpy arrays with given rank."""
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        return U, V, W
    
    def check_exact(U, V, W, tol=1e-10):
        """Check if decomposition is exact."""
        # For each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should be 1
        # Efficient check: for each pair (i,j), the vector [U[i*m+j,c]*V[j*p+k,c] for k]
        # dotted with W[k*n+i,c] over c should give 1 for all k
        nm = n*m
        mp = m*p
        pn = p*n
        rank = U.shape[1]
        
        # Check all entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def check_exact_fast(U, V, W, tol=1e-8):
        """Faster exact check using matrix operations."""
        rank = U.shape[1]
        # For each i: build matrix A_i where A_i[j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # = sum_c (U[:,c] * W[i,c]) outer with V[:,c]... 
        # Actually: A_i = sum_c (U[:,c] * W[i,c]) ⊗ V[:,c] reshaped
        # Simpler: for each c, contribution to (i,j,k) is U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def random_init(rank, seed=None):
        """Random initialization with some structure."""
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
        
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Initialize with random values
        for c in range(rank):
            U[:, c] = rng.randn(n*m)
            V[:, c] = rng.randn(m*p)
            W[:, c] = rng.randn(p*n)
        
        return U, V, W
    
    def gradient_step(U, V, W, lr=0.01):
        """One step of gradient descent to minimize sum of (T - sum U*V*W)^2."""
        rank = U.shape[1]
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i*m + j
                    idx_v = j*p + k
                    idx_w = k*n + i
                    err = 1.0 - np.dot(U[idx_u], V[idx_v] * W[idx_w])
                    dU[idx_u] += err * V[idx_v] * W[idx_w]
                    dV[idx_v] += err * U[idx_u] * W[idx_w]
                    dW[idx_w] += err * U[idx_u] * V[idx_v]
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        return U, V, W
    
    def try_rank(rank, max_iters=5000, seed=None):
        """Try to find a valid decomposition of given rank."""
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
        
        best_U, best_V, best_W = None, None, None
        best_err = float('inf')
        
        # Multiple random restarts
        for restart in range(5):
            if time_left() < 5:
                break
                
            U, V, W = random_init(rank, seed=seed + restart * 1000 if seed else None)
            
            # Adaptive learning rate
            lr = 0.05
            for it in range(max_iters):
                if time_left() < 5:
                    break
                    
                # Compute error
                err_sum = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            e = 1.0 - np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                            err_sum += e * e
                
                if err_sum < best_err:
                    best_err = err_sum
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if err_sum < 1e-12:
                    # Check exact
                    if check_exact_fast(U, V, W, tol=1e-10):
                        return U, V, W
                
                # Gradient step
                U, V, W = gradient_step(U, V, W, lr)
                
                # Adaptive lr
                if err_sum > best_err * 1.5:
                    lr *= 0.5
                elif it % 100 == 0 and err_sum < best_err * 0.9:
                    lr = min(lr * 1.1, 0.1)
        
        return best_U, best_V, best_W
    
    # Start with schoolbook as fallback
    schoolbook_rank = n * m * p
    U_sb = np.zeros((n*m, schoolbook_rank))
    V_sb = np.zeros((m*p, schoolbook_rank))
    W_sb = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_result = (U_sb, V_sb, W_sb)
    best_rank = schoolbook_rank
    
    # Try to find lower rank
    # Known lower bounds: max(n*m, m*p, p*n) = max(8, 32, 16) = 32
    # Try ranks from 32 upward
    for rank in range(32, min(schoolbook_rank, 64)):
        if time_left() < 10:
            break
        
        U, V, W = try_rank(rank, max_iters=3000, seed=rank * 7)
        
        if U is not None:
            if check_exact_fast(U, V, W, tol=1e-10):
                best_result = (U, V, W)
                best_rank = rank
                break  # Found minimal, stop
        elif U is not None:
            # Check if this is better than current best
            if rank < best_rank:
                best_result = (U, V, W)
                best_rank = rank
    
    # Convert to list format
    U, V, W = best_result
    rank = best_rank
    
    U_list = [[float(U[i, c]) for c in range(rank)] for i in range(n*m)]
    V_list = [[float(V[i, c]) for c in range(rank)] for i in range(m*p)]
    W_list = [[float(W[i, c]) for c in range(rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
