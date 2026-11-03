"""Matrix multiplication 3,3,5 rank minimization."""
DIMENSIONS = [3, 3, 5]

import time
import numpy as np
from itertools import product

def construct():
    """
    Return a rank-minimal exact tensor decomposition for 3x3 * 3x5.
    """
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe
    
    n, m, p = DIMENSIONS  # 3, 3, 5
    nm, mp, pn = n*m, m*p, p*n  # 9, 15, 15
    
    # Schoolbook: rank 45
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb
    
    def check_decomposition(U, V, W, rank, tol=1e-8):
        """Check if the decomposition is valid."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def tensor_residual(U, V, W):
        """Compute the maximum residual of the tensor decomposition."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        max_res = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    max_res = max(max_res, abs(val - 1.0))
        return max_res
    
    # Try to find lower rank decompositions using optimization
    # Strategy: Start with rank 44 and try to find a valid decomposition
    # Use alternating minimization / gradient descent
    
    def try_rank_optimization(rank, max_iters=500, tol=1e-8):
        """Try to find a rank decomposition using alternating minimization."""
        if time.time() > deadline - 5:
            return None
        
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(nm, rank) * 0.1
        V = np.random.randn(mp, rank) * 0.1
        W = np.random.randn(pn, rank) * 0.1
        
        # Also try schoolbook-based initialization for rank close to 45
        if rank <= rank_sb:
            U_init = np.zeros((nm, rank))
            V_init = np.zeros((mp, rank))
            W_init = np.zeros((pn, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col < rank:
                            U_init[i*m + j, col] = 1
                            V_init[j*p + k, col] = 1
                            W_init[k*n + i, col] = 1
                            col += 1
                        else:
                            break
                    if col >= rank:
                        break
                if col >= rank:
                    break
            
            # For ranks close to schoolbook, use this initialization
            if rank >= 40:
                U = U_init.copy()
                V = V_init.copy()
                W = W_init.copy()
        
        # Alternating minimization
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Fix V and W, solve for U (linear least squares)
            # For each (i,j), we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
            # This is a linear system in U[i*m+j, :]
            
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    # Build the linear system: A * u = b
                    # A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    # b[k] = 1
                    A = np.zeros((p, rank))
                    b = np.ones(p)
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    
                    # Solve least squares
                    if rank <= p:
                        u, residuals, rank_s, sv = np.linalg.lstsq(A, b, rcond=None)
                        U_new[i*m+j, :] = u
                    else:
                        # Use pseudo-inverse
                        u = np.linalg.lstsq(A, b, rcond=None)[0]
                        U_new[i*m+j, :] = u
            
            U = U_new
            
            # Fix U and W, solve for V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    # Build linear system: A * v = b
                    # A[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    # b[i] = 1
                    A = np.zeros((n, rank))
                    b = np.ones(n)
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    
                    if rank <= n:
                        v, residuals, rank_s, sv = np.linalg.lstsq(A, b, rcond=None)
                        V_new[j*p+k, :] = v
                    else:
                        v = np.linalg.lstsq(A, b, rcond=None)[0]
                        V_new[j*p+k, :] = v
            
            V = V_new
            
            # Fix U and V, solve for W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    # Build linear system: A * w = b
                    # A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    # b[j] = 1
                    A = np.zeros((m, rank))
                    b = np.ones(m)
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    
                    if rank <= m:
                        w, residuals, rank_s, sv = np.linalg.lstsq(A, b, rcond=None)
                        W_new[k*n+i, :] = w
                    else:
                        w = np.linalg.lstsq(A, b, rcond=None)[0]
                        W_new[k*n+i, :] = w
            
            W = W_new
            
            # Check convergence
            res = tensor_residual(U, V, W)
            if res < tol:
                return (U, V, W)
        
        # Check if we converged
        res = tensor_residual(U, V, W)
        if res < tol:
            return (U, V, W)
        return None
    
    # Try ranks from 44 down to 40
    for rank in range(44, 39, -1):
        if time.time() > deadline - 15:
            break
        
        result = try_rank_optimization(rank, max_iters=300)
        if result is not None:
            U, V, W = result
            if check_decomposition(U, V, W, rank):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
                break
    
    # If we found a better rank, try to go lower
    if best_rank < rank_sb:
        for rank in range(best_rank - 1, 35, -1):
            if time.time() > deadline - 10:
                break
            
            result = try_rank_optimization(rank, max_iters=200)
            if result is not None:
                U, V, W = result
                if check_decomposition(U, V, W, rank):
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = rank
                    break
    
    return best_result
