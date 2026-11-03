"""Improved portfolio: Better ALS with structured initialization for 2x5x5 matrix multiplication tensor."""

import time
import numpy as np
import sys

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = 145.0  # 145 seconds safety margin (must be < 150s)
    
    n, m, p = 2, 5, 5
    
    # Helper to build schoolbook
    def build_schoolbook(rank):
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    # Verify a decomposition
    def is_valid(U, V, W, tol=1e-8):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(total - 1) > tol:
                        return False
        return True
    
    # Compute error for a decomposition
    def compute_error(U, V, W):
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    max_err = max(max_err, abs(total - 1))
        return max_err
    
    # Start with schoolbook as best known
    best_rank = n * m * p
    U_best, V_best, W_best = build_schoolbook(best_rank)
    
    # Structured initialization: exploit the 2-dimension
    def structured_init(R, seed=42):
        """Initialize with structure that may help escape local minima."""
        np.random.seed(seed)
        
        # Strategy: split rank into two halves, one for each i-value
        # For i=0: U[0*m+j, :R//2], V[j*p+k, :R//2], W[k*n+0, :R//2]
        # For i=1: U[1*m+j, R//2:], V[j*p+k, R//2:], W[k*n+1, R//2:]
        # This reduces to two independent 1x5x5 problems
        
        half = R // 2
        remainder = R - 2 * half
        
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        
        # Initialize each half independently
        for i in range(n):
            offset = i * half
            # Random initialization for the i-th block
            U[i*m:(i+1)*m, offset:offset+half] = np.random.randn(m, half) * 0.5
            V[:, offset:offset+half] = np.random.randn(m*p, half) * 0.5
            W[i::n, offset:offset+half] = np.random.randn(p, half) * 0.5
        
        # If there's a remainder, add a small random component
        if remainder > 0:
            U[:, 2*half:] = np.random.randn(n*m, remainder) * 0.3
            V[:, 2*half:] = np.random.randn(m*p, remainder) * 0.3
            W[:, 2*half:] = np.random.randn(p*n, remainder) * 0.3
        
        # Normalize
        U /= np.max(np.abs(U)) + 1e-10
        V /= np.max(np.abs(V)) + 1e-10
        W /= np.max(np.abs(W)) + 1e-10
        
        return U, V, W
    
    # Try to find a valid decomposition of rank R using optimization
    def try_rank(R, seed=42, use_structured=True):
        """Try to find a valid decomposition of rank R using optimization."""
        if time.time() > deadline:
            return None
        
        if use_structured:
            U, V, W = structured_init(R, seed=seed)
        else:
            np.random.seed(seed)
            U = np.random.randn(n*m, R) * 0.5
            V = np.random.randn(m*p, R) * 0.5
            W = np.random.randn(p*n, R) * 0.5
            U /= np.max(np.abs(U)) + 1e-10
            V /= np.max(np.abs(V)) + 1e-10
            W /= np.max(np.abs(W)) + 1e-10
        
        # Alternating least squares
        for it in range(300):
            if time.time() > deadline - 5:
                return None
            
            # Solve for U given V and W
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    A_sub = np.zeros((p, R))
                    for k in range(p):
                        A_sub[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b_sub = np.ones(p)
                    try:
                        U_new[i*m+j, :] = np.linalg.lstsq(A_sub, b_sub, rcond=None)[0]
                    except:
                        U_new[i*m+j, :] = U[i*m+j, :]
            U = U_new
            
            # Solve for V given U and W
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    A_sub = np.zeros((n, R))
                    for i in range(n):
                        A_sub[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b_sub = np.ones(n)
                    try:
                        V_new[j*p+k, :] = np.linalg.lstsq(A_sub, b_sub, rcond=None)[0]
                    except:
                        V_new[j*p+k, :] = V[j*p+k, :]
            V = V_new
            
            # Solve for W given U and V
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    A_sub = np.zeros((m, R))
                    for j in range(m):
                        A_sub[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b_sub = np.ones(m)
                    try:
                        W_new[k*n+i, :] = np.linalg.lstsq(A_sub, b_sub, rcond=None)[0]
                    except:
                        W_new[k*n+i, :] = W[k*n+i, :]
            W = W_new
            
            # Check convergence
            err = compute_error(U, V, W)
            if err < 1e-7:
                return U, V, W
        
        return None
    
    # Try structured initialization first with various ranks
    for R in range(25, 50):
        if time.time() > deadline - 10:
            break
        found = False
        for seed in range(20):
            if time.time() > deadline - 5:
                break
            result = try_rank(R, seed=seed + 200, use_structured=True)
            if result is not None:
                U_best, V_best, W_best = result
                best_rank = R
                found = True
                break
        if found:
            break
    
    # If structured didn't work, try random initialization with more seeds
    if best_rank == 50:
        for R in range(30, 50):
            if time.time() > deadline - 10:
                break
            found = False
            for seed in range(15):
                if time.time() > deadline - 5:
                    break
                result = try_rank(R, seed=seed + 500, use_structured=False)
                if result is not None:
                    U_best, V_best, W_best = result
                    best_rank = R
                    found = True
                    break
            if found:
                break
    
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
