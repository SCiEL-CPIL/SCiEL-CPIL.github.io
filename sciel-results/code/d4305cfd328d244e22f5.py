"""
Rank-minimal exact tensor decomposition for 2x3 * 3x3 matrix multiplication.

Uses a structural approach: systematic search over small half-integer values
for low ranks, leveraging the all-ones structure of the multiplication tensor.
"""

import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 3]
n, m, p = DIMENSIONS
N = n * m  # 6
M = m * p  # 9
P = p * n  # 6
SCHOOLBOOK = n * m * p  # 18

def build_tensor():
    T = np.ones((n, m, p), dtype=np.float64)
    return T

def verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that U, V, W exactly decompose T."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                if abs(val - T[i, j, k]) > tol:
                    return False
    return True

def is_half_integer(arr, tol=1e-9):
    """Check if all entries are integer or half-integer."""
    flat = arr.flatten()
    v = flat * 2
    return np.all(np.abs(v - np.round(v)) < tol)

def round_to_half(arr):
    return np.round(arr * 2) / 2

def schoolbook():
    rank = SCHOOLBOOK
    U = np.zeros((N, rank))
    V = np.zeros((M, rank))
    W = np.zeros((P, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def tensor_residual(U, V, W, T):
    """Compute sum of squared residuals."""
    rank = U.shape[1]
    total = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                total += (s - T[i,j,k])**2
    return total

def try_rank_algebraic(r, T, max_time=30):
    """
    Try to find a rank-r decomposition using algebraic structure.
    
    Key insight: T[i,j,k] = 1 for all (i,j,k). We need:
    sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    
    Strategy: Try to find solutions where the decomposition has a regular structure.
    For example, try to use the fact that we can "share" computations.
    """
    start = time.time()
    
    # Try different initialization strategies
    for strategy in range(10):
        if time.time() - start > max_time:
            break
        
        np.random.seed(12345 + r * 100 + strategy)
        
        # Initialize with small half-integer values
        U = np.random.randint(-2, 3, size=(N, r)) / 2.0
        V = np.random.randint(-2, 3, size=(M, r)) / 2.0
        W = np.random.randint(-2, 3, size=(P, r)) / 2.0
        
        # Alternating least squares with half-integer rounding
        for it in range(500):
            if time.time() - start > max_time:
                break
            
            # Fix V, W; solve for U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((r, p))
                    for k in range(p):
                        for c in range(r):
                            A[c, k] = V[j*p+k, c] * W[k*n+i, c]
                    b = np.array([T[i,j,k] for k in range(p)])
                    try:
                        u, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        U[i*m+j, :] = u
                    except:
                        pass
            
            # Fix U, W; solve for V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((r, n))
                    for i in range(n):
                        for c in range(r):
                            A[c, i] = U[i*m+j, c] * W[k*n+i, c]
                    b = np.array([T[i,j,k] for i in range(n)])
                    try:
                        v, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        V[j*p+k, :] = v
                    except:
                        pass
            
            # Fix U, V; solve for W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((r, m))
                    for j in range(m):
                        for c in range(r):
                            A[c, j] = U[i*m+j, c] * V[j*p+k, c]
                    b = np.array([T[i,j,k] for j in range(m)])
                    try:
                        w, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        W[k*n+i, :] = w
                    except:
                        pass
            
            # Check convergence
            res = tensor_residual(U, V, W, T)
            if res < 1e-10:
                # Round to half-integers and verify
                U_r = round_to_half(U)
                V_r = round_to_half(V)
                W_r = round_to_half(W)
                if is_half_integer(U_r) and is_half_integer(V_r) and is_half_integer(W_r):
                    if verify_decomposition(U_r, V_r, W_r, T):
                        return U_r, V_r, W_r
                # Also check without rounding
                if is_half_integer(U, 1e-6) and is_half_integer(V, 1e-6) and is_half_integer(W, 1e-6):
                    if verify_decomposition(U, V, W, T):
                        return U, V, W
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    T = build_tensor()
    
    # Schoolbook fallback
    U_sb, V_sb, W_sb = schoolbook()
    best = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = SCHOOLBOOK
    
    # Try ranks from low to high
    for r in range(13, SCHOOLBOOK):
        if time.time() > deadline - 10:
            break
        
        result = try_rank_algebraic(r, T, max_time=min(20, deadline - 10 - (time.time() - start_time)))
        if result is not None:
            U, V, W = result
            if verify_decomposition(U, V, W, T):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = r
                break  # Found minimal rank, stop
    
    return best
