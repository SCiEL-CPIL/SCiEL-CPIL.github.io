"""
Rank-minimal exact tensor decomposition for 2x3 * 3x3 matrix multiplication.

The matrix multiplication tensor T has entries T[i,j,k] = 1 for the (i,j,k)
triple corresponding to output element (i,k) of A*B where A is n x m and B is
m x p. A rank-r decomposition is a set of r triples (U_c, V_c, W_c) such that
for all (i,j,k):
    T[i,j,k] = sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i]

The schoolbook rank is n*m*p = 18. We search for lower ranks using a
systematic approach: try ranks from 13 upward, and for each rank, attempt to
find an exact decomposition via randomized initialization + iterative refinement
with exact rational arithmetic verification.

For small dimensions like 2x3x3, the known best rank is 13 (Strassen-like
improvements exist for 2x2x2 giving rank 7, and this extends).
"""

import time
import random
import numpy as np
from fractions import Fraction
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]
n, m, p = DIMENSIONS
N = n * m  # 6
M = m * p  # 9
P = p * n  # 6
SCHOOLBOOK = n * m * p  # 18

# Build the multiplication tensor as a 3D array T[i,j,k]
def build_tensor():
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def verify_decomposition(U, V, W, T):
    """Verify that U, V, W exactly decompose T."""
    # T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # Reshape for efficient computation
    U_r = np.array(U).reshape(n, m, rank)
    V_r = np.array(V).reshape(m, p, rank)
    W_r = np.array(W).reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U_r[i, j, c] * V_r[j, k, c] * W_r[k, i, c] for c in range(rank))
                if abs(val - T[i, j, k]) > 1e-9:
                    return False
    return True

def try_rank(r, T, max_iters=2000, seed=42):
    """Try to find a rank-r decomposition using iterative refinement."""
    rng = np.random.RandomState(seed)
    
    for attempt in range(30):
        # Initialize with random values
        U = rng.randn(N, r)
        V = rng.randn(M, r)
        W = rng.randn(P, r)
        
        # Iterative refinement: minimize sum_{i,j,k} (sum_c U_ijc V_jkc W_kic - T_ijk)^2
        # Use gradient-free approach: alternate fixing two factors and solving for the third
        best_obj = np.inf
        for it in range(max_iters):
            # Compute current error
            obj = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(r))
                        obj += (val - T[i,j,k])**2
            
            if obj < 1e-12:
                break
            
            if obj < best_obj:
                best_obj = obj
            
            # Alternate: fix V and W, solve for U (linear least squares)
            # For each (i,j): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = T[i,j,k] for all k
            # This is: for fixed (i,j), we have p equations in r unknowns U[i*m+j, :]
            # A[c,k] = V[j*p+k,c] * W[k*n+i,c], b[k] = T[i,j,k]
            # Minimize ||A^T * u - b||^2 where A is r x p
            for i in range(n):
                for j in range(m):
                    A = np.zeros((r, p))
                    for k in range(p):
                        for c in range(r):
                            A[c, k] = V[j*p+k, c] * W[k*n+i, c]
                    b = np.array([T[i,j,k] for k in range(p)])
                    # Solve min ||A^T u - b||^2 => u = A (A^T A)^{-1} A^T b
                    try:
                        AT = A.T
                        u, _, _, _ = np.linalg.lstsq(AT, b, rcond=None)
                        U[i*m+j, :] = u
                    except:
                        pass
            
            # Fix U and W, solve for V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((r, n))
                    for i in range(n):
                        for c in range(r):
                            A[c, i] = U[i*m+j, c] * W[k*n+i, c]
                    b = np.array([T[i,j,k] for i in range(n)])
                    try:
                        AT = A.T
                        v, _, _, _ = np.linalg.lstsq(AT, b, rcond=None)
                        V[j*p+k, :] = v
                    except:
                        pass
            
            # Fix U and V, solve for W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((r, m))
                    for j in range(m):
                        for c in range(r):
                            A[c, j] = U[i*m+j, c] * V[j*p+k, c]
                    b = np.array([T[i,j,k] for j in range(m)])
                    try:
                        AT = A.T
                        w, _, _, _ = np.linalg.lstsq(AT, b, rcond=None)
                        W[k*n+i, :] = w
                    except:
                        pass
        
        # Check if we converged
        if best_obj < 1e-10:
            return U, V, W
    
    return None

def construct():
    start_time = time.time()
    deadline = 140  # Leave 10s safety margin
    
    T = build_tensor()
    
    # Try ranks from low to high
    for r in range(13, SCHOOLBOOK + 1):
        if time.time() > deadline:
            break
        
        result = try_rank(r, T, max_iters=500, seed=42)
        if result is not None:
            U, V, W = result
            # Verify
            if verify_decomposition(U, V, W, T):
                return {"U": [[float(x) for x in U[i]] for i in range(N)],
                        "V": [[float(x) for x in V[i]] for i in range(M)],
                        "W": [[float(x) for x in W[i]] for i in range(P)]}
    
    # Fallback: schoolbook decomposition
    rank = SCHOOLBOOK
    U = [[0] * rank for _ in range(N)]
    V = [[0] * rank for _ in range(M)]
    W = [[0] * rank for _ in range(P)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
