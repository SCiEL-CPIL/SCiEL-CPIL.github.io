"""Rewritten program: search for a rank-minimal exact tensor decomposition
for 3x5 * 5x6 matrix multiplication using ALS-style rank reduction from a
block/schoolbook starting point, with a deadline guard."""

import time
import random
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 5, 6]
DEADLINE = 150.0
SAFETY_MARGIN = 10.0  # seconds before kill to stop searching

def _build_tensor(n, m, p):
    """Return the matrix-multiplication tensor T[i,j,k] = 1 for all i,j,k
    (the 'flat' tensor that schoolbook decomposes). The actual decomposition
    condition is: sum_c U[i*m+j, c] * V[j*p+k, c] == W[k*n+i, c] for all i,j,k.
    
    More precisely, the tensor T[i,j,k] = 1 and we need:
    T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * ... 
    Actually the standard form: for matrix mult A(3x5)*B(5x6)=C(3x6),
    C[i,k] = sum_j A[i,j] B[j,k].
    The tensor T[i,j,k] = 1 (for all i in 0..2, j in 0..4, k in 0..5).
    Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]?
    
    Wait, the interface says:
    U: (n*m, rank)
    V: (m*p, rank)  
    W: (p*n, rank)
    
    And the output factor follows W[k*n+i, r].
    
    The standard tensor decomposition for matmul:
    T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    But here U is indexed by (i*m+j), V by (j*p+k), W by (k*n+i).
    So: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    And T[i,j,k] = 1 for all valid i,j,k.
    """
    T = np.ones((n, m, p), dtype=np.float64)
    return T

def _reconstruct(U, V, W, n, m, p):
    """Reconstruct T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]"""
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        u = U[:, c]  # (n*m,)
        v = V[:, c]  # (m*p,)
        w = W[:, c]  # (p*n,)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u[i * m + j] * v[j * p + k] * w[k * n + i]
    return T

def _reconstruct_fast(U, V, W, n, m, p):
    """Vectorized reconstruction."""
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=np.float64)
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # Use einsum
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return T

def _check_exact(U, V, W, n, m, p, tol=1e-9):
    """Check if decomposition is exact (within tolerance)."""
    T = _reconstruct_fast(U, V, W, n, m, p)
    return np.max(np.abs(T - 1.0)) < tol

def _als_step(U, V, W, n, m, p, max_iter=50, tol=1e-10):
    """One ALS iteration: fix two, solve for the third."""
    rank = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for it in range(max_iter):
        # Solve for U: for each (i,j), T[i,j,k] = sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c])
        # So U[i*m+j,:] * diag(V[j*p+k,:]*W[k*n+i:]) @ ... 
        # Actually: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # For fixed j,k: T[i,j,k] = sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c])
        # This is a linear system in U[i*m+j,:] for each i,j,k... 
        # But U[i*m+j,:] is shared across k. So:
        # T[i,j,:] = U[i*m+j,:] @ (V[j*p,:rank] * W[:,n+i?]) -- need to be careful.
        
        # Let's think differently. For fixed i,j:
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] for all k
        # This is: T[i,j,:] = U[i*m+j,:] * M_{i,j} where M_{i,j}[c,k] = V[j*p+k,c]*W[k*n+i,c]
        # So U[i*m+j,:] is a row vector (1 x rank), M is (rank x p)
        # T[i,j,:] = U_row @ M  =>  U_row @ M = T[i,j,:]
        # This is underdetermined (rank >= p typically). We want the solution that
        # minimizes ||U_row @ M - T[i,j,:]||^2, i.e., U_row = T[i,j,:] @ M^T (M M^T)^{-1}
        # if rank > p. Or use least squares.
        
        # Similarly for V and W.
        
        # Solve for U
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                # M[c,k] = V[j*p+k,c] * W[k*n+i,c]
                M = np.zeros((rank, p), dtype=np.float64)
                for k in range(p):
                    M[:, k] = V[j * p + k, :] * W[k * n + i, :]
                # U[i*m+j,:] @ M = T[i,j,:] = ones(p)
                # Least squares: U_row = argmin ||U_row @ M - ones||
                # U_row = ones @ M^T (M M^T)^{-1} if full rank, else pinv
                try:
                    U_row = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                except:
                    U_row = np.ones(rank) / rank
                U_new[i * m + j, :] = U_row
        U = U_new
        
        # Solve for V
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # For fixed j,k: T[:,j,k] = V[j*p+k,:] * N where N[c,i] = U[i*m+j,c]*W[k*n+i,c]
                N = np.zeros((rank, n), dtype=np.float64)
                for i in range(n):
                    N[:, i] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    V_row = np.linalg.lstsq(N.T, np.ones(n), rcond=None)[0]
                except:
                    V_row = np.ones(rank) / rank
                V_new[j * p + k, :] = V_row
        V = V_new
        
        # Solve for W
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # For fixed k,i: T[i,:,k] = W[k*n+i,:] * P where P[c,j] = U[i*m+j,c]*V[j*p+k,c]
                P = np.zeros((rank, m), dtype=np.float64)
                for j in range(m):
                    P[:, j] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    W_row = np.linalg.lstsq(P.T, np.ones(m), rcond=None)[0]
                except:
                    W_row = np.ones(rank) / rank
                W_new[k * n + i, :] = W_row
        W = W_new
        
        # Check convergence
        T = _reconstruct_fast(U, V, W, n, m, p)
        err = np.max(np.abs(T - 1.0))
        if err < tol:
            break
    
    return U, V, W, err

def _try_reduce_rank(U, V, W, n, m, p, target_rank, max_iter=30, tol=1e-9):
    """Try to reduce rank by zeroing out some columns and re-solving."""
    if target_rank >= U.shape[1]:
        return None
    
    # Pick target_rank columns to keep, zero the rest
    best = None
    # Try different subsets of columns to keep
    cols = list(range(U.shape[1]))
    # For simplicity, try keeping the first target_rank columns
    # and also try random subsets
    attempts = 0
    max_attempts = 20
    
    while attempts < max_attempts:
        U_try = U.copy()
        V_try = V.copy()
        W_try = W.copy()
        
        if attempts == 0:
            keep = list(range(target_rank))
        else:
            keep = random.sample(cols, target_rank)
        
        # Zero out non-kept columns
        for c in range(U.shape[1]):
            if c not in keep:
                U_try[:, c] = 0
                V_try[:, c] = 0
                W_try[:, c] = 0
        
        # Run ALS to see if we can converge
        U_try, V_try, W_try, err = _als_step(U_try, V_try, W_try, n, m, p, max_iter=max_iter, tol=tol)
        
        if err < tol:
            # Verify
            T = _reconstruct_fast(U_try, V_try, W_try, n, m, p)
            if np.max(np.abs(T - 1.0)) < tol:
                # Count effective rank
                effective_rank = 0
                for c in range(U_try.shape[1]):
                    if np.max(np.abs(U_try[:, c])) > 1e-12 or \
                       np.max(np.abs(V_try[:, c])) > 1e-12 or \
                       np.max(np.abs(W_try[:, c])) > 1e-12:
                        effective_rank += 1
                if best is None or effective_rank < best[0]:
                    best = (effective_rank, U_try, V_try, W_try)
        
        attempts += 1
    
    return best

def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    current_rank = n * m * p
    
    # Run ALS to clean up
    U, V, W, err = _als_step(U, V, W, n, m, p, max_iter=20, tol=1e-10)
    
    # Try to reduce rank
    best_rank = current_rank
    best_result = (U, V, W)
    
    # Try target ranks from low to high
    # Theoretical lower bound is max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # Known results for 3x5x6: let's try from 30 upward
    for target_rank in range(30, current_rank + 1):
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        result = _try_reduce_rank(U, V, W, n, m, p, target_rank, max_iter=20, tol=1e-9)
        if result is not None:
            eff_rank, U_r, V_r, W_r = result
            if eff_rank < best_rank:
                best_rank = eff_rank
                best_result = (U_r, V_r, W_r)
                # Update U,V,W for further reductions
                U, V, W = U_r, V_r, W_r
    
    # Also try random restarts if time permits
    if time.time() - start_time < DEADLINE - SAFETY_MARGIN - 10:
        for restart in range(5):
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
            
            # Random initialization
            rank_init = best_rank + 5
            U_r = np.random.randn(n * m, rank_init)
            V_r = np.random.randn(m * p, rank_init)
            W_r = np.random.randn(p * n, rank_init)
            
            U_r, V_r, W_r, err = _als_step(U_r, V_r, W_r, n, m, p, max_iter=50, tol=1e-10)
            
            if err < 1e-9:
                # Try to reduce
                for target_rank in range(30, rank_init + 1):
                    if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                        break
                    result = _try_reduce_rank(U_r, V_r, W_r, n, m, p, target_rank, max_iter=15, tol=1e-9)
                    if result is not None:
                        eff_rank, Ur, Vr, Wr = result
                        if eff_rank < best_rank:
                            best_rank = eff_rank
                            best_result = (Ur, Vr, Wr)
    
    U_best, V_best, W_best = best_result
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U_best]
    V_list = [[float(x) for x in row] for row in V_best]
    W_list = [[float(x) for x in row] for row in W_best]
    
    return {"U": U_list, "V": V_list, "W": W_list}
