"""Matrix multiplication 3x3 x 3x5 — minimal-rank exact tensor decomposition."""
import time
import math
import itertools
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n=3, m=3, p=5):
    """Check that U * V^T * W^T reproduces the matrix multiplication tensor.
    T[i,j,k] = 1 if j==k else 0."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                expected = 1 if j == k else 0
                if abs(total - expected) > 1e-9:
                    return False
    return True


def _verify_np(U, V, W, n=3, m=3, p=5):
    """Vectorized verification."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    UV = np.einsum('ijc,jkc->ijkc', U3, V3)
    T = np.einsum('ijkc,kic->ijk', UV, W3)
    expected = np.zeros((n, m, p), dtype=np.float64)
    for j in range(min(m, p)):
        expected[:, j, j] = 1.0
    return np.allclose(T, expected, atol=1e-9)


def _schoolbook(n=3, m=3, p=5):
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
    return {"U": U, "V": V, "W": W}


def _als_decompose(n, m, p, rank, max_iter=500, tol=1e-12, seed=42):
    """ALS to find rank-r decomposition of the matmul tensor T[i,j,k]=1 if j==k else 0."""
    rng = np.random.RandomState(seed)
    
    U = rng.randn(n * m, rank) * 0.3
    V = rng.randn(m * p, rank) * 0.3
    W = rng.randn(p * n, rank) * 0.3
    
    # Build target tensor
    T = np.zeros((n, m, p))
    for j in range(min(m, p)):
        T[:, j, j] = 1.0
    
    for it in range(max_iter):
        # Update U
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                rhs = M.T @ T[i, j, :]
                A = M.T @ M + 1e-10 * np.eye(rank)
                try:
                    U[i * m + j, :] = np.linalg.solve(A, rhs)
                except:
                    U[i * m + j, :] = np.linalg.lstsq(A, rhs, rcond=None)[0]
        
        # Update V
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                rhs = M.T @ T[:, j, k]
                A = M.T @ M + 1e-10 * np.eye(rank)
                try:
                    V[j * p + k, :] = np.linalg.solve(A, rhs)
                except:
                    V[j * p + k, :] = np.linalg.lstsq(A, rhs, rcond=None)[0]
        
        # Update W
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                rhs = M.T @ T[i, :, k]
                A = M.T @ M + 1e-10 * np.eye(rank)
                try:
                    W[k * n + i, :] = np.linalg.solve(A, rhs)
                except:
                    W[k * n + i, :] = np.linalg.lstsq(A, rhs, rcond=None)[0]
        
        # Error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    expected = 1.0 if j == k else 0.0
                    err += (val - expected) ** 2
        if err < tol:
            break
    
    return U, V, W, err


def _try_round_and_verify(U, V, W, n=3, m=3, p=5):
    """Try rounding to half-integers and verify."""
    U_r = np.round(U * 2) / 2.0
    V_r = np.round(V * 2) / 2.0
    W_r = np.round(W * 2) / 2.0
    
    if _verify_np(U_r, V_r, W_r, n, m, p):
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
    return None


def _try_integer_round_and_verify(U, V, W, n=3, m=3, p=5):
    """Try rounding to integers and verify."""
    U_r = np.round(U)
    V_r = np.round(V)
    W_r = np.round(W)
    
    if _verify_np(U_r, V_r, W_r, n, m, p):
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
    return None


def _try_merge_pair(U, V, W, rank, c1, c2, n=3, m=3, p=5):
    """
    Try to merge two columns c1, c2 into a single column, reducing rank by 1.
    Returns (new_U, new_V, new_W, new_rank) or None.
    """
    if c1 == c2:
        return None
    
    # We want to find new column vectors u_new, v_new, w_new such that
    # for all (i,j,k): u_new[i*m+j]*v_new[j*p+k]*w_new[k*n+i] 
    #   = u1[i*m+j]*v1[j*p+k]*w1[k*n+i] + u2[i*m+j]*v2[j*p+k]*w2[k*n+i]
    
    # This is generally not possible with a single term unless the two terms
    # are proportional in a specific way. Instead, we try a different approach:
    # perturb the system and see if we can find a valid rank-(rank-1) solution.
    
    # Remove columns c1 and c2, add one new column
    new_rank = rank - 1
    keep = [c for c in range(rank) if c != c1 and c != c2]
    
    U_base = U[:, keep]
    V_base = V[:, keep]
    W_base = W[:, keep]
    
    # Try random perturbation of the new column
    rng = np.random.RandomState(c1 * 1000 + c2)
    best_err = float('inf')
    best_result = None
    
    for attempt in range(20):
        u_new = rng.randn(n * m) * 0.5
        v_new = rng.randn(m * p) * 0.5
        w_new = rng.randn(p * n) * 0.5
        
        U_test = np.column_stack([U_base, u_new])
        V_test = np.column_stack([V_base, v_new])
        W_test = np.column_stack([W_base, w_new])
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_test[i * m + j, :], V_test[j * p + k, :] * W_test[k * n + i, :])
                    expected = 1.0 if j == k else 0.0
                    err += (val - expected) ** 2
        
        if err < best_err:
            best_err = err
            best_result = (U_test, V_test, W_test, new_rank)
    
    if best_err < 1e-8:
        U_r = np.round(best_result[0] * 2) / 2.0
        V_r = np.round(best_result[1] * 2) / 2.0
        W_r = np.round(best_result[2] * 2) / 2.0
        if _verify_np(U_r, V_r, W_r, n, m, p):
            return U_r, V_r, W_r, new_rank
    
    return None


def construct():
    t0 = time.time()
    deadline = t0 + 110  # Safety margin
    
    n, m, p = 3, 3, 5
    schoolbook_rank = n * m * p  # 45
    
    # First, try to find a decomposition with rank < 45
    # Known lower bound is around 43 for (3,3,5)
    
    for target_rank in range(43, 45):  # Try 43, 44
        if time.time() > deadline - 15:
            break
        
        max_iter = max(200, 600 - (target_rank - 43) * 50)
        n_restarts = max(5, 15 - (target_rank - 43) * 3)
        
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            seed = 123 + restart * 7919 + target_rank * 100
            U, V, W, err = _als_decompose(n, m, p, target_rank, max_iter=max_iter, seed=seed)
            
            if err < 1e-8:
                # Try rounding to half-integers
                result = _try_round_and_verify(U, V, W, n, m, p)
                if result is not None:
                    U_list, V_list, W_list = result
                    if _verify(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}
                
                # Try integer rounding
                result = _try_integer_round_and_verify(U, V, W, n, m, p)
                if result is not None:
                    U_list, V_list, W_list = result
                    if _verify(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}
        
        # If ALS found a near-solution, try to polish with more iterations
        if time.time() < deadline - 20:
            for restart in range(3):
                seed = 999 + restart * 31337 + target_rank * 200
                U, V, W, err = _als_decompose(n, m, p, target_rank, max_iter=1000, seed=seed)
                if err < 1e-10:
                    result = _try_round_and_verify(U, V, W, n, m, p)
                    if result is not None:
                        U_list, V_list, W_list = result
                        if _verify(U_list, V_list, W_list):
                            return {"U": U_list, "V": V_list, "W": W_list}
                    result = _try_integer_round_and_verify(U, V, W, n, m, p)
                    if result is not None:
                        U_list, V_list, W_list = result
                        if _verify(U_list, V_list, W_list):
                            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Try rank 44 with more aggressive search
    if time.time() < deadline - 10:
        for restart in range(10):
            if time.time() > deadline - 5:
                break
            seed = 777 + restart * 4444
            U, V, W, err = _als_decompose(n, m, p, 44, max_iter=800, seed=seed)
            if err < 1e-8:
                result = _try_round_and_verify(U, V, W, n, m, p)
                if result is not None:
                    U_list, V_list, W_list = result
                    if _verify(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}
                result = _try_integer_round_and_verify(U, V, W, n, m, p)
                if result is not None:
                    U_list, V_list, W_list = result
                    if _verify(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-END
