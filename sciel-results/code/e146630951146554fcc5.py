"""Tensor decomposition search for 3x3 x 3x4 matrix multiplication."""
import time
import random
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 3, 4]
n, m, p = DIMENSIONS
N = n * m  # 9
M = m * p  # 12
P = p * n  # 12
RANK_SCHOOLBOOK = n * m * p  # 36


def _schoolbook():
    rank = RANK_SCHOOLBOOK
    U = [[0] * rank for _ in range(N)]
    V = [[0] * rank for _ in range(M)]
    W = [[0] * rank for _ in range(P)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _tensor_tensor(U, V, W):
    """Compute T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]."""
    rank = len(U[0])
    T = np.zeros((n, m, p), dtype=np.float64)
    Uc = np.array(U, dtype=np.float64).reshape(N, rank)
    Vc = np.array(V, dtype=np.float64).reshape(M, rank)
    Wc = np.array(W, dtype=np.float64).reshape(P, rank)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = np.dot(Uc[i * m + j], np.dot(Vc[j * p + k], Wc[k * n + i]))
    return T


def _verify_exact(U, V, W):
    """Check if decomposition is exact (all entries of T are integers)."""
    T = _tensor_tensor(U, V, W)
    # Check if all entries are close to integers
    if not np.allclose(T, np.round(T), atol=1e-6):
        return False
    return True


def _try_rationalize(U, V, W, max_denom=100):
    """Try to find rational/half-integer entries that exactly decompose."""
    # This is a simplified version - we'll use it after numerical optimization
    pass


def _als_decompose(rank, init='random', n_iter=200, tol=1e-8):
    """Alternating Least Squares for tensor decomposition."""
    # Build the target tensor T
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0

    # Initialize
    Uc = np.random.randn(N, rank)
    Vc = np.random.randn(M, rank)
    Wc = np.random.randn(P, rank)

    for it in range(n_iter):
        # Update W: for each (k,i), solve Wc[k*n+i,:] = (Uc[i*m+j,:] * Vc[j*p+k,:]) dot product
        # T[i,j,k] = sum_c Uc[i*m+j,c] * Vc[j*p+k,c] * Wc[k*n+i,c]
        # For fixed (k,i): T[i,:,k] = sum_j Uc[i*m+j,:] * Vc[j*p+k,:] * Wc[k*n+i,c]
        # This is: T_i_k[j] = (Uc[i*m+j] * Vc[j*p+k]) dot Wc[k*n+i]
        # So Wc[k*n+i] should minimize sum_j (T[i,j,k] - Uc[i*m+j]*Vc[j*p+k]*Wc[k*n+i])^2
        # = sum_j T[i,j,k]^2 - 2*Wc[k*n+i] dot (sum_j T[i,j,k]*Uc[i*m+j]*Vc[j*p+k]) + ||Wc[k*n+i]||^2 * sum_j (Uc[i*m+j]*Vc[j*p+k])^2
        # Optimal: Wc[k*n+i] = (sum_j T[i,j,k]*Uc[i*m+j]*Vc[j*p+k]) / (sum_j (Uc[i*m+j]*Vc[j*p+k])^2)

        for k in range(p):
            for i in range(n):
                num = np.zeros(rank)
                den = 0.0
                for j in range(m):
                    prod = Uc[i * m + j] * Vc[j * p + k]
                    num += T[i, j, k] * prod
                    den += np.dot(prod, prod)
                if den > 1e-12:
                    Wc[k * n + i] = num / den

        # Update V: for fixed (j,k): T[:,j,k] = sum_i Uc[i*m+j] * Vc[j*p+k] * Wc[k*n+i]
        for j in range(m):
            for k in range(p):
                num = np.zeros(rank)
                den = 0.0
                for i in range(n):
                    prod = Uc[i * m + j] * Wc[k * n + i]
                    num += T[i, j, k] * prod
                    den += np.dot(prod, prod)
                if den > 1e-12:
                    Vc[j * p + k] = num / den

        # Update U: for fixed (i,j): T[i,j,:] = sum_k Uc[i*m+j] * Vc[j*p+k] * Wc[k*n+i]
        for i in range(n):
            for j in range(m):
                num = np.zeros(rank)
                den = 0.0
                for k in range(p):
                    prod = Vc[j * p + k] * Wc[k * n + i]
                    num += T[i, j, k] * prod
                    den += np.dot(prod, prod)
                if den > 1e-12:
                    Uc[i * m + j] = num / den

    # Check residual
    T_recon = _tensor_tensor(Uc.reshape(N, rank).tolist(), Vc.reshape(M, rank).tolist(), Wc.reshape(P, rank).tolist())
    residual = np.max(np.abs(T - T_recon))
    return Uc, Vc, Wc, residual


def _snap_to_halfint(mat, tol=0.01):
    """Snap entries to nearest half-integer if close."""
    snapped = np.round(mat * 2) / 2
    if np.max(np.abs(mat - snapped)) < tol:
        return snapped
    return None


def _try_rank(rank, time_budget=10):
    """Try to find a rank decomposition using ALS with multiple restarts."""
    start = time.time()
    best_residual = float('inf')
    best_Uc = None
    best_Vc = nbest_Wc = None

    n_restarts = 5
    for r in range(n_restarts):
        Uc, Vc, Wc, resid = _als_decompose(rank, n_iter=100)
        if resid < best_residual:
            best_residual = resid
            best_Uc = Uc.copy()
            best_Vc = Vc.copy()
            best_Wc = Wc.copy()
        if time.time() - start > time_budget:
            break

    if best_residual < 1e-6:
        # Try to snap to half-integers
        Uc_snapped = _snap_to_halfint(best_Uc, tol=0.01)
        Vc_snapped = _snap_to_halfint(best_Vc, tol=0.01)
        Wc_snapped = _snap_to_halfint(best_Wc, tol=0.01)
        if Uc_snapped is not None and Vc_snapped is not None and Wc_snapped is not None:
            # Verify exactness
            T = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] = np.dot(Uc_snapped[i * m + j], np.dot(Vc_snapped[j * p + k], Wc_snapped[k * n + i]))
            if np.allclose(T, np.round(T), atol=1e-9) and np.allclose(np.round(T), 1.0, atol=1e-9):
                U = Uc_snapped.tolist()
                V = Vc_snapped.tolist()
                W = Wc_snapped.tolist()
                return {"U": U, "V": V, "W": W}

    return None


def construct():
    deadline = time.time() + 110  # Safety margin from 120s limit

    # First, try to find a better decomposition than schoolbook
    # The known minimal rank for 3x3 x 3x4 is unknown, but schoolbook is 36
    # Let's try ranks from 35 down to some lower bound

    # Try rank 35 first (just one less than schoolbook)
    for rank in range(35, 30, -1):
        if time.time() > deadline:
            break
        result = _try_rank(rank, time_budget=15)
        if result is not None:
            return result

    # If we couldn't find a better one, return schoolbook
    return _schoolbook()
