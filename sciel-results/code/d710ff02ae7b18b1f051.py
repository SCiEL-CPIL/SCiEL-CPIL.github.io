"""Slice-method decomposition for 3x4x7 matrix multiplication."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [3, 4, 7]


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify exact decomposition: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1"""
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for c in range(rank):
        Uc = U[:, c]
        Vc = V[:, c]
        Wc = W[:, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += Uc[i * m + j] * Vc[j * p + k] * Wc[k * n + i]
    return np.allclose(T, np.ones((n, m, p)), atol=tol)


def _search_decomp(n, m, p, max_rank, coeff_set, timeout=50, num_trials=20000):
    """Search for rank-minimal decomposition using random U,V and solving for W."""
    start = time.time()
    nm, mp, pn = n * m, m * p, p * n

    for r in range(1, max_rank + 1):
        for trial in range(num_trials):
            if time.time() - start > timeout:
                return None

            U_rand = np.random.choice(coeff_set, size=(nm, r))
            V_rand = np.random.choice(coeff_set, size=(mp, r))

            # Build linear system for W
            # Variables: W[k*n+i, c] for k in [0,p), i in [0,n), c in [0,r)
            # Equations: for each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            n_eqs = n * m * p
            n_vars = p * n * r
            A = np.zeros((n_eqs, n_vars))
            b = np.ones(n_eqs)
            eq = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(r):
                            col = (k * n + i) * r + c
                            A[eq, col] = U_rand[i * m + j, c] * V_rand[j * p + k, c]
                        eq += 1

            try:
                W_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                if np.allclose(A @ W_flat, b, atol=1e-10):
                    W_rand = W_flat.reshape((pn, r))
                    # Check if entries are half-integers
                    W_check = np.round(W_rand * 2) / 2
                    if np.allclose(W_check, W_rand, atol=1e-10):
                        if _verify(U_rand, V_rand, W_check, n, m, p):
                            return (r, U_rand, V_rand, W_check)
            except:
                pass
        # If we get here, no solution found for this rank, try next
    return None


def _construct_sub(n, m, p, max_rank, timeout=50):
    """Find rank-minimal decomposition for n x m x p."""
    # Try increasingly larger coefficient sets and more trials
    coeff_sets = [
        [-1, 0, 1],
        [-2, -1, 0, 1, 2],
        [-2, -1, -0.5, 0, 0.5, 1, 2],
        [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3],
    ]
    for cs in coeff_sets:
        result = _search_decomp(n, m, p, max_rank, cs, timeout=timeout, num_trials=30000)
        if result:
            return result
    # Fallback: schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return (rank, U, V, W)


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7

    # Partition k dimension into blocks: 3, 2, 2
    blocks = [3, 2, 2]
    k_offsets = [0, 3, 5]

    # Find sub-decompositions
    sub_decomps = []
    r0, U0, V0, W0 = _construct_sub(n, m, 3, max_rank=15, timeout=35)
    sub_decomps.append((r0, U0, V0, W0, k_offsets[0], blocks[0]))
    r1, U1, V1, W1 = _construct_sub(n, m, 2, max_rank=12, timeout=35)
    sub_decomps.append((r1, U1, V1, W1, k_offsets[1], blocks[1]))
    r2, U2, V2, W2 = _construct_sub(n, m, 2, max_rank=12, timeout=35)
    sub_decomps.append((r2, U2, V2, W2, k_offsets[2], blocks[2]))

    total_rank = sum(r for r, _, _, _, _, _ in sub_decomps)

    # Build full matrices
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))

    col_offset = 0
    for r, U_sub, V_sub, W_sub, k_off, b in sub_decomps:
        # U: shared input indices, just copy
        U_full[:, col_offset:col_offset + r] = U_sub

        # V: for block starting at k_off with size b
        # V_sub has shape (m*b, r) with indices (j, k_local)
        # V_full has shape (m*p, r) with indices (j, k_global)
        for j in range(m):
            for kk in range(b):
                k_global = k_off + kk
                V_full[j * p + k_global, col_offset:col_offset + r] = V_sub[j * b + kk, :]

        # W: W_sub has shape (b*n, r) with indices (k_local, i)
        # W_full has shape (p*n, r) with indices (k_global, i)
        for kk in range(b):
            k_global = k_off + kk
            for i in range(n):
                W_full[k_global * n + i, col_offset:col_offset + r] = W_sub[kk * n + i, :]

        col_offset += r

    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
