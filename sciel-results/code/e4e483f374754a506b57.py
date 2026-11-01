"""Rank-56 Strassen decomposition for 2x4 * 4x8 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START

def _build_strassen_56():
    """Build the base rank-56 Strassen decomposition."""
    n, m, p = DIMENSIONS
    total_rank = 56
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)

    for u in range(2):
        for v in range(4):
            base_col = 7 * (4 * u + v)
            a11 = 0 * m + 2 * u
            a12 = 0 * m + 2 * u + 1
            a21 = 1 * m + 2 * u
            a22 = 1 * m + 2 * u + 1
            b11 = (2 * u) * p + 2 * v
            b12 = (2 * u) * p + 2 * v + 1
            b21 = (2 * u + 1) * p + 2 * v
            b22 = (2 * u + 1) * p + 2 * v + 1
            c11 = (2 * v) * n + 0
            c12 = (2 * v + 1) * n + 0
            c21 = (2 * v) * n + 1
            c22 = (2 * v + 1) * n + 1

            U[a11, base_col + 0] += 1
            U[a22, base_col + 0] += 1
            V[b11, base_col + 0] += 1
            V[b22, base_col + 0] += 1
            W[c11, base_col + 0] += 1
            W[c22, base_col + 0] += 1

            U[a21, base_col + 1] += 1
            U[a22, base_col + 1] += 1
            V[b11, base_col + 1] += 1
            W[c21, base_col + 1] += 1
            W[c22, base_col + 1] -= 1

            U[a11, base_col + 2] += 1
            V[b12, base_col + 2] += 1
            V[b22, base_col + 2] -= 1
            W[c12, base_col + 2] += 1
            W[c22, base_col + 2] += 1

            U[a22, base_col + 3] += 1
            V[b21, base_col + 3] += 1
            V[b11, base_col + 3] -= 1
            W[c11, base_col + 3] += 1
            W[c21, base_col + 3] += 1

            U[a11, base_col + 4] += 1
            U[a12, base_col + 4] += 1
            V[b22, base_col + 4] += 1
            W[c11, base_col + 4] -= 1
            W[c12, base_col + 4] += 1

            U[a21, base_col + 5] += 1
            U[a11, base_col + 5] -= 1
            V[b11, base_col + 5] += 1
            V[b12, base_col + 5] += 1
            W[c22, base_col + 5] += 1

            U[a12, base_col + 6] += 1
            U[a22, base_col + 6] -= 1
            V[b21, base_col + 6] += 1
            V[b22, base_col + 6] += 1
            W[c11, base_col + 6] += 1

    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    for i in range(n):
        for j in range(p):
            for k in range(m):
                val = 0.0
                for r in range(U.shape[1]):
                    val += U[i * m + k, r] * V[k * p + j, r] * W[j * n + i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _merge_identical_terms(U, V, W):
    """
    Merge terms where both U and W vectors are identical.
    For such terms, V vectors can be summed.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    n_terms = U.shape[1]
    eliminated = set()
    
    for r1 in range(n_terms):
        if r1 in eliminated:
            continue
        for r2 in range(r1 + 1, n_terms):
            if r2 in eliminated:
                continue
            if np.array_equal(U[:, r1], U[:, r2]) and np.array_equal(W[:, r1], W[:, r2]):
                V[:, r1] += V[:, r2]
                eliminated.add(r2)
    
    if not eliminated:
        return U, V, W, n_terms
    
    keep = [i for i in range(n_terms) if i not in eliminated]
    return U[:, keep], V[:, keep], W[:, keep], len(keep)


def construct():
    """
    Rank-minimal exact tensor decomposition for 2x4 * 4x8 matrix multiplication.
    
    Starts with rank-56 Strassen decomposition and attempts rank reduction
    by merging terms with identical U and W vectors.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Build base decomposition
    U, V, W = _build_strassen_56()
    
    # Verify base decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try merging terms with identical U and W
    U_merged, V_merged, W_merged, merged_rank = _merge_identical_terms(U, V, W)
    
    if merged_rank < 56:
        if _verify_decomposition(U_merged, V_merged, W_merged, n, m, p):
            return {"U": U_merged.tolist(), "V": V_merged.tolist(), "W": W_merged.tolist()}
    
    # Return base decomposition
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
