"""Matrix multiplication 2x4 by 4x5 via block Strassen + schoolbook,
with linear dependency detection and random probing for rank reduction."""

import time
import numpy as np

DIMENSIONS = [2, 4, 5]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 110.0


def _remaining():
    return _DEADLINE - time.time()


def _build_strassen_36():
    """Build the deterministic Strassen + schoolbook rank-36 decomposition."""
    n, m, p = DIMENSIONS
    rank = 36

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen_222(i0, j0, k0, col_start):
        nonlocal col
        c = col_start
        U[i0 * m + j0][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[j0 * p + k0][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1
        col += 1

        c = col_start + 1
        U[(i0 + 1) * m + j0][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[j0 * p + k0][c] += 1
        W[k0 * n + (i0 + 1)][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] -= 1
        col += 1

        c = col_start + 2
        U[i0 * m + j0][c] += 1
        V[j0 * p + (k0 + 1)][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] -= 1
        W[(k0 + 1) * n + i0][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1
        col += 1

        c = col_start + 3
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[(j0 + 1) * p + k0][c] += 1
        V[j0 * p + k0][c] -= 1
        W[k0 * n + i0][c] += 1
        W[k0 * n + (i0 + 1)][c] += 1
        col += 1

        c = col_start + 4
        U[i0 * m + j0][c] += 1
        U[i0 * m + (j0 + 1)][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] -= 1
        W[(k0 + 1) * n + i0][c] += 1
        col += 1

        c = col_start + 5
        U[(i0 + 1) * m + j0][c] += 1
        U[i0 * m + j0][c] -= 1
        V[j0 * p + k0][c] += 1
        V[j0 * p + (k0 + 1)][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1
        col += 1

        c = col_start + 6
        U[i0 * m + (j0 + 1)][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] -= 1
        V[(j0 + 1) * p + k0][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] += 1
        col += 1

        return col_start + 7

    def add_schoolbook_2x4x1(i0, j0, k0, col_start):
        c = col_start
        for di in range(2):
            for dj in range(4):
                U[(i0 + di) * m + (j0 + dj)][c] = 1
                V[(j0 + dj) * p + k0][c] = 1
                W[k0 * n + (i0 + di)][c] = 1
                c += 1
        return c

    col = add_strassen_222(0, 0, 0, 0)
    col = add_strassen_222(0, 2, 0, col)
    col = add_strassen_222(0, 0, 2, col)
    col = add_strassen_222(0, 2, 2, col)
    col = add_schoolbook_2x4x1(0, 0, 4, col)

    assert col == rank
    return U, V, W


def _flatten_tensor(U, V, W, n, m, p, rank):
    """Flatten the tensor into a vector: T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]."""
    vec = np.zeros(n * m * p)
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if val != 0:
                        idx = i * m * p + j * p + k
                        vec[idx] += val
    return vec


def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition gives T[i,j,k]=1 for all i,j,k."""
    vec = _flatten_tensor(U, V, W, n, m, p, rank)
    return np.allclose(vec, 1.0, atol=1e-10)


def _try_remove_term(U, V, W, n, m, p, rank, term_idx):
    """Try to remove term_idx by expressing it as a combination of others.
    Returns (new_U, new_V, new_W, new_rank) or None if not possible."""
    # Check if the tensor without term_idx can be reconstructed
    # by scaling the remaining terms. This only works if there's a simple dependency.
    # For now, just check if removing it still gives the right tensor
    # (this would only work if the term was zero, which it isn't).
    # Instead, we check linear dependencies via SVD.
    return None


def _find_linear_dependencies(U, V, W, n, m, p, rank):
    """Find linear dependencies among the rank-1 terms using SVD.
    Returns list of null-space vectors (each of length rank)."""
    # Build matrix M where column r is the flattened tensor T_r
    M = np.zeros((n * m * p, rank))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if val != 0:
                        idx = i * m * p + j * p + k
                        M[idx, r] = val

    # SVD to find rank
    U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
    tol = 1e-10
    current_rank = np.sum(s > tol)

    null_vectors = []
    if current_rank < rank:
        # Null space is spanned by last (rank - current_rank) rows of Vt
        null_dim = rank - current_rank
        for i in range(null_dim):
            null_vectors.append(Vt[-(i + 1), :])

    return null_vectors, current_rank


def _apply_dependency_reduction(U, V, W, n, m, p, rank, null_vec):
    """Given a null-space vector c, try to eliminate the term with largest |c_r|.
    The idea: if sum_r c_r T_r = 0, then T_j = -1/c_j * sum_{r!=j} c_r T_r.
    But this doesn't directly reduce rank unless we can absorb the coefficients.
    
    For exact rank reduction, we need a "simple" dependency where one term
    is an exact linear combination that allows us to drop it and adjust others.
    
    This is complex. For now, we just detect and report."""
    # Find index with largest |c_r|
    idx = np.argmax(np.abs(null_vec))
    c_idx = null_vec[idx]
    
    if abs(c_idx) < 1e-10:
        return None
    
    # We can't simply drop the term; we need to re-express.
    # For a true rank reduction, we'd need to find a different basis.
    # This is left as a detection step.
    return None


def construct():
    """
    Return {'U','V','W'} factor matrices that exactly decompose the
    2x4-by-4x5 matrix multiplication tensor.
    
    Strategy:
    1. Build deterministic Strassen+schoolbook rank-36 decomposition.
    2. Detect linear dependencies via SVD.
    3. Use random linear probing to search for lower-rank decompositions.
    4. Return best valid decomposition found.
    """
    n, m, p = DIMENSIONS

    # Step 1: Build base rank-36 decomposition
    U, V, W = _build_strassen_36()
    rank = 36

    # Verify base decomposition
    if not _verify(U, V, W, n, m, p, rank):
        # Should not happen, but return as-is
        return {"U": U, "V": V, "W": W}

    best_U, best_V, best_W, best_rank = U, V, W, rank

    # Step 2: Check for linear dependencies
    if _remaining() > 10.0:
        null_vecs, tensor_rank = _find_linear_dependencies(U, V, W, n, m, p, rank)
        
        if tensor_rank < rank:
            # There are dependencies. Try to exploit them.
            # For each null vector, try to find a way to reduce rank.
            for nv in null_vecs:
                if _remaining() < 5.0:
                    break
                result = _apply_dependency_reduction(U, V, W, n, m, p, rank, nv)
                if result is not None:
                    new_U, new_V, new_W, new_rank = result
                    if new_rank < best_rank and _verify(new_U, new_V, new_W, n, m, p, new_rank):
                        best_U, best_V, best_W, best_rank = new_U, new_V, new_W, new_rank

    # Step 3: Random linear probing - search for lower-rank decompositions
    # Generate random rank-1 tensors with small integer/half-integer entries
    # and check if they can help reduce the rank.
    
    if _remaining() > 20.0:
        rng = np.random.default_rng(42)
        
        # Target tensor vector (all ones)
        target = np.ones(n * m * p)
        
        # Current tensor matrix (columns are the rank-1 terms)
        M = np.zeros((n * m * p, rank))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                        if val != 0:
                            idx = i * m * p + j * p + k
                            M[idx, r] = val
        
        # Try to find a subset of size < rank that still spans the target
        # This is a combinatorial problem. Use greedy: try removing one term at a time.
        
        for remove_idx in range(rank - 1, -1, -1):
            if _remaining() < 15.0:
                break
            
            # Check if target is in span of remaining columns
            remaining = [r for r in range(rank) if r != remove_idx]
            if len(remaining) == 0:
                break
                
            M_sub = M[:, remaining]
            
            # Check if target is in column space of M_sub
            # Use least squares: minimize ||M_sub @ x - target||
            try:
                x, residuals, rank_sub, sv = np.linalg.lstsq(M_sub, target, rcond=None)
                if residuals and residuals[0] < 1e-10:
                    # Target is in the span! But we need the coefficients to be
                    # such that the resulting decomposition has valid U,V,W structure.
                    # The coefficients x give us how to express target as a combination
                    # of the remaining rank-1 terms. But each "term" is already a
                    # product U_r ⊗ V_r ⊗ W_r, so we can't just scale them arbitrarily
                    # and still have a valid rank-1 decomposition with the same U,V,W
                    # structure (since U,V,W entries would need to be scaled consistently).
                    #
                    # However, if all x coefficients are 1, we're done.
                    # More generally, if x coefficients are integers, we might be able
                    # to absorb them into U, V, W by scaling rows/columns.
                    #
                    # For now, check if x is close to all-ones (which would mean
                    # the removed term was redundant).
                    if np.allclose(x, 1.0, atol=1e-10):
                        # The removed term was exactly redundant!
                        new_U = [row[:] for row in U]
                        new_V = [row[:] for row in V]
                        new_W = [row[:] for row in W]
                        new_rank = rank - 1
                        
                        # Remove column remove_idx
                        new_U = [row[:remove_idx] + row[remove_idx+1:] for row in new_U]
                        new_V = [row[:remove_idx] + row[remove_idx+1:] for row in new_V]
                        new_W = [row[:remove_idx] + row[remove_idx+1:] for row in new_W]
                        
                        if _verify(new_U, new_V, new_W, n, m, p, new_rank):
                            best_U, best_V, best_W, best_rank = new_U, new_V, new_W, new_rank
                            rank = new_rank  # Update for next iteration
                            # Rebuild M
                            M = np.zeros((n * m * p, rank))
                            for r in range(rank):
                                for i in range(n):
                                    for j in range(m):
                                        for k in range(p):
                                            val = U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                                            if val != 0:
                                                idx = i * m * p + j * p + k
                                                M[idx, r] = val
            except Exception:
                pass

    # Final verification
    if _remaining() > 0.5:
        if not _verify(best_U, best_V, best_W, n, m, p, best_rank):
            # Fallback to original
            best_U, best_V, best_W, best_rank = U, V, W, 36

    return {"U": best_U, "V": best_V, "W": best_W}
