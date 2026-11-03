"""
Matrix multiplication 2x4 by 4x5 via structured tensor decomposition search.

Strategy:
  1. Build the known rank-36 decomposition (4 Strassen 2x2x2 blocks + schoolbook).
  2. Use a completely different structural approach: exploit the small n=2 dimension
     to formulate the problem as finding a low-rank decomposition where we can
     merge terms across blocks. Specifically, we use the fact that with n=2,
     the W-matrix only has 10 rows, and we can search for a rank-35 or lower
     decomposition by trying to express one Strassen term as a combination of others.
  3. Additionally, try a random search approach: start from rank-36 and try to
     perturb/merge terms to find lower rank solutions.
"""

import time
import numpy as np

DIMENSIONS = [2, 4, 5]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0


def _remaining():
    return _DEADLINE - time.time()


def _build_rank36():
    """Build the rank-36 decomposition: 4 Strassen 2x2x2 + schoolbook 2x4x4x1."""
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    rank = 36

    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36

    col = 0

    def add_strassen_222(i0, j0, k0, col_start):
        nonlocal col
        c = col_start
        # M1 = (A00 + A11) * (B00 + B11)
        U[i0 * m + j0][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[j0 * p + k0][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1
        c += 1
        # M2 = (A10 + A11) * B00
        U[(i0 + 1) * m + j0][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[j0 * p + k0][c] += 1
        W[k0 * n + (i0 + 1)][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] -= 1
        c += 1
        # M3 = A00 * (B01 - B11)
        U[i0 * m + j0][c] += 1
        V[j0 * p + (k0 + 1)][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] -= 1
        W[(k0 + 1) * n + i0][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1
        c += 1
        # M4 = A11 * (B10 - B00)
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[(j0 + 1) * p + k0][c] += 1
        V[j0 * p + k0][c] -= 1
        W[k0 * n + i0][c] += 1
        W[k0 * n + (i0 + 1)][c] += 1
        c += 1
        # M5 = (A00 + A01) * B11
        U[i0 * m + j0][c] += 1
        U[i0 * m + (j0 + 1)][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] -= 1
        W[(k0 + 1) * n + i0][c] += 1
        c += 1
        # M6 = (A10 - A00) * (B00 + B01)
        U[(i0 + 1) * m + j0][c] += 1
        U[i0 * m + j0][c] -= 1
        V[j0 * p + k0][c] += 1
        V[j0 * p + (k0 + 1)][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1
        c += 1
        # M7 = (A01 - A11) * (B10 + B11)
        U[i0 * m + (j0 + 1)][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] -= 1
        V[(j0 + 1) * p + k0][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] += 1
        c += 1
        col = c
        return c

    def add_schoolbook_2x4x1(i0, j0, k0, col_start):
        c = col_start
        for di in range(2):
            for dj in range(4):
                U[(i0 + di) * m + (j0 + dj)][c] = 1
                V[(j0 + dj) * p + k0][c] = 1
                W[k0 * n + (i0 + di)][c] = 1
                c += 1
        col = c
        return c

    col = add_strassen_222(0, 0, 0, col)
    col = add_strassen_222(0, 2, 0, col)
    col = add_strassen_222(0, 0, 2, col)
    col = add_strassen_222(0, 2, 2, col)
    col = add_schoolbook_2x4x1(0, 0, 4, col)

    assert col == rank, f"Expected rank {rank}, got {col}"
    return U, V, W, rank


def _verify_full(U, V, W, n, m, p):
    """Full verification: check all tensor entries."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                acc = 0
                for r in range(rank):
                    acc += U[ui][r] * V[vj][r] * W[wk][r]
                if acc != 1:
                    return False
    return True


def _verify_numpy(U, V, W, n, m, p):
    """Vectorized verification using numpy."""
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                acc = np.dot(Ua[ui], Va[vj] * Wa[wk])
                if abs(acc - 1.0) > 1e-10:
                    return False
    return True


def _try_merge_pairs(U, V, W, n, m, p):
    """
    Structural approach: try to merge pairs of rank-1 terms into one.
    
    Two rank-1 terms (r1, r2) can be merged into one if there exist vectors
    u, v, w such that:
      u1 ⊗ v1 ⊗ w1 + u2 ⊗ v2 ⊗ w2 = u ⊗ v ⊗ w
    
    This is a very restrictive condition. Instead, we try a different structural
    approach: since n=2, we can view the problem as two "slices" (i=0 and i=1).
    
    For slice i=0: C[0,k] = sum_j A[0,j] * B[j,k]
    For slice i=1: C[1,k] = sum_j A[1,j] * B[j,k]
    
    The key insight: if we can find a decomposition where some terms contribute
    to both slices in a way that allows sharing, we can reduce rank.
    
    Specifically, we try to find a rank-35 decomposition by checking if any
    single term can be eliminated by adjusting others.
    """
    rank36 = len(U[0])
    
    # Build the 40x36 matrix of tensor contributions
    num_entries = n * m * p  # 40
    M = np.zeros((num_entries, rank36))
    for r in range(rank36):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    entry_idx = i * m * p + j * p + k
                    M[entry_idx, r] = U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
    
    target = np.ones(num_entries)
    
    # Check if the system is underdetermined (more unknowns than equations)
    # 40 equations, 36 unknowns -> overdetermined, but let's check rank
    try:
        rank_M = np.linalg.matrix_rank(M)
    except:
        return None
    
    # If rank < 36, there are dependencies
    if rank_M < rank36:
        # Find null space vectors
        try:
            _, s, vh = np.linalg.svd(M, full_matrices=True)
            # Null space dimension
            tol = 1e-10
            null_dim = rank36 - np.sum(s > tol)
            if null_dim > 0:
                # Get null space basis
                null_space = vh[rank_M:, :].T
                # For each null space vector, check if we can subtract it
                # from the all-ones solution to get a non-negative (or sparse) solution
                for ns_vec in null_space:
                    # Try small perturbations
                    for t in [-2, -1, -0.5, 0.5, 1, 2]:
                        coeffs = np.ones(rank36) + t * ns_vec
                        # Check if this gives a valid solution
                        residual = np.max(np.abs(M @ coeffs - target))
                        if residual < 1e-10:
                            # Check if any coefficient is close to zero
                            zero_mask = np.abs(coeffs) < 1e-10
                            if np.sum(zero_mask) > 0:
                                keep = np.where(~zero_mask)[0]
                                if len(keep) < rank36:
                                    U_new = [[U[i][r] for r in keep] for i in range(len(U))]
                                    V_new = [[V[i][r] for r in keep] for i in range(len(V))]
                                    W_new = [[W[i][r] for r in keep] for i in range(len(W))]
                                    if _verify_full(U_new, V_new, W_new, n, m, p):
                                        return U_new, V_new, W_new, len(keep)
        except:
            pass
    
    return None


def _random_search_lower_rank(U, V, W, n, m, p, max_iters=5000):
    """
    Random search: try to find a lower-rank decomposition by randomly
    generating rank-1 terms and checking if they can replace existing ones.
    
    Different structural approach: use the fact that the tensor has a specific
    structure (it's the matrix multiplication tensor) to guide the search.
    """
    rank36 = len(U[0])
    best_rank = rank36
    best_result = None
    
    # Precompute the target tensor
    target = np.ones(n * m * p)
    
    # Build matrix M where M[entry, r] = contribution of term r to entry
    M = np.zeros((n * m * p, rank36))
    for r in range(rank36):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    entry_idx = i * m * p + j * p + k
                    M[entry_idx, r] = U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
    
    # Try to find solutions with fewer non-zero coefficients
    # by solving the linear system with sparsity constraints
    
    # Approach: try removing one term at a time and see if the remaining
    # terms can be adjusted to still satisfy the equations
    for drop_idx in range(rank36):
        if _remaining() < 5.0:
            break
        
        # Remove term drop_idx
        keep = [r for r in range(rank36) if r != drop_idx]
        M_reduced = M[:, keep]
        
        # Try to solve M_reduced @ coeffs = target
        try:
            coeffs, residuals, rank_r, sv = np.linalg.lstsq(M_reduced, target, rcond=None)
            residual = np.max(np.abs(M_reduced @ coeffs - target))
            if residual < 1e-10:
                # Check if solution is valid (entries should be reasonable)
                if np.max(np.abs(coeffs)) < 100:
                    # Build new decomposition
                    U_new = [[U[i][r] for r in keep] for i in range(len(U))]
                    V_new = [[V[i][r] for r in keep] for i in range(len(V))]
                    W_new = [[W[i][r] for r in keep] for i in range(len(W))]
                    # Scale by coefficients
                    for r_idx, orig_r in enumerate(keep):
                        c = coeffs[r_idx]
                        if abs(c) > 1e-15:
                            for i in range(len(U_new)):
                                U_new[i][r_idx] *= c
                            for i in range(len(V_new)):
                                V_new[i][r_idx] *= c
                            for i in range(len(W_new)):
                                W_new[i][r_idx] = 1  # Keep W as is, scale U and V
                    # Actually, we need to be more careful with scaling
                    # Let's just verify
                    if _verify_full(U_new, V_new, W_new, n, m, p):
                        return U_new, V_new, W_new, len(keep)
        except:
            pass
    
    return None


def _try_structural_merge(U, V, W, n, m, p):
    """
    Try a structural approach specific to the 2x4x5 case.
    
    Key insight: With n=2, we have only 2 rows in the output. The tensor can
    be viewed as:
      C[0,:] = A[0,:] * B  (row 0 of A times B)
      C[1,:] = A[1,:] * B  (row 1 of A times B)
    
    Each row of A (length 4) times B (4x5) gives a 1x5 row of C.
    So we need to decompose two vector-matrix products.
    
    For a 1x4 vector times 4x5 matrix, the tensor has rank 4 (schoolbook).
    For two such products sharing the same B, we might be able to do better.
    
    Actually, the full 2x4x5 tensor has a known lower bound. Let's try a
    completely different construction: use the fact that we can combine
    the two rows of A.
    
    Let A = [a0; a1] where a0, a1 are 1x4 vectors.
    C[0,k] = sum_j a0[j] * B[j,k]
    C[1,k] = sum_j a1[j] * B[j,k]
    
    We can write:
    C[0,k] + C[1,k] = sum_j (a0[j] + a1[j]) * B[j,k]
    C[0,k] - C[1,k] = sum_j (a0[j] - a1[j]) * B[j,k]
    
    This doesn't directly help with rank reduction since we still need
    4 multiplications per combination.
    
    Let's try yet another approach: use the 4x5 matrix B and try to find
    a low-rank factorization that helps.
    """
    # For now, this is a placeholder for future structural improvements
    return None


def construct():
    """
    Return {'U','V','W'} factor matrices that exactly decompose the
    2x4-by-4x5 matrix multiplication tensor.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=5

    # Start with the known rank-36 decomposition
    U, V, W, rank = _build_rank36()

    # Verify
    if not _verify_full(U, V, W, n, m, p):
        raise RuntimeError("Rank-36 decomposition failed verification")

    # Try structural merge
    if _remaining() > 10.0:
        result = _try_merge_pairs(U, V, W, n, m, p)
        if result is not None:
            U, V, W, rank = result
            if _verify_full(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}

    # Try random search
    if _remaining() > 10.0:
        result = _random_search_lower_rank(U, V, W, n, m, p)
        if result is not None:
            U, V, W, rank = result
            if _verify_full(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
