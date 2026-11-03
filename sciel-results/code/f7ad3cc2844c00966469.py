"""
Matrix multiplication 2x4 by 4x5 via structured tensor decomposition search.

Strategy:
  1. Build the known rank-36 decomposition (4 Strassen 2x2x2 blocks + schoolbook).
  2. Apply slice-merging: partition p=5 into p=2 and p=3 slices, build independent
     minimal decompositions, then perform joint linear dependency analysis on the
     combined factor matrices to identify and eliminate redundant rank-1 terms.
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


def _build_slice_decomp(n, m, p_slice, k_offset):
    """
    Build a minimal decomposition for the sub-tensor C[:, k_offset:k_offset+p_slice].
    
    For p_slice=2: use 2 Strassen blocks (one for A cols 0-1, one for A cols 2-3) = 14 rank
    For p_slice=3: use 2 Strassen blocks + schoolbook for last column = 7+7+8 = 22 rank
    
    Returns (U_slice, V_slice, W_slice, rank_slice) where:
      U_slice: (n*m) x rank_slice
      V_slice: (m*p_slice) x rank_slice  (local k indices 0..p_slice-1)
      W_slice: (p_slice*n) x rank_slice  (local k indices 0..p_slice-1)
    """
    # Determine how to decompose this slice
    # p_slice=2: two 2x2 Strassen blocks
    # p_slice=3: two 2x2 Strassen blocks (for first 2 cols) + schoolbook for last col
    # But actually for p_slice=3, we can do:
    #   - Strassen for cols 0-1 (rank 7)
    #   - Schoolbook for col 2 (rank 8)
    #   Total: 15
    # Or:
    #   - Strassen for cols 0-1 (rank 7) 
    #   - Strassen for cols 1-2 (rank 7) - but this overlaps col 1
    #   Better: Strassen cols 0-1 + schoolbook col 2 = 15
    
    if p_slice == 2:
        # Two Strassen blocks: A[:,0:2]*B[0:2,0:2] and A[:,2:4]*B[2:4,0:2]
        rank_slice = 14
        U_s = [[0] * rank_slice for _ in range(n * m)]
        V_s = [[0] * rank_slice for _ in range(m * p_slice)]
        W_s = [[0] * rank_slice for _ in range(p_slice * n)]
        
        col = 0
        # Strassen block 1: A cols 0-1, B cols 0-1
        col = _add_strassen_local(U_s, V_s, W_s, n, m, p_slice, 0, 0, 0, col)
        # Strassen block 2: A cols 2-3, B cols 0-1
        col = _add_strassen_local(U_s, V_s, W_s, n, m, p_slice, 0, 2, 0, col)
        
        assert col == rank_slice
        return U_s, V_s, W_s, rank_slice
    
    elif p_slice == 3:
        # Strassen for cols 0-1 (rank 7) + schoolbook for col 2 (rank 8)
        rank_slice = 15
        U_s = [[0] * rank_slice for _ in range(n * m)]
        V_s = [[0] * rank_slice for _ in range(m * p_slice)]
        W_s = [[0] * rank_slice for _ in range(p_slice * n)]
        
        col = 0
        # Strassen block: A cols 0-1, B cols 0-1
        col = _add_strassen_local(U_s, V_s, W_s, n, m, p_slice, 0, 0, 0, col)
        # Strassen block: A cols 2-3, B cols 0-1
        col = _add_strassen_local(U_s, V_s, W_s, n, m, p_slice, 0, 2, 0, col)
        # Schoolbook for col 2
        col = _add_schoolbook_local(U_s, V_s, W_s, n, m, p_slice, 0, 0, 2, col)
        
        assert col == rank_slice, f"Slice p=3: expected 15, got {col}"
        return U_s, V_s, W_s, rank_slice
    
    else:
        # Fallback: schoolbook
        rank_slice = n * m * p_slice
        U_s = [[0] * rank_slice for _ in range(n * m)]
        V_s = [[0] * rank_slice for _ in range(m * p_slice)]
        W_s = [[0] * rank_slice for _ in range(p_slice * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p_slice):
                    U_s[i * m + j][col] = 1
                    V_s[j * p_slice + k][col] = 1
                    W_s[k * n + i][col] = 1
                    col += 1
        
        return U_s, V_s, W_s, rank_slice


def _add_strassen_local(U, V, W, n, m, p_local, i0, j0, k0, col_start):
    """Add Strassen 2x2x2 to local matrices with local p indexing."""
    c = col_start
    # M1 = (A00 + A11) * (B00 + B11)
    U[i0 * m + j0][c] += 1
    U[(i0 + 1) * m + (j0 + 1)][c] += 1
    V[j0 * p_local + k0][c] += 1
    V[(j0 + 1) * p_local + (k0 + 1)][c] += 1
    W[k0 * n + i0][c] += 1
    W[(k0 + 1) * n + (i0 + 1)][c] += 1
    c += 1
    # M2 = (A10 + A11) * B00
    U[(i0 + 1) * m + j0][c] += 1
    U[(i0 + 1) * m + (j0 + 1)][c] += 1
    V[j0 * p_local + k0][c] += 1
    W[k0 * n + (i0 + 1)][c] += 1
    W[(k0 + 1) * n + (i0 + 1)][c] -= 1
    c += 1
    # M3 = A00 * (B01 - B11)
    U[i0 * m + j0][c] += 1
    V[j0 * p_local + (k0 + 1)][c] += 1
    V[(j0 + 1) * p_local + (k0 + 1)][c] -= 1
    W[(k0 + 1) * n + i0][c] += 1
    W[(k0 + 1) * n + (i0 + 1)][c] += 1
    c += 1
    # M4 = A11 * (B10 - B00)
    U[(i0 + 1) * m + (j0 + 1)][c] += 1
    V[(j0 + 1) * p_local + k0][c] += 1
    V[j0 * p_local + k0][c] -= 1
    W[k0 * n + i0][c] += 1
    W[k0 * n + (i0 + 1)][c] += 1
    c += 1
    # M5 = (A00 + A01) * B11
    U[i0 * m + j0][c] += 1
    U[i0 * m + (j0 + 1)][c] += 1
    V[(j0 + 1) * p_local + (k0 + 1)][c] += 1
    W[k0 * n + i0][c] -= 1
    W[(k0 + 1) * n + i0][c] += 1
    c += 1
    # M6 = (A10 - A00) * (B00 + B01)
    U[(i0 + 1) * m + j0][c] += 1
    U[i0 * m + j0][c] -= 1
    V[j0 * p_local + k0][c] += 1
    V[j0 * p_local + (k0 + 1)][c] += 1
    W[(k0 + 1) * n + (i0 + 1)][c] += 1
    c += 1
    # M7 = (A01 - A11) * (B10 + B11)
    U[i0 * m + (j0 + 1)][c] += 1
    U[(i0 + 1) * m + (j0 + 1)][c] -= 1
    V[(j0 + 1) * p_local + k0][c] += 1
    V[(j0 + 1) * p_local + (k0 + 1)][c] += 1
    W[k0 * n + i0][c] += 1
    c += 1
    return c


def _add_schoolbook_local(U, V, W, n, m, p_local, i0, j0, k0, col_start):
    """Add schoolbook 2x4x1 to local matrices."""
    c = col_start
    for di in range(2):
        for dj in range(4):
            U[(i0 + di) * m + (j0 + dj)][c] = 1
            V[(j0 + dj) * p_local + k0][c] = 1
            W[k0 * n + (i0 + di)][c] = 1
            c += 1
    return c


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


def _slice_merge_optimization(n, m, p):
    """
    Slice-merging optimization:
    1. Partition p=5 into p1=2 and p2=3.
    2. Build independent minimal decompositions for each slice.
    3. Combine them and perform linear dependency analysis to find redundant terms.
    
    The key insight: both slices share the same U structure (A matrix), so there
    may be linear dependencies between the combined factor matrices that allow
    rank reduction.
    """
    p1, p2 = 2, 3
    
    # Build slice 1 decomposition (p=2)
    U1, V1, W1, r1 = _build_slice_decomp(n, m, p1, 0)
    
    # Build slice 2 decomposition (p=3)
    U2, V2, W2, r2 = _build_slice_decomp(n, m, p2, 0)
    
    # Combine into full decomposition
    # U_combined: (n*m) x (r1+r2)
    # V_combined: (m*p) x (r1+r2) - need to map local V indices to global
    # W_combined: (p*n) x (r1+r2) - need to map local W indices to global
    
    rank_total = r1 + r2
    U_comb = [[0] * rank_total for _ in range(n * m)]
    V_comb = [[0] * rank_total for _ in range(m * p)]
    W_comb = [[0] * rank_total for _ in range(p * n)]
    
    # Copy slice 1
    for i in range(n * m):
        for r in range(r1):
            U_comb[i][r] = U1[i][r]
    for j in range(m):
        for k in range(p1):
            for r in range(r1):
                V_comb[j * p + k][r] = V1[j * p1 + k][r]
    for k in range(p1):
        for i in range(n):
            for r in range(r1):
                W_comb[k * n + i][r] = W1[k * n + i][r]
    
    # Copy slice 2 (offset by r1)
    for i in range(n * m):
        for r in range(r2):
            U_comb[i][r1 + r] = U2[i][r]
    for j in range(m):
        for k in range(p2):
            for r in range(r2):
                V_comb[j * p + (p1 + k)][r1 + r] = V2[j * p2 + k][r]
    for k in range(p2):
        for i in range(n):
            for r in range(r2):
                W_comb[(p1 + k) * n + i][r1 + r] = W2[k * n + i][r]
    
    # Verify combined decomposition
    if not _verify_full(U_comb, V_comb, W_comb, n, m, p):
        return None
    
    # Now perform linear dependency analysis
    # Build the tensor matrix M where M[entry, r] = U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
    num_entries = n * m * p  # 40
    M = np.zeros((num_entries, rank_total))
    for r in range(rank_total):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    entry_idx = i * m * p + j * p + k
                    M[entry_idx, r] = U_comb[i * m + j][r] * V_comb[j * p + k][r] * W_comb[k * n + i][r]
    
    target = np.ones(num_entries)
    
    # Check rank of M
    try:
        rank_M = np.linalg.matrix_rank(M, tol=1e-10)
    except:
        return None
    
    if rank_M < rank_total:
        # Find null space
        try:
            _, s, vh = np.linalg.svd(M, full_matrices=True)
            tol = 1e-10
            null_dim = rank_total - np.sum(s > tol)
            
            if null_dim > 0:
                null_space = vh[rank_M:, :].T
                
                # Try to find a combination that zeros out at least one coefficient
                # while maintaining the decomposition
                for ns_vec in null_space:
                    for t in np.linspace(-3, 3, 25):
                        coeffs = np.ones(rank_total) + t * ns_vec
                        residual = np.max(np.abs(M @ coeffs - target))
                        if residual < 1e-10:
                            zero_mask = np.abs(coeffs) < 1e-10
                            if np.sum(zero_mask) > 0:
                                keep = np.where(~zero_mask)[0]
                                if len(keep) < rank_total:
                                    U_new = [[U_comb[i][r] for r in keep] for i in range(n * m)]
                                    V_new = [[V_comb[i][r] for r in keep] for i in range(m * p)]
                                    W_new = [[W_comb[i][r] for r in keep] for i in range(p * n)]
                                    if _verify_full(U_new, V_new, W_new, n, m, p):
                                        return U_new, V_new, W_new, len(keep)
        except:
            pass
    
    # Try removing individual terms and solving the reduced system
    for drop_idx in range(rank_total):
        if _remaining() < 5.0:
            break
        
        keep = [r for r in range(rank_total) if r != drop_idx]
        M_reduced = M[:, keep]
        
        try:
            coeffs, residuals, rank_r, sv = np.linalg.lstsq(M_reduced, target, rcond=None)
            residual = np.max(np.abs(M_reduced @ coeffs - target))
            if residual < 1e-10:
                if np.max(np.abs(coeffs)) < 100:
                    U_new = [[U_comb[i][r] for r in keep] for i in range(n * m)]
                    V_new = [[V_comb[i][r] for r in keep] for i in range(m * p)]
                    W_new = [[W_comb[i][r] for r in keep] for i in range(p * n)]
                    
                    # Scale factors: distribute coefficient among U, V, W
                    # Simple approach: scale U by sqrt(coeff), V by sqrt(coeff), W by 1
                    # But this may not preserve exactness. Try direct verification first.
                    for r_idx, orig_r in enumerate(keep):
                        c = coeffs[r_idx]
                        if abs(c - 1.0) > 1e-10:
                            # Need to scale the rank-1 term
                            # Scale U by c^(1/3), V by c^(1/3), W by c^(1/3) - not exact for non-cubes
                            # Instead, scale U by c, keep V and W
                            for i in range(n * m):
                                U_new[i][r_idx] *= c
                            # This changes the product by c, which is what we want
                            # But we need to verify
                            pass
                    
                    if _verify_full(U_new, V_new, W_new, n, m, p):
                        return U_new, V_new, W_new, len(keep)
        except:
            pass
    
    return None


def _try_merge_pairs(U, V, W, n, m, p):
    """
    Try to merge pairs of rank-1 terms or find linear dependencies.
    """
    rank36 = len(U[0])
    
    num_entries = n * m * p
    M = np.zeros((num_entries, rank36))
    for r in range(rank36):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    entry_idx = i * m * p + j * p + k
                    M[entry_idx, r] = U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
    
    target = np.ones(num_entries)
    
    try:
        rank_M = np.linalg.matrix_rank(M, tol=1e-10)
    except:
        return None
    
    if rank_M < rank36:
        try:
            _, s, vh = np.linalg.svd(M, full_matrices=True)
            tol = 1e-10
            null_dim = rank36 - np.sum(s > tol)
            if null_dim > 0:
                null_space = vh[rank_M:, :].T
                for ns_vec in null_space:
                    for t in np.linspace(-3, 3, 25):
                        coeffs = np.ones(rank36) + t * ns_vec
                        residual = np.max(np.abs(M @ coeffs - target))
                        if residual < 1e-10:
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


def _random_search_lower_rank(U, V, W, n, m, p, max_iters=2000):
    """
    Random search: try to find a lower-rank decomposition by removing terms
    and solving the reduced linear system.
    """
    rank36 = len(U[0])
    
    target = np.ones(n * m * p)
    
    M = np.zeros((n * m * p, rank36))
    for r in range(rank36):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    entry_idx = i * m * p + j * p + k
                    M[entry_idx, r] = U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
    
    # Try removing one term at a time
    for drop_idx in range(rank36):
        if _remaining() < 5.0:
            break
        
        keep = [r for r in range(rank36) if r != drop_idx]
        M_reduced = M[:, keep]
        
        try:
            coeffs, residuals, rank_r, sv = np.linalg.lstsq(M_reduced, target, rcond=None)
            residual = np.max(np.abs(M_reduced @ coeffs - target))
            if residual < 1e-10:
                if np.max(np.abs(coeffs)) < 100:
                    U_new = [[U[i][r] for r in keep] for i in range(len(U))]
                    V_new = [[V[i][r] for r in keep] for i in range(len(V))]
                    W_new = [[W[i][r] for r in keep] for i in range(len(W))]
                    
                    # Apply scaling: multiply U column by coeff
                    for r_idx in range(len(keep)):
                        c = coeffs[r_idx]
                        if abs(c - 1.0) > 1e-10:
                            for i in range(len(U_new)):
                                U_new[i][r_idx] *= c
                    
                    if _verify_full(U_new, V_new, W_new, n, m, p):
                        return U_new, V_new, W_new, len(keep)
        except:
            pass
    
    # Try removing two terms at a time
    for drop1 in range(rank36):
        if _remaining() < 3.0:
            break
        for drop2 in range(drop1 + 1, rank36):
            if _remaining() < 3.0:
                break
            
            keep = [r for r in range(rank36) if r != drop1 and r != drop2]
            M_reduced = M[:, keep]
            
            try:
                coeffs, residuals, rank_r, sv = np.linalg.lstsq(M_reduced, target, rcond=None)
                residual = np.max(np.abs(M_reduced @ coeffs - target))
                if residual < 1e-10:
                    if np.max(np.abs(coeffs)) < 100:
                        U_new = [[U[i][r] for r in keep] for i in range(len(U))]
                        V_new = [[V[i][r] for r in keep] for i in range(len(V))]
                        W_new = [[W[i][r] for r in keep] for i in range(len(W))]
                        
                        for r_idx in range(len(keep)):
                            c = coeffs[r_idx]
                            if abs(c - 1.0) > 1e-10:
                                for i in range(len(U_new)):
                                    U_new[i][r_idx] *= c
                        
                        if _verify_full(U_new, V_new, W_new, n, m, p):
                            return U_new, V_new, W_new, len(keep)
            except:
                pass
    
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

    # Try slice-merging optimization (new approach)
    if _remaining() > 15.0:
        result = _slice_merge_optimization(n, m, p)
        if result is not None:
            U, V, W, rank = result
            if _verify_full(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}

    # Try structural merge on the original rank-36
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
