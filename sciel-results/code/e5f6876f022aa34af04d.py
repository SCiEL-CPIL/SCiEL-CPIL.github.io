"""
Rank-minimal exact tensor decomposition for 3x4 x 4x4 matrix multiplication.

Strategy:
  1. Build a structured rank-44 decomposition (4 Strassen 2x2x2 blocks + 4 schoolbook 1x2x2 blocks).
  2. Attempt to reduce rank by finding linear dependencies among the tensor terms and eliminating them.
  3. Verify each candidate decomposition exactly.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]


def _build_target_tensor(n, m, p):
    """Return the 3D tensor T[i,j,k] = 1 (the matrix multiplication tensor)."""
    return np.ones((n, m, p), dtype=np.float64)


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U,V,W exactly decompose the tensor (within tol)."""
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
    return np.allclose(T, 1.0, atol=tol, rtol=0)


def _construct_structured_decomposition():
    """Construct a rank-44 decomposition using Strassen for 2x2x2 blocks and schoolbook for 1x2x2 blocks."""
    n, m, p = 3, 4, 4
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0

    def add_strassen(rA, cA, rB, cB):
        nonlocal col
        # Term 1: (A11+A22)(B11+B22)
        U[(rA + 0) * m + (cA + 0), col] = 1
        U[(rA + 1) * m + (cA + 1), col] = 1
        V[(rB + 0) * p + (cB + 0), col] = 1
        V[(rB + 1) * p + (cB + 1), col] = 1
        W[(cB + 0) * n + (rA + 0), col] = 1
        W[(cB + 1) * n + (rA + 1), col] = 1
        col += 1
        # Term 2: (A21+A22)B11
        U[(rA + 1) * m + (cA + 0), col] = 1
        U[(rA + 1) * m + (cA + 1), col] = 1
        V[(rB + 0) * p + (cB + 0), col] = 1
        W[(cB + 0) * n + (rA + 1), col] = 1
        W[(cB + 1) * n + (rA + 1), col] = -1
        col += 1
        # Term 3: A11(B21-B11)
        U[(rA + 0) * m + (cA + 0), col] = 1
        V[(rB + 0) * p + (cB + 1), col] = 1
        V[(rB + 1) * p + (cB + 1), col] = -1
        W[(cB + 1) * n + (rA + 0), col] = 1
        W[(cB + 1) * n + (rA + 1), col] = 1
        col += 1
        # Term 4: A22(B12-B22)
        U[(rA + 1) * m + (cA + 1), col] = 1
        V[(rB + 1) * p + (cB + 0), col] = 1
        V[(rB + 0) * p + (cB + 0), col] = -1
        W[(cB + 0) * n + (rA + 0), col] = 1
        W[(cB + 0) * n + (rA + 1), col] = 1
        col += 1
        # Term 5: (A11+A12)B22
        U[(rA + 0) * m + (cA + 0), col] = 1
        U[(rA + 0) * m + (cA + 1), col] = 1
        V[(rB + 1) * p + (cB + 1), col] = 1
        W[(cB + 1) * n + (rA + 0), col] = 1
        W[(cB + 0) * n + (rA + 0), col] = -1
        col += 1
        # Term 6: (A21-A11)(B11+B12)
        U[(rA + 1) * m + (cA + 0), col] = 1
        U[(rA + 0) * m + (cA + 0), col] = -1
        V[(rB + 0) * p + (cB + 0), col] = 1
        V[(rB + 0) * p + (cB + 1), col] = 1
        W[(cB + 1) * n + (rA + 1), col] = 1
        col += 1
        # Term 7: (A12-A22)(B21+B22)
        U[(rA + 0) * m + (cA + 1), col] = 1
        U[(rA + 1) * m + (cA + 1), col] = -1
        V[(rB + 1) * p + (cB + 0), col] = 1
        V[(rB + 1) * p + (cB + 1), col] = 1
        W[(cB + 0) * n + (rA + 0), col] = 1
        col += 1

    def add_schoolbook_1x2x2(rA, cA, rB, cB):
        nonlocal col
        U[(rA + 0) * m + (cA + 0), col] = 1
        V[(rB + 0) * p + (cB + 0), col] = 1
        W[(cB + 0) * n + (rA + 0), col] = 1
        col += 1
        U[(rA + 0) * m + (cA + 1), col] = 1
        V[(rB + 1) * p + (cB + 0), col] = 1
        W[(cB + 0) * n + (rA + 0), col] = 1
        col += 1
        U[(rA + 0) * m + (cA + 0), col] = 1
        V[(rB + 0) * p + (cB + 1), col] = 1
        W[(cB + 1) * n + (rA + 0), col] = 1
        col += 1
        U[(rA + 0) * m + (cA + 1), col] = 1
        V[(rB + 1) * p + (cB + 1), col] = 1
        W[(cB + 1) * n + (rA + 0), col] = 1
        col += 1

    # Apply Strassen to the four 2x2x2 sub-blocks
    add_strassen(0, 0, 0, 0)
    add_strassen(0, 2, 2, 0)
    add_strassen(0, 0, 0, 2)
    add_strassen(0, 2, 2, 2)
    # Apply schoolbook to the four 1x2x2 sub-blocks (row 2 of A, columns 0-1 or 2-3)
    add_schoolbook_1x2x2(2, 0, 0, 0)
    add_schoolbook_1x2x2(2, 2, 2, 0)
    add_schoolbook_1x2x2(2, 0, 0, 2)
    add_schoolbook_1x2x2(2, 2, 2, 2)

    return U, V, W


def _compute_outer_products(U, V, W, n, m, p):
    """Precompute the flattened outer product for each term."""
    rank = U.shape[1]
    outers = np.zeros((n * m * p, rank))
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        outers[:, r] = np.einsum('ij,jk,ki->ijk', u, v, w).flatten()
    return outers


def _try_eliminate_one_term(U, V, W, n, m, p):
    """Try to eliminate one term by finding linear dependency in U (or V or W) columns.
    
    If column r of U is a linear combination of other U columns, we can absorb
    the coefficients into V and W to eliminate that term.
    """
    rank = U.shape[1]
    for r_to_remove in range(rank):
        indices = [i for i in range(rank) if i != r_to_remove]
        
        # Check if U[:, r_to_remove] is in span of U[:, indices]
        try:
            # Solve U[:, indices] @ delta = U[:, r_to_remove]
            delta_u, residuals_u, rank_u, sv_u = np.linalg.lstsq(
                U[:, indices], U[:, r_to_remove], rcond=1e-10
            )
            if rank_u < len(indices) and np.allclose(
                U[:, indices] @ delta_u, U[:, r_to_remove], atol=1e-9, rtol=0
            ):
                # U[:, r] = sum_j delta_u[j] * U[:, indices[j]]
                # So term r: U_r V_r W_r = sum_j delta_u[j] U_j V_r W_r
                # We can add delta_u[j] * V_r to V_j and delta_u[j] * W_r to W_j
                U_new = U[:, indices].copy()
                V_new = V[:, indices].copy()
                W_new = W[:, indices].copy()
                for j, orig_idx in enumerate(indices):
                    V_new[:, j] += delta_u[j] * V[:, r_to_remove]
                    W_new[:, j] += delta_u[j] * W[:, r_to_remove]
                
                if _verify_decomposition(U_new, V_new, W_new, n, m, p, tol=1e-9):
                    return U_new, V_new, W_new
        except Exception:
            pass
        
        # Also try V dependency
        try:
            delta_v, residuals_v, rank_v, sv_v = np.linalg.lstsq(
                V[:, indices], V[:, r_to_remove], rcond=1e-10
            )
            if rank_v < len(indices) and np.allclose(
                V[:, indices] @ delta_v, V[:, r_to_remove], atol=1e-9, rtol=0
            ):
                U_new = U[:, indices].copy()
                V_new = V[:, indices].copy()
                W_new = W[:, indices].copy()
                for j, orig_idx in enumerate(indices):
                    U_new[:, j] += delta_v[j] * U[:, r_to_remove]
                    W_new[:, j] += delta_v[j] * W[:, r_to_remove]
                
                if _verify_decomposition(U_new, V_new, W_new, n, m, p, tol=1e-9):
                    return U_new, V_new, W_new
        except Exception:
            pass
        
        # Also try W dependency
        try:
            delta_w, residuals_w, rank_w, sv_w = np.linalg.lstsq(
                W[:, indices], W[:, r_to_remove], rcond=1e-10
            )
            if rank_w < len(indices) and np.allclose(
                W[:, indices] @ delta_w, W[:, r_to_remove], atol=1e-9, rtol=0
            ):
                U_new = U[:, indices].copy()
                V_new = V[:, indices].copy()
                W_new = W[:, indices].copy()
                for j, orig_idx in enumerate(indices):
                    U_new[:, j] += delta_w[j] * U[:, r_to_remove]
                    V_new[:, j] += delta_w[j] * V[:, r_to_remove]
                
                if _verify_decomposition(U_new, V_new, W_new, n, m, p, tol=1e-9):
                    return U_new, V_new, W_new
        except Exception:
            pass

    return U, V, W


def _try_reduce_rank(U, V, W, n, m, p, max_reductions, time_budget):
    """Iteratively try to reduce the rank by eliminating terms."""
    start_time = time.time()
    U_cur, V_cur, W_cur = U.copy(), V.copy(), W.copy()
    current_rank = U_cur.shape[1]

    for _ in range(max_reductions):
        if time.time() - start_time > time_budget:
            break
        old_rank = U_cur.shape[1]
        U_cur, V_cur, W_cur = _try_eliminate_one_term(U_cur, V_cur, W_cur, n, m, p)
        if U_cur.shape[1] == old_rank:
            # No reduction was made, so stop
            break
        current_rank = U_cur.shape[1]

    return U_cur, V_cur, W_cur


def _try_rank_search(U, V, W, n, m, p, target_rank, time_budget):
    """Try to find a valid decomposition at target_rank by selecting the best subset of terms."""
    start_time = time.time()
    current_rank = U.shape[1]
    
    if target_rank >= current_rank:
        return U, V, W, current_rank
    
    # Precompute the tensor contribution of each term
    # We need to find a subset of size target_rank whose sum equals the target tensor
    
    # Strategy: greedily select terms that contribute most to the remaining tensor
    target_tensor = np.ones((n, m, p), dtype=np.float64)
    
    # Compute contribution of each term
    term_contribs = np.zeros((current_rank, n, m, p), dtype=np.float64)
    for r in range(current_rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        term_contribs[r] = np.einsum('ij,jk,ki->ijk', u, v, w)
    
    # Greedy selection: at each step, pick the term that best matches the remaining tensor
    remaining = target_tensor.copy()
    selected = []
    
    for step in range(target_rank):
        if time.time() - start_time > time_budget:
            break
        
        best_idx = -1
        best_score = -np.inf
        
        for r in range(current_rank):
            if r in selected:
                continue
            # Score: how well term r matches the remaining tensor
            # Use inner product as similarity measure
            score = np.sum(remaining * term_contribs[r])
            if score > best_score:
                best_score = score
                best_idx = r
        
        if best_idx < 0:
            break
        
        selected.append(best_idx)
        remaining -= term_contribs[best_idx]
    
    if len(selected) < target_rank:
        return U, V, W, current_rank
    
    # Check if the selected terms sum to the target
    total = np.zeros((n, m, p), dtype=np.float64)
    for r in selected:
        total += term_contribs[r]
    
    if np.allclose(total, target_tensor, atol=1e-8, rtol=0):
        U_new = U[:, selected].copy()
        V_new = V[:, selected].copy()
        W_new = W[:, selected].copy()
        return U_new, V_new, W_new, len(selected)
    
    return U, V, W, current_rank


def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS

    # Start with the structured rank-44 decomposition
    U44, V44, W44 = _construct_structured_decomposition()
    best_U, best_V, best_W = U44.copy(), V44.copy(), W44.copy()
    best_rank = 44

    # Verify the initial decomposition
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p, tol=1e-9):
        # Fallback: schoolbook rank-48 decomposition
        U48 = np.zeros((n * m, 48))
        V48 = np.zeros((m * p, 48))
        W48 = np.zeros((p * n, 48))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U48[i * m + j, col] = 1
                    V48[j * p + k, col] = 1
                    W48[k * n + i, col] = 1
                    col += 1
        best_U, best_V, best_W = U48, V48, W48
        best_rank = 48

    # Try to reduce rank iteratively
    if time.time() < deadline - 40:
        U_try, V_try, W_try = _try_reduce_rank(
            best_U, best_V, best_W, n, m, p,
            max_reductions=best_rank - 30,
            time_budget=40
        )
        if U_try.shape[1] < best_rank:
            best_U, best_V, best_W = U_try, V_try, W_try
            best_rank = U_try.shape[1]

    # Try direct rank search for lower ranks
    if time.time() < deadline - 20:
        for target in range(best_rank - 1, max(30, best_rank - 10), -1):
            if time.time() > deadline - 15:
                break
            U_try, V_try, W_try, achieved_rank = _try_rank_search(
                best_U, best_V, best_W, n, m, p, target, time_budget=10
            )
            if achieved_rank < best_rank:
                best_U, best_V, best_W = U_try, V_try, W_try
                best_rank = achieved_rank
                break

    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
