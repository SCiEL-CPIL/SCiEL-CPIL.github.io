"""
Matrix multiplication 2x4x7 via tensor decomposition.
Attempts a targeted search for a rank-49 decomposition by optimizing the
last 3 columns (2x4x3 sub-problem) to rank 21, combined with the
known rank-28 decomposition for the first 4 columns.

Score = seed / rank. seed = 56 (schoolbook).
Rank 50 -> 1.12, Rank 49 -> 1.1429.
"""

import time
import numpy as np

# ---------------------------------------------------------------------------
# Deadline guard
# ---------------------------------------------------------------------------
_EVAL_DEADLINE = 150.0
_START = time.time()
TIME_BUDGET = _EVAL_DEADLINE - (time.time() - _START) - 5.0

def _time_left():
    return _EVAL_DEADLINE - 5.0 - time.time()

# ---------------------------------------------------------------------------
# Build the multiplication tensor T in [r,i,j] layout
# r = k*n + i, i is row of A, j = row of B / col of B (flattened j*p+k)
# ---------------------------------------------------------------------------
def _build_tensor(n, m, p):
    r_dim = p * n
    i_dim = n
    j_dim = m * p
    T = np.zeros((r_dim, i_dim, j_dim))
    for k in range(p):
        for i in range(n):
            for j in range(m):
                r = k * n + i
                jj = j * p + k
                T[r, i, jj] = 1.0
    return T

# ---------------------------------------------------------------------------
# Known rank-50 construction
# ---------------------------------------------------------------------------
def _construct_rank50(n=2, m=4, p=7):
    rank = 50
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    col = [0]

    def add_strassen(i0, j0, k0):
        c = col[0]
        # M1
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        c += 1
        # M2
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] -= 1
        c += 1
        # M3
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] -= 1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        c += 1
        # M4
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] -= 1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        c += 1
        # M5
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] -= 1
        W[(k0+1)*n + i0][c] += 1
        c += 1
        # M6
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] -= 1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        c += 1
        # M7
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] -= 1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        c += 1
        col[0] = c

    # 2x4x4 block: four 2x2x2 Strassen blocks
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    
    # 2x4x2 block: two 2x2x2 Strassen blocks (for k=4,5)
    add_strassen(0, 0, 4)
    add_strassen(0, 2, 4)
    
    # 2x4x1 block: direct terms (for k=6)
    for i in range(n):
        for j in range(m):
            U[i*m + j][col[0]] = 1.0
            V[j*p + 6][col[0]] = 1.0
            W[6*n + i][col[0]] = 1.0
            col[0] += 1

    assert col[0] == rank, f"Rank mismatch: used {col[0]}, expected {rank}"
    return {"U": U, "V": V, "W": W}

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def _verify(U, V, W, n, m, p, tol=1e-10):
    r_dim = p * n
    i_dim = n
    j_dim = m * p
    T = _build_tensor(n, m, p)
    T_mat = T.reshape(r_dim, i_dim * j_dim)
    
    rank = len(W[0])
    Ur = np.array(U).T   # (rank, i_dim)
    Vr = np.array(V).T   # (rank, j_dim)
    Wr = np.array(W).T   # (rank, r_dim)
    
    cand = np.zeros_like(T_mat)
    for t in range(rank):
        wt = Wr[t, :]
        ut = Ur[t, :]
        vt = Vr[t, :]
        outer = ut[:, None] * vt[None, :]
        cand += np.outer(wt, outer.ravel())
        
    max_err = np.max(np.abs(cand - T_mat))
    return max_err < tol

# ---------------------------------------------------------------------------
# Targeted Search for Rank 49
# ---------------------------------------------------------------------------
def _search_rank49(n=2, m=4, p=7, time_budget=30.0):
    """
    Focuses on the last 3 columns (k=4,5,6).
    Base: 14 terms (Strassen) + 8 terms (Direct) = 22 terms.
    Goal: Reduce to 21 terms.
    
    Strategy: For each pair of terms to remove, compute the residual and check
    if it's rank-1. If so, extract (u,v,w) from the rank-1 structure.
    """
    if _time_left() < 5:
        return None

    # Get base rank 50
    base = _construct_rank50(n, m, p)
    U_base = np.array(base["U"])
    V_base = np.array(base["V"])
    W_base = np.array(base["W"])
    
    # Extract the sub-tensor for k=4,5,6
    r_sub_indices = [8, 9, 10, 11, 12, 13]
    i_sub_indices = [0, 1]
    j_sub_indices = [4, 11, 18, 25, 5, 12, 19, 26, 6, 13, 20, 27]
    
    T_full = _build_tensor(n, m, p)
    T_sub = T_full[r_sub_indices][:, i_sub_indices][:, j_sub_indices]
    T_sub_mat = T_sub.reshape(6, 24)
    
    # Initial 22 terms for this sub-tensor
    U_sub_base = U_base[:, 28:50]  # (8, 22)
    V_sub_base_mapped = np.zeros((12, 22))
    for local_j, global_j in enumerate(j_sub_indices):
        V_sub_base_mapped[local_j, :] = V_base[global_j, 28:50]
        
    W_sub_base_mapped = np.zeros((6, 22))
    for local_r, global_r in enumerate(r_sub_indices):
        W_sub_base_mapped[local_r, :] = W_base[global_r, 28:50]
    
    # Verify base sub-decomposition
    def verify_sub(U_s, V_s, W_s, T_s_mat, tol=1e-10):
        rank = U_s.shape[1]
        cand = np.zeros_like(T_s_mat)
        for t in range(rank):
            wt = W_s[:, t]
            ut = U_s[:, t]
            vt = V_s[:, t]
            outer = ut[:, None] * vt[None, :]
            cand += np.outer(wt, outer.ravel())
        return np.max(np.abs(cand - T_s_mat)) < tol

    if not verify_sub(U_sub_base, V_sub_base_mapped, W_sub_base_mapped, T_sub_mat):
        return None

    target_rank = 21
    base_rank = 22
    
    # Precompute term contributions for fast residual calculation
    term_contribs = []
    for t in range(base_rank):
        wt = W_sub_base_mapped[:, t]
        ut = U_sub_base[:, t]
        vt = V_sub_base_mapped[:, t]
        outer = ut[:, None] * vt[None, :]
        term_contribs.append(np.outer(wt, outer.ravel()))
    
    # Total tensor = sum of all terms
    total = np.sum(term_contribs, axis=0)
    
    # Try merging each pair (t1, t2)
    for t1 in range(base_rank):
        if _time_left() < 5:
            break
        for t2 in range(t1 + 1, base_rank):
            if _time_left() < 5:
                break
            
            # Residual = total - term_t1 - term_t2
            residual = total - term_contribs[t1] - term_contribs[t2]
            
            # Check if residual is rank-1: all 2x2 minors should be zero
            # residual is (6, 24). Rank-1 means all 2x2 subdeterminants are 0.
            # Quick check: SVD
            try:
                svals = np.linalg.svd(residual, full_matrices=False, compute_uv=False)
            except np.linalg.LinAlgError:
                continue
            
            if len(svals) < 2 or svals[1] < 1e-8:
                # Residual is (numerically) rank-1
                # Extract (w, u, v) from residual
                # residual[r, i*12+j] = w[r] * u[i] * v[j]
                
                # Find a nonzero entry to normalize
                nz_mask = np.abs(residual) > 1e-10
                if not np.any(nz_mask):
                    continue
                    
                # Get first nonzero entry
                nz_idx = np.argwhere(nz_mask)[0]
                r0, col0 = nz_idx
                i0 = col0 // 12
                j0 = col0 % 12
                
                val = residual[r0, col0]
                
                # Try to extract w, u, v
                # w[r0] * u[i0] * v[j0] = val
                # We need to determine the sparsity pattern.
                
                # For rank-1, each row is proportional to each column.
                # w[r] = residual[r, col0] / (u[i0] * v[j0])
                # u[i] = residual[r0, i*12+j0] / (w[r0] * v[j0])
                # v[j] = residual[r0, i0*12+j] / (w[r0] * u[i0])
                
                # We need to determine which entries are nonzero.
                # Start with the known nonzero at (r0, i0, j0).
                
                # Determine support of w: rows where residual[r, col0] != 0
                w_support = np.where(np.abs(residual[:, col0]) > 1e-10)[0]
                if len(w_support) == 0:
                    continue
                    
                # Determine support of u: for a fixed r in w_support, columns where residual[r, :] != 0
                r_ref = w_support[0]
                u_col_support = np.where(np.abs(residual[r_ref, :]) > 1e-10)[0]
                if len(u_col_support) == 0:
                    continue
                u_support = set()
                for c in u_col_support:
                    u_support.add(c // 12)
                u_support = sorted(u_support)
                
                # Determine support of v: for a fixed i in u_support, columns where residual[:, i*12+j] != 0
                i_ref = u_support[0]
                v_support = set()
                for r in w_support:
                    for j in range(12):
                        if np.abs(residual[r, i_ref * 12 + j]) > 1e-10:
                            v_support.add(j)
                v_support = sorted(v_support)
                
                if len(u_support) == 0 or len(v_support) == 0:
                    continue
                
                # Now extract values
                # Use the known entry to normalize
                # Let's set u[i0] = 1, then w[r0] * v[j0] = val
                # Or better: use multiple entries to solve.
                
                # Build linear system
                # unknowns: w[r] for r in w_support, u[i] for i in u_support, v[j] for j in v_support
                # But we have scale invariance: (w*c, u/c, v) or (w, u*c, v/c) etc.
                
                # Fix: set u[i0] = 1
                u_new = np.zeros(2)
                u_new[i0] = 1.0
                
                # Then w[r0] * v[j0] = val
                # And for each r in w_support: w[r] * u[i0] * v[j0] = residual[r, col0]
                # So w[r] * v[j0] = residual[r, col0] (since u[i0]=1)
                
                # For each i in u_support: w[r0] * u[i] * v[j0] = residual[r0, i*12+j0]
                # So w[r0] * u[i] = residual[r0, i*12+j0] / v[j0]
                
                # This is underdetermined. Let's use a different approach.
                # Use SVD to get the rank-1 factors.
                U_svd, s, Vt_svd = np.linalg.svd(residual, full_matrices=False)
                if s[0] < 1e-10:
                    continue
                    
                w_svd = U_svd[:, 0] * np.sqrt(s[0])
                uv_svd = Vt_svd[0, :] * np.sqrt(s[0])
                
                # uv_svd is (24,) = flatten(u * v)
                # Reshape to (2, 12)
                uv_matrix = uv_svd.reshape(2, 12)
                
                # Check if uv_matrix is rank-1
                s_uv = np.linalg.svd(uv_matrix, full_matrices=False, compute_uv=False)
                if len(s_uv) > 1 and s_uv[1] > 1e-8:
                    continue
                
                # Extract u and v from uv_matrix
                # uv_matrix[i, j] = u[i] * v[j]
                # Find nonzero entries
                nz_u = np.where(np.any(np.abs(uv_matrix) > 1e-10, axis=1))[0]
                nz_v = np.where(np.any(np.abs(uv_matrix) > 1e-10, axis=0))[0]
                
                if len(nz_u) == 0 or len(nz_v) == 0:
                    continue
                
                # Normalize: set u[nz_u[0]] = 1
                u_new = np.zeros(2)
                v_new = np.zeros(12)
                
                i_ref2 = nz_u[0]
                j_ref2 = nz_v[0]
                
                # uv_matrix[i_ref2, j_ref2] = u[i_ref2] * v[j_ref2]
                # Set u[i_ref2] = 1, then v[j_ref2] = uv_matrix[i_ref2, j_ref2]
                u_new[i_ref2] = 1.0
                v_new[j_ref2] = uv_matrix[i_ref2, j_ref2]
                
                # Fill in u and v
                for i in nz_u:
                    if i == i_ref2:
                        continue
                    # Find a j where uv_matrix[i, j] != 0
                    nz_j_for_i = np.where(np.abs(uv_matrix[i, :]) > 1e-10)[0]
                    if len(nz_j_for_i) == 0:
                        continue
                    j_pick = nz_j_for_i[0]
                    if v_new[j_pick] != 0:
                        u_new[i] = uv_matrix[i, j_pick] / v_new[j_pick]
                    else:
                        # Find another j
                        for j in nz_j_for_i:
                            if v_new[j] != 0:
                                u_new[i] = uv_matrix[i, j] / v_new[j]
                                break
                
                for j in nz_v:
                    if j == j_ref2:
                        continue
                    nz_i_for_j = np.where(np.abs(uv_matrix[:, j]) > 1e-10)[0]
                    if len(nz_i_for_j) == 0:
                        continue
                    i_pick = nz_i_for_j[0]
                    if u_new[i_pick] != 0:
                        v_new[j] = uv_matrix[i_pick, j] / u_new[i_pick]
                    else:
                        for i in nz_i_for_j:
                            if u_new[i] != 0:
                                v_new[j] = uv_matrix[i, j] / u_new[i]
                                break
                
                w_new = w_svd.copy()
                
                # Verify the extracted term matches the residual
                term = np.outer(w_new, np.outer(u_new, v_new).flatten())
                if np.max(np.abs(term - residual)) < 1e-8:
                    # Round to nearest half-integer
                    def round_half(x):
                        return np.round(x * 2) / 2
                    
                    w_rounded = round_half(w_new)
                    u_rounded = round_half(u_new)
                    v_rounded = round_half(v_new)
                    
                    term_rounded = np.outer(w_rounded, np.outer(u_rounded, v_rounded).flatten())
                    if np.max(np.abs(term_rounded - residual)) < 1e-8:
                        # Found a valid rank-1 term with half-integer values!
                        # Construct full rank 49 solution
                        mask = [x for x in range(base_rank) if x != t1 and x != t2]
                        U_rem = U_sub_base[:, mask]
                        V_rem = V_sub_base_mapped[:, mask]
                        W_rem = W_sub_base_mapped[:, mask]
                        
                        U_full = np.zeros((8, 49))
                        V_full = np.zeros((28, 49))
                        W_full = np.zeros((14, 49))
                        
                        # Copy first 28 terms
                        U_full[:, :28] = U_base[:, :28]
                        V_full[:, :28] = V_base[:, :28]
                        W_full[:, :28] = W_base[:, :28]
                        
                        # Copy remaining 20 terms
                        U_full[:, 28:48] = U_rem
                        for local_j, global_j in enumerate(j_sub_indices):
                            V_full[global_j, 28:48] = V_rem[local_j, :]
                        for local_r, global_r in enumerate(r_sub_indices):
                            W_full[global_r, 28:48] = W_rem[local_r, :]
                            
                        # Add new term
                        U_full[:, 48] = u_rounded
                        for local_j, global_j in enumerate(j_sub_indices):
                            V_full[global_j, 48] = v_rounded[local_j]
                        for local_r, global_r in enumerate(r_sub_indices):
                            W_full[global_r, 48] = w_rounded[local_r]
                            
                        # Verify full solution
                        if _verify(U_full.tolist(), V_full.tolist(), W_full.tolist(), n, m, p):
                            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    return None

# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------
def construct():
    n, m, p = 2, 4, 7
    
    # Known rank-50 solution
    fallback = _construct_rank50(n, m, p)
    
    # Try targeted search for rank 49
    if _time_left() > 10:
        result = _search_rank49(n, m, p, min(_time_left(), 40.0))
        if result is not None:
            return result
            
    return fallback
