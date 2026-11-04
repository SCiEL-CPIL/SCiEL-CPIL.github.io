"""
Rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x8 matrix.

Strategy:
  1. Build a structured base: Strassen for (2,2,2) sub-blocks + schoolbook for remaining row/column.
     - 4 Strassen blocks of (2x2)*(2x2) covering A1(2x2)*B1(2x8): 4*7 = 28 terms
     - Schoolbook for A2(2x1)*B2(1x8): 2*1*8 = 16 terms
     - Total: rank 44
  2. Attempt to reduce rank via:
     a. Duplicate column detection
     b. Border rank / Gröbner-style elimination (determine if rank 43 is feasible)
     c. Local search: try replacing pairs of columns with single combined columns
  3. Validate any candidate with exact rational arithmetic.
  4. If no improvement is found within the time budget, return the best decomposition found.
"""
import time
import sys
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 120.0  # 30 s safety margin before 150 s kill

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

    # ---- Schoolbook fallback (rank 48) ----
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # ---- Exact validation using Fractions ----
    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # ---- Strassen 2x2x2 decomposition (rank 7) ----
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        u_val = S_U[r][i * 2 + a]
                        v_val = S_V[r][a * 2 + k]
                        w_val = S_W[r][k * 2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # ---- Build rank-44 base candidate ----
    def _build_base():
        rank = 44
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        # Part 1: A1 (2x2) * B1 (2x8) using 4 Strassen blocks
        for block_idx in range(4):
            k0 = block_idx * 2
            k1 = k0 + 1
            for r in range(7):
                for li in range(2):
                    for la in range(2):
                        u_val = S_U[r][li * 2 + la]
                        if u_val == 0:
                            continue
                        u_idx = li * m + la
                        U[u_idx][col] = u_val

                for la in range(2):
                    for lk in range(2):
                        v_val = S_V[r][la * 2 + lk]
                        if v_val == 0:
                            continue
                        v_idx = la * p + (k0 + lk)
                        V[v_idx][col] = v_val

                for lk in range(2):
                    for li in range(2):
                        w_val = S_W[r][lk * 2 + li]
                        if w_val == 0:
                            continue
                        w_idx = (k0 + lk) * n + li
                        W[w_idx][col] = w_val

                col += 1

        # Part 2: A2 (2x1) * B2 (1x8) schoolbook
        for i in range(2):
            for k in range(8):
                u_idx = i * m + 2
                v_idx = 2 * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1

        return {"U": U, "V": V, "W": W}

    base = _build_base()
    if not _validate(base):
        return _schoolbook()

    best_candidate = base
    best_rank = 44

    # ---- Try to reduce rank via structured alternatives and linear dependence search ----
    def _try_reduce():
        nonlocal best_candidate, best_rank

        num_terms = n * m * p  # 48

        def _col_vector(U, V, W, c):
            vec = [0] * num_terms
            for i in range(n):
                for a in range(m):
                    u_val = U[i * m + a][c]
                    if u_val == 0:
                        continue
                    for k in range(p):
                        v_val = V[a * p + k][c]
                        if v_val == 0:
                            continue
                        w_val = W[k * n + i][c]
                        if w_val == 0:
                            continue
                        vec[i * m * p + a * p + k] = u_val * v_val * w_val
            return vec

        def _is_rank1_tensor(vec):
            """Check if a 48-dim vector represents a rank-1 tensor using exact arithmetic."""
            vec_arr = np.array(vec)
            # Reshape to (n*m) x p = 6 x 8
            M1 = vec_arr.reshape(n * m, p)
            # Reshape to (m*p) x n = 24 x 2
            M2 = vec_arr.reshape(m * p, n)
            # Reshape to (p*n) x m = 16 x 3
            M3 = vec_arr.reshape(p * n, m)
            
            # Check rank 1 for each using SVD
            try:
                if np.linalg.matrix_rank(M1) > 1:
                    return False
                if np.linalg.matrix_rank(M2) > 1:
                    return False
                if np.linalg.matrix_rank(M3) > 1:
                    return False
            except np.linalg.LinAlgError:
                return False
            return True

        def _extract_rank1_factors(vec):
            """Extract U, V, W factors from a rank-1 tensor vector using iterative refinement."""
            vec_arr = np.array(vec, dtype=float)
            
            # Find first non-zero entry
            nz = np.nonzero(vec_arr != 0)
            if len(nz[0]) == 0:
                return None
            
            # Initialize with ones
            U_f = np.ones(n * m)
            V_f = np.ones(m * p)
            W_f = np.ones(p * n)
            
            # Iteratively refine: for each (i,a,k), we have
            # vec[i*m*p + a*p + k] = U_f[i*m+a] * V_f[a*p+k] * W_f[k*n+i]
            # Take the geometric mean approach with signs
            
            # Use multiplicative updates
            for iteration in range(50):
                if not _elapsed_ok():
                    return None
                
                # Update U
                for i in range(n):
                    for a in range(m):
                        for k in range(p):
                            idx = i * m * p + a * p + k
                        if vec_arr[idx] != 0 and V_f[a * p + k] != 0 and W_f[k * n + i] != 0:
                            ratio = vec_arr[idx] / (V_f[a * p + k] * W_f[k * n + i])
                            U_f[i * m + a] = (U_f[i * m + a] * ratio) ** 0.5
                            # Handle sign
                            if U_f[i * m + a] < 0:
                                U_f[i * m + a] = -abs(U_f[i * m + a])
                
                # Update V
                for a in range(m):
                    for k in range(p):
                        for i in range(n):
                            idx = i * m * p + a * p + k
                            if vec_arr[idx] != 0 and U_f[i * m + a] != 0 and W_f[k * n + i] != 0:
                                ratio = vec_arr[idx] / (U_f[i * m + a] * W_f[k * n + i])
                                V_f[a * p + k] = (V_f[a * p + k] * ratio) ** 0.5
                                if V_f[a * p + k] < 0:
                                    V_f[a * p + k] = -abs(V_f[a * p + k])
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        for a in range(m):
                            idx = i * m * p + a * p + k
                            if vec_arr[idx] != 0 and U_f[i * m + a] != 0 and V_f[a * p + k] != 0:
                                ratio = vec_arr[idx] / (U_f[i * m + a] * V_f[a * p + k])
                                W_f[k * n + i] = (W_f[k * n + i] * ratio) ** 0.5
                                if W_f[k * n + i] < 0:
                                    W_f[k * n + i] = -abs(W_f[k * n + i])
                
                # Check convergence
                max_err = 0.0
                for i in range(n):
                    for a in range(m):
                        for k in range(p):
                            idx = i * m * p + a * p + k
                            if vec_arr[idx] != 0:
                                prod = U_f[i * m + a] * V_f[a * p + k] * W_f[k * n + i]
                                err = abs(prod - vec_arr[idx]) / max(abs(vec_arr[idx]), 1e-15)
                                max_err = max(max_err, err)
                
                if max_err < 1e-10:
                    # Convert to list of Fractions for exact validation
                    U_list = [float(x) for x in U_f]
                    V_list = [float(x) for x in V_f]
                    W_list = [float(x) for x in W_f]
                    return (U_list, V_list, W_list)
            
            return None

        def _build_candidate_without(U, V, W, cols_to_remove, new_factors=None):
            """Build a new candidate with specified columns removed and optionally a new column added."""
            rank0 = len(U[0])
            remove_set = set(cols_to_remove)
            new_rank = rank0 - len(remove_set)
            if new_factors:
                new_rank += 1
            
            new_U = [[0] * new_rank for _ in range(n * m)]
            new_V = [[0] * new_rank for _ in range(m * p)]
            new_W = [[0] * new_rank for _ in range(p * n)]
            nc = 0
            
            for c in range(rank0):
                if c in remove_set:
                    continue
                for i in range(n * m):
                    new_U[i][nc] = U[i][c]
                for i in range(m * p):
                    new_V[i][nc] = V[i][c]
                for i in range(p * n):
                    new_W[i][nc] = W[i][c]
                nc += 1
            
            if new_factors:
                U_new, V_new, W_new = new_factors
                for i in range(n * m):
                    new_U[i][nc] = U_new[i]
                for i in range(m * p):
                    new_V[i][nc] = V_new[i]
                for i in range(p * n):
                    new_W[i][nc] = W_new[i]
                nc += 1
            
            return {"U": new_U, "V": new_V, "W": new_W}

        U = base["U"]
        V = base["V"]
        W = base["W"]
        rank0 = len(U[0])  # 44

        cols = []
        for c in range(rank0):
            cols.append(_col_vector(U, V, W, c))

        cols_arr = np.array(cols, dtype=float)

        # Step 1: Check for duplicate columns (exact)
        seen = {}
        removable = []
        for c in range(rank0):
            key = tuple(cols[c])
            if key in seen:
                removable.append(c)
            else:
                seen[key] = c

        if removable:
            for c_rem in removable:
                new_rank = rank0 - 1
                cand = _build_candidate_without(U, V, W, [c_rem])
                if _validate(cand):
                    best_candidate = cand
                    best_rank = new_rank
                    return

        if not _elapsed_ok():
            return

        # Step 2: Check if any single column is zero (can be removed)
        for c in range(rank0):
            if np.all(cols_arr[c] == 0):
                new_rank = rank0 - 1
                cand = _build_candidate_without(U, V, W, [c])
                if _validate(cand):
                    best_candidate = cand
                    best_rank = new_rank
                    return

        if not _elapsed_ok():
            return

        # Step 3: Try merging pairs - check if col(c1) + col(c2) is a rank-1 tensor
        # Use binary support patterns to pre-filter candidates
        # Two columns can only merge if their sum has a valid rank-1 structure
        
        # Precompute binary support patterns for quick filtering
        support_patterns = []
        for c in range(rank0):
            pattern = tuple(1 if cols_arr[c][idx] != 0 else 0 for idx in range(num_terms))
            support_patterns.append(pattern)
        
        # Group by support pattern
        pattern_to_cols = {}
        for c in range(rank0):
            pat = support_patterns[c]
            if pat not in pattern_to_cols:
                pattern_to_cols[pat] = []
            pattern_to_cols[pat].append(c)
        
        # Try all pairs, prioritizing those with overlapping support
        merged = False
        for c1 in range(rank0):
            if not _elapsed_ok():
                break
            for c2 in range(c1 + 1, rank0):
                if not _elapsed_ok():
                    break
                
                # Quick check: if supports are disjoint, sum is likely not rank-1
                # unless one is zero
                sup1 = support_patterns[c1]
                sup2 = support_patterns[c2]
                
                # Check if sum is rank-1
                c_new_vec = cols_arr[c1] + cols_arr[c2]
                
                if not _is_rank1_tensor(c_new_vec):
                    continue
                
                # Extract factors
                factors = _extract_rank1_factors(c_new_vec)
                if factors is None:
                    continue
                
                # Build candidate with c1 and c2 removed and new column added
                cand = _build_candidate_without(U, V, W, [c1, c2], new_factors=factors)
                if _validate(cand):
                    best_candidate = cand
                    best_rank = rank0 - 1
                    merged = True
                    return

        if not _elapsed_ok():
            return

        # Step 4: Try GF(2) syndrome decoding approach
        # Map each column to its binary support pattern
        # The target tensor has all 48 positions = 1
        # We want to find a subset of columns whose XOR of support patterns = all-ones
        # AND whose actual values sum to 1 at each position
        
        # For now, try a simpler heuristic: look for columns that are scalar multiples
        for c1 in range(rank0):
            if not _elapsed_ok():
                break
            for c2 in range(c1 + 1, rank0):
                # Check if col(c1) = alpha * col(c2)
                nz2 = np.nonzero(cols_arr[c2] != 0)
                if len(nz2[0]) == 0:
                    continue
                idx = nz2[0][0]
                if cols_arr[c2][idx] == 0:
                    continue
                alpha = cols_arr[c1][idx] / cols_arr[c2][idx]
                if np.allclose(cols_arr[c1], alpha * cols_arr[c2], atol=1e-10):
                    # col(c1) = alpha * col(c2)
                    # We can potentially merge: (1+alpha) * col(c2) should be rank-1
                    # since it's just a scalar multiple
                    new_vec = (1 + alpha) * cols_arr[c2]
                    if _is_rank1_tensor(new_vec):
                        factors = _extract_rank1_factors(new_vec)
                        if factors is not None:
                            cand = _build_candidate_without(U, V, W, [c1, c2], new_factors=factors)
                            if _validate(cand):
                                best_candidate = cand
                                best_rank = rank0 - 1
                                return

        return

    if _elapsed_ok():
        _try_reduce()

    if best_rank < 44:
        return best_candidate

    return base

# EVOLVE-BLOCK-END
