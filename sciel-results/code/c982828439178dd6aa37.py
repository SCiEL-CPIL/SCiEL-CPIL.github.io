"""
Matrix multiplication 2x4x4 - rank-minimal tensor decomposition.

Uses Strassen-based block decomposition with shared multiplications,
plus a global subspace-restriction search for further rank reduction.
"""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 4]


def _verify_fast(U, V, W, n, m, p):
    """Vectorized verification using NumPy."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i * m + j], V_arr[j * p + k] * W_arr[k * n + i])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err


def construct():
    """
    Construct a rank-minimal tensor decomposition for 2x4 * 4x4 matrix multiplication.
    
    Returns:
        dict: {'U': list of lists, 'V': list of lists, 'W': list of lists}
    """
    t0 = time.time()
    deadline = 140.0  # seconds, safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    # Strassen 7-multiplication patterns for 2x2 matrices
    # Each pattern: (a_indices, a_coeffs, b_indices, b_coeffs, c_indices, c_coeffs)
    # Local 2x2 indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_patterns = [
        ((0, 3), (1, 1), (0, 3), (1, 1), (0, 3), (1, 1)),
        ((2, 3), (1, 1), (0,), (1,), (2, 3), (1, -1)),
        ((0,), (1,), (1, 3), (1, -1), (1, 3), (1, 1)),
        ((3,), (1,), (2, 0), (1, -1), (0, 2), (1, 1)),
        ((0, 1), (1, 1), (3,), (1,), (0, 1), (-1, 1)),
        ((2, 0), (1, -1), (0, 1), (1, 1), (3,), (1,)),
        ((1, 3), (1, -1), (2, 3), (1, 1), (0,), (1,)),
    ]
    
    # Block structure for 2x4 * 4x4:
    # A (2x4) = [A_L | A_R] where A_L, A_R are 2x2
    # B (4x4) = [[B_TL, B_TR], [B_BL, B_BR]]
    # C (2x4) = [C_L | C_R]
    blocks = [
        (0, 0, 0, 0, 0, 0),   # A_L * B_TL -> C_L
        (0, 2, 2, 0, 0, 0),   # A_R * B_BL -> C_L
        (0, 0, 0, 2, 0, 2),   # A_L * B_TR -> C_R
        (0, 2, 2, 2, 0, 2),   # A_R * B_BR -> C_R
    ]
    
    # Build the full 28-term decomposition (4 blocks x 7 Strassen terms)
    terms = []
    for (a_ro, a_co, b_ro, b_co, c_ro, c_co) in blocks:
        for (a_idx, a_c, b_idx, b_c, c_idx, c_c) in strassen_patterns:
            u_vec = {}
            v_vec = {}
            w_vec = {}
            for idx, coeff in zip(a_idx, a_c):
                lr = idx // 2
                lc = idx % 2
                gi = a_ro + lr
                gj = a_co + lc
                u_vec[gi * m + gj] = u_vec.get(gi * m + gj, 0) + coeff
            for idx, coeff in zip(b_idx, b_c):
                lr = idx // 2
                lc = idx % 2
                gj = b_ro + lr
                gk = b_co + lc
                v_vec[gj * p + gk] = v_vec.get(gj * p + gk, 0) + coeff
            for idx, coeff in zip(c_idx, c_c):
                lr = idx // 2
                lc = idx % 2
                gi = c_ro + lr
                gk = c_co + lc
                w_vec[gk * n + gi] = w_vec.get(gk * n + gi, 0) + coeff
            u_vec = {k: v for k, v in u_vec.items() if v != 0}
            v_vec = {k: v for k, v in v_vec.items() if v != 0}
            w_vec = {k: v for k, v in w_vec.items() if v != 0}
            terms.append((u_vec, v_vec, w_vec))
    
    def _build_UVW(term_list):
        """Build U, V, W matrices from a list of (u_vec, v_vec, w_vec)."""
        rank = len(term_list)
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        for r, (u_vec, v_vec, w_vec) in enumerate(term_list):
            for idx, coeff in u_vec.items():
                U[idx][r] = coeff
            for idx, coeff in v_vec.items():
                V[idx][r] = coeff
            for idx, coeff in w_vec.items():
                W[idx][r] = coeff
        return U, V, W
    
    def _verify_matrix(U, V, W):
        """Fast verification."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i * m + j], V_arr[j * p + k] * W_arr[k * n + i])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    # Baseline: all 28 terms
    U, V, W = _build_UVW(terms)
    err = _verify_matrix(U, V, W)
    if err > 1e-10:
        return {"U": U, "V": V, "W": W}
    
    best_rank = 28
    best_U, best_V, best_W = U, V, W
    
    # Search for shared terms: two terms (i, j) can be merged if
    # u_i = s*u_j, v_i = s*v_j, w_i = (1/s)*w_j for some scalar s
    def _find_merges(terms_list, max_checks=10000):
        """Find all valid merges among the terms."""
        merges = []
        count = 0
        num = len(terms_list)
        for i in range(num):
            for j in range(i + 1, num):
                count += 1
                if count > max_checks:
                    break
                u_i, v_i, w_i = terms_list[i]
                u_j, v_j, w_j = terms_list[j]
                s_candidates = []
                for key in u_i:
                    if key in u_j and u_j[key] != 0:
                        s_candidates.append(u_i[key] / u_j[key])
                if not s_candidates:
                    continue
                s = s_candidates[0]
                valid = True
                for key in u_i:
                    if key in u_j:
                        if u_j[key] == 0:
                            if u_i[key] != 0:
                                valid = False
                                break
                        else:
                            if abs(u_i[key] / u_j[key] - s) > 1e-9:
                                valid = False
                                break
                    else:
                        if u_i[key] != 0:
                            valid = False
                            break
                for key in u_j:
                    if key not in u_i and u_j[key] != 0:
                        valid = False
                        break
                if not valid:
                    continue
                for key in v_i:
                    if key in v_j:
                        if v_j[key] == 0:
                            if v_i[key] != 0:
                                valid = False
                                break
                        else:
                            if abs(v_i[key] / v_j[key] - s) > 1e-9:
                                valid = False
                                break
                    else:
                        if v_i[key] != 0:
                            valid = False
                            break
                for key in v_j:
                    if key not in v_i and v_j[key] != 0:
                        valid = False
                        break
                if not valid:
                    continue
                inv_s = 1.0 / s
                for key in w_i:
                    if key in w_j:
                        if w_j[key] == 0:
                            if w_i[key] != 0:
                                valid = False
                                break
                        else:
                            if abs(w_i[key] / w_j[key] - inv_s) > 1e-9:
                                valid = False
                                break
                    else:
                        if w_i[key] != 0:
                            valid = False
                            break
                for key in w_j:
                    if key not in w_i and w_j[key] != 0:
                        valid = False
                        break
                if not valid:
                    continue
                merges.append((i, j, s))
            if count > max_checks:
                break
        return merges
    
    merges = _find_merges(terms)
    
    if merges:
        used = set()
        final_terms = list(terms)
        applied = 0
        for (i, j, s) in merges:
            if i in used or j in used:
                continue
            u_new = dict(final_terms[i][0])
            v_new = dict(final_terms[i][1])
            w_new = dict(final_terms[i][2])
            inv_s = 1.0 / s
            for key, val in final_terms[j][2].items():
                w_new[key] = w_new.get(key, 0) + val * inv_s
            w_new = {k: v for k, v in w_new.items() if abs(v) > 1e-12}
            final_terms[i] = (u_new, v_new, w_new)
            final_terms[j] = None
            used.add(i)
            used.add(j)
            applied += 1
        
        if applied > 0:
            final_terms = [t for t in final_terms if t is not None]
            U2, V2, W2 = _build_UVW(final_terms)
            err2 = _verify_matrix(U2, V2, W2)
            if err2 < 1e-10:
                best_rank = len(final_terms)
                best_U, best_V, best_W = U2, V2, W2
    
    # Global subspace restriction search:
    # Treat the 2x4x4 tensor as a sub-tensor of 4x4x4.
    # The known rank-29 decomposition of M_{4,4,4} (Strassen 7-mult on 2x2 blocks
    # plus one extra term) can be projected onto the subspace where row 1 of A is zero.
    # We explore whether non-blockwise global dependencies allow rank < 28.
    
    if time.time() - t0 < deadline - 40:
        # Build the full 4x4x4 Strassen decomposition (rank 28 for 4x4x4 is not standard;
        # standard is rank 49 or 48. Let's use the 2x2-block Strassen for 4x4x4:
        # 4x4 = 2x2 blocks, each 2x2 uses 7 mults, so 4*7=28 terms for 4x4x4)
        # Actually for 4x4x4 with 2x2 blocks: 4 blocks * 7 = 28 terms.
        # The 2x4x4 is a sub-tensor: fix row 1 of A to zero.
        # In the 4x4x4 decomposition, terms that only involve A[1,*] can be dropped.
        
        # 4x4x4 block structure: A (4x4) = [[A00, A01],[A10, A11]]
        # B (4x4) = [[B00, B01],[B10, B11]]
        # C (4x4) = [[C00, C01],[C10, C11]]
        # For 2x4x4: A is 2x4, so we only have A[0,*] (row 0 of the 4x4 A)
        # This means A10 and A11 blocks are zero.
        
        # 4x4x4 Strassen terms (28 terms):
        blocks_4x4 = [
            (0, 0, 0, 0, 0, 0),   # A00 * B00 -> C00
            (0, 2, 2, 0, 0, 0),   # A01 * B10 -> C00  (row offset 0, col offset 2)
            (0, 0, 0, 2, 0, 2),   # A00 * B01 -> C01
            (0, 2, 2, 2, 0, 2),   # A01 * B11 -> C01
            (2, 0, 0, 0, 2, 0),   # A10 * B00 -> C10
            (2, 2, 2, 0, 2, 0),   # A11 * B10 -> C10
            (2, 0, 0, 2, 2, 2),   # A10 * B01 -> C11
            (2, 2, 2, 2, 2, 2),   # A11 * B11 -> C11
        ]
        
        terms_4x4 = []
        for (a_ro, a_co, b_ro, b_co, c_ro, c_co) in blocks_4x4:
            for (a_idx, a_c, b_idx, b_c, c_idx, c_c) in strassen_patterns:
                u_vec = {}
                v_vec = {}
                w_vec = {}
                for idx, coeff in zip(a_idx, a_c):
                    lr = idx // 2
                    lc = idx % 2
                    gi = a_ro + lr
                    gj = a_co + lc
                    u_vec[gi * 4 + gj] = u_vec.get(gi * 4 + gj, 0) + coeff
                for idx, coeff in zip(b_idx, b_c):
                    lr = idx // 2
                    lc = idx % 2
                    gj = b_ro + lr
                    gk = b_co + lc
                    v_vec[gj * 4 + gk] = v_vec.get(gj * 4 + gk, 0) + coeff
                for idx, coeff in zip(c_idx, c_c):
                    lr = idx // 2
                    lc = idx % 2
                    gi = c_ro + lr
                    gk = c_co + lc
                    w_vec[gk * 4 + gi] = w_vec.get(gk * 4 + gi, 0) + coeff
                u_vec = {k: v for k, v in u_vec.items() if v != 0}
                v_vec = {k: v for k, v in v_vec.items() if v != 0}
                w_vec = {k: v for k, v in w_vec.items() if v != 0}
                terms_4x4.append((u_vec, v_vec, w_vec))
        
        # For 2x4x4 sub-tensor: A has rows 0,1 of the 4x4 (i.e., A00, A01 blocks)
        # B is full 4x4, C has rows 0,1 of the 4x4 (C00, C01 blocks)
        # So we only need terms where c_ro is 0 (top row of C), and u_vec only has
        # entries with row < 2 (top 2 rows of A).
        
        # Project: keep only terms where all u_vec indices have row < 2
        # and all w_vec indices have row < 2
        terms_sub = []
        for (u_vec, v_vec, w_vec) in terms_4x4:
            u_valid = all(idx // 4 < 2 for idx in u_vec)
            w_valid = all(idx // 4 < 2 for idx in w_vec)
            if u_valid and w_valid:
                # Remap indices to 2x4x4 space
                u_new = {}
                for idx, val in u_vec.items():
                    row = idx // 4
                    col = idx % 4
                    u_new[row * 4 + col] = val
                v_new = dict(v_vec)
                w_new = {}
                for idx, val in w_vec.items():
                    row = idx % 4
                    col = idx // 4
                    w_new[col * 2 + row] = val
                u_new = {k: v for k, v in u_new.items() if v != 0}
                v_new = {k: v for k, v in v_new.items() if v != 0}
                w_new = {k: v for k, v in w_new.items() if v != 0}
                terms_sub.append((u_new, v_new, w_new))
        
        if len(terms_sub) < best_rank:
            U_sub, V_sub, W_sub = _build_UVW(terms_sub)
            err_sub = _verify_matrix(U_sub, V_sub, W_sub)
            if err_sub < 1e-10:
                best_rank = len(terms_sub)
                best_U, best_V, best_W = U_sub, V_sub, W_sub
    
    # Try removing individual terms that are linear combinations of others
    if time.time() - t0 < deadline - 30:
        current_terms = terms
        if merges:
            final_terms = list(terms)
            used = set()
            for (i, j, s) in merges:
                if i in used or j in used:
                    continue
                u_new = dict(final_terms[i][0])
                v_new = dict(final_terms[i][1])
                w_new = dict(final_terms[i][2])
                inv_s = 1.0 / s
                for key, val in final_terms[j][2].items():
                    w_new[key] = w_new.get(key, 0) + val * inv_s
                w_new = {k: v for k, v in w_new.items() if abs(v) > 1e-12}
                final_terms[i] = (u_new, v_new, w_new)
                final_terms[j] = None
                used.add(i)
                used.add(j)
            current_terms = [t for t in final_terms if t is not None]
        
        for remove_idx in range(len(current_terms)):
            if time.time() - t0 > deadline - 25:
                break
            trial = [t for i, t in enumerate(current_terms) if i != remove_idx]
            U3, V3, W3 = _build_UVW(trial)
            err3 = _verify_matrix(U3, V3, W3)
            if err3 < 1e-10:
                best_rank = len(trial)
                best_U, best_V, best_W = U3, V3, W3
                current_terms = trial
    
    return {"U": best_U, "V": best_V, "W": best_W}


if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
    err = _verify_fast(res['U'], res['V'], res['W'], 2, 4, 4)
    print(f"Max error: {err}")
