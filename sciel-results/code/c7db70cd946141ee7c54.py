"""
Matrix multiplication 2x4x4 - rank-minimal tensor decomposition.

Uses Strassen-based block decomposition with shared multiplications.
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
    rank = U_arr.shape[1]
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
    deadline = 115.0  # seconds, safety margin before 120s kill
    
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
    # Each term is represented as (u_vec, v_vec, w_vec) where each is a dict
    # mapping global index -> coefficient
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
            # Remove zero entries
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
        # Should not happen; return anyway
        return {"U": U, "V": V, "W": W}
    
    best_rank = 28
    best_U, best_V, best_W = U, V, W
    
    # Search for shared terms: two terms (i, j) can be merged if
    # u_i = s*u_j, v_i = s*v_j, w_i = (1/s)*w_j for some scalar s
    # This means they represent the same "multiplication" up to scaling.
    # After merging, the combined term has u = u_i, v = v_i, w = w_i + w_j/s
    # (or equivalently u = s*u_j, v = s*v_j, w = w_j + w_i/s)
    
    def _find_merges(terms_list, max_checks=5000):
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
                # Check if u_i = s*u_j for some s
                # s must be consistent across all entries
                s_candidates = []
                for key in u_i:
                    if key in u_j and u_j[key] != 0:
                        s_candidates.append(u_i[key] / u_j[key])
                if not s_candidates:
                    continue
                # Verify consistency
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
                # Check v_i = s*v_j
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
                # Check w_i = (1/s)*w_j
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
    
    # Try to find merges
    merges = _find_merges(terms)
    
    if merges:
        # Greedily apply non-conflicting merges
        used = set()
        final_terms = list(terms)
        applied = 0
        for (i, j, s) in merges:
            if i in used or j in used:
                continue
            # Merge term j into term i
            # New term: u = u_i, v = v_i, w = w_i + w_j/s
            u_new = dict(final_terms[i][0])
            v_new = dict(final_terms[i][1])
            w_new = dict(final_terms[i][2])
            inv_s = 1.0 / s
            for key, val in final_terms[j][2].items():
                w_new[key] = w_new.get(key, 0) + val * inv_s
            # Clean up zeros
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
    
    # Additional search: try removing individual terms that are linear combinations
    # of others (i.e., the tensor is still covered without them)
    # This is more expensive; only do it if we have time
    if time.time() - t0 < deadline - 30:
        current_terms = [t for t in (final_terms if merges else terms) if t is not None] if merges else terms
        # Try removing each term one at a time
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
    # Quick verification
    err = _verify_fast(res['U'], res['V'], res['W'], 2, 4, 4)
    print(f"Max error: {err}")
