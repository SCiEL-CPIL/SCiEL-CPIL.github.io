"""Search for a rank-minimal exact tensor decomposition of 3x5 * 5x6 multiplication."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 140.0


def _strassen_2x2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication."""
    terms = []
    terms.append(({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}))
    terms.append(({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(0, 1): 1, (1, 1): -1}))
    terms.append(({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}))
    terms.append(({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}))
    terms.append(({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 0): -1, (1, 0): 1}))
    terms.append(({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}))
    terms.append(({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1}))
    return terms


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _build_recursive_strassen(n, m, p, i_groups, j_groups, k_groups):
    """Build a decomposition by splitting each dimension into groups.
    Any (2,2,2) block uses the 7-term Strassen trick; others use schoolbook."""
    nm, mp, pn = n * m, m * p, p * n
    strassen_terms = _strassen_2x2x2_terms()

    total_rank = 0
    block_info = []
    for ig in i_groups:
        for jg in j_groups:
            for kg in k_groups:
                si, sj, sk = len(ig), len(jg), len(kg)
                r = 7 if (si == 2 and sj == 2 and sk == 2) else si * sj * sk
                total_rank += r
                block_info.append((ig, jg, kg, r))

    U = [[0] * total_rank for _ in range(nm)]
    V = [[0] * total_rank for _ in range(mp)]
    W = [[0] * total_rank for _ in range(pn)]

    col = 0
    for ig, jg, kg, r in block_info:
        si, sj, sk = len(ig), len(jg), len(kg)
        if si == 2 and sj == 2 and sk == 2:
            for (u_pat, v_pat, w_pat) in strassen_terms:
                for (li, lj), val_u in u_pat.items():
                    gi, gj = ig[li], jg[lj]
                    U[gi * m + gj][col] = val_u
                for (lj, lk), val_v in v_pat.items():
                    gj, gk = jg[lj], kg[lk]
                    V[gj * p + gk][col] = val_v
                for (lk, li), val_w in w_pat.items():
                    gk, gi = kg[lk], ig[li]
                    W[gk * n + gi][col] = val_w
                col += 1
        else:
            for li in range(si):
                for lj in range(sj):
                    for lk in range(sk):
                        gi, gj, gk = ig[li], jg[lj], kg[lk]
                        U[gi * m + gj][col] = 1
                        V[gj * p + gk][col] = 1
                        W[gk * n + gi][col] = 1
                        col += 1

    return {"U": U, "V": V, "W": W}


def _verify_fast(U, V, W, n, m, p):
    """Fast verification using numpy. Returns True if decomposition is exact."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3, optimize=True)
    
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True


def _try_reduce_rank(U, V, W, n, m, p, max_iterations=200):
    """Try to reduce rank by removing columns that are linearly dependent
    across all three factor matrices simultaneously."""
    U = np.array(U, dtype=float).copy()
    V = np.array(V, dtype=float).copy()
    W = np.array(W, dtype=float).copy()
    r = U.shape[1]

    for _ in range(max_iterations):
        if time.time() > _DEADLINE:
            break
        reduced = False
        for target in range(r):
            if time.time() > _DEADLINE:
                break
            others = [i for i in range(r) if i != target]
            if len(others) == 0:
                continue

            ok = True
            for M in (U, V, W):
                M_others = M[:, others]
                M_target = M[:, target]
                r1 = np.linalg.matrix_rank(M_others, tol=1e-10)
                r2 = np.linalg.matrix_rank(np.column_stack([M_others, M_target]), tol=1e-10)
                if r2 > r1:
                    ok = False
                    break
            if not ok:
                continue

            U = np.delete(U, target, axis=1)
            V = np.delete(V, target, axis=1)
            W = np.delete(W, target, axis=1)
            r -= 1
            reduced = True
            break

        if not reduced:
            break

    return U.tolist(), V.tolist(), W.tolist()


def _generate_partitions(dim):
    """Generate all ways to partition [0..dim) into contiguous chunks of size 1 or 2."""
    results = []
    def rec(start, cur):
        if start == dim:
            results.append(list(cur))
            return
        if start + 1 <= dim:
            cur.append([start])
            rec(start + 1, cur)
            cur.pop()
        if start + 2 <= dim:
            cur.append([start, start + 1])
            rec(start + 2, cur)
            cur.pop()
    rec(0, [])
    return results


def _expected_rank(ig, jg, kg):
    tot = 0
    for a in ig:
        for b in jg:
            for c in kg:
                tot += 7 if (len(a) == 2 and len(b) == 2 and len(c) == 2) \
                    else len(a) * len(b) * len(c)
    return tot


def _polish(U, V, W, n, m, p):
    """Final polish: snap values to nearest integer or half-integer if close."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for M in (U, V, W):
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                snapped = round(v * 2) / 2
                if abs(v - snapped) < 1e-8:
                    M[i, j] = snapped
                else:
                    snapped_i = round(v)
                    if abs(v - snapped_i) < 1e-8:
                        M[i, j] = snapped_i
    
    return U.tolist(), V.tolist(), W.tolist()


def _generate_all_set_partitions(size):
    """Generate all set partitions of {0, 1, ..., size-1}."""
    if size == 0:
        yield []
        return
    if size == 1:
        yield [[0]]
        return
    
    base_partitions = list(_generate_all_set_partitions(size - 1))
    
    seen = set()
    for part in base_partitions:
        # Option 1: (size-1) starts a new block
        new_part = part + [[size - 1]]
        key = tuple(tuple(sorted(b)) for b in sorted(new_part, key=lambda x: min(x)))
        if key not in seen:
            seen.add(key)
            yield new_part
        
        # Option 2: (size-1) joins each existing block
        for idx in range(len(part)):
            new_part = [b[:] for b in part]
            new_part[idx].append(size - 1)
            new_part = [sorted(b) for b in new_part]
            new_part.sort(key=lambda x: min(x))
            key = tuple(tuple(b) for b in new_part)
            if key not in seen:
                seen.add(key)
                yield new_part


def _build_set_partition_decomp(n, m, p, i_groups, j_groups, k_groups):
    """Build a decomposition using arbitrary set partitions (not just contiguous).
    Uses Strassen 7-term for any 2x2x2 block, schoolbook otherwise."""
    nm, mp, pn = n * m, m * p, p * n
    strassen_terms = _strassen_2x2x2_terms()

    total_rank = 0
    block_info = []
    for ig in i_groups:
        for jg in j_groups:
            for kg in k_groups:
                si, sj, sk = len(ig), len(jg), len(kg)
                r = 7 if (si == 2 and sj == 2 and sk == 2) else si * sj * sk
                total_rank += r
                block_info.append((ig, jg, kg, r))

    U = [[0] * total_rank for _ in range(nm)]
    V = [[0] * total_rank for _ in range(mp)]
    W = [[0] * total_rank for _ in range(pn)]

    col = 0
    for ig, jg, kg, r in block_info:
        si, sj, sk = len(ig), len(jg), len(kg)
        if si == 2 and sj == 2 and sk == 2:
            for (u_pat, v_pat, w_pat) in strassen_terms:
                for (li, lj), val_u in u_pat.items():
                    gi, gj = ig[li], jg[lj]
                    U[gi * m + gj][col] = val_u
                for (lj, lk), val_v in v_pat.items():
                    gj, gk = jg[lj], kg[lk]
                    V[gj * p + gk][col] = val_v
                for (lk, li), val_w in w_pat.items():
                    gk, gi = kg[lk], ig[li]
                    W[gk * n + gi][col] = val_w
                col += 1
        else:
            for li in range(si):
                for lj in range(sj):
                    for lk in range(sk):
                        gi, gj, gk = ig[li], jg[lj], kg[lk]
                        U[gi * m + gj][col] = 1
                        V[gj * p + gk][col] = 1
                        W[gk * n + gi][col] = 1
                        col += 1

    return {"U": U, "V": V, "W": W}


def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    full = n * m * p

    best = _schoolbook(n, m, p)
    best_rank = full

    def timed():
        return time.time() > _DEADLINE

    # Strategy 1: Contiguous partitions (original approach)
    ip = _generate_partitions(n)
    jp = _generate_partitions(m)
    kp = _generate_partitions(p)

    candidates = []
    for ig in ip:
        for jg in jp:
            for kg in kp:
                candidates.append((ig, jg, kg))

    candidates.sort(key=lambda t: _expected_rank(t[0], t[1], t[2]))

    for (ig, jg, kg) in candidates:
        if timed():
            break
        cand = _build_recursive_strassen(n, m, p, ig, jg, kg)
        if _verify_fast(cand["U"], cand["V"], cand["W"], n, m, p):
            r = len(cand["U"][0])
            if r < best_rank:
                best = cand
                best_rank = r
            U_red, V_red, W_red = _try_reduce_rank(cand["U"], cand["V"], cand["W"], n, m, p, max_iterations=50)
            if _verify_fast(U_red, V_red, W_red, n, m, p):
                new_rank = len(U_red[0])
                if new_rank < best_rank:
                    best = {"U": U_red, "V": V_red, "W": W_red}
                    best_rank = new_rank

    # Strategy 2: Set partitions (non-contiguous) - may find better Strassen alignments
    if not timed():
        i_sets = list(_generate_all_set_partitions(n))
        j_sets = list(_generate_all_set_partitions(m))
        k_sets = list(_generate_all_set_partitions(p))
        
        set_candidates = []
        for ig in i_sets:
            for jg in j_sets:
                for kg in k_sets:
                    set_candidates.append((ig, jg, kg))
        
        set_candidates.sort(key=lambda t: _expected_rank(t[0], t[1], t[2]))
        
        for (ig, jg, kg) in set_candidates:
            if timed():
                break
            er = _expected_rank(ig, jg, kg)
            if er >= best_rank:
                continue
            cand = _build_set_partition_decomp(n, m, p, ig, jg, kg)
            if _verify_fast(cand["U"], cand["V"], cand["W"], n, m, p):
                r = len(cand["U"][0])
                if r < best_rank:
                    best = cand
                    best_rank = r
                U_red, V_red, W_red = _try_reduce_rank(cand["U"], cand["V"], cand["W"], n, m, p, max_iterations=50)
                if _verify_fast(U_red, V_red, W_red, n, m, p):
                    new_rank = len(U_red[0])
                    if new_rank < best_rank:
                        best = {"U": U_red, "V": V_red, "W": W_red}
                        best_rank = new_rank

    # Strategy 3: Polish the best result
    if not timed():
        U_pol, V_pol, W_pol = _polish(best["U"], best["V"], best["W"], n, m, p)
        if _verify_fast(U_pol, V_pol, W_pol, n, m, p):
            best = {"U": U_pol, "V": V_pol, "W": W_pol}
            U_red, V_red, W_red = _try_reduce_rank(best["U"], best["V"], best["W"], n, m, p, max_iterations=50)
            if _verify_fast(U_red, V_red, W_red, n, m, p):
                new_rank = len(U_red[0])
                if new_rank < best_rank:
                    best = {"U": U_red, "V": V_red, "W": W_red}
                    best_rank = new_rank

    # Final verification
    if not _verify_fast(best["U"], best["V"], best["W"], n, m, p):
        best = _schoolbook(n, m, p)

    return best
# EVOLVE-BLOCK-END
