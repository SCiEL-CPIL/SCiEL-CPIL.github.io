"""Search for a rank-minimal exact tensor decomposition of 3x5 * 5x6 multiplication."""
import time
import random
import numpy as np
from itertools import combinations

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 110.0


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
    """Vectorized verification via einsum. Returns True if decomposition is exact."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    if U.shape[1] == 0:
        return False
    nm, mp, pn = n * m, m * p, p * n
    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    M_all = np.einsum('ar,br,cr->cab', U, V, W)  # (nm, mp, pn)
    # Expected: T[i*m+j, j*p+k, k*n+i] == 1, else 0
    # Build expected by index mapping
    expected = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                expected[a, b, c] = 1.0
    return bool(np.max(np.abs(M_all - expected)) < 1e-9)


def _is_in_span(vec, matrix, tol=1e-10):
    """Check if vec is in the column span of matrix."""
    if matrix.shape[1] == 0:
        return bool(np.allclose(vec, 0, atol=tol))
    try:
        coeffs, residuals, rank, sv = np.linalg.lstsq(matrix, vec, rcond=tol)
        residual = np.linalg.norm(vec - matrix @ coeffs)
        return residual < tol * max(1.0, np.linalg.norm(vec))
    except np.linalg.LinAlgError:
        return False


def _try_reduce_rank(U, V, W, n, m, p, max_iterations=50):
    """Remove columns that are linearly dependent across all three factors."""
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
                if not _is_in_span(M_target, M_others, tol=1e-10):
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
    """All ways to partition [0..dim) into contiguous chunks of size 1 or 2."""
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
    """Snap values to nearest integer or half-integer if close."""
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


# ---------------------------------------------------------------------------
# Finite-field / minimal-perfect-hashing support search.
#
# Idea: the multiplication tensor is a 3-tensor T with T[i,j,k]=1. A rank-r
# decomposition means T = sum_{c=1}^r u_c (x) v_c (x) w_c. Equivalently we
# want a sparse set of rank-1 "support" terms. We build the full linear system
# over the reals: for each (i,j,k) the equation sum_c U[i,j,c] V[j,k,c]
# W[k,i,c] = 1 is bilinear, not linear. Instead we use a linearized relaxation:
# fix the U and V factors to a random sparse pattern and solve for W linearly,
# then check exactness. We iterate with greedy column selection.
# ---------------------------------------------------------------------------

def _finite_field_support_search(n, m, p, max_rank, rng, max_iters=2000):
    """Attempt to find a low-rank exact decomposition by greedy column selection
    on a randomized sparse factor pattern, with linear solve for the third factor.
    Returns (U,V,W) lists if a valid decomposition of rank <= max_rank is found,
    else None."""
    nm, mp, pn = n * m, m * p, p * n
    # Precompute index mapping for the target tensor entries
    target_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_entries.append((i * m + j, j * p + k, k * n + i))

    best = None
    for trial in range(40):
        if time.time() > _DEADLINE:
            return best
        r = max_rank
        # Random sparse U and V with entries from {-1,0,1}
        density_u = 0.25
        density_v = 0.25
        U = np.zeros((nm, r), dtype=float)
        V = np.zeros((mp, r), dtype=float)
        for c in range(r):
            for a in range(nm):
                if rng.random() < density_u:
                    U[a, c] = rng.choice([-1.0, 1.0])
            for b in range(mp):
                if rng.random() < density_v:
                    V[b, c] = rng.choice([-1.0, 1.0])
        # For each target (a,b,c) we need sum_r U[a,r] V[b,r] W[c,r] = 1
        # This is linear in W: for each c, sum_r (U[a,r] V[b,r]) W[c,r] = 1
        # But the same c-index (k*n+i) appears in multiple target entries, so we
        # get a system: for each output index c, we have up to m equations
        # (one per j). We solve a least-squares / exact system for W.
        # Build A[c, r] = sum over j of ... no: each target (a,b,c) gives
        #   sum_r U[a,r] V[b,r] W[c,r] = 1
        # Group by c: for fixed c, we have equations over r for W[c, :].
        # For a given c, the equations are: for each (a,b) with c = k*n+i,
        #   (U[a,:] * V[b,:]) @ W[c,:] = 1
        # So W[c,:] must satisfy a set of linear equations. Solve them.
        W = np.zeros((pn, r), dtype=float)
        feasible = True
        for c in range(pn):
            # collect all (a,b) target pairs mapping to this c
            eqs = []
            for (a, b, cc) in target_entries:
                if cc == c:
                    row = U[a, :] * V[b, :]
                    eqs.append(row)
            if not eqs:
                continue
            A = np.array(eqs, dtype=float)  # (m, r)
            rhs = np.ones(A.shape[0], dtype=float)
            # Solve A @ w = rhs (overdetermined or square)
            try:
                sol, residuals, rank, sv = np.linalg.lstsq(A, rhs, rcond=1e-10)
                resid = np.linalg.norm(A @ sol - rhs)
                if resid > 1e-7:
                    feasible = False
                    break
                W[c, :] = sol
            except np.linalg.LinAlgError:
                feasible = False
                break
        if not feasible:
            continue
        # Verify full exactness
        if _verify_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            best = (U.tolist(), V.tolist(), W.tolist())
            return best
    return best


def _greedy_rank_search(n, m, p, start_rank, end_rank, rng):
    """Try finite-field support search for ranks in [start_rank, end_rank]."""
    for r in range(start_rank, end_rank + 1):
        if time.time() > _DEADLINE:
            break
        res = _finite_field_support_search(n, m, p, r, rng)
        if res is not None:
            return res[0], res[1], res[2], r
    return None


def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    full = n * m * p

    best = _schoolbook(n, m, p)
    best_rank = full

    rng = random.Random(12345)

    def timed():
        return time.time() > _DEADLINE

    # --- Strategy 1: recursive Strassen partitioning ---
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
            max_iter = 30 if r <= best_rank + 5 else 15
            U_red, V_red, W_red = _try_reduce_rank(
                cand["U"], cand["V"], cand["W"], n, m, p, max_iterations=max_iter)
            if _verify_fast(U_red, V_red, W_red, n, m, p):
                new_rank = len(U_red[0])
                if new_rank < best_rank:
                    best = {"U": U_red, "V": V_red, "W": W_red}
                    best_rank = new_rank

    # --- Strategy 2: finite-field / support search for lower rank ---
    if not timed():
        # Search from a modest rank upward; cap to keep within budget.
        lo = max(30, best_rank - 20)
        hi = best_rank
        res = _greedy_rank_search(n, m, p, lo, hi, rng)
        if res is not None:
            Uf, Vf, Wf, rf = res
            # polish + reduce
            U_pol, V_pol, W_pol = _polish(Uf, Vf, Wf, n, m, p)
            if _verify_fast(U_pol, V_pol, W_pol, n, m, p):
                U_red, V_red, W_red = _try_reduce_rank(
                    U_pol, V_pol, W_pol, n, m, p, max_iterations=30)
                if _verify_fast(U_red, V_red, W_red, n, m, p):
                    new_rank = len(U_red[0])
                    if new_rank < best_rank:
                        best = {"U": U_red, "V": V_red, "W": W_red}
                        best_rank = new_rank
                else:
                    if rf < best_rank:
                        best = {"U": U_pol, "V": V_pol, "W": W_pol}
                        best_rank = rf

    # --- Strategy 3: polish the best result ---
    if not timed():
        U_pol, V_pol, W_pol = _polish(best["U"], best["V"], best["W"], n, m, p)
        if _verify_fast(U_pol, V_pol, W_pol, n, m, p):
            best = {"U": U_pol, "V": V_pol, "W": W_pol}
            U_red, V_red, W_red = _try_reduce_rank(
                best["U"], best["V"], best["W"], n, m, p, max_iterations=30)
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
