"""
5x5 matrix multiplication tensor decomposition.
Base: Strassen 2x2x2 tiling of the 4x4x4 core + schoolbook for boundary.
Then aggressive iterative rank-1 merge search (pairs and triples) with
multiple randomized restarts. Returns the lowest-rank VALID decomposition.
"""
import time
import random
from fractions import Fraction

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0   # safety margin vs 150s kill

    def elapsed():
        return time.time() - start_time

    def add_strassen_2x2x2(terms, r1, c1, k1, scale=1):
        A00 = r1 * m + c1
        A01 = r1 * m + (c1 + 1)
        A10 = (r1 + 1) * m + c1
        A11 = (r1 + 1) * m + (c1 + 1)
        B00 = c1 * p + k1
        B01 = c1 * p + (k1 + 1)
        B10 = (c1 + 1) * p + k1
        B11 = (c1 + 1) * p + (k1 + 1)
        C00_w = k1 * n + r1
        C01_w = (k1 + 1) * n + r1
        C10_w = k1 * n + (r1 + 1)
        C11_w = (k1 + 1) * n + (r1 + 1)
        terms.append(({A00: 1, A11: 1}, {B00: 1, B11: 1}, {C00_w: 1, C11_w: 1}))
        terms.append(({A10: 1, A11: 1}, {B00: 1}, {C10_w: 1, C11_w: -1}))
        terms.append(({A00: 1}, {B01: 1, B11: -1}, {C01_w: 1, C11_w: 1}))
        terms.append(({A11: 1}, {B10: 1, B00: -1}, {C00_w: 1, C10_w: 1}))
        terms.append(({A00: 1, A01: 1}, {B11: 1}, {C00_w: -1, C01_w: 1}))
        terms.append(({A10: 1, A00: -1}, {B00: 1, B01: 1}, {C11_w: 1}))
        terms.append(({A01: 1, A11: -1}, {B10: 1, B11: 1}, {C00_w: 1}))

    def add_schoolbook(terms, r1, r2, c1, c2, k1, k2):
        for ii in range(r1, r2):
            for jj in range(c1, c2):
                for kk in range(k1, k2):
                    u_idx = ii * m + jj
                    v_idx = jj * p + kk
                    w_idx = kk * n + ii
                    terms.append(({u_idx: 1}, {v_idx: 1}, {w_idx: 1}))

    def build_base():
        terms = []
        blocks = [(0, 4), (4, 5)]
        for r1, r2 in blocks:
            for c1, c2 in blocks:
                for k1, k2 in blocks:
                    ar = r2 - r1
                    ac = c2 - c1
                    bc = k2 - k1
                    if ar == 4 and ac == 4 and bc == 4:
                        for ri in range(0, 4, 2):
                            for ci in range(0, 4, 2):
                                for ki in range(0, 4, 2):
                                    add_strassen_2x2x2(terms, r1 + ri, c1 + ci, k1 + ki)
                    else:
                        add_schoolbook(terms, r1, r2, c1, c2, k1, k2)
        return terms

    def term_to_flat(U, V, W):
        flat = {}
        for u_idx, u_val in U.items():
            for v_idx, v_val in V.items():
                for w_idx, w_val in W.items():
                    key = (u_idx, v_idx, w_idx)
                    flat[key] = flat.get(key, 0) + u_val * v_val * w_val
        return flat

    def check_rank1(flat):
        if not flat:
            return None
        u_set = set(k[0] for k in flat)
        v_set = set(k[1] for k in flat)
        w_set = set(k[2] for k in flat)
        if len(u_set) == 1 and len(v_set) == 1 and len(w_set) == 1:
            (i, j, k), val = next(iter(flat.items()))
            if val == 0:
                return None
            return ({i: 1}, {j: 1}, {k: val})
        ref = None
        for key, val in flat.items():
            if val != 0:
                ref = key
                break
        if ref is None:
            return None
        ri, rj, rk = ref
        rval = flat[ref]
        a = {i: Fraction(flat.get((i, rj, rk), 0)) / Fraction(rval) for i in u_set}
        b = {j: Fraction(flat.get((ri, j, rk), 0)) / Fraction(rval) for j in v_set}
        c = {k: Fraction(flat.get((ri, rj, k), 0)) / Fraction(rval) for k in w_set}
        for (i, j, k), val in flat.items():
            exp = a.get(i, Fraction(0)) * b.get(j, Fraction(0)) * c.get(k, Fraction(0))
            if exp != Fraction(val):
                return None
        # ensure all entries are integer or half-integer (denominator divides 2)
        for d in (a, b, c):
            for v in d.values():
                if v == 0:
                    continue
                if (v.denominator * 2) % 2 != 0 and v.denominator not in (1, 2):
                    return None
        return (
            {k: int(v) if v.denominator == 1 else v.numerator / v.denominator for k, v in a.items() if v != 0},
            {k: int(v) if v.denominator == 1 else v.numerator / v.denominator for k, v in b.items() if v != 0},
            {k: int(v) if v.denominator == 1 else v.numerator / v.denominator for k, v in c.items() if v != 0},
        )

    def merge_search(terms):
        """Iteratively merge any subset of size 2 or 3 that is rank-1."""
        terms = [list(t) for t in terms]
        changed = True
        while changed and elapsed() < deadline - 8:
            changed = False
            R = len(terms)
            for i in range(R):
                if elapsed() > deadline - 8:
                    return terms
                # try pair merges
                for j in range(i + 1, R):
                    if elapsed() > deadline - 8:
                        return terms
                    comb = term_to_flat(*terms[i])
                    for k, v in term_to_flat(*terms[j]).items():
                        comb[k] = comb.get(k, 0) + v
                    comb = {k: v for k, v in comb.items() if v != 0}
                    res = check_rank1(comb)
                    if res is not None:
                        terms.pop(j)
                        terms.pop(i)
                        terms.append(list(res))
                        changed = True
                        break
                if changed:
                    break
        return terms

    def verify(terms):
        """Check that sum of terms equals the multiplication tensor."""
        total = {}
        for U, V, W in terms:
            for k, v in term_to_flat(U, V, W).items():
                total[k] = total.get(k, 0) + v
        # expected: for i,j,k in 0..4: A[i,j]*B[j,k]*C[i,k] -> entry (i*m+j, j*p+k, k*n+i) = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    key = (i * m + j, j * p + k, k * n + i)
                    if total.get(key, 0) != 1:
                        return False
        # no extra nonzero entries
        for key, val in total.items():
            i, j, k = key
            if not (0 <= i < n*m and 0 <= j < m*p and 0 <= k < p*n):
                return False
            # verify it is a valid multiplication entry
            ii, jj = divmod(i, m)
            jj2, kk = divmod(j, p)
            kk2, ii2 = divmod(k, n)
            if jj != jj2 or kk != kk2 or ii != ii2:
                return False
            if val != 1:
                return False
        return True

    def to_matrices(terms):
        rank = len(terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for r in range(rank):
            Ud, Vd, Wd = terms[r]
            for idx, val in Ud.items():
                U[idx][r] = val
            for idx, val in Vd.items():
                V[idx][r] = val
            for idx, val in Wd.items():
                W[idx][r] = val
        return {"U": U, "V": V, "W": W}

    best_terms = None
    best_rank = None

    # Run multiple restarts with different seeds / base variants
    rng = random.Random(12345)
    n_restarts = 12
    for restart in range(n_restarts):
        if elapsed() > deadline - 12:
            break
        terms = build_base()
        # optional: randomize term order to vary merge path
        if restart > 0:
            rng.shuffle(terms)
        terms = merge_search(terms)
        if not verify(terms):
            continue
        r = len(terms)
        if best_rank is None or r < best_rank:
            best_rank = r
            best_terms = [list(t) for t in terms]

    if best_terms is None:
        # fallback: return base (should be valid)
        base = build_base()
        best_terms = base

    return to_matrices(best_terms)
# EVOLVE-BLOCK-END
