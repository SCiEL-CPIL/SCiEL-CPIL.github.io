"""Combined: structured binomial-product generation + fast incremental local search."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 84.0  # safety margin before 90s kill
    rng = random.Random(12345)

    def ok():
        return time.time() < deadline

    def square_nonzero(exponents, coefficients):
        """Nonzero count of P^2 for sparse P = sum c_i x^{e_i}."""
        sq = {}
        n = len(exponents)
        for i in range(n):
            ci = coefficients[i]
            ei = exponents[i]
            for j in range(i, n):
                e = ei + exponents[j]
                if i == j:
                    sq[e] = sq.get(e, 0) + ci * ci
                else:
                    sq[e] = sq.get(e, 0) + 2 * ci * coefficients[j]
        return sum(1 for v in sq.values() if v != 0)

    def make_poly(exponents, coefficients):
        max_deg = max(exponents) if exponents else 0
        coeffs = [0] * (max_deg + 1)
        for e, c in zip(exponents, coefficients):
            coeffs[e] += c
        while coeffs and coeffs[-1] == 0:
            coeffs.pop()
        return coeffs

    def is_valid(coeffs):
        if len(coeffs) > 41:
            return False
        if not coeffs or coeffs[-1] == 0:
            return False
        if sum(1 for c in coeffs if c != 0) != 12:
            return False
        for c in coeffs:
            if c < -64 or c > 64:
                return False
        return True

    def expand_product(factors):
        """Expand list of (a, b, e) binomials (a + b x^e) -> (exponents, coeffs)."""
        terms = [(0, 1)]  # (exponent, coeff)
        for a, b, e in factors:
            new = []
            for exp, c in terms:
                if a != 0:
                    new.append((exp, c * a))
                if b != 0:
                    new.append((exp + e, c * b))
            terms = new
        # merge by exponent
        merged = {}
        for exp, c in terms:
            merged[exp] = merged.get(exp, 0) + c
        exponents = []
        coeffs = []
        for exp in sorted(merged):
            if merged[exp] != 0:
                exponents.append(exp)
                coeffs.append(merged[exp])
        return exponents, coeffs

    def valid_state(exponents, coefficients):
        if len(exponents) != 12:
            return False
        if len(set(exponents)) != 12:
            return False
        if max(exponents) > 40:
            return False
        for c in coefficients:
            if c == 0 or c < -64 or c > 64:
                return False
        return True

    # ---------- incremental local search ----------
    def local_search(exponents, coefficients, max_iters=3000):
        exponents = list(exponents)
        coefficients = list(coefficients)
        n = 12
        # build square map
        sq = {}
        for i in range(n):
            ci = coefficients[i]
            ei = exponents[i]
            for j in range(i, n):
                e = ei + exponents[j]
                if i == j:
                    sq[e] = sq.get(e, 0) + ci * ci
                else:
                    sq[e] = sq.get(e, 0) + 2 * ci * coefficients[j]
        nz = sum(1 for v in sq.values() if v != 0)
        best_nz = nz

        def apply_move(idx, new_e, new_c):
            """Return (delta_nz, new_nz) for moving term idx to (new_e, new_c)."""
            nonlocal nz
            # remove term idx contribution
            ei0 = exponents[idx]
            ci0 = coefficients[idx]
            for j in range(n):
                if j == idx:
                    continue
                e = ei0 + exponents[j]
                sq[e] -= 2 * ci0 * coefficients[j]
            e = 2 * ei0
            sq[e] -= ci0 * ci0
            # add new term
            for j in range(n):
                if j == idx:
                    continue
                e = new_e + exponents[j]
                sq[e] = sq.get(e, 0) + 2 * new_c * coefficients[j]
            e = 2 * new_e
            sq[e] = sq.get(e, 0) + new_c * new_c
            # recompute nz
            nnz = sum(1 for v in sq.values() if v != 0)
            return nnz - nz, nnz

        improved = True
        iters = 0
        while improved and iters < max_iters and ok():
            improved = False
            iters += 1
            for i in range(n):
                if not ok():
                    break
                ei0 = exponents[i]
                ci0 = coefficients[i]
                others = exponents[:i] + exponents[i + 1:]
                best_delta = 0
                best_e = ei0
                best_c = ci0
                # exponent candidates
                for d in (-1, 1, -2, 2, -3, 3, -5, 5, -8, 8, -13, 13):
                    ne = ei0 + d
                    if ne < 0 or ne > 40 or ne in others:
                        continue
                    delta, _ = apply_move(i, ne, ci0)
                    if delta < best_delta:
                        best_delta = delta
                        best_e = ne
                        best_c = ci0
                # coefficient candidates
                for nc in (-2, -1, 1, 2, -3, 3, -4, 4, -8, 8, -16, 16, -32, 32, -64, 64):
                    if nc == ci0:
                        continue
                    delta, _ = apply_move(i, ei0, nc)
                    if delta < best_delta:
                        best_delta = delta
                        best_e = ei0
                        best_c = nc
                if best_delta < 0:
                    # commit best move
                    _, nz = apply_move(i, best_e, best_c)
                    exponents[i] = best_e
                    coefficients[i] = best_c
                    improved = True
                    if nz < best_nz:
                        best_nz = nz
        return exponents, coefficients, best_nz

    best_score = None
    best_exponents = None
    best_coefficients = None

    def consider(exponents, coefficients):
        nonlocal best_score, best_exponents, best_coefficients
        if not valid_state(exponents, coefficients):
            return
        nz = square_nonzero(exponents, coefficients)
        if best_score is None or nz < best_score:
            best_score = nz
            best_exponents = list(exponents)
            best_coefficients = list(coefficients)

    # ---------- generation phase ----------
    # 1. Sign patterns on consecutive exponents
    for mask in range(4096):
        signs = [1 if (mask >> i) & 1 else -1 for i in range(12)]
        exponents = list(range(12))
        consider(exponents, signs)

    # 2. Products of binomials: choose 6 exponents, each factor (a + b x^e)
    #    with a,b in {1,-1} -> 2^6=64 terms... too many. Instead use factors
    #    that yield exactly 12 nonzero terms after merging.
    #    Strategy: pick 6 "base" exponents e_k; factors (1 + s_k x^{e_k}).
    #    The product has 2^6 = 64 terms generally; we want merges to reduce to 12.
    #    Hard to control; instead use FEWER factors with larger powers.
    #    e.g. (1 + x^a)(1 + x^b)(1 + x^c)(1 + x^d)(1 + x^e)(1 + x^f) -> 64 terms.
    #    To get 12 terms, use factors like (1 + x)(1 + x^2)(1 + x^4)(1 + x^8) = 16 terms (too many)
    #    Actually (1+x)(1+x^2)(1+x^4)(1+x^8)(1+x^16) = 32 terms.
    #    12 terms is awkward for pure binomial products (powers of 2).
    #    So use a MIX: some binomial factors + a linear tail to hit exactly 12.
    #    Simpler: generate candidate exponent sets and let local search fix counts.

    # 3. Structured exponent sets with sign patterns
    exponent_sets = []
    for start in range(0, 8):
        exponent_sets.append(list(range(start, start + 12)))
    # gaps
    for start in range(0, 6):
        for gp in range(2, 12):
            exps = list(range(start, start + gp)) + list(range(start + gp + 1, start + gp + 13))
            if len(exps) == 12 and max(exps) <= 40:
                exponent_sets.append(exps)
    # arithmetic progressions
    for step in (2, 3, 4):
        for start in range(0, 4):
            exps = [start + i * step for i in range(12)]
            if max(exps) <= 40:
                exponent_sets.append(exps)
    # symmetric / palindromic
    exponent_sets.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    exponent_sets.append([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
    exponent_sets.append([0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16])

    # For each exponent set, try a handful of sign patterns + local search
    sign_patterns = []
    for mask in range(0, 4096, 64):  # 64 patterns per set
        sign_patterns.append([1 if (mask >> i) & 1 else -1 for i in range(12)])
    # add some targeted patterns
    sign_patterns.append([1] * 12)
    sign_patterns.append([-1] * 12)
    sign_patterns.append([1 if i % 2 == 0 else -1 for i in range(12)])
    sign_patterns.append([1 if i % 3 == 0 else -1 for i in range(12)])
    sign_patterns.append([1] * 6 + [-1] * 6)

    for exps in exponent_sets:
        if not ok():
            break
        if len(exps) != 12 or len(set(exps)) != 12 or max(exps) > 40:
            continue
        for signs in sign_patterns:
            if not ok():
                break
            consider(exps, signs)
            # local search from this candidate
            le, lc, lnz = local_search(exps, signs, max_iters=200)
            if lnz < (best_score if best_score is not None else 10**9):
                best_score = lnz
                best_exponents = le
                best_coefficients = lc

    # 4. Random restarts + local search
    restarts = 0
    while ok() and restarts < 40:
        restarts += 1
        # random exponent set
        exps = sorted(rng.sample(range(0, 41), 12))
        signs = [rng.choice((-1, 1)) for _ in range(12)]
        le, lc, lnz = local_search(exps, signs, max_iters=400)
        if lnz < (best_score if best_score is not None else 10**9):
            best_score = lnz
            best_exponents = le
            best_coefficients = lc

    # 5. Final deep local search from current best
    if best_exponents is not None and ok():
        le, lc, lnz = local_search(best_exponents, best_coefficients, max_iters=4000)
        if lnz < best_score:
            best_score = lnz
            best_exponents = le
            best_coefficients = lc

    # Build and validate result
    if best_exponents is not None:
        poly = make_poly(best_exponents, best_coefficients)
        if is_valid(poly):
            return poly

    # Fallback: seed
    return [1] * 12
# EVOLVE-BLOCK-END
