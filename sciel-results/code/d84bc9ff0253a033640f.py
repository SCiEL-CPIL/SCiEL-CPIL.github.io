import time

DEADLINE = time.time() + 78.0   # safety margin well under the 90 s kill
SEED_VALUE = 21.0               # [1]*12 -> square has 21 nonzero terms


def _sq_nonzero(coeffs):
    """Return the number of nonzero terms in the exact square of the
    coefficient list (ascending degree)."""
    n = len(coeffs)
    sq = [0] * (2 * n - 1)
    for i in range(n):
        ci = coeffs[i]
        if ci:
            for j in range(i, n):
                cj = coeffs[j]
                if cj:
                    sq[i + j] += ci * cj
    return sum(1 for v in sq if v)


def _exponents_from_3(a, b, c):
    # square of x^a + x^b + x^c: {2a,2b,2c, a+b, a+c, b+c}
    return {2 * a, 2 * b, 2 * c, a + b, a + c, b + c}


def construct():
    best = [1] * 12  # seed: 21 nonzero terms in the square
    best_terms = _sq_nonzero(best)  # 21

    # --- 1) exhaustive search over 3-term monomial sums x^a+x^b+x^c ---
    # Square has at most 6 distinct exponents; look for a collision (5 terms).
    D = 40
    for a in range(0, D + 1):
        for b in range(a + 1, D + 1):
            for c in range(b + 1, D + 1):
                if a + c > D:
                    break
                exps = _exponents_from_3(a, b, c)
                if len(exps) < best_terms:
                    # build the coefficient list and confirm
                    coeffs = [0] * (a + c + 1)
                    coeffs[a] = 1
                    coeffs[b] = 1
                    coeffs[c] = 1
                    t = _sq_nonzero(coeffs)
                    if t < best_terms:
                        best_terms = t
                        best = coeffs
                        if best_terms <= 4:
                            return best
                if time.time() > DEADLINE:
                    return best

    # --- 2) exhaustive search over 4-term monomial sums (coarser) ---
    # Square has at most 10 distinct exponents; collisions may drop it below best.
    D = 40
    step = 1
    for a in range(0, D + 1, step):
        for b in range(a + 1, D + 1, step):
            for c in range(b + 1, D + 1, step):
                for d in range(c + 1, D + 1, step):
                    if a + d > D:
                        break
                    # exponents of (x^a+x^b+x^c+x^d)^2
                    ex = {2 * a, 2 * b, 2 * c, 2 * d,
                          a + b, a + c, a + d,
                          b + c, b + d,
                          c + d}
                    if len(ex) < best_terms:
                        coeffs = [0] * (a + d + 1)
                        coeffs[a] = 1
                        coeffs[b] = 1
                        coeffs[c] = 1
                        coeffs[d] = 1
                        t = _sq_nonzero(coeffs)
                        if t < best_terms:
                            best_terms = t
                            best = coeffs
                            if best_terms <= 4:
                                return best
                if time.time() > DEADLINE:
                    return best

    # --- 3) safety net: guaranteed Bhargava 3-term construction (6 terms) ---
    if best_terms > 6:
        coeffs = [0] * 11
        coeffs[0] = 1
        coeffs[2] = 1
        coeffs[5] = 1
        best = coeffs
        best_terms = _sq_nonzero(best)

    # --- final validation: trim trailing zeros, enforce constraints ---
    while len(best) > 1 and best[-1] == 0:
        best.pop()
    if best[-1] == 0:
        return [1] * 12
    if len(best) > 41:
        return [1] * 12
    for v in best:
        if not (-64 <= v <= 64):
            return [1] * 12
    nz = sum(1 for v in best if v)
    if nz != 12:
        return [1] * 12
    return best


if __name__ == "__main__":
    c = construct()
    print("terms in square:", _sq_nonzero(c))
    print("poly length:", len(c), "nonzero:", sum(1 for v in c if v))
    print(c)
