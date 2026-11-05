import time
from fractions import Fraction

def _dfs_high(need, lo, D, best, deadline):
    """Find distinct denominators in [lo, D] summing reciprocally to need.
    Append the found tuple to best and return True if found."""
    if time.time() > deadline:
        return False
    if need <= 0:
        return False
    num, den = need.numerator, need.denominator
    d_min = (den + num - 1) // num  # ceil(1/need)
    if d_min < lo:
        d_min = lo
    if d_min > D:
        return False
    # quick: single term
    if num == 1 and lo <= den <= D:
        best.append((den,))
        return True
    for d in range(d_min, D + 1):
        nt = need - Fraction(1, d)
        if nt < 0:
            continue
        if nt == 0:
            best.append((d,))
            return True
        sub = []
        if _dfs_high(nt, d + 1, D, sub, deadline):
            best.append((d,) + sub[0])
            return True
        if time.time() > deadline:
            return False
    return False


def _low_sums(M, cap):
    sums = {Fraction(0, 1): [()]}
    for d in range(1, M + 1):
        inv = Fraction(1, d)
        additions = {}
        for s, lst in sums.items():
            ns = s + inv
            if ns > cap:
                continue
            additions.setdefault(ns, []).extend(lst)
        for s, lst in additions.items():
            sums.setdefault(s, []).extend(lst)
    return sums


def construct():
    deadline = time.time() + 50.0
    target = Fraction(5, 31)
    M = 45
    low = _low_sums(M, target)

    def H(n):
        return sum(Fraction(1, i) for i in range(1, n + 1))
    H_M = H(M)

    best = None
    for D in range(M + 1, 300):
        if time.time() > deadline:
            break
        max_high = H(D) - H_M
        for s, lst in low.items():
            need = target - s
            if need <= 0 or need > max_high:
                continue
            found = []
            if _dfs_high(need, M + 1, D, found, deadline):
                for tup in found:
                    cand = list(lst[0]) + list(tup)
                    if len(cand) <= 256 and len(set(cand)) == len(cand):
                        if best is None or max(cand) < max(best):
                            best = cand
                break
        if best is not None and max(best) <= D:
            break

    if best is None:
        r = target
        values = []
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values
    return best
