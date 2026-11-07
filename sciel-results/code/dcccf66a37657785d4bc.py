import time
from itertools import combinations

_P0 = time.time()
_DEADLINE = _P0 + 80.0  # safety margin under the 90s kill


def _can_add(prods, x):
    """True iff adding x keeps all subset products distinct.

    Adding x maps each existing product p to p*x; a collision occurs iff
    p*x is already in the set for some p. Equivalently x divides some
    product q in the set with q//x also in the set. We use the divisibility
    form to avoid constructing p*x for every product.
    """
    for q in prods:
        if q % x == 0 and (q // x) in prods:
            return False
    return True


def _try_add(prods, x):
    """Add x to the product set; return (success, new_prods)."""
    if not _can_add(prods, x):
        return False, prods
    new_prods = set(prods)
    for p in prods:
        new_prods.add(p * x)
    return True, new_prods


def _build_prods(selected):
    """Build the full subset-product set for a selected list; None if invalid."""
    prods = {1}
    for x in selected:
        if not _can_add(prods, x):
            return None
        for p in list(prods):
            prods.add(p * x)
    return prods


def _is_sqfree(n):
    """True iff n has no squared prime factor."""
    d = 2
    m = n
    while d * d <= m:
        if m % (d * d) == 0:
            return False
        d += 1
    return True


def construct():
    if time.time() > _DEADLINE:
        return [2]

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    nprimes = len(primes)

    # Non-square-free composites in 2..40: the only composites that can ever
    # be added once all primes are present (square-free composites collide
    # immediately because they are already subset products of the primes).
    candidates = []
    for c in range(4, 41):
        if c in primes:
            continue
        if not _is_sqfree(c):
            candidates.append(c)
    # candidates = [4,8,9,12,16,18,20,24,25,27,28,32,36,40] (14 items)

    # Base product set for all primes (2^12 = 4096 products).
    base_prods = {1}
    for x in primes:
        for p in list(base_prods):
            base_prods.add(p * x)

    best_count = nprimes
    best_sel = list(primes)
    best_prods = base_prods

    def _update(selected, prods):
        nonlocal best_count, best_sel, best_prods
        if len(selected) > best_count:
            best_count = len(selected)
            best_sel = list(selected)
            best_prods = prods

    # Fast path: greedy ascending (cheapest, often enough).
    sel = list(primes)
    prods = set(base_prods)
    for x in candidates:
        if len(sel) >= 18:
            break
        ok, prods2 = _try_add(prods, x)
        if ok:
            sel.append(x)
            prods = prods2
    _update(sel, prods)
    if best_count >= 18:
        return [int(x) for x in best_sel[:18]]

    # Exact exhaustive check of C(14,6)=3003 combos of 6 composites.
    # Each build is 2^18 products of small ints; total fits the budget.
    target = 6
    found = False
    for combo in combinations(candidates, target):
        if time.time() > _DEADLINE - 8:
            break
        prods = _build_prods(list(primes) + list(combo))
        if prods is not None:
            _update(list(primes) + list(combo), prods)
            found = True
            break
    if best_count >= 18:
        return [int(x) for x in best_sel[:18]]

    # Fallback: branch-and-bound over candidates (covers k=1..5 too).
    def _search(idx, prods, selected):
        nonlocal best_count
        if time.time() > _DEADLINE - 5:
            return
        if len(selected) > best_count:
            best_count = len(selected)
            best_sel = list(selected)
            best_prods = prods
        # Upper bound: current size + remaining candidates.
        if len(selected) + (len(candidates) - idx) <= best_count:
            return
        if idx >= len(candidates):
            return
        x = candidates[idx]
        # Include x if it fits.
        ok, prods2 = _try_add(prods, x)
        if ok:
            selected.append(x)
            _search(idx + 1, prods2, selected)
            selected.pop()
        # Exclude x.
        _search(idx + 1, prods, selected)

    if time.time() <= _DEADLINE - 5:
        _search(0, set(base_prods), list(primes))

    result = best_sel[:best_count]
    return [int(x) for x in result]
