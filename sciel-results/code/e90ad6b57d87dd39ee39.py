"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

# --- Core data: primes and exponent vectors ---
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
NPRIMES = len(PRIMES)

# Exponent vectors for numbers 2..40
EXP_VECTORS = {}
for _i in range(2, 41):
    _vec = [0] * NPRIMES
    _n = _i
    for _j, _p in enumerate(PRIMES):
        while _n % _p == 0:
            _vec[_j] += 1
            _n //= _p
    EXP_VECTORS[_i] = tuple(_vec)

ZERO = tuple([0] * NPRIMES)

# Fixed known-good 15-set (8 primes >=11 + 7 composites in {2,3,5,7}-space)
KNOWN_GOOD = [11, 13, 17, 19, 23, 29, 31, 37, 2, 4, 16, 3, 9, 5, 25, 7]


def _valid_with_diff_set(diff_set, vec):
    """Return (valid, new_diff_set). Fast: O(|diff_set| * NPRIMES)."""
    if vec in diff_set:
        return False, None
    new_diff = set(diff_set)
    for d in diff_set:
        nd = tuple(d[i] + vec[i] for i in range(NPRIMES))
        if nd in new_diff:
            return False, None
        new_diff.add(nd)
    return True, new_diff


def _build_diff(nums):
    """Build the difference set for a list of numbers."""
    ds = {ZERO}
    for x in nums:
        v = EXP_VECTORS[x]
        new_ds = set(ds)
        for d in ds:
            nd = tuple(d[i] + v[i] for i in range(NPRIMES))
            new_ds.add(nd)
        ds = new_ds
    return ds


def _direct_verify(nums):
    """Exact check: all 2^k subset products distinct (incl. empty product 1)."""
    k = len(nums)
    products = set()
    for mask in range(1 << k):
        prod = 1
        for i in range(k):
            if mask & (1 << i):
                prod *= nums[i]
        if prod in products:
            return False
        products.add(prod)
    return True


def _constrain_score(x):
    v = EXP_VECTORS[x]
    num_distinct = sum(1 for e in v if e > 0)
    total_exp = sum(v)
    return num_distinct * 2 + total_exp


def _greedy(ordered, deadline):
    """Greedy build: add elements in `ordered` that stay valid."""
    ds = {ZERO}
    best = []
    for x in ordered:
        if time.time() > deadline:
            break
        v = EXP_VECTORS[x]
        ok, nds = _valid_with_diff_set(ds, v)
        if ok:
            best.append(x)
            ds = nds
    return best


def _swap1_pass(nums, deadline):
    """1-swap local search: replace one element with an excluded one if it grows."""
    current = list(nums)
    current_set = set(current)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for ri in range(len(current)):
            if time.time() > deadline:
                break
            removed = current[ri]
            remaining = [x for i, x in enumerate(current) if i != ri]
            ds_rem = _build_diff(remaining)
            for c in range(2, 41):
                if c in current_set:
                    continue
                ok, _ = _valid_with_diff_set(ds_rem, EXP_VECTORS[c])
                if ok:
                    # Only accept if it strictly grows the set (we removed 1, added 1)
                    # A same-size swap helps if it enables further growth; accept
                    # any valid swap that isn't a no-op, then re-loop.
                    new_set = remaining + [c]
                    current = new_set
                    current_set = set(current)
                    improved = True
                    break
            if improved:
                break
    return current


def _swap2_pass(nums, deadline):
    """2-swap polish: replace two elements with two excluded ones if it grows."""
    current = list(nums)
    current_set = set(current)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(current)
        for a in range(n):
            if time.time() > deadline:
                break
            for b in range(a + 1, n):
                if time.time() > deadline:
                    break
                remaining = [x for i, x in enumerate(current) if i != a and i != b]
                ds_rem = _build_diff(remaining)
                # Try to add two excluded elements that keep validity and grow size
                found = False
                for c in range(2, 41):
                    if found or c in current_set:
                        continue
                    ok1, nds1 = _valid_with_diff_set(ds_rem, EXP_VECTORS[c])
                    if not ok1:
                        continue
                    for d in range(2, 41):
                        if d == c or d in current_set:
                            continue
                        ok2, _ = _valid_with_diff_set(nds1, EXP_VECTORS[d])
                        if ok2:
                            new_set = remaining + [c, d]
                            current = new_set
                            current_set = set(current)
                            improved = True
                            found = True
                            break
                if found:
                    break
            if improved:
                break
    return current


def _backtrack(ordered, idx, diff_set, selected, best, deadline):
    if time.time() > deadline:
        return
    if len(selected) > len(best):
        best.append(len(selected))
        best[1:] = list(selected)
        if len(best) > 1:
            best.clear()
            best.append(len(selected))
            best.extend(selected)
    if idx >= len(ordered):
        return
    if len(selected) + (len(ordered) - idx) <= len(best):
        return
    num = ordered[idx]
    vec = EXP_VECTORS[num]
    valid, new_diff = _valid_with_diff_set(diff_set, vec)
    if valid:
        selected.append(num)
        _backtrack(ordered, idx + 1, new_diff, selected, best, deadline)
        selected.pop()
    _backtrack(ordered, idx + 1, diff_set, selected, best, deadline)


def _construct():
    t0 = time.time()
    deadline = t0 + 55.0
    all_nums = list(range(2, 41))
    primes_in = [x for x in all_nums if all(x % d != 0 for d in range(2, isqrt(x) + 1))]
    composites_in = [x for x in all_nums if x not in primes_in]

    best = list(KNOWN_GOOD)

    # --- Greedy with multiple orderings (fast, uses difference-set check) ---
    orderings = [
        primes_in + composites_in,
        sorted(all_nums, key=lambda x: _constrain_score(x)),
        sorted(all_nums, key=lambda x: -_constrain_score(x)),
        sorted(all_nums, reverse=True),
        sorted(all_nums),
        composites_in + primes_in,
    ]
    import random
    rng = random.Random(42)
    for _ in range(6):
        r = list(all_nums)
        rng.shuffle(r)
        orderings.append(r)

    for ordered in orderings:
        if time.time() > deadline - 20:
            break
        g = _greedy(ordered, deadline - 20)
        if len(g) > len(best):
            best = g

    # --- 1-swap local search ---
    if time.time() < deadline - 10:
        best = _swap1_pass(best, deadline - 10)

    # --- 2-swap polish pass (refinement) ---
    if time.time() < deadline - 6:
        best = _swap2_pass(best, deadline - 6)

    # --- Final 1-swap pass on the polished result ---
    if time.time() < deadline - 3:
        best = _swap1_pass(best, deadline - 3)

    # --- Optional short backtracking to try to beat current best size ---
    if time.time() < deadline - 12 and len(best) < 18:
        # Backtrack seeded with current best size to try to exceed it
        target_order = sorted(all_nums, key=lambda x: _constrain_score(x))
        bt_best = [len(best)] + list(best)
        _backtrack(target_order, 0, {ZERO}, [], bt_best, deadline - 12)
        if len(bt_best) - 1 > len(best):
            best = bt_best[1:]
            if time.time() < deadline - 3:
                best = _swap1_pass(best, deadline - 3)
                if time.time() < deadline - 2:
                    best = _swap2_pass(best, deadline - 2)

    # --- Final exact verification ---
    if not _direct_verify(best):
        best = list(KNOWN_GOOD)
        if not _direct_verify(best):
            # Extremely unlikely; fall back to primes
            best = list(primes_in)

    best = [int(x) for x in best]
    assert len(best) >= 12
    assert all(isinstance(x, int) for x in best)
    assert all(2 <= x <= 40 for x in best)
    assert len(set(best)) == len(best)
    assert _direct_verify(best)
    return best


def construct():
    return _construct()
# EVOLVE-BLOCK-END
