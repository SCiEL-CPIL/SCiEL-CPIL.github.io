"""Combined program: Prime-exponent vector greedy with many random restarts and local search."""
# EVOLVE-BLOCK-START
import time
import random

# Precompute prime exponent vectors for 2..40
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
NUM_PRIMES = len(PRIMES)
ZERO = tuple([0] * NUM_PRIMES)

# vectors[n] = tuple of exponents of each prime in factorization of n
_VECTORS = {}
for _n in range(2, 41):
    _v = [0] * NUM_PRIMES
    _t = _n
    for _i, _p in enumerate(PRIMES):
        while _t % _p == 0:
            _v[_i] += 1
            _t //= _p
    _VECTORS[_n] = tuple(_v)

# Precompute norms (sum of exponents) for ordering heuristics
_NORMS = {n: sum(v) for n, v in _VECTORS.items()}
_NORMS2 = {n: sum(x * x for x in v) for n, v in _VECTORS.items()}


def _add_vector(current_sums, v_new):
    """Try to add v_new to current_sums (set of exponent tuples).
    Returns (success, updated_sums). Uses the fact that new sums are old_sums + v_new.
    We check collisions by seeing if any old_sum + v_new is already in current_sums.
    """
    # Check: for each s in current_sums, is s + v_new in current_sums?
    # If yes, collision. Otherwise, add all s+v_new to the set.
    # Optimization: since all exponents are non-negative, s + v_new > s component-wise
    # (strictly in at least one component where v_new has a positive entry).
    # So s + v_new can only collide with an element t in current_sums where t >= s component-wise.
    # But simplest correct check: iterate.
    for s in current_sums:
        ns = tuple(s[i] + v_new[i] for i in range(NUM_PRIMES))
        if ns in current_sums:
            return False, current_sums
    # No collision: add all new sums
    for s in current_sums:
        ns = tuple(s[i] + v_new[i] for i in range(NUM_PRIMES))
        current_sums.add(ns)
    return True, current_sums


def _greedy_build(order, max_size=18, deadline=None):
    """Greedily add elements in given order, maintaining distinct subset products."""
    current_set = []
    current_sums = {ZERO}
    for n in order:
        if len(current_set) >= max_size:
            break
        if deadline and time.time() > deadline:
            break
        v = _VECTORS[n]
        ok, _ = _add_vector(current_sums, v)
        if ok:
            current_set.append(n)
    return current_set


def _local_search_improve(S, all_candidates, max_iter=40, deadline=None):
    """
    Local search: try removing one element and adding a different one to increase size.
    """
    current = list(S)
    best = list(current)
    best_len = len(current)
    all_set = set(all_candidates)

    for _ in range(max_iter):
        if deadline and time.time() > deadline:
            break
        improved = False
        for i in range(len(current)):
            if deadline and time.time() > deadline:
                break
            trial = current[:i] + current[i + 1:]
            # Rebuild sums for trial
            trial_sums = {ZERO}
            for t in trial:
                _add_vector(trial_sums, _VECTORS[t])
            # Try to add any candidate not in trial
            for cand in all_candidates:
                if cand in trial:
                    continue
                v = _VECTORS[cand]
                ok, _ = _add_vector(trial_sums, v)
                if ok:
                    new_trial = trial + [cand]
                    if len(new_trial) > best_len:
                        best = new_trial
                        best_len = len(new_trial)
                        current = new_trial
                        improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return best


def _build_max_set():
    """
    Multi-strategy search for max set.
    """
    all_candidates = list(range(2, 41))
    primes = [p for p in range(2, 41) if _NORMS[p] == 1]
    composites = [c for c in range(2, 41) if _NORMS[c] > 1]

    t0 = time.time()
    deadline = t0 + 75  # Safety margin (90s kill, leave 15s)

    best_set = []
    best_len = 0

    def update_best(S):
        nonlocal best_set, best_len
        if len(S) > best_len:
            best_set = list(S)
            best_len = len(S)

    # Strategy 1: Deterministic orders
    orders = [
        ('primes_asc', primes + sorted(composites, key=lambda x: x)),
        ('primes_desc', primes + sorted(composites, key=lambda x: -x)),
        ('norm_asc', sorted(all_candidates, key=lambda x: _NORMS[x])),
        ('norm_desc', sorted(all_candidates, key=lambda x: -_NORMS[x])),
        ('norm2_desc', sorted(all_candidates, key=lambda x: -_NORMS2[x])),
        ('value_desc', sorted(all_candidates, key=lambda x: -x)),
        ('value_asc', sorted(all_candidates, key=lambda x: x)),
    ]

    for name, order in orders:
        if time.time() > deadline:
            break
        S = _greedy_build(order, max_size=18, deadline=deadline)
        update_best(S)

    # Strategy 2: Try removing individual primes from full prime set, then greedy
    for i in range(len(primes)):
        if time.time() > deadline - 5:
            break
        psubset = primes[:i] + primes[i + 1:]
        for comp_order in [sorted(composites, key=lambda x: x), sorted(composites, key=lambda x: -_NORMS[x])]:
            S = _greedy_build(psubset + comp_order, max_size=18, deadline=deadline)
            update_best(S)

    # Strategy 3: Many random restarts (fast due to vector approach)
    num_restarts = 5000
    for r in range(num_restarts):
        if time.time() > deadline - 15:
            break
        shuffled = all_candidates[:]
        random.shuffle(shuffled)
        S = _greedy_build(shuffled, max_size=18, deadline=deadline)
        update_best(S)

    # Strategy 4: Local search on best found
    if best_set and time.time() < deadline - 10:
        improved = _local_search_improve(best_set, all_candidates, max_iter=50, deadline=deadline)
        update_best(improved)

    # Strategy 5: Local search on a few other good sets
    if time.time() < deadline - 5:
        for name, order in orders[:3]:
            if time.time() > deadline - 5:
                break
            S = _greedy_build(order, max_size=18, deadline=deadline)
            if len(S) >= max(best_len - 1, 10):
                improved = _local_search_improve(S, all_candidates, max_iter=20, deadline=deadline)
                update_best(improved)

    return best_set


def construct():
    """Return up to 18 distinct integers in 2..40 with all subset products distinct."""
    t0 = time.time()
    random.seed(42)

    S = _build_max_set()

    # Ensure list of Python ints
    S = [int(x) for x in S]

    # Trim to max 18
    if len(S) > 18:
        S = S[:18]

    # Final verification using actual products (exact)
    if time.time() - t0 < 85:
        prods = [1]
        for a in S:
            prods = prods + [x * a for x in prods]
        if len(prods) != len(set(prods)):
            # Fallback: return just primes
            S = [p for p in range(2, 41) if _NORMS[p] == 1]

    return S
# EVOLVE-BLOCK-END
