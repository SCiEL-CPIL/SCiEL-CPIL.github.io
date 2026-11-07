"""Fast linear-independence (subset-product distinctness) search with heuristic greedy + local search."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt

P = {'n': 40, 'maximum_size': 18}

def _is_prime(a):
    if a < 2:
        return False
    if a < 4:
        return True
    if a % 2 == 0 or a % 3 == 0:
        return False
    d = 5
    while d * d <= a:
        if a % d == 0 or a % (d + 2) == 0:
            return False
        d += 6
    return True

def _factor(n):
    """Return dict prime -> exponent for n (n <= 40)."""
    d = {}
    x = n
    p = 2
    while p * p <= x:
        while x % p == 0:
            d[p] = d.get(p, 0) + 1
            x //= p
        p += 1
    if x > 1:
        d[x] = d.get(x, 0) + 1
    return d

# Precompute exponent vectors (as dict) for all candidates 2..40
_FACTORS = {v: _factor(v) for v in range(2, P['n'] + 1)}
_PRIMES = [v for v in range(2, P['n'] + 1) if _is_prime(v)]
_COMPS = [v for v in range(2, P['n'] + 1) if not _is_prime(v)]
_CANDS = list(range(2, P['n'] + 1))

# A large prime modulus for exact linear rank over GF(2) of exponent vectors.
# Exponents are small (<=5) and we only ever test linear dependence over Z;
# dependence over Z implies dependence over any field, and for these small
# vectors dependence over a large prime field is equivalent (no spurious
# dependence because a true integer relation has small coefficients).
_MOD = 1000000007

def _vec_of(v):
    return _FACTORS[v]

def _check_valid(S):
    """Exact check: all subset products distinct iff exponent vectors are
    linearly independent over Z. We test via rank over a large prime field,
    which is exact for these small integer vectors."""
    # Build list of exponent vectors; rank over GF(2)-like via modular Gaussian.
    # Represent each vector as a dict prime->exp mod _MOD.
    vecs = []
    for v in S:
        vecs.append({p: e % _MOD for p, e in _FACTORS[v].items()})
    # Gaussian elimination over the field mod _MOD on the union of coordinates.
    coords = set()
    for vec in vecs:
        coords.update(vec.keys())
    coords = list(coords)
    # rows = vecs
    basis = []  # list of (pivot_coord, row_dict)
    for vec in vecs:
        row = {c: vec.get(c, 0) % _MOD for c in coords}
        # eliminate
        for pc, brow in basis:
            if row.get(pc, 0) != 0:
                inv = pow(brow[pc], _MOD - 2, _MOD)
                mult = row[pc] * inv % _MOD
                if mult != 0:
                    for c in coords:
                        row[c] = (row.get(c, 0) - mult * brow.get(c, 0)) % _MOD
        # find pivot
        pivot = None
        for c in coords:
            if row.get(c, 0) != 0:
                pivot = c
                break
        if pivot is None:
            return False  # dependent -> collision
        inv = pow(row[pivot], _MOD - 2, _MOD)
        for c in coords:
            row[c] = row.get(c, 0) * inv % _MOD
        basis.append((pivot, row))
    return True

def _check_valid_products(S):
    """Direct enumeration check (used for final validation, small sets)."""
    current = {1}
    for x in S:
        new_vals = {p * x for p in current}
        if current & new_vals:
            return False
        current |= new_vals
    return True

def _try_greedy(order):
    """Greedy: add v if it keeps exponent vectors independent."""
    current = []
    vecs = []  # list of dicts
    coords = set()
    basis = []  # (pivot, normalized_row)
    for v in order:
        f = _FACTORS[v]
        row = {p: e % _MOD for p, e in f.items()}
        # eliminate against basis
        ok = True
        for pc, brow in basis:
            if row.get(pc, 0) != 0:
                inv = pow(brow[pc], _MOD - 2, _MOD)
                mult = row[pc] * inv % _MOD
                for p in brow:
                    row[p] = (row.get(p, 0) - mult * brow[p]) % _MOD
        # find pivot
        pivot = None
        for p in f:
            if row.get(p, 0) != 0:
                pivot = p
                break
        if pivot is None:
            continue  # dependent, skip
        inv = pow(row[pivot], _MOD - 2, _MOD)
        for p in row:
            row[p] = row[p] * inv % _MOD
        basis.append((pivot, row))
        current.append(v)
        vecs.append(f)
    return current

def _collision_risk(v, current):
    """Surrogate: number of subset products of current that collide with
    any subset product involving v. Approximated via shared-prime overlap and
    a modular rank test cost. Lower is better."""
    # Heuristic: count how many current elements share a prime factor with v
    f = _FACTORS[v]
    risk = 0
    for u in current:
        fu = _FACTORS[u]
        for p in f:
            if p in fu:
                risk += 1
    # Also penalize by size of v's factor set (more factors -> more coords)
    risk += 0.1 * len(f)
    return risk

def _heuristic_order(current, remaining, rng, mode):
    """Order remaining candidates by a risk-based key."""
    def key(v):
        return _collision_risk(v, current)
    if mode == 0:
        return sorted(remaining, key=key)
    elif mode == 1:
        # random perturbation
        return sorted(remaining, key=lambda v: key(v) + rng.random() * 0.5)
    else:
        # random
        lst = list(remaining)
        rng.shuffle(lst)
        return lst

def _local_search(S, deadline, rng):
    """Try swaps: remove one element, add a different one if valid and same size,
    then try to grow. Keep best size."""
    best = list(S)
    best_size = len(best)
    in_set = set(best)
    # Precompute candidate pool
    while time.time() < deadline:
        improved = False
        # Try to grow by adding any candidate
        for v in _CANDS:
            if v in in_set:
                continue
            trial = best + [v]
            if _check_valid(trial):
                best = trial
                in_set.add(v)
                best_size += 1
                improved = True
                break
        if improved:
            continue
        # Try a swap: remove r, add a
        for r in list(best):
            for a in _CANDS:
                if a in in_set or a == r:
                    continue
                trial = [x for x in best if x != r] + [a]
                if _check_valid(trial):
                    # accept swap (same size) to enable future growth
                    best = trial
                    in_set.remove(r)
                    in_set.add(a)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin vs 90s kill

    rng = random.Random(20240501)

    # Initial best: primes (safe base, 12 elements)
    best_set = _PRIMES[:]
    best_size = len(best_set)

    # 1. Deterministic heuristic greedy orders
    # Order A: primes first, then composites by collision risk
    def _risk_key(v):
        return _collision_risk(v, best_set)
    orderA = _PRIMES[:] + sorted(_COMPS, key=_risk_key)
    solA = _try_greedy(orderA)
    if len(solA) > best_size:
        best_size = len(solA)
        best_set = solA[:]

    # Order B: composites sorted by number of distinct prime factors (fewer first)
    def _nfac(v):
        return len(_FACTORS[v])
    orderB = sorted(_CANDS, key=_nfac)
    solB = _try_greedy(orderB)
    if len(solB) > best_size:
        best_size = len(solB)
        best_set = solB[:]

    # Order C: descending value (large primes/composites first)
    orderC = sorted(_CANDS, reverse=True)
    solC = _try_greedy(orderC)
    if len(solC) > best_size:
        best_size = len(solC)
        best_set = solC[:]

    # 2. Local search polish on the best found so far
    if time.time() < deadline:
        polished = _local_search(best_set, deadline, rng)
        if len(polished) > best_size:
            best_size = len(polished)
            best_set = polished[:]

    # 3. Randomized restarts with heuristic ordering
    while time.time() < deadline:
        mode = rng.choice([0, 0, 1, 2])  # bias toward heuristic
        remaining = _CANDS[:]
        current = []
        # Build order incrementally with risk
        order = _heuristic_order(current, remaining, rng, mode)
        sol = _try_greedy(order)
        if len(sol) > best_size:
            best_size = len(sol)
            best_set = sol[:]
            # polish immediately
            if time.time() < deadline:
                polished = _local_search(best_set, deadline, rng)
                if len(polished) > best_size:
                    best_size = len(polished)
                    best_set = polished[:]
        # Occasionally restart from a random subset of primes + a few composites
        if mode == 2 and rng.random() < 0.3:
            seed_set = _PRIMES[:]
            rng.shuffle(seed_set)
            # add a couple of random composites if valid
            for c in _COMPS:
                if rng.random() < 0.3:
                    trial = seed_set + [c]
                    if _check_valid(trial):
                        seed_set = trial
            sol = _try_greedy(seed_set + _heuristic_order(seed_set,
                       [v for v in _CANDS if v not in set(seed_set)], rng, 0))
            if len(sol) > best_size:
                best_size = len(sol)
                best_set = sol[:]

    # Final exact validation via direct product enumeration
    if not _check_valid_products(best_set):
        best_set = _PRIMES[:]

    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
