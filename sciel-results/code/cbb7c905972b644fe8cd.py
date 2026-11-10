"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 40, 'maximum_size': 18}

# Precompute primes up to 40
def _get_primes(limit):
    sieve = [True] * (limit + 1)
    if limit >= 0:
        sieve[0] = False
    if limit >= 1:
        sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i]]

N = P['n']
MAX_SIZE = P['maximum_size']
PRIMES = _get_primes(N)
N_PRIMES = len(PRIMES)
CANDIDATES = list(range(2, N + 1))

# Precompute prime-exponent vectors for all candidates
def _factorize(n):
    fac = {}
    for p in PRIMES:
        if p * p > n:
            break
        while n % p == 0:
            fac[p] = fac.get(p, 0) + 1
            n //= p
    if n > 1:
        fac[n] = fac.get(n, 0) + 1
    return fac

ALL_VECS = {}
for num in CANDIDATES:
    fac = _factorize(num)
    vec = [0] * N_PRIMES
    for i, p in enumerate(PRIMES):
        if p in fac:
            vec[i] = fac[p]
    ALL_VECS[num] = vec

# Complexity metrics for ordering heuristics
def _total_prime_factors(num):
    n = num
    count = 0
    for p in PRIMES:
        if p * p > n:
            break
        while n % p == 0:
            count += 1
            n //= p
    if n > 1:
        count += 1
    return count

def _distinct_prime_factors(num):
    n = num
    count = 0
    for p in PRIMES:
        if p * p > n:
            break
        if n % p == 0:
            count += 1
        while n % p == 0:
            n //= p
    if n > 1:
        count += 1
    return count

TOTAL_PF = {x: _total_prime_factors(x) for x in CANDIDATES}
DISTINCT_PF = {x: _distinct_prime_factors(x) for x in CANDIDATES}

# Precompute a few deterministic orderings
ORDER_TOTAL_PF = sorted(CANDIDATES, key=lambda x: (TOTAL_PF[x], x))
ORDER_ASC = sorted(CANDIDATES)
ORDER_DESC = sorted(CANDIDATES, reverse=True)
ORDER_PRIMES_FIRST = sorted(CANDIDATES, key=lambda x: (DISTINCT_PF[x], TOTAL_PF[x], x))

ZERO_VEC = (0,) * N_PRIMES

def _add_element_check(existing_products, new_vec):
    """Return (True, new_prods) if adding new_vec causes no collision, else (False, None)."""
    new_products = set()
    for p in existing_products:
        np = tuple(p[i] + new_vec[i] for i in range(N_PRIMES))
        if np in existing_products:
            return (False, None)
        new_products.add(np)
    return (True, new_products)

def _greedy_fill(ordered, max_size, deadline, start_set=None):
    """Greedy fill from an ordering. Returns (selected, products)."""
    if start_set is not None:
        selected = list(start_set)
        selected_set = set(selected)
        # Build products from start_set
        products = {ZERO_VEC}
        for x in selected:
            v = ALL_VECS[x]
            new_prods = set()
            for p in products:
                np = tuple(p[i] + v[i] for i in range(N_PRIMES))
                new_prods.add(np)
            products |= new_prods
    else:
        selected = []
        selected_set = set()
        products = {ZERO_VEC}

    for cand in ordered:
        if time.time() > deadline:
            break
        if len(selected) >= max_size:
            break
        if cand in selected_set:
            continue
        v = ALL_VECS[cand]
        ok, new_prods = _add_element_check(products, v)
        if ok:
            selected.append(cand)
            selected_set.add(cand)
            products |= new_prods
    return selected, products

def _swap_improve(selected, products, deadline):
    """Try single-element swaps to improve. Returns improved (selected, products) or None."""
    best_selected = list(selected)
    best_products = products
    best_size = len(selected)
    n = len(selected)

    for i in range(n):
        if time.time() > deadline:
            break
        if best_size >= MAX_SIZE:
            break
        removed = best_selected[i]
        trial = best_selected[:i] + best_selected[i + 1:]
        trial_set = set(trial)
        # Recompute products
        trial_products = {ZERO_VEC}
        for x in trial:
            v = ALL_VECS[x]
            new_prods = set()
            for p in trial_products:
                np = tuple(p[j] + v[j] for j in range(N_PRIMES))
                new_prods.add(np)
            trial_products |= new_prods
        # Greedily add candidates not in trial
        for cand in ORDER_TOTAL_PF:
            if time.time() > deadline:
                break
            if len(trial) >= MAX_SIZE:
                break
            if cand in trial_set:
                continue
            v = ALL_VECS[cand]
            ok, new_prods = _add_element_check(trial_products, v)
            if ok:
                trial.append(cand)
                trial_set.add(cand)
                trial_products |= new_prods
        if len(trial) > best_size:
            best_selected = trial
            best_products = trial_products
            best_size = len(trial)
    if best_size > len(selected):
        return best_selected, best_products
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin

    best_selected = []
    best_len = 0

    # Deterministic first pass: total prime factors ordering
    sel, _ = _greedy_fill(ORDER_TOTAL_PF, MAX_SIZE, deadline)
    if len(sel) > best_len:
        best_len = len(sel)
        best_selected = sel

    # Try a couple of deterministic orderings
    for ordering in (ORDER_PRIMES_FIRST, ORDER_ASC, ORDER_DESC):
        if time.time() > deadline:
            break
        sel, _ = _greedy_fill(ordering, MAX_SIZE, deadline)
        if len(sel) > best_len:
            best_len = len(sel)
            best_selected = sel

    # Random restarts
    rng = random.Random(12345)
    restarts = 0
    while time.time() < deadline - 5:
        restarts += 1
        # Pick a strategy
        r = rng.random()
        if r < 0.3:
            ordered = CANDIDATES[:]
            rng.shuffle(ordered)
        elif r < 0.5:
            # Weighted by complexity, with randomness
            ordered = sorted(CANDIDATES, key=lambda x: (TOTAL_PF[x] + rng.random() * 2, x))
        elif r < 0.7:
            # Prime-powers favored: numbers with few distinct prime factors
            ordered = sorted(CANDIDATES, key=lambda x: (DISTINCT_PF[x] + rng.random(), x))
        elif r < 0.85:
            ordered = CANDIDATES[:]
            rng.shuffle(ordered)
        else:
            # Descending with noise
            ordered = sorted(CANDIDATES, key=lambda x: (-x + rng.random() * 10))

        sel, _ = _greedy_fill(ordered, MAX_SIZE, deadline)
        if len(sel) > best_len:
            best_len = len(sel)
            best_selected = sel
            # Try swap improvement on the new best
            if best_len < MAX_SIZE and time.time() < deadline - 5:
                # Recompute products for best_selected
                products = {ZERO_VEC}
                for x in best_selected:
                    v = ALL_VECS[x]
                    new_prods = set()
                    for p in products:
                        np = tuple(p[j] + v[j] for j in range(N_PRIMES))
                        new_prods.add(np)
                    products |= new_prods
                improved = _swap_improve(best_selected, products, deadline)
                if improved is not None:
                    best_selected, _ = improved
                    best_len = len(best_selected)

        # Early exit if we hit the maximum
        if best_len >= MAX_SIZE:
            break

    # Final: try to fill the current best with any remaining candidates
    if best_len < MAX_SIZE and time.time() < deadline:
        products = {ZERO_VEC}
        for x in best_selected:
            v = ALL_VECS[x]
            new_prods = set()
            for p in products:
                np = tuple(p[j] + v[j] for j in range(N_PRIMES))
                new_prods.add(np)
            products |= new_prods
        sel_set = set(best_selected)
        for cand in ORDER_TOTAL_PF:
            if time.time() > deadline:
                break
            if len(best_selected) >= MAX_SIZE:
                break
            if cand in sel_set:
                continue
            v = ALL_VECS[cand]
            ok, new_prods = _add_element_check(products, v)
            if ok:
                best_selected.append(cand)
                sel_set.add(cand)
                products |= new_prods
                best_len = len(best_selected)

    # Ensure return type is a Python list of Python ints
    result = [int(x) for x in best_selected]
    return result
# EVOLVE-BLOCK-END
