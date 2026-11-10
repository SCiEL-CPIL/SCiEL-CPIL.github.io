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

# Pack vectors into single integers for fast set operations
# Bit widths: prime 2: 3 bits (max exp 5), prime 3: 2 bits (max exp 3),
# prime 5: 2 bits (max exp 2), prime 7: 1 bit, primes 11-37: 1 bit each (8 primes)
# Total: 3+2+2+1+8 = 16 bits
_BIT_WIDTHS = [3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]
_BIT_OFFSETS = [0] * N_PRIMES
_offset = 0
for i in range(N_PRIMES):
    _BIT_OFFSETS[i] = _offset
    _offset += _BIT_WIDTHS[i]

def _pack(vec):
    """Pack a prime-exponent vector into a single integer."""
    result = 0
    for i in range(N_PRIMES):
        result |= vec[i] << _BIT_OFFSETS[i]
    return result

# Precompute packed vectors
ALL_PACKED = {num: _pack(vec) for num, vec in ALL_VECS.items()}
ZERO_PACKED = 0

# Precompute packed sums for pairs (for faster checking)
# Actually, we need to check if adding a vector creates collisions
# The check is: for all existing products p, (p + new_vec) must not be in existing products
# With packed ints, we just do integer addition

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

def _add_element_check_packed(products, new_packed):
    """Return (True, new_prods) if adding new_packed causes no collision, else (False, None)."""
    new_products = set()
    for p in products:
        np = p + new_packed
        if np in products:
            return (False, None)
        new_products.add(np)
    return (True, new_products)

def _greedy_fill(ordered, max_size, deadline, start_set=None):
    """Greedy fill from an ordering. Returns (selected, products)."""
    if start_set is not None:
        selected = list(start_set)
        selected_set = set(selected)
        products = {ZERO_PACKED}
        for x in selected:
            v = ALL_PACKED[x]
            new_prods = set()
            for p in products:
                new_prods.add(p + v)
            products |= new_prods
    else:
        selected = []
        selected_set = set()
        products = {ZERO_PACKED}

    for cand in ordered:
        if time.time() > deadline:
            break
        if len(selected) >= max_size:
            break
        if cand in selected_set:
            continue
        v = ALL_PACKED[cand]
        ok, new_prods = _add_element_check_packed(products, v)
        if ok:
            selected.append(cand)
            selected_set.add(cand)
            products |= new_prods
    return selected, products

def _compute_products(selected):
    """Compute all subset product packed values for a given selected set."""
    products = {ZERO_PACKED}
    for x in selected:
        v = ALL_PACKED[x]
        new_prods = set()
        for p in products:
            new_prods.add(p + v)
        products |= new_prods
    return products

def _try_fill(selected, products, deadline, ordering=None):
    """Try to fill selected set with more elements from ordering. Returns (new_selected, new_products, improved)."""
    if ordering is None:
        ordering = ORDER_TOTAL_PF
    sel = list(selected)
    sel_set = set(sel)
    prods = set(products)
    original_len = len(sel)
    
    for cand in ordering:
        if time.time() > deadline:
            break
        if len(sel) >= MAX_SIZE:
            break
        if cand in sel_set:
            continue
        v = ALL_PACKED[cand]
        ok, new_prods = _add_element_check_packed(prods, v)
        if ok:
            sel.append(cand)
            sel_set.add(cand)
            prods |= new_prods
    improved = len(sel) > original_len
    return sel, prods, improved

def _swap_improve(selected, products, deadline):
    """Try single-element swaps and removals to improve. Returns improved (selected, products) or None."""
    best_selected = list(selected)
    best_products = set(products)
    best_size = len(selected)
    n = len(selected)
    
    # Try removing each element and refilling
    for i in range(n):
        if time.time() > deadline:
            break
        if best_size >= MAX_SIZE:
            break
        trial = best_selected[:i] + best_selected[i + 1:]
        trial_products = _compute_products(trial)
        
        trial_sel, trial_prods, improved = _try_fill(trial, trial_products, deadline)
        if improved and len(trial_sel) > best_size:
            best_selected = trial_sel
            best_products = trial_prods
            best_size = len(trial_sel)
    
    # Try direct additions
    trial_sel, trial_prods, improved = _try_fill(best_selected, best_products, deadline)
    if improved:
        best_selected = trial_sel
        best_products = trial_prods
        best_size = len(trial_sel)
    
    if best_size > len(selected):
        return best_selected, best_products
    return None

def _two_swap_improve(selected, products, deadline):
    """Try 2-element swaps: remove one, add one. More aggressive local search."""
    best_selected = list(selected)
    best_products = set(products)
    best_size = len(selected)
    n = len(selected)
    sel_set = set(best_selected)
    
    # For each element in selected, try removing it and adding a non-selected element
    for i in range(n):
        if time.time() > deadline:
            break
        if best_size >= MAX_SIZE:
            break
        removed = best_selected[i]
        trial = best_selected[:i] + best_selected[i + 1:]
        trial_products = _compute_products(trial)
        trial_set = set(trial)
        
        # Try adding each non-selected candidate
        for cand in ORDER_TOTAL_PF:
            if time.time() > deadline:
                break
            if cand in trial_set:
                continue
            v = ALL_PACKED[cand]
            ok, new_prods = _add_element_check_packed(trial_products, v)
            if ok:
                new_sel = trial + [cand]
                if len(new_sel) > best_size:
                    best_selected = new_sel
                    best_products = new_prods
                    best_size = len(new_sel)
                    sel_set = set(best_selected)
                    break
    
    if best_size > len(selected):
        return best_selected, best_products
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin for 60s limit

    best_selected = []
    best_len = 0

    # Deterministic first pass: total prime factors ordering
    sel, _ = _greedy_fill(ORDER_TOTAL_PF, MAX_SIZE, deadline)
    if len(sel) > best_len:
        best_len = len(sel)
        best_selected = sel

    # Try other deterministic orderings
    for ordering in (ORDER_PRIMES_FIRST, ORDER_ASC, ORDER_DESC):
        if time.time() > deadline:
            break
        sel, _ = _greedy_fill(ordering, MAX_SIZE, deadline)
        if len(sel) > best_len:
            best_len = len(sel)
            best_selected = sel

    # Random restarts with diverse strategies
    rng = random.Random(12345)
    restarts = 0
    while time.time() < deadline - 5:
        restarts += 1
        r = rng.random()
        if r < 0.20:
            ordered = CANDIDATES[:]
            rng.shuffle(ordered)
        elif r < 0.35:
            ordered = sorted(CANDIDATES, key=lambda x: (TOTAL_PF[x] + rng.random() * 3, x))
        elif r < 0.50:
            ordered = sorted(CANDIDATES, key=lambda x: (DISTINCT_PF[x] + rng.random() * 2, x))
        elif r < 0.65:
            ordered = sorted(CANDIDATES, key=lambda x: (DISTINCT_PF[x] * 3 + TOTAL_PF[x] + rng.random() * 2, x))
        elif r < 0.80:
            ordered = sorted(CANDIDATES, key=lambda x: (-x + rng.random() * 15))
        else:
            # Weighted random: prefer numbers with fewer prime factors
            weights = [1.0 / (TOTAL_PF[x] + 1) for x in CANDIDATES]
            total_w = sum(weights)
            ordered = []
            remaining = list(CANDIDATES)
            for _ in range(len(CANDIDATES)):
                w = [1.0 / (TOTAL_PF[x] + 1) for x in remaining]
                tw = sum(w)
                r2 = rng.random() * tw
                cum = 0
                for j, x in enumerate(remaining):
                    cum += w[j]
                    if r2 <= cum:
                        ordered.append(x)
                        remaining.remove(x)
                        break
            if remaining:
                ordered.extend(remaining)

        sel, _ = _greedy_fill(ordered, MAX_SIZE, deadline)
        if len(sel) > best_len:
            best_len = len(sel)
            best_selected = sel
            # Try local improvements
            if best_len < MAX_SIZE and time.time() < deadline - 5:
                products = _compute_products(best_selected)
                improved = _swap_improve(best_selected, products, deadline)
                if improved is not None:
                    best_selected, _ = improved
                    best_len = len(best_selected)
                if best_len < MAX_SIZE and time.time() < deadline - 5:
                    products = _compute_products(best_selected)
                    improved = _two_swap_improve(best_selected, products, deadline)
                    if improved is not None:
                        best_selected, _ = improved
                        best_len = len(best_selected)

        if best_len >= MAX_SIZE:
            break

    # Final fill attempt
    if best_len < MAX_SIZE and time.time() < deadline:
        products = _compute_products(best_selected)
        sel_set = set(best_selected)
        for cand in ORDER_TOTAL_PF:
            if time.time() > deadline:
                break
            if len(best_selected) >= MAX_SIZE:
                break
            if cand in sel_set:
                continue
            v = ALL_PACKED[cand]
            ok, new_prods = _add_element_check_packed(products, v)
            if ok:
                best_selected.append(cand)
                sel_set.add(cand)
                products |= new_prods
                best_len = len(best_selected)

    result = [int(x) for x in best_selected]
    return result
# EVOLVE-BLOCK-END
