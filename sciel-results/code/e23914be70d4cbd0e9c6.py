"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt, gcd

# --- Precomputation: primes, exponent vectors, conflict graph ---
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
NPRIMES = len(PRIMES)

def _sieve_primes(n):
    if n < 2:
        return []
    is_p = [True] * (n + 1)
    is_p[0] = is_p[1] = False
    for i in range(2, isqrt(n) + 1):
        if is_p[i]:
            for j in range(i * i, n + 1, i):
                is_p[j] = False
    return [i for i in range(2, n + 1) if is_p[i]]

def _factorize(x, primes):
    exponents = []
    for p in primes:
        e = 0
        while x % p == 0:
            x //= p
            e += 1
        exponents.append(e)
    return exponents

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

# Precompute all subset sums for each number's "conflict set"
# Two numbers a, b conflict if a*b can be represented as a product of other numbers
# More precisely: a and b conflict if there exist disjoint subsets S, T of the full set
# such that product(S) = product(T) and a in S, b in T (or a in T, b in S)
# Simpler: a and b conflict if a divides b, or b divides a, or a*b is a product of two
# other numbers in the set... Actually the cleanest check is via exponent vectors.

# Build conflict graph: edge between a and b if they are NOT Q-linearly independent
# with respect to each other (i.e., one is a power of the other, or they share a
# multiplicative relation)
# Actually, the condition is: the set {a, b} has all 4 subset products distinct.
# Products: 1, a, b, a*b. These are distinct iff a != b, a != 1, b != 1, a*b != a, a*b != b.
# Since a, b >= 2, a*b > a and a*b > b. So a and b are always compatible as a pair!
# The conflict arises with 3+ elements.

# So the problem is NOT a simple graph problem. We need to find the largest set where
# all 2^k subset products are distinct. This is equivalent to Q-linear independence
# of the exponent vectors.

# For a set of exponent vectors, all subset sums are distinct iff the vectors are
# Q-linearly independent (no non-trivial integer linear combination equals zero).

# Precompute: for each number, its exponent vector
# For the search, we'll use the fact that we need Q-linear independence.

def _check_q_independent_fast(vecs):
    """Check Q-linear independence of a list of exponent vectors."""
    if not vecs:
        return True
    nrows = len(vecs)
    ncols = len(vecs[0])
    # Gaussian elimination over Q (using fractions to avoid overflow)
    mat = [list(row) for row in vecs]
    rank = 0
    for col in range(ncols):
        pivot_row = -1
        for r in range(rank, nrows):
            if mat[r][col] != 0:
                pivot_row = r
                break
        if pivot_row == -1:
            continue
        mat[rank], mat[pivot_row] = mat[pivot_row], mat[rank]
        pivot_val = mat[rank][col]
        for r in range(rank + 1, nrows):
            factor = mat[r][col]
            if factor != 0:
                for c in range(col, ncols):
                    mat[r][c] = mat[r][c] * pivot_val - factor * mat[rank][c]
                g = 0
                for c in range(col, ncols):
                    g = gcd(g, abs(mat[r][c]))
                if g > 1:
                    for c in range(col, ncols):
                        mat[r][c] //= g
        rank += 1
        if rank == nrows:
            return True
    return rank == nrows

def _check_valid_subset(nums):
    """Check if all subset products are distinct by checking Q-independence."""
    if len(nums) <= 1:
        return True
    vecs = [EXP_VECTORS[x] for x in nums]
    return _check_q_independent_fast(vecs)

def _check_valid_subset_direct(nums):
    """Check by direct enumeration (for small sets)."""
    k = len(nums)
    if k > 20:
        return _check_valid_subset(nums)
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

# --- Branch and bound for maximum independent set in a "conflict" sense ---
# Since the problem is about Q-linear independence, we can use a backtracking
# approach with pruning.

def _max_independent_search(candidates, current, deadline, best_size, best_set):
    """Backtracking search to find largest Q-independent set."""
    if time.time() > deadline:
        return best_size, best_set
    if len(current) > best_size:
        best_size = len(current)
        best_set = list(current)
    if not candidates:
        return best_size, best_set
    # Try adding the first candidate
    first = candidates[0]
    rest = candidates[1:]
    # Check if we can add first
    trial = current + [first]
    if _check_q_independent_fast([EXP_VECTORS[x] for x in trial]):
        best_size, best_set = _max_independent_search(rest, trial, deadline, best_size, best_set)
    # Try not adding first
    best_size, best_set = _max_independent_search(rest, current, deadline, best_size, best_set)
    return best_size, best_set

def _construct():
    t0 = time.time()
    deadline = t0 + 55.0  # Safety margin before 60s limit
    
    # Known good 16-element set
    initial_set = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Verify initial set
    if not _check_valid_subset(initial_set):
        # Fallback to primes
        initial_set = [x for x in range(2, 41) if all(x % d != 0 for d in range(2, isqrt(x) + 1))]
    
    best_set = list(initial_set)
    best_size = len(best_set)
    
    # Candidates: numbers not in best_set
    candidates = [x for x in range(2, 41) if x not in best_set]
    
    # Strategy 1: Greedy addition with multiple orderings
    strategies = [
        sorted(candidates, key=lambda x: -x),
        sorted(candidates, key=lambda x: x),
        sorted(candidates, key=lambda x: sum(EXP_VECTORS[x])),
        sorted(candidates, key=lambda x: -sum(EXP_VECTORS[x])),
        sorted(candidates, key=lambda x: len([p for p in PRIMES if EXP_VECTORS[x][PRIMES.index(p)] > 0])),
        sorted(candidates, key=lambda x: -len([p for p in PRIMES if EXP_VECTORS[x][PRIMES.index(p)] > 0])),
        sorted(candidates, key=lambda x: max(EXP_VECTORS[x])),
        sorted(candidates, key=lambda x: -max(EXP_VECTORS[x])),
    ]
    
    for ordering in strategies:
        if time.time() > deadline - 5:
            break
        current = list(best_set)
        current_vecs = [EXP_VECTORS[x] for x in current]
        for c in ordering:
            if time.time() > deadline - 5:
                break
            trial_vecs = current_vecs + [EXP_VECTORS[c]]
            if _check_q_independent_fast(trial_vecs):
                current.append(c)
                current_vecs.append(EXP_VECTORS[c])
        if len(current) > best_size:
            best_size = len(current)
            best_set = current
    
    # Strategy 2: Local search - swap one element for another
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        current = list(best_set)
        current_set = set(current)
        for remove_idx in range(len(current)):
            if time.time() > deadline - 5:
                break
            remaining = [x for i, x in enumerate(current) if i != remove_idx]
            removed = current[remove_idx]
            for c in range(2, 41):
                if c in current_set:
                    continue
                trial = remaining + [c]
                if _check_q_independent_fast([EXP_VECTORS[x] for x in trial]):
                    if len(trial) > best_size:
                        best_set = trial
                        best_size = len(trial)
                        current = trial
                        current_set = set(current)
                        improved = True
                        break
    
    # Strategy 3: Try adding pairs: remove 2, add 2
    if time.time() < deadline - 10:
        current = list(best_set)
        current_set = set(current)
        for i in range(len(current)):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, len(current)):
                if time.time() > deadline - 5:
                    break
                remaining = [x for idx, x in enumerate(current) if idx != i and idx != j]
                removed_pair = {current[i], current[j]}
                for c1 in range(2, 41):
                    if c1 in current_set or c1 in removed_pair:
                        continue
                    for c2 in range(c1 + 1, 41):
                        if c2 in current_set or c2 in removed_pair or c2 == c1:
                            continue
                        trial = remaining + [c1, c2]
                        if _check_q_independent_fast([EXP_VECTORS[x] for x in trial]):
                            if len(trial) > best_size:
                                best_set = trial
                                best_size = len(trial)
                                current = trial
                                current_set = set(current)
                                improved = True
    
    # Strategy 4: Branch and bound on remaining candidates
    if time.time() < deadline - 20 and best_size < 18:
        remaining_candidates = [x for x in range(2, 41) if x not in best_set]
        # Sort by "independence potential" - numbers with fewer prime factors first
        remaining_candidates.sort(key=lambda x: sum(1 for e in EXP_VECTORS[x] if e > 0))
        # Try to add from remaining candidates using backtracking
        current = list(best_set)
        current_vecs = [EXP_VECTORS[x] for x in current]
        
        def _backtrack_add(cands, cur_vecs, cur_set):
            nonlocal best_size, best_set
            if time.time() > deadline:
                return
            if len(cur_set) > best_size:
                best_size = len(cur_set)
                best_set = list(cur_set)
            if not cands:
                return
            # Try adding first candidate
            first = cands[0]
            rest = cands[1:]
            trial_vecs = cur_vecs + [EXP_VECTORS[first]]
            if _check_q_independent_fast(trial_vecs):
                _backtrack_add(rest, trial_vecs, cur_set + [first])
            # Try not adding
            _backtrack_add(rest, cur_vecs, cur_set)
        
        _backtrack_add(remaining_candidates, current_vecs, current)
    
    # Final verification
    assert len(best_set) >= 12
    assert all(isinstance(x, int) for x in best_set)
    assert all(2 <= x <= 40 for x in best_set)
    assert len(set(best_set)) == len(best_set)
    assert _check_valid_subset(best_set)
    
    return list(best_set)

def construct():
    return _construct()
# EVOLVE-BLOCK-END
