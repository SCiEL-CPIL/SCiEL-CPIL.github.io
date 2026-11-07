"""Combined program: fast sieve-based SFP precomputation + systematic pair/triple search with deadline guard."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from collections import defaultdict

def _sieve_sfp(limit):
    """Precompute squarefree part for all numbers 1..limit using a sieve."""
    spf = list(range(limit + 1))
    for i in range(2, int(limit ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    sfp = [0] * (limit + 1)
    sfp[1] = 1
    for x in range(2, limit + 1):
        v = x
        r = 1
        while v > 1:
            p = spf[v]
            cnt = 0
            while v % p == 0:
                v //= p
                cnt ^= 1
            if cnt:
                r *= p
        sfp[x] = r
    return sfp

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _build_prime_index(limit):
    """Build list of primes up to limit and mapping prime->index."""
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, limit + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    return primes, prime_to_idx

def _sfp_to_vec(s, primes, prime_to_idx):
    """Convert squarefree number to parity vector (as Python int for speed)."""
    vec = 0
    v = s
    for i, p in enumerate(primes):
        if p * p > v:
            break
        if v % p == 0:
            vec |= (1 << i)
            v //= p
            while v % p == 0:
                v //= p
    if v > 1:
        idx = prime_to_idx.get(v)
        if idx is not None:
            vec |= (1 << idx)
    return vec

def _find_subset_gf2(M, target_vec, num_primes, sfp, primes, prime_to_idx):
    """
    Find a non-empty subset of {211, ..., M} whose XOR of parity vectors equals target_vec.
    Uses Gaussian elimination to find a minimal subset.
    Returns list of numbers or None.
    """
    if M < 211:
        return None
    
    # Build parity vectors for numbers 211..M
    # We want to find a subset whose XOR = target_vec
    # This is equivalent to finding a linear dependency in the augmented system
    
    # Use Gaussian elimination over GF(2)
    # Rows are parity vectors, we track which numbers contribute
    
    # For efficiency, use integer representation of vectors
    # Each vector is an integer where bit i corresponds to prime i
    
    # We'll do Gaussian elimination to find a basis, then check if target is in span
    # If so, backtrack to find the actual subset
    
    # Build list of (vector, number) pairs
    vecs = []
    for x in range(211, M + 1):
        v = _sfp_to_vec(sfp[x], primes, prime_to_idx)
        vecs.append((v, x))
    
    # Gaussian elimination: find basis and check if target is in span
    # We'll store basis as list of (pivot_bit, vector, set_of_numbers)
    # For memory efficiency with up to ~10000 numbers, use bitsets
    
    # Actually, for finding ANY subset, we can use a simpler approach:
    # Process vectors in order, maintaining a basis.
    # When we see a vector that's in the span of the current basis,
    # we can find the subset that produces zero, but we need target.
    
    # Better approach: augment target as an extra row
    # We want: sum_{i in S} v_i = target
    # Equivalently: sum_{i in S} v_i + target = 0
    # So we add target as an extra vector and look for a non-trivial dependency
    # that includes the target row.
    
    # Let's use a different approach: 
    # Process all vectors, maintain basis with backtracking info.
    # At the end, check if target is in the span, and if so, recover the subset.
    
    # For simplicity and correctness, use the standard approach:
    # 1. Build basis from all vectors
    # 2. Check if target is in span
    # 3. If yes, recover subset by backtracking
    
    # Step 1: Gaussian elimination with tracking
    # basis[p] = (vector, frozenset of numbers that produce this vector)
    # This could be memory intensive, so let's use a smarter approach
    
    # Alternative: process vectors one by one, maintain basis.
    # For each new vector, if it reduces to 0, we found a dependency.
    # But we need the dependency to equal target, not 0.
    
    # Let me use a cleaner approach:
    # Add target as an extra row (index = len(vecs))
    # Find any non-trivial linear dependency among all rows
    # If the dependency includes the target row, then the other rows form our subset
    
    n_rows = len(vecs) + 1  # +1 for target
    # Store each row as (vector, row_index)
    all_rows = [(v, i) for i, (v, x) in enumerate(vecs)]
    all_rows.append((target_vec, n_rows - 1))  # target row
    
    # Gaussian elimination
    # basis[bit] = (row_index, vector) for the row that has leading bit at 'bit'
    basis = {}
    row_to_basis = {}  # row_index -> list of row indices used to reduce it
    
    for row_idx, (vec, _) in enumerate(all_rows):
        # Try to reduce vec using existing basis
        current = vec
        used_rows = []
        for bit in range(num_primes - 1, -1, -1):
            if not (current & (1 << bit)):
                continue
            if bit in basis:
                basis_row_idx, basis_vec = basis[bit]
                current ^= basis_vec
                used_rows.append(basis_row_idx)
            else:
                # This row becomes the basis for this bit
                basis[bit] = (row_idx, current)
                break
        if current == 0:
            # We found a dependency: row_idx XOR (all used_rows) = 0
            # The dependency includes row_idx and all used_rows
            dep_rows = [row_idx] + used_rows
            # Check if target row (n_rows - 1) is in dep_rows
            if (n_rows - 1) in dep_rows:
                # The subset is all rows in dep_rows except the target row
                subset_indices = [r for r in dep_rows if r != (n_rows - 1)]
                if subset_indices:
                    subset_nums = [vecs[i][1] for i in subset_indices]
                    return subset_nums
            # If target row not in dependency, this dependency doesn't help
            # Continue processing
            # But we need to remember this dependency for later...
            # Actually, if we find ANY dependency that includes the target row, we're done.
            # If not, we continue. But we might miss later dependencies.
            # Let's just check at the end if target is in span.
    
    # After processing all rows, check if target is in the span
    # Reduce target using basis
    current = target_vec
    used_rows = []
    for bit in range(num_primes - 1, -1, -1):
        if not (current & (1 << bit)):
            continue
        if bit in basis:
            basis_row_idx, basis_vec = basis[bit]
            current ^= basis_vec
            used_rows.append(basis_row_idx)
        else:
            # Target not in span
            return None
    
    if current == 0:
        # Target is in span, subset is the used_rows
        if used_rows:
            subset_nums = [vecs[i][1] for i in used_rows]
            return subset_nums
    
    return None

def construct():
    t0 = time.time()
    deadline = t0 + 85.0  # safety margin before 90s kill
    LO = 210
    MAXV = 10210
    MAX_SIZE = 100

    # Precompute squarefree parts for all numbers up to MAXV
    sfp = _sieve_sfp(MAXV)
    
    # Build prime index
    primes, prime_to_idx = _build_prime_index(MAXV)
    num_primes = len(primes)
    
    # Target vector: parity of 210
    target_sfp = sfp[210]  # should be 210
    target_vec = _sfp_to_vec(target_sfp, primes, prime_to_idx)
    
    # Binary search on M: find minimal M such that a valid subset exists in {211, ..., M}
    # Lower bound: 211 (need at least one number)
    # Upper bound: MAXV
    
    lo = 211
    hi = MAXV
    best_result = None
    
    # First, check if a single element works
    for x in range(LO + 1, MAXV + 1):
        if sfp[x] == target_sfp:
            prod = LO * x
            if _is_square(prod):
                return [x]
    
    # Binary search on M
    while lo <= hi:
        if time.time() > deadline - 10:
            break
        mid = (lo + hi) // 2
        
        result = _find_subset_gf2(mid, target_vec, num_primes, sfp, primes, prime_to_idx)
        
        if result is not None:
            # Verify the result
            prod = LO
            for x in result:
                prod *= x
            if _is_square(prod) and len(result) <= MAX_SIZE:
                best_result = result
                hi = mid - 1
            else:
                # Invalid result, continue searching
                lo = mid + 1
        else:
            lo = mid + 1
    
    if best_result is not None:
        # Verify once more
        prod = LO
        for x in best_result:
            prod *= x
        if _is_square(prod) and len(best_result) <= MAX_SIZE:
            return list(best_result)
    
    # Fallback: try direct search for small subsets
    # Single element
    for x in range(LO + 1, MAXV + 1):
        if sfp[x] == target_sfp:
            prod = LO * x
            if _is_square(prod):
                return [x]
    
    # Pairs
    sfp_to_nums = defaultdict(list)
    for x in range(LO + 1, MAXV + 1):
        sfp_to_nums[sfp[x]].append(x)
    
    sfp_values = sorted(sfp_to_nums.keys())
    found = None
    
    for i in range(len(sfp_values)):
        if time.time() > deadline - 5:
            break
        a = sfp_values[i]
        nums_a = sfp_to_nums[a]
        for j in range(i, len(sfp_values)):
            b = sfp_values[j]
            nums_b = sfp_to_nums[b]
            g = 1
            aa, bb = a, b
            while bb:
                aa, bb = bb, aa % bb
            g = aa
            prod_sfp = (a * b) // (g * g)
            if prod_sfp == target_sfp:
                if a == b:
                    if len(nums_a) >= 2:
                        x = nums_a[0]
                        y = nums_a[1]
                    else:
                        continue
                else:
                    x = nums_a[0]
                    y = nums_b[0]
                prod = LO * x * y
                if _is_square(prod):
                    mx = max(x, y)
                    if found is None or mx < found[0]:
                        found = (mx, [min(x,y), max(x,y)])
    
    if found is not None:
        return found[1]
    
    # Fallback: return known valid [840]
    return [840]
# EVOLVE-BLOCK-END
