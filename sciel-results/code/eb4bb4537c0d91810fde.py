import math
import time
from itertools import combinations

def _sqfree_part(n):
    """Return (sqfree_part, perfect_square_part) where n = sf * ps^2."""
    ps = 1
    i = 2
    while i * i <= n:
        while n % (i * i) == 0:
            n //= (i * i)
            ps *= i
        i += 1
    return n, ps

def construct():
    start = time.time()
    deadline = start + 82.0  # leave 8s safety margin for 90s limit
    target = 210
    max_val = 10210

    # --- Precomputation ---
    # Sieve for smallest prime factor (SPF) to speed up factorization
    spf = list(range(max_val + 1))
    for i in range(2, int(max_val ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # Precompute squarefree signatures for all numbers up to max_val
    # sig[x] = set of primes p where x has an odd exponent of p
    signatures = [set() for _ in range(max_val + 1)]
    for x in range(2, max_val + 1):
        temp = x
        sig = set()
        last_p = 0
        count = 0
        while temp > 1:
            p = spf[temp]
            if p == last_p:
                count += 1
            else:
                if count % 2 == 1:
                    sig.add(last_p)
                last_p = p
                count = 1
            temp //= p
        if count % 2 == 1:
            sig.add(last_p)
        signatures[x] = sig

    # Map primes to indices for bit manipulation
    prime_index = {}
    primes = []
    for i in range(2, max_val + 1):
        if spf[i] == i:
            prime_index[i] = len(primes)
            primes.append(i)

    def sig_to_mask(sig_set):
        mask = 0
        for p in sig_set:
            mask |= (1 << prime_index[p])
        return mask

    target_mask = sig_to_mask(signatures[target])

    # --- Search ---
    # Iterate M from 211 upwards.
    # For each M, check if target_mask is in the span of {211..M}.
    # Use Gaussian elimination with index tracking.
    # If solution has >100 elements, try BFS for minimal-cardinality solution.

    basis = {}
    
    for M in range(211, max_val + 1):
        if time.time() > deadline:
            # Fallback: try to find a small solution quickly
            return _fallback_solution()

        v = sig_to_mask(signatures[M])
        
        # Insert M into basis
        temp_v = v
        temp_indices = frozenset([M])
        
        while temp_v != 0:
            p = temp_v.bit_length() - 1
            if p in basis:
                b_mask, b_indices = basis[p]
                temp_v ^= b_mask
                temp_indices = frozenset(temp_indices ^ b_indices)
            else:
                basis[p] = (temp_v, temp_indices)
                break

        # Check if target is in span
        t_v = target_mask
        t_indices = frozenset()
        
        while t_v != 0:
            p = t_v.bit_length() - 1
            if p in basis:
                b_mask, b_indices = basis[p]
                t_v ^= b_mask
                t_indices = frozenset(t_indices ^ b_indices)
            else:
                t_v = -1  # Not in span
                break
        
        if t_v == 0:
            # Solution found
            sol = list(t_indices)
            if len(sol) <= 100:
                # Verify
                prod = target
                for x in sol:
                    prod *= x
                if math.isqrt(prod) ** 2 == prod:
                    return sol
                else:
                    continue
            else:
                # Too many elements. Try BFS for minimal-cardinality solution
                # within {211..M}
                bfs_sol = _bfs_minimal_solution(211, M, target_mask, signatures, prime_index, deadline)
                if bfs_sol is not None:
                    return bfs_sol
                # If BFS fails, continue to next M (larger M may give better basis)

    return _fallback_solution()

def _bfs_minimal_solution(lo, hi, target_mask, signatures, prime_index, deadline):
    """BFS to find minimal-cardinality subset of {lo..hi} XORing to target_mask."""
    start = time.time()
    
    # Build list of (mask, number) for numbers in [lo, hi]
    items = []
    for x in range(lo, hi + 1):
        m = 0
        for p in signatures[x]:
            m |= (1 << prime_index[p])
        items.append((m, x))
    
    if not items:
        return None
    
    # BFS: state is (mask, frozenset of indices)
    # Start from empty set
    visited = {0: frozenset()}
    queue = [(0, frozenset())]
    
    # Limit BFS depth to avoid explosion
    max_depth = min(100, len(items))
    
    for depth in range(1, max_depth + 1):
        if time.time() > deadline - 5.0:
            return None
        
        next_queue = []
        for mask, indices in queue:
            for i, (item_mask, item_val) in enumerate(items):
                new_mask = mask ^ item_mask
                new_indices = indices | frozenset([i])
                
                if new_mask == target_mask:
                    # Found solution
                    sol = [items[j][1] for j in new_indices]
                    if len(sol) <= 100:
                        # Verify
                        prod = 210
                        for x in sol:
                            prod *= x
                        if math.isqrt(prod) ** 2 == prod:
                            return sol
                    return None
                
                if new_mask not in visited:
                    visited[new_mask] = new_indices
                    next_queue.append((new_mask, new_indices))
        
        queue = next_queue
        if not queue:
            break
    
    return None

def _fallback_solution():
    """Find a valid solution as fallback."""
    # Try small sets first
    # 210 * 840 = 176400 = 420^2, so [840] works
    # But let's try to find something better
    
    # Check if any single number x > 210 makes 210*x a perfect square
    # 210 = 2*3*5*7, so we need x to be of form 2^a*3^b*5^c*7^d*other^2
    # where a,b,c,d are all odd (to make total exponent even)
    # Simplest: x = 2*3*5*7 = 210, but that's not > 210
    # Next: x = 210 * k^2 for k >= 1, but 210*1=210 not > 210
    # x = 210*4 = 840 works
    
    # Try pairs: 210*x*y = square
    # We need x*y to have the same squarefree part as 210
    # i.e., sf(x*y) = 210
    # This means sf(x) XOR sf(y) = 210
    
    # Let's just return [840] as a safe fallback
    return [840]
