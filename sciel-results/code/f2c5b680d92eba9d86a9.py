import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    base = 210
    max_value = 10210
    max_size = 100
    
    # Precompute primes up to sqrt(10210) ~ 101, use up to 200 for safety
    limit = 200
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit+1, i):
                sieve[j] = False
    primes = [i for i in range(limit+1) if sieve[i]]
    
    def factorize_parity(n):
        """Return frozenset of primes dividing n to odd power."""
        result = set()
        temp = n
        for p in primes:
            if p * p > temp:
                break
            count = 0
            while temp % p == 0:
                count += 1
                temp //= p
            if count % 2 == 1:
                result.add(p)
        if temp > 1:
            result.add(temp)
        return frozenset(result)
    
    # Precompute parity vectors for all numbers in (210, max_value]
    parity_cache = {}
    for n in range(211, max_value + 1):
        parity_cache[n] = factorize_parity(n)
    
    # Target
    target = frozenset([2, 3, 5, 7])
    
    # Precompute all primes that could appear in any parity vector up to max_value
    all_primes_global = set()
    for n in range(211, max_value + 1):
        all_primes_global.update(parity_cache[n])
    all_primes_global.update(target)
    prime_list_global = sorted(all_primes_global)
    prime_to_idx_global = {p: i for i, p in enumerate(prime_list_global)}
    num_primes_global = len(prime_list_global)
    
    def to_bitset_global(fs):
        bits = 0
        for p in fs:
            idx = prime_to_idx_global.get(p)
            if idx is not None:
                bits |= (1 << idx)
        return bits
    
    target_bits = to_bitset_global(target)
    
    # Precompute bit vectors for all candidates
    bit_vectors = {}
    for n in range(211, max_value + 1):
        bit_vectors[n] = to_bitset_global(parity_cache[n])
    
    def find_solution(m):
        """Find subset of (210, m] with XOR = target, size <= 100."""
        if m <= 210:
            return None
        
        n_cand = m - 210
        # Use bit vectors directly
        rows = [bit_vectors[n] for n in range(211, m + 1)]
        rows.append(target_bits)
        
        # Track combinations using Python ints as bitsets
        comb = []
        for i in range(n_cand):
            comb.append(1 << i)
        comb.append(0)
        
        n_rows = len(rows)
        rank = 0
        for col in range(num_primes_global):
            pivot = -1
            for r in range(rank, n_rows):
                if (rows[r] >> col) & 1:
                    pivot = r
                    break
            if pivot == -1:
                continue
            rows[rank], rows[pivot] = rows[pivot], rows[rank]
            comb[rank], comb[pivot] = comb[pivot], comb[rank]
            for r in range(n_rows):
                if r != rank and ((rows[r] >> col) & 1):
                    rows[r] ^= rows[rank]
                    comb[r] ^= comb[rank]
            rank += 1
        
        # Check if target is in span
        if rows[-1] != 0:
            return None
        
        # Extract solution
        c = comb[-1]
        subset_indices = [i for i in range(n_cand) if (c >> i) & 1]
        
        if len(subset_indices) == 0:
            return None
        if len(subset_indices) > max_size:
            return None
        
        subset = [211 + i for i in subset_indices]
        
        # Verify
        prod = base
        for x in subset:
            prod *= x
        s = math.isqrt(prod)
        if s * s != prod:
            return None
        
        return subset
    
    # Binary search on M
    # We know M=840 works (single element 840)
    # Check if smaller M works
    
    best_solution = [840]
    
    # Binary search: find smallest M in [211, 840] where solution exists
    lo = 211
    hi = 840
    
    while lo < hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        result = find_solution(mid)
        if result is not None:
            hi = mid
            best_solution = result
        else:
            lo = mid + 1
    
    # Verify the final answer
    if time.time() > deadline:
        return list(best_solution)
    
    result = find_solution(lo)
    if result is not None:
        best_solution = result
    
    # Final verification
    prod = base
    for x in best_solution:
        prod *= x
    s = math.isqrt(prod)
    if s * s != prod:
        return [840]
    
    if len(best_solution) < 1 or len(best_solution) > max_size:
        return [840]
    
    if any(not isinstance(x, int) or isinstance(x, bool) for x in best_solution):
        return [840]
    
    if any(x <= 210 or x > max_value for x in best_solution):
        return [840]
    
    if len(set(best_solution)) != len(best_solution):
        return [840]
    
    return list(best_solution)
