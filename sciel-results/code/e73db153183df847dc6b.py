"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin before 60s kill
    
    n = 256
    
    # Sieve of Eratosthenes
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    num_primes = len(primes)
    
    # For each prime, find the largest power <= 256
    max_power = {}
    for p in primes:
        mp = p
        while mp * p <= n:
            mp *= p
        max_power[p] = mp
    
    # Precompute prime factor masks for all numbers 2..256
    # mask is a bitmask over the primes list
    def get_mask(x):
        mask = 0
        for idx, p in enumerate(primes):
            if x % p == 0:
                mask |= (1 << idx)
        return mask
    
    masks = [0] * (n + 1)
    for x in range(2, n + 1):
        masks[x] = get_mask(x)
    
    # Initial free set: all primes
    all_free_mask = (1 << num_primes) - 1
    
    # Precompute for each possible number (2..256) its mask and value
    # Group numbers by their mask to avoid duplicates
    # For each mask, we only need the maximum value number with that mask
    mask_to_max_val = {}
    for x in range(2, n + 1):
        m = masks[x]
        if m not in mask_to_max_val or x > mask_to_max_val[m]:
            mask_to_max_val[m] = x
    
    # Sort candidate numbers by value descending for better pruning
    candidates = []
    for m, v in mask_to_max_val.items():
        if m != 0:  # skip mask 0 (number 1) for now, handle separately
            candidates.append((m, v))
    # Sort by value descending
    candidates.sort(key=lambda x: -x[1])
    
    # Precompute the contribution of each prime
    prime_contrib = [0] * num_primes
    for idx, p in enumerate(primes):
        prime_contrib[idx] = max_power[p]
    
    # Precompute for each prime index, which candidate indices include that prime
    # This allows us to quickly find candidates for a given free set
    # Actually, let's precompute a tighter upper bound:
    # For a given free mask, the upper bound is the maximum sum we can get
    # by partitioning the free primes into groups, where each group's value
    # is the max number whose prime factors are exactly that group.
    # This is the same as the original problem on the free primes.
    # But we can compute a simpler bound: for each prime p in free,
    # the best we can do with p is max_power[p], but if we combine p with q,
    # we might get a higher value.
    # 
    # A better bound: compute the maximum independent set weight on the
    # conflict graph of free primes, where edge weights represent the
    # "penalty" for not combining.
    #
    # Simpler approach: use a greedy bound. For each prime in free,
    # estimate its contribution as max_power[p], but if p and q can be
    # combined into p^a * q^b <= 256, the combined value might be higher
    # than max_power[p] + max_power[q].
    #
    # Let's precompute for every pair of primes whether combining them
    # gives a better value than separate prime powers.
    
    # For the upper bound, we'll use a recursive memoized computation
    # that computes the optimal value for a given free mask, but with
    # memoization to avoid recomputation.
    
    # Actually, let's use a different strategy:
    # 1. Precompute for each subset of primes (up to some size) the optimal value.
    # 2. For larger subsets, use the sum of max_power as an upper bound.
    
    # Let's precompute the optimal value for all masks up to a certain bit count.
    # Since 256 has at most 8 distinct prime factors (2^8=256), any number uses
    # at most 8 primes. So any valid number has a mask with at most 8 bits set.
    
    # Precompute: for each mask with popcount <= 8, the max value achievable
    # by choosing numbers that together cover exactly those primes.
    # This is a set packing problem on small sets.
    
    # Actually, let's just improve the upper bound by considering that
    # each free prime p can contribute at most max_power[p], and we can
    # also add 1 for the number 1.
    
    best_sum = 0
    best_set = [1]  # Always include 1
    
    # Memoize upper bounds for free masks
    ub_memo = {}
    
    def compute_ub(free_mask):
        """Compute a tighter upper bound for the given free mask."""
        if free_mask == 0:
            return 1  # just the number 1
        
        if free_mask in ub_memo:
            return ub_memo[free_mask]
        
        # Simple bound: sum of max_power for each free prime + 1
        ub = 1
        fm = free_mask
        while fm:
            lowbit = fm & (-fm)
            idx = lowbit.bit_length() - 1
            ub += prime_contrib[idx]
            fm ^= lowbit
        
        # Try to tighten: for each candidate that is a subset of free_mask,
        # the value v covers mask m. The remaining free primes are free_mask & ~m.
        # The upper bound is v + compute_ub(free_mask & ~m), but we take the max
        # over all valid candidates. However, this is expensive.
        # 
        # Instead, let's use a greedy approach: sort free primes by max_power
        # descending, and for each, check if combining with previously processed
        # primes gives a better value.
        
        # For now, return the simple bound
        ub_memo[free_mask] = ub
        return ub
    
    # DFS with branch and bound
    # free_mask: bitmask of primes not yet covered
    # current_sum: sum of values chosen so far
    # chosen: list of chosen numbers
    
    def dfs(free_mask, current_sum, chosen, depth):
        nonlocal best_sum, best_set, deadline
        
        if time.time() > deadline:
            return
        
        if free_mask == 0:
            # All primes covered
            total = current_sum + 1  # add 1
            if total > best_sum:
                best_sum = total
                best_set = chosen + [1]
            return
        
        # Upper bound
        ub = current_sum + compute_ub(free_mask)
        
        if ub <= best_sum:
            return
        
        # Find candidates that are subsets of free_mask
        # Sort them by value descending
        valid_candidates = []
        for m, v in candidates:
            if (m & ~free_mask) == 0:
                valid_candidates.append((m, v))
        
        if not valid_candidates:
            return
        
        # Try each valid candidate
        for m, v in valid_candidates:
            new_free = free_mask & ~m
            dfs(new_free, current_sum + v, chosen + [v], depth + 1)
            
            # Prune: if time is up, stop
            if time.time() > deadline:
                return
    
    dfs(all_free_mask, 0, [], 0)
    
    # If no solution found (shouldn't happen), use the simple prime powers
    if not best_set or best_sum == 0:
        result = [1]
        for p in primes:
            result.append(max_power[p])
    else:
        result = best_set
    
    # Validation
    if len(result) != len(set(result)):
        return [1]
    if any(x < 1 or x > 256 for x in result):
        return [1]
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if math.gcd(result[i], result[j]) != 1:
                return [1]
    
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
