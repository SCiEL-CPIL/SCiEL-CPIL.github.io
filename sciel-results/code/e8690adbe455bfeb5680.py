"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    N = 256
    
    # 1. Sieve to get primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # 2. Calculate largest prime power for each prime
    prime_powers = {}
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        prime_powers[p] = power
        
    # Base sum is 1 + sum of all prime powers
    base_sum = 1 + sum(prime_powers.values())
    
    # 3. Generate potential merge groups
    # A group is a set of distinct primes {p1, ..., pk}
    # Valid if product <= N and product > sum(prime_powers[pi])
    # Gain = product - sum(prime_powers[pi])
    
    # Only primes p where p <= N/2 can be in a product of 2+ primes <= N
    # Actually, if p > N/2, p*q > N for any q >= 2.
    # So active primes for merging are p <= N // 2
    active_primes = [p for p in primes if p <= N // 2]
    
    potential_groups = []
    
    def time_check():
        return time.time() > deadline

    # Generate pairs
    n_active = len(active_primes)
    for i in range(n_active):
        if time_check():
            break
        p1 = active_primes[i]
        for j in range(i + 1, n_active):
            p2 = active_primes[j]
            prod = p1 * p2
            if prod > N:
                break
            cost = prime_powers[p1] + prime_powers[p2]
            gain = prod - cost
            if gain > 0:
                potential_groups.append((frozenset([p1, p2]), gain, prod))

    # Generate triples
    for i in range(n_active):
        if time_check():
            break
        p1 = active_primes[i]
        for j in range(i + 1, n_active):
            p2 = active_primes[j]
            prod12 = p1 * p2
            # If prod12 * min_active_prime > N, skip inner loop
            # min_active_prime is active_primes[0] which is 2
            if prod12 * 2 > N:
                continue
            for k in range(j + 1, n_active):
                p3 = active_primes[k]
                prod = prod12 * p3
                if prod > N:
                    break
                cost = prime_powers[p1] + prime_powers[p2] + prime_powers[p3]
                gain = prod - cost
                if gain > 0:
                    potential_groups.append((frozenset([p1, p2, p3]), gain, prod))

    # Generate quads
    # Only possible with very small primes
    # 2*3*5*7 = 210 <= 256
    # 2*3*5*11 = 330 > 256
    # So only quads involving 2,3,5,7 or similar small sets
    # We can just check combinations of the first few active primes
    small_active = active_primes[:10]
    from itertools import combinations
    for combo in combinations(small_active, 4):
        prod = 1
        for p in combo:
            prod *= p
        if prod > N:
            continue
        cost = sum(prime_powers[p] for p in combo)
        gain = prod - cost
        if gain > 0:
            potential_groups.append((frozenset(combo), gain, prod))

    # Sort groups by gain descending
    potential_groups.sort(key=lambda x: x[1], reverse=True)
    
    # 4. Search for max weight independent set of groups
    # State: idx, used_primes (frozenset), current_gain
    best_gain = 0
    best_groups = []
    
    memo = {}
    
    def backtrack(idx, used_primes, current_gain, current_groups):
        nonlocal best_gain, best_groups
        
        if time.time() > deadline:
            return
            
        if current_gain > best_gain:
            best_gain = current_gain
            best_groups = list(current_groups)
            
        if idx == len(potential_groups):
            return
            
        state_key = (idx, used_primes)
        if state_key in memo:
            if memo[state_key] >= current_gain:
                return
        else:
            memo[state_key] = current_gain
            
        # Option 1: Skip
        backtrack(idx + 1, used_primes, current_gain, current_groups)
        
        # Option 2: Take
        group_set, gain, prod = potential_groups[idx]
        if not used_primes.isdisjoint(group_set):
            return
        backtrack(idx + 1, used_primes | group_set, current_gain + gain, current_groups + [group_set])

    backtrack(0, frozenset(), 0, [])
    
    # 5. Construct final result
    result = [1]
    used_in_merge = set()
    for group in best_groups:
        prod = 1
        for p in group:
            prod *= p
            used_in_merge.add(p)
        result.append(prod)
        
    for p in primes:
        if p not in used_in_merge:
            result.append(prime_powers[p])
            
    # Ensure types are Python int
    result = [int(x) for x in result]
    
    return result

# EVOLVE-BLOCK-END
