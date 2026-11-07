"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    n = 256
    # Generate primes up to n using Sieve of Eratosthenes
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    
    # Initial assignment: largest power of each prime
    # Map: prime -> value
    assignment = {}
    for p in primes:
        val = p
        while val * p <= n:
            val *= p
        assignment[p] = val
        
    # Local Search: Attempt to merge pairs of primes if it increases the sum.
    # We represent the current state as a list of (set_of_primes, value).
    # Initially, each is ({p}, p^k).
    # A merge replaces ({p}, v1) and ({q}, v2) with ({p,q}, v_merged) if v_merged > v1+v2.
    # Note: Merged items cannot be merged further in this simple pass, but we can run multiple passes.
    
    items = [({p}, assignment[p]) for p in primes]
    
    improved = True
    while improved:
        improved = False
        new_items = []
        used_indices = set()
        for i in range(len(items)):
            if i in used_indices:
                continue
            for j in range(i + 1, len(items)):
                if j in used_indices:
                    continue
                s1, v1 = items[i]
                s2, v2 = items[j]
                # Only merge if they are disjoint (they should be) and result in valid number
                # Check if we can form a number with primes s1 U s2
                # Since s1 and s2 are sets of primes, we look for max x <= n with P(x) = s1 U s2
                # But wait, if we already have a merged item, s1 might have size > 1.
                # We should only merge singletons or allow general merges?
                # General merges are complex. Let's stick to merging singletons or existing merges?
                # For now, let's just try to merge any two items.
                
                union_primes = s1 | s2
                # Find max value
                max_val = 0
                # Brute force check for max_val is expensive if done often.
                # But n=256 is small.
                # We can precompute best value for any subset of primes? No, too many subsets.
                # We can just iterate x from n down to 1 and check factors?
                # Better: iterate powers.
                
                # Simple check: Is there any number?
                # If len(union_primes) > 2, it's less likely to be beneficial.
                # We'll just do it for all pairs.
                
                # Optimization: Only merge if sum of individual values is small enough to be beaten?
                # max possible is 256.
                if v1 + v2 >= 256:
                    continue # Cannot beat sum if sum is already >= max possible composite? 
                    # Wait, composite <= 256. If v1+v2 >= 256, composite <= 256 <= v1+v2.
                    # So no improvement possible.
                
                # Find max x with prime factors exactly union_primes
                # We can just check all x from 256 down to 1
                found = False
                for x in range(n, 1, -1):
                    # Check if P(x) == union_primes
                    temp = x
                    factors = set()
                    for p in primes:
                        while temp % p == 0:
                            factors.add(p)
                            temp //= p
                    if temp != 1:
                        continue # Has prime factor > n? No, x<=n. But x might have prime factor not in primes? 
                                 # primes contains all primes <= n. So temp should be 1.
                    if factors == union_primes:
                        max_val = x
                        found = True
                        break
                
                if max_val > v1 + v2:
                    # Merge
                    new_items.append((union_primes, max_val))
                    used_indices.add(i)
                    used_indices.add(j)
                    improved = True
                    break # Restart scan with new list
            
            if improved:
                break
        
        if improved:
            # Rebuild items list
            final_items = []
            for i in range(len(items)):
                if i not in used_indices:
                    final_items.append(items[i])
            final_items.extend(new_items)
            items = final_items
            
    # Construct result
    result = [1]
    for s, v in items:
        result.append(v)
        
    return result

# EVOLVE-BLOCK-END
