"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    # Deadline guard: aim to finish well within 90s
    t0 = time.time()
    deadline = t0 + 80.0  # 10s safety margin

    n = 256

    # Step 1: Generate all primes up to 256
    def sieve(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit+1, i):
                    is_prime[j] = False
        return [i for i in range(2, limit+1) if is_prime[i]]

    primes = sieve(n)

    # Step 2: For each prime, find the maximum power p^k <= n
    prime_powers = []
    for p in primes:
        max_power = p
        while max_power * p <= n:
            max_power *= p
        prime_powers.append(max_power)

    # The set of candidate numbers: 1 and all prime powers
    # Any pairwise-coprime set can be transformed into a set of prime powers (and possibly 1)
    # with at least as large sum, because if a number has two distinct prime factors, we can
    # replace it with the product of the max powers of those primes... wait, that's not right.
    # 
    # Actually, the key insight is: any valid set S of pairwise-coprime integers can be mapped
    # to a set S' of prime powers (one per distinct prime in S) such that sum(S') >= sum(S).
    # This is because for each prime p appearing in S, the maximum contribution is p^k where
    # p^k <= n and p^k is coprime to all other elements. But since elements are pairwise coprime,
    # each prime appears in at most one element. So the optimal solution consists of choosing
    # for each prime p either 0 (not used) or p^k (the max power <= n), such that the chosen
    # prime powers are pairwise coprime (which they always are, since they're powers of distinct primes).
    #
    # Wait, but we could also have composite numbers that are products of prime powers of distinct primes.
    # For example, 6 = 2*3. But 2 and 3 are both prime powers. And 2+3 = 5 < 6. So including 6 is better
    # than including 2 and 3 separately!
    #
    # So the problem is: choose a subset of {1} ∪ {numbers in 2..256} that are pairwise coprime, maximizing sum.
    # The numbers in the optimal set can be composite (products of distinct primes' powers).
    #
    # Key observation: In the optimal set, each number is a product of prime powers of distinct primes,
    # and no prime appears in more than one number. So we're partitioning some subset of primes into groups,
    # where each group's product (of their max powers) is <= 256, and we want to maximize the sum of these products.
    #
    # This is a weighted set partitioning / knapsack-like problem.
    #
    # Since there are only 55 primes <= 256, but most prime powers > 128 can't be multiplied with anything else,
    # the effective search space is small. Let's think about which primes can be "merged":
    #
    # A prime p can be merged with prime q if p^a * q^b <= 256 where p^a and q^b are the max powers.
    # For large primes (> 128), p^a = p, and p * q > 256 for any q >= 2, so they can't be merged.
    # For primes in (64, 128], p * 2 > 128, so they can only be merged with 2 if 2*p <= 256, i.e., p <= 128.
    #
    # Let's enumerate the primes and their max powers:
    # This approach is getting complex. Let me use a different strategy.
    #
    # Alternative: Since the numbers must be pairwise coprime and in 1..256, and we want max sum,
    # let's use the fact that the optimal solution uses only "prime power products" where each prime
    # appears at most once. We can do a DP over primes, but the state space is large.
    #
    # Better approach: The primes > 128 (i.e., 129..256) must appear alone (as p^1) since p*2 > 256.
    # There are 24 such primes: 131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257... wait 257>256.
    # Let me count: primes from 129 to 256.
    
    # Let me just implement a smart search. Given the small range, I'll use the following:
    # 1. All primes > 128 must be included alone (they contribute p, and merging is impossible).
    # 2. For primes <= 128, we need to decide how to group them into products <= 256.
    #
    # Actually, let me reconsider. Is it always optimal to include all primes > 128?
    # Yes, because they can't conflict with each other (each is a distinct prime power) and each contributes positively.
    
    # Primes > 128:
    large_primes = [p for p in primes if p > 128]
    
    # For primes <= 128, we need to find the optimal grouping.
    # The number of such primes is manageable. Let's list them:
    small_primes = [p for p in primes if p <= 128]
    
    # For each small prime, compute its max power <= 256:
    def max_prime_power(p, limit):
        power = p
        while power * p <= limit:
            power *= p
        return power
    
    # Now, the problem for small primes: partition some subset into groups, each group's product of max powers <= 256.
    # This is still complex. Let me use a greedy + local search approach, or just enumerate.
    
    # Given time constraints, let me try a different approach:
    # The optimal set consists of numbers that are pairwise coprime. Each number's prime factorization
    # uses distinct primes. So we're assigning each prime to at most one number.
    
    # Let me just include 1 and all prime powers, then check if we can do better by merging.
    # Start with: 1 + all prime powers
    base_set = [1] + prime_powers
    
    # Check pairwise coprimality (should be fine since they're powers of distinct primes)
    # Sum of base set:
    base_sum = sum(base_set)
    
    # Now try to improve by merging some prime powers into products.
    # For each pair of prime powers (p^a, q^b), if p^a * q^b <= 256, we can merge them.
    # Merging changes the sum from p^a + q^b to p^a * q^b, which is better if p^a * q^b > p^a + q^b.
    # This is true when p^a * q^b - p^a - q^b > 0, i.e., (p^a - 1)(q^b - 1) > 1.
    # For p^a >= 2 and q^b >= 2, this is always true (minimum is (2-1)(2-1)=1, not > 1).
    # Wait: 2*3 = 6 > 2+3=5. Yes. 2*2 is not possible (distinct primes).
    # So merging any two prime powers p^a and q^b (p != q) with product <= 256 always improves the sum.
    
    # So we want to maximize the number of merges, subject to product <= 256.
    # This is a matching problem: find the maximum-weight matching where edge weight = p^a*q^b - p^a - q^b.
    
    # Let me implement a greedy matching:
    # 1. Compute all possible pairs (i, j) with i < j, where prime_powers[i] * prime_powers[j] <= 256.
    # 2. Sort by weight descending.
    # 3. Greedily add edges that don't conflict.
    
    # But we can also have triples! p^a * q^b * r^c <= 256.
    # And even larger groups.
    
    # This is getting complex. Let me use a simpler approach:
    # Since the numbers are small, let me just enumerate all valid sets using backtracking,
    # but prune aggressively.
    
    # Actually, let me try a different insight:
    # The optimal solution includes 1, and for each prime, we decide whether to use it.
    # The numbers in the solution are of the form prod(p_i^a_i) for distinct primes p_i.
    # Each such number must be <= 256.
    
    # Let me use a DP approach where we process primes in order and decide how to group them.
    # But the state is the current "active" products, which is complex.
    
    # Given the small size, let me just try all subsets of small primes and for each,
    # find the optimal grouping. But 2^55 is too large.
    
    # Let me try a heuristic: include all primes > 128, and for primes <= 128,
    # greedily merge pairs when beneficial.
    
    # Actually, let me just compute the answer by careful reasoning:
    # Primes > 128: all included alone. Count them.
    # Primes in (64, 128]: can be merged with 2 only. 2*p <= 256 iff p <= 128. So yes.
    #   Merging 2 and p: 2*p > 2+p always. So merge.
    #   But 2 can only be in one group.
    # Primes in (32, 64]: can be merged with 2 and/or 3.
    #   2*3*p = 6p <= 256 iff p <= 42.66, so p <= 41.
    #   2*p <= 256 for all p <= 128.
    #   3*p <= 256 for p <= 85.
    # etc.
    
    # This is getting too complex for manual reasoning. Let me implement a proper algorithm.
    
    # Approach: Since the maximum number in the set is 256, and numbers are pairwise coprime,
    # the number of elements in the set is at most the number of primes <= 256, which is 54.
    # But most elements will be prime powers or small products.
    
    # Let me use the following algorithm:
    # 1. Start with all prime powers (and 1).
    # 2. Repeatedly try to merge pairs of elements whose product is <= 256 and coprime (always true for distinct primes).
    # 3. Use a priority queue to always merge the pair with the largest gain.
    
    # But this greedy might not be optimal. Let me try it anyway, as it's simple and fast.
    
    # Elements: list of (value, prime_set) where prime_set is the set of primes in the value.
    # Initially: 1 -> (1, frozenset()), and p^k -> (p^k, frozenset({p}))
    
    elements = [(1, frozenset())]
    for p in primes:
        mp = max_prime_power(p, n)
        elements.append((mp, frozenset([p])))
    
    # Try to merge: repeatedly find the best pair to merge.
    # A pair (i, j) can be merged if elements[i][1] & elements[j][1] == empty set (disjoint primes)
    # and elements[i][0] * elements[j][0] <= n.
    # Gain = elements[i][0] * elements[j][0] - elements[i][0] - elements[j][0].
    
    # Use a simple greedy: repeatedly find the best merge and apply it.
    # To avoid O(n^2) per iteration, and since n is small (~55), this is fine.
    
    changed = True
    while changed:
        changed = False
        best_gain = -1
        best_i = -1
        best_j = -1
        for i in range(len(elements)):
            for j in range(i+1, len(elements)):
                if elements[i][1] & elements[j][1]:
                    continue
                prod = elements[i][0] * elements[j][0]
                if prod > n:
                    continue
                gain = prod - elements[i][0] - elements[j][0]
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
                    best_j = j
        if best_i >= 0:
            # Merge best_i and best_j
            new_val = elements[best_i][0] * elements[best_j][0]
            new_primes = elements[best_i][1] | elements[best_j][1]
            # Remove best_i and best_j, add new element
            if best_i < best_j:
                elements.pop(best_j)
                elements.pop(best_i)
            else:
                elements.pop(best_i)
                elements.pop(best_j)
            elements.append((new_val, new_primes))
            changed = True
    
    # Now elements is our set. Convert to list of ints.
    result = [e[0] for e in elements]
    
    # Ensure it's a list of Python ints
    result = [int(x) for x in result]
    
    # Verify pairwise coprimality
    from math import gcd
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if gcd(result[i], result[j]) != 1:
                # This shouldn't happen, but fall back to prime powers
                return [1] + [max_prime_power(p, n) for p in primes]
    
    # Check deadline
    if time.time() > deadline:
        # Return what we have
        pass
    
    return result

# EVOLVE-BLOCK-END
