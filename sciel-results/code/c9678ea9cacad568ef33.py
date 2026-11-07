# EVOLVE-BLOCK-START
import time
import numpy as np
from math import isqrt
from functools import lru_cache

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _conflict(a, b, c):
    """
    Check if there's a conflict among three distinct integers a, b, c.
    Conflict exists if any one divides the product of the other two.
    """
    return (a % (b * c) == 0) or (b % (a * c) == 0) or (c % (a * b) == 0)

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s limit
    
    # Generate all candidates in [2, 256]
    candidates = list(range(2, 257))
    n = len(candidates)
    
    # Build conflict graph: edge between i and j if there exists k such that
    # {candidates[i], candidates[j], candidates[k]} has a conflict
    # Actually, we need: for every triple (a,b,c) of distinct members, no a|b*c
    # This means: if we have a,b,c in our set, we need a∤b*c, b∤a*c, c∤a*b
    # 
    # Let's think of it as: we want to find the largest subset S such that for all
    # distinct a,b,c in S, none of a|b*c, b|a*c, c|a*b holds.
    #
    # This is equivalent to: for all distinct a,b in S, and for all c in S\{a,b},
    # we need a∤b*c AND b∤a*c.
    #
    # Let's define a "2-conflict" between a and b if there exists c in candidates
    # (c ≠ a, c ≠ b) such that a|b*c OR b|a*c.
    # But that's not quite right because c must also be in our selected set.
    #
    # Better approach: This is a 3-uniform hypergraph independent set problem.
    # Each hyperedge is a triple {a,b,c} where at least one divisibility condition fails.
    # We want the maximum independent set in this hypergraph.
    #
    # For small n (255 candidates), we can use a different approach:
    # 1. First, identify which triples are "bad" (have a conflict)
    # 2. Then use a greedy + local search approach, or
    # 3. Reformulate as a maximum independent set in a derived graph
    
    # Let's use a different structural approach:
    # Key insight: If a | b*c, then for prime factorization, 
    # the exponents in a must be ≤ sum of exponents in b and c for each prime.
    # 
    # Alternative: Use the fact that if we restrict to numbers with limited 
    # prime factor structure, we might avoid conflicts.
    #
    # Let me try a constraint programming / backtracking approach with pruning.
    
    # First, let's identify "safe" numbers that are less likely to cause conflicts.
    # Primes are safe because if a is prime, a|b*c implies a|b or a|c.
    # So if we only have primes and numbers not divisible by those primes, we're safe.
    
    # Let's try a multi-stage approach:
    # Stage 1: Start with all primes
    # Stage 2: Add composites that don't create conflicts with existing set
    # Stage 3: Local search to try to improve
    
    primes = [x for x in range(2, 257) if _is_prime(x)]
    composites = [x for x in range(2, 257) if not _is_prime(x)]
    
    # For each composite, determine which primes divide it
    def prime_divisors(n):
        divs = set()
        d = 2
        temp = n
        while d * d <= temp:
            while temp % d == 0:
                divs.add(d)
                temp //= d
            d += 1
        if temp > 1:
            divs.add(temp)
        return divs
    
    comp_prime_divs = {c: prime_divisors(c) for c in composites}
    
    # A composite a can only "conflict" with primes p where p|a
    # If a|b*c and a is composite, this is more complex.
    # Let's just use direct checking.
    
    S = set(primes)
    S_list = list(primes)
    
    # Sort composites by "restrictiveness" - fewer prime factors first
    def factor_count(n):
        count = 0
        temp = n
        d = 2
        while d * d <= temp:
            while temp % d == 0:
                count += 1
                temp //= d
            d += 1
        if temp > 1:
            count += 1
        return count
    
    composites_sorted = sorted(composites, key=factor_count)
    
    def can_add(a, S_set):
        """Check if a can be added to S_set without creating any triple conflict."""
        if time.time() > deadline:
            return False
        for b in S_set:
            if b == a:
                continue
            for c in S_set:
                if c == a or c == b:
                    continue
                # Check a|b*c, b|a*c, c|a*b
                if (b * c) % a == 0 or (a * c) % b == 0 or (a * b) % c == 0:
                    return False
        return True
    
    # Greedy addition
    for a in composites_sorted:
        if time.time() > deadline - 5.0:
            break
        if can_add(a, S):
            S.add(a)
            S_list.append(a)
    
    # Local search: try to swap out one element for another to gain more
    # This is a different structural approach - simulated annealing style
    best_S = set(S_list)
    best_size = len(best_S)
    
    # Try random swaps
    import random
    random.seed(42)
    
    for iteration in range(500):
        if time.time() > deadline - 2.0:
            break
        
        # Try removing one element and adding another
        if len(best_S) < 2:
            break
            
        removed = random.choice(list(best_S))
        temp_S = best_S - {removed}
        
        # Try to add a different element
        candidates_to_add = [x for x in candidates if x not in temp_S and x != removed]
        random.shuffle(candidates_to_add)
        
        added = None
        for cand in candidates_to_add:
            if can_add(cand, temp_S):
                added = cand
                break
        
        if added is not None and added != removed:
            new_S = temp_S | {added}
            if len(new_S) >= best_size:
                best_S = new_S
                best_size = len(best_S)
    
    # Convert to list of Python ints
    result = [int(x) for x in sorted(best_S)]
    return result
# EVOLVE-BLOCK-END
