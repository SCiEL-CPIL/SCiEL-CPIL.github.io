"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from itertools import combinations

P = {'n': 256}

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True

def _prime_factors(n):
    """Return set of distinct prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.add(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def _valid_set(lst):
    """Check if list of distinct ints satisfies the constraint:
    For every three distinct a,b,c in lst, a does NOT divide b*c."""
    n = len(lst)
    for i in range(n):
        a = lst[i]
        for j in range(i + 1, n):
            b = lst[j]
            for k in range(j + 1, n):
                c = lst[k]
                if (b * c) % a == 0:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    n = P['n']
    candidates = list(range(2, n + 1))
    
    # Precompute prime factors for each candidate
    pf_cache = {x: _prime_factors(x) for x in candidates}
    
    # Strategy: We want to maximize the size of a set S ⊆ {2,...,256}
    # such that for all distinct a,b,c ∈ S, a ∤ b*c.
    
    # Key insight: If a | b*c, this happens when the prime factorization of a
    # is "covered" by the union of prime factors (with multiplicities) of b and c.
    # 
    # A simple sufficient condition for a ∤ b*c: a has a prime factor p^e 
    # such that p^e does not divide b*c. This can happen if:
    # - p appears in a but not in b or c (but p is in pf_cache[a])
    # - p^e appears in a but b*c has insufficient power of p
    #
    # Greedy approach: Start with all candidates, iteratively remove elements
    # that cause the most violations.
    
    # Better approach: Build up the set greedily, adding elements that don't
    # violate the constraint.
    
    # First, let's try a greedy construction with multiple orderings and
    # take the best result.
    
    def greedy_construct(ordering, time_limit):
        """Greedily build a valid set following the given ordering."""
        s = set()
        s_list = []
        
        for x in ordering:
            if time.time() > time_limit:
                break
            # Check if adding x violates any constraint
            valid = True
            for i in range(len(s_list)):
                a = s_list[i]
                for j in range(i + 1, len(s_list)):
                    b = s_list[j]
                    c = x
                    if (b * c) % a == 0:
                        valid = False
                        break
                    if (a * c) % b == 0:
                        valid = False
                        break
                    if (a * b) % c == 0:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                s.add(x)
                s_list.append(x)
        
        return s
    
    # Try different orderings
    import random
    
    best_set = set()
    
    # Ordering 1: natural order
    s1 = greedy_construct(candidates, deadline - 30)
    if len(s1) > len(best_set):
        best_set = s1
    
    # Ordering 2: primes first, then composites
    primes = [x for x in candidates if _is_prime(x)]
    composites = [x for x in candidates if not _is_prime(x)]
    ordering2 = primes + composites
    s2 = greedy_construct(ordering2, deadline - 30)
    if len(s2) > len(best_set):
        best_set = s2
    
    # Ordering 3: reverse natural order
    s3 = greedy_construct(candidates[::-1], deadline - 30)
    if len(s3) > len(best_set):
        best_set = s3
    
    # Ordering 4: descending by number of distinct prime factors
    by_npf = sorted(candidates, key=lambda x: -len(pf_cache[x]))
    s4 = greedy_construct(by_npf, deadline - 30)
    if len(s4) > len(best_set):
        best_set = s4
    
    # Ordering 5: ascending by smallest prime factor
    def smallest_pf(x):
        return min(pf_cache[x])
    by_spf = sorted(candidates, key=smallest_pf)
    s5 = greedy_construct(by_spf, deadline - 30)
    if len(s5) > len(best_set):
        best_set = s5
    
    # Now try local improvement: try to add elements not in best_set
    # and try to swap elements
    if time.time() < deadline - 15:
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            missing = [x for x in candidates if x not in best_set]
            for x in missing:
                if time.time() > deadline - 15:
                    break
                # Try adding x
                trial = best_set | {x}
                if _valid_set(sorted(trial)):
                    best_set = trial
                    improved = True
                    break
            
            # Try swapping: remove one element, add another
            if time.time() < deadline - 15:
                for y in list(best_set):
                    if time.time() > deadline - 15:
                        break
                    for x in missing:
                        if time.time() > deadline - 10:
                            break
                        trial = (best_set - {y}) | {x}
                        if _valid_set(sorted(trial)):
                            best_set = trial
                            improved = True
                            break
                    if improved:
                        break
    
    result = sorted(best_set)
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
