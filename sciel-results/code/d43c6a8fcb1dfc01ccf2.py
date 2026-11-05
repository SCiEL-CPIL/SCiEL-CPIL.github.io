import time
import random
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 87.0  # Use more of the 90s budget
    
    n = 128
    numbers = list(range(1, n + 1))
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    # Precompute LCM table for all pairs
    lcm_table = {}
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[(i, j)] = lcm(i, j)
    
    def get_lcm(a, b):
        if a <= b:
            return lcm_table[(a, b)]
        else:
            return lcm_table[(b, a)]
    
    # Precompute conflict graph: conflict[x][y] = True if x and y are in a bad triple with some z
    # Actually, we need: for a given x, which pairs (a,b) in S would form a bad triple with x?
    # A bad triple (x, a, b) exists if lcm(x,a) == lcm(x,b) == lcm(a,b)
    # So for each x, we can precompute which pairs (a,b) would conflict with x
    
    # Better approach: For each x, precompute the set of "conflict pairs"
    # conflict_pairs[x] = set of frozenset({a,b}) such that lcm(x,a)==lcm(x,b)==lcm(a,b)
    
    # Even better: For each x, precompute which elements a would be "compatible" with x
    # in the sense that they don't form a bad triple with any other element in S
    
    # Let's think differently: 
    # For a fixed x, adding x to S is bad iff there exist a,b in S such that 
    # lcm(x,a) == lcm(x,b) == lcm(a,b)
    
    # For each x, precompute: for each a, what is lcm(x,a)?
    # Then for each L, the elements a with lcm(x,a)=L form a group.
    # Within each group, if any pair (a,b) has lcm(a,b)=L, then x conflicts with {a,b}.
    
    # Precompute for each x:
    #   conflict_with[x] = set of frozensets {a,b} that form a bad triple with x
    # But this is O(n^3) which is 2M, manageable.
    
    # Actually, let's precompute: for each x, which pairs (a,b) are "x-conflicting"
    # Then can_add(x, S) checks if any x-conflicting pair is entirely in S
    
    # Precompute x_conflicts[x] = list of (a, b) where a < b and lcm(x,a)==lcm(x,b)==lcm(a,b)
    x_conflicts = [set() for _ in range(n + 1)]
    for x in range(1, n + 1):
        for a in range(1, n + 1):
            if a == x:
                continue
            for b in range(a + 1, n + 1):
                if b == x:
                    continue
                if get_lcm(x, a) == get_lcm(x, b) == get_lcm(a, b):
                    x_conflicts[x].add(frozenset((a, b)))
    
    # Check if adding x to set S creates a bad triple
    def can_add_fast(x, S):
        for conflict_pair in x_conflicts[x]:
            if conflict_pair <= S:  # Both elements are in S
                return False
        return True
    
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(n)
    prime_powers = set()
    for p in primes:
        power = p
        while power <= n:
            prime_powers.add(power)
            power *= p
    
    core_set = prime_powers | {1}
    
    def num_distinct_primes(x):
        count = 0
        d = 2
        while d * d <= x:
            if x % d == 0:
                count += 1
                while x % d == 0:
                    x //= d
            d += 1
        if x > 1:
            count += 1
        return count
    
    # Precompute number of conflicts for each x (how many pairs it conflicts with)
    # This helps prioritize which elements to try first
    conflict_count = [0] * (n + 1)
    for x in range(1, n + 1):
        conflict_count[x] = len(x_conflicts[x])
    
    remaining = [x for x in range(1, n + 1) if x not in core_set]
    
    # Multiple randomized greedy passes
    best_set = set(core_set)
    
    orderings = []
    # Sort by fewest conflicts first (more flexible elements)
    orderings.append(sorted(remaining, key=lambda x: (conflict_count[x], num_distinct_primes(x), x)))
    orderings.append(sorted(remaining, key=lambda x: (conflict_count[x], num_distinct_primes(x), -x)))
    orderings.append(sorted(remaining, key=lambda x: x))
    orderings.append(sorted(remaining, key=lambda x: -x))
    
    for _ in range(12):
        rand_order = remaining[:]
        random.shuffle(rand_order)
        orderings.append(rand_order)
    
    for ordering in orderings:
        if time.time() > deadline - 20.0:
            break
        
        current = set(core_set)
        for x in ordering:
            if time.time() > deadline - 20.0:
                break
            if x in current:
                continue
            if can_add_fast(x, current):
                current.add(x)
        
        if len(current) > len(best_set):
            best_set = set(current)
    
    # Local search: try swaps (1-for-k and 2-for-k)
    all_numbers = set(range(1, n + 1))
    
    improved = True
    pass_count = 0
    while improved and time.time() < deadline - 3.0:
        improved = False
        pass_count += 1
        rejected = [x for x in all_numbers if x not in best_set]
        rejected.sort(key=lambda x: (conflict_count[x], num_distinct_primes(x), x))
        
        # 1-for-k swaps
        for e in list(best_set):
            if time.time() > deadline - 3.0:
                break
            
            best_set.remove(e)
            
            added = []
            for r in rejected:
                if r in best_set:
                    continue
                if can_add_fast(r, best_set):
                    best_set.add(r)
                    added.append(r)
            
            if len(added) > 1:
                improved = True
                rejected = [x for x in rejected if x not in best_set]
            else:
                best_set.add(e)
                for r in added:
                    best_set.remove(r)
        
        # 2-for-k swaps (only if still improving)
        if improved and time.time() < deadline - 3.0:
            improved = False
            rejected = [x for x in all_numbers if x not in best_set]
            rejected.sort(key=lambda x: (conflict_count[x], num_distinct_primes(x), x))
            
            best_list = list(best_set)
            for i in range(len(best_list)):
                if time.time() > deadline - 3.0:
                    break
                for j in range(i + 1, len(best_list)):
                    if time.time() > deadline - 3.0:
                        break
                    
                    e1, e2 = best_list[i], best_list[j]
                    best_set.discard(e1)
                    best_set.discard(e2)
                    
                    added = []
                    for r in rejected:
                        if r in best_set:
                            continue
                        if can_add_fast(r, best_set):
                            best_set.add(r)
                            added.append(r)
                    
                    if len(added) > 2:
                        improved = True
                        rejected = [x for x in rejected if x not in best_set]
                        best_list = list(best_set)
                    else:
                        best_set.add(e1)
                        best_set.add(e2)
                        for r in added:
                            best_set.discard(r)
    
    # Final validation with optimized check using precomputed conflicts
    S = sorted(best_set)
    valid = True
    for i in range(len(S)):
        if time.time() > deadline - 1.0:
            break
        for j in range(i + 1, len(S)):
            si, sj = S[i], S[j]
            # Check if there exists k in S such that lcm(si,sj)==lcm(si,k)==lcm(sj,k)
            target_lcm = get_lcm(si, sj)
            for k in range(j + 1, len(S)):
                sk = S[k]
                if get_lcm(si, sk) == target_lcm and get_lcm(sj, sk) == target_lcm:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: try to fix by removing problematic elements
        S = sorted(best_set)
        changed = True
        while changed and time.time() < deadline - 1.0:
            changed = False
            S = sorted(best_set)
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    si, sj = S[i], S[j]
                    target_lcm = get_lcm(si, sj)
                    for k in range(j + 1, len(S)):
                        sk = S[k]
                        if get_lcm(si, sk) == target_lcm and get_lcm(sj, sk) == target_lcm:
                            best_set.discard(sk)
                            changed = True
                            break
                    if changed:
                        break
                if changed:
                    break
    
    return list(best_set) if len(best_set) > 0 else list(core_set)
