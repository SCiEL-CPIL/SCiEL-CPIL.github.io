"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

DEADLINE = 85.0  # safety margin before 90s kill

def _violates(S, a, b, c):
    """Check if a divides b*c (a, b, c distinct)."""
    return (b * c) % a == 0

def _is_valid_set(S):
    """Check all triples of distinct elements."""
    n = len(S)
    for i in range(n):
        a = S[i]
        for j in range(n):
            if j == i:
                continue
            b = S[j]
            for k in range(n):
                if k == i or k == j:
                    continue
                c = S[k]
                if (b * c) % a == 0:
                    return False
    return True

def _count_violations(S):
    """Count number of violating triples (for optimization)."""
    n = len(S)
    count = 0
    for i in range(n):
        a = S[i]
        for j in range(n):
            if j == i:
                continue
            b = S[j]
            for k in range(n):
                if k == i or k == j:
                    continue
                c = S[k]
                if (b * c) % a == 0:
                    count += 1
    return count

def _fast_check_add(S, x):
    """Check if adding x to S creates any violation. S is a set."""
    # For each a in S: check if a divides x*c for any c in S, c != x (x not in S yet)
    # For x: check if x divides a*b for any a,b in S, a != b
    from math import gcd
    for a in S:
        # a | x * c  for some c in S?
        # a | x*c means: let g = gcd(a, x), then (a/g) | c
        g = gcd(a, x)
        rem = a // g
        if rem == 1:
            # a/g = 1, so a | x, meaning a divides x. Then for ANY c, a | x*c.
            # Violation if there exists c in S (any c, since a, x, c must be distinct)
            # a and x are distinct (x not in S), so need c in S, c != a.
            if len(S) >= 2:  # there's some c != a
                return False
        else:
            # Need some c in S such that rem | c
            for c in S:
                if c != a and c % rem == 0:
                    return False
    # Check x | a*b for any a, b in S, a != b
    for a in S:
        for b in S:
            if a == b:
                continue
            if (a * b) % x == 0:
                return False
    return True

def _conflict_count(S, x):
    """Count how many triples would violate if x were in S."""
    from math import gcd
    count = 0
    for a in S:
        # a | x * c for some c in S?
        g = gcd(a, x)
        rem = a // g
        if rem == 1:
            if len(S) >= 2:
                count += max(0, len(S) - 1)  # all c != a
        else:
            for c in S:
                if c != a and c % rem == 0:
                    count += 1
    # x | a*b for any a, b in S, a != b
    for a in S:
        for b in S:
            if a == b:
                continue
            if (a * b) % x == 0:
                count += 1
    return count

def _most_conflicting(S):
    """Find the element in S that causes the most conflicts if removed from consideration."""
    # For each element, count how many other pairs (a,b) it conflicts with
    # i.e., for element x, count triples where x is the 'a' that divides b*c
    from math import gcd
    conflicts = {}
    S_list = list(S)
    for x in S_list:
        cnt = 0
        for a in S_list:
            if a == x:
                continue
            # Check if x | a * b for some b in S, b != a, b != x
            for b in S_list:
                if b == a or b == x:
                    continue
                if (a * b) % x == 0:
                    cnt += 1
            # Also check if a | x * b for some b
            g = gcd(a, x)
            rem = a // g
            if rem == 1:
                # a | x, so a | x*b for any b != a
                for b in S_list:
                    if b != a and b != x:
                        cnt += 1
            else:
                for b in S_list:
                    if b != a and b != x and b % rem == 0:
                        cnt += 1
        conflicts[x] = cnt
    if not conflicts:
        return None
    return max(conflicts, key=conflicts.get)

def construct():
    start_time = time.time()
    
    # Start with all primes in 2..256
    from math import isqrt, gcd
    
    def is_prime(n):
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
    
    primes = [p for p in range(2, 257) if is_prime(p)]
    composites = [x for x in range(2, 257) if not is_prime(x)]
    
    # Strategy: Try multiple greedy orderings and local search
    best_set = set(primes)
    best_size = len(primes)
    
    def try_greedy(candidate_order):
        """Greedily add elements in given order."""
        S = set()
        for x in candidate_order:
            if _fast_check_add(S, x):
                S.add(x)
        return S
    
    # Try different orderings
    orderings = [
        primes + composites,          # primes first
        composites + primes,          # composites first
        sorted(range(2, 257), key=lambda x: -x),  # descending
        sorted(range(2, 257), key=lambda x: x),   # ascending
    ]
    
    for ordering in orderings:
        S = try_greedy(ordering)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
        if time.time() - start_time > DEADLINE - 10:
            break
    
    # Local search: try to add composites to best set
    S = set(best_set)
    for x in composites:
        if x not in S and _fast_check_add(S, x):
            S.add(x)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
    
    # Improved local search: try removing the most conflicting elements and adding better ones
    if time.time() - start_time < DEADLINE - 5:
        S = set(best_set)
        # Try 1-for-1 swaps, prioritizing removal of high-conflict elements
        max_swaps = 50
        for _swap_iter in range(max_swaps):
            if time.time() - start_time > DEADLINE - 5:
                break
            improved = False
            # Find most conflicting element to try removing
            candidates_to_remove = list(S)
            # Sort by conflict count (most conflicts first)
            candidates_to_remove.sort(key=lambda x: -_conflict_count(S - {x}, x))
            
            for a in candidates_to_remove:
                if time.time() - start_time > DEADLINE - 5:
                    break
                S_without_a = S - {a}
                # Try adding each composite not in S
                for x in composites:
                    if x in S:
                        continue
                    if _fast_check_add(S_without_a, x):
                        S = S_without_a
                        S.add(x)
                        improved = True
                        best_set = set(S)
                        best_size = len(S)
                        break
                if improved:
                    break
            if not improved:
                break
        
        # Try 2-for-2 swaps: remove two, add two
        if time.time() - start_time < DEADLINE - 5:
            S = set(best_set)
            S_list = list(S)
            for i in range(len(S_list)):
                if time.time() - start_time > DEADLINE - 5:
                    break
                a1 = S_list[i]
                for j in range(i+1, len(S_list)):
                    if time.time() - start_time > DEADLINE - 5:
                        break
                    a2 = S_list[j]
                    S_without = S - {a1, a2}
                    # Try adding two composites
                    added = []
                    for x1 in composites:
                        if x1 in S:
                            continue
                        if _fast_check_add(S_without, x1):
                            added.append(x1)
                            S2 = S_without | {x1}
                            for x2 in composites:
                                if x2 in S or x2 == x1:
                                    continue
                                if _fast_check_add(S2, x2):
                                    S = S2
                                    S.add(x2)
                                    best_set = set(S)
                                    best_size = len(S)
                                    break
                            break
    
    # Random restarts with perturbation
    if time.time() - start_time < DEADLINE - 10:
        rng = random.Random(42)
        num_restarts = 300
        for trial in range(num_restarts):
            if time.time() - start_time > DEADLINE - 10:
                break
            # Start from a random subset of primes + random composites
            start_primes = [p for p in primes if rng.random() < 0.8]
            start_comps = [c for c in composites if rng.random() < 0.2]
            ordering = start_primes + start_comps + [x for x in range(2, 257) if x not in start_primes and x not in start_comps]
            rng.shuffle(ordering)
            S = try_greedy(ordering)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S
                # Do a quick local search on this new best
                if time.time() - start_time < DEADLINE - 10:
                    S2 = set(S)
                    for _ in range(20):
                        improved = False
                        candidates = list(S2)
                        candidates.sort(key=lambda x: -_conflict_count(S2 - {x}, x))
                        for a in candidates:
                            S_without_a = S2 - {a}
                            for x in composites:
                                if x in S2:
                                    continue
                                if _fast_check_add(S_without_a, x):
                                    S2 = S_without_a
                                    S2.add(x)
                                    improved = True
                                    break
                            if improved:
                                break
                        if not improved:
                            break
                    if len(S2) > best_size:
                        best_size = len(S2)
                        best_set = S2
    
    # Final: ensure we have at least the primes
    if len(best_set) < best_size:
        best_set = set(primes)
    
    # Verify
    if not _is_valid_set(list(best_set)):
        # Fallback to primes
        best_set = set(primes)
    
    result = list(best_set)
    # Ensure all are Python ints
    result = [int(x) for x in result]
    return result
# EVOLVE-BLOCK-END
