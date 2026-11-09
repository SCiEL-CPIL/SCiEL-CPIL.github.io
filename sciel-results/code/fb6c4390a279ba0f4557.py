"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def _precompute_conflicts(N=128):
    """
    For each number x in [1,N], compute the set of pairs (y,z) such that
    {x,y,z} is a bad triple (all three pairwise gcds equal).
    We store conflicts[x] as a set of frozenset({y,z}) for efficient lookup.
    """
    conflicts = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                gab = gcd(a, b)
                gbc = gcd(b, c)
                gca = gcd(a, c)
                if gab == gbc == gca:
                    # {a,b,c} is a bad triple
                    conflicts[a].add(frozenset((b, c)))
                    conflicts[b].add(frozenset((a, c)))
                    conflicts[c].add(frozenset((a, b)))
    return conflicts

def _can_add(x, selected_set, conflicts):
    """Check if x can be added to selected_set without creating a bad triple."""
    # For each pair (a,b) in selected_set, check if {x,a,b} is bad
    # i.e., check if frozenset({a,b}) is in conflicts[x]
    for a in selected_set:
        for b in selected_set:
            if a == b:
                continue
            if frozenset((a, b)) in conflicts[x]:
                return False
    return True

def _greedy_build(order, conflicts, deadline):
    """Greedily build a valid set following the given order."""
    selected = []
    selected_set = set()
    for x in order:
        if time.time() > deadline:
            break
        if _can_add(x, selected_set, conflicts):
            selected.append(x)
            selected_set.add(x)
    return selected

def _local_search(selected, conflicts, deadline, max_iters=200):
    """Try to improve by swapping elements or adding missing ones."""
    selected_set = set(selected)
    best = list(selected)
    best_size = len(best)
    
    # Try to add elements not in the set
    all_numbers = set(range(1, 129))
    missing = all_numbers - selected_set
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        improved = False
        # Try adding a missing element
        for x in missing:
            if _can_add(x, selected_set, conflicts):
                selected_set.add(x)
                best = list(selected_set)
                best_size = len(best)
                missing.discard(x)
                improved = True
                break
        if improved:
            continue
        
        # Try removing one element and adding two others
        for x in list(selected_set):
            if time.time() > deadline:
                break
            trial_set = selected_set - {x}
            # Try to add two missing elements
            added = []
            for y in missing:
                if _can_add(y, trial_set, conflicts):
                    trial_set.add(y)
                    added.append(y)
                    break
            if added:
                for z in missing:
                    if z in trial_set:
                        continue
                    if _can_add(z, trial_set, conflicts):
                        trial_set.add(z)
                        added.append(z)
                        break
            if len(added) >= 2 and len(trial_set) > best_size:
                selected_set = trial_set
                best = list(selected_set)
                best_size = len(best)
                missing = all_numbers - selected_set
                improved = True
                break
            elif len(added) == 1 and len(trial_set) > best_size:
                selected_set = trial_set
                best = list(selected_set)
                best_size = len(best)
                missing = all_numbers - selected_set
                improved = True
                break
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin
    
    # Precompute conflicts
    conflicts = _precompute_conflicts(128)
    
    # Try multiple greedy orderings
    best_set = []
    
    # Ordering 1: natural order
    order1 = list(range(1, 129))
    s1 = _greedy_build(order1, conflicts, deadline)
    if len(s1) > len(best_set):
        best_set = s1
    
    # Ordering 2: reverse
    if time.time() < deadline:
        order2 = list(range(128, 0, -1))
        s2 = _greedy_build(order2, conflicts, deadline)
        if len(s2) > len(best_set):
            best_set = s2
    
    # Ordering 3: primes first, then composites
    if time.time() < deadline:
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True
        primes = [x for x in range(1, 129) if is_prime(x)]
        composites = [x for x in range(1, 129) if not is_prime(x)]
        order3 = primes + composites
        s3 = _greedy_build(order3, conflicts, deadline)
        if len(s3) > len(best_set):
            best_set = s3
    
    # Ordering 4: by number of prime factors (fewer first)
    if time.time() < deadline:
        def num_prime_factors(n):
            count = 0
            d = 2
            while d * d <= n:
                while n % d == 0:
                    count += 1
                    n //= d
                d += 1
            if n > 1:
                count += 1
            return count
        order4 = sorted(range(1, 129), key=lambda x: (num_prime_factors(x), x))
        s4 = _greedy_build(order4, conflicts, deadline)
        if len(s4) > len(best_set):
            best_set = s4
    
    # Local search on the best
    if time.time() < deadline:
        best_set = _local_search(best_set, conflicts, deadline, max_iters=50)
    
    # Verify validity
    def is_valid(s):
        s = set(s)
        for a, b, c in combinations(sorted(s), 3):
            if gcd(a, b) == gcd(b, c) == gcd(c, a):
                return False
        return True
    
    if not is_valid(best_set):
        # Fallback: simple greedy
        selected = []
        for x in range(1, 129):
            if all(len({gcd(a, b), gcd(a, x), gcd(b, x)}) > 1 for a, b in combinations(selected, 2)):
                selected.append(x)
        return selected
    
    return best_set
# EVOLVE-BLOCK-END
