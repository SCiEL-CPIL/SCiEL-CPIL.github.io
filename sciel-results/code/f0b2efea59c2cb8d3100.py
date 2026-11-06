import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    limit = 512
    deadline = start_time + 85  # safety margin from 90s kill
    n = limit

    # Precompute LCM compatibility matrix
    compat = [[False] * (n + 1) for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a, n + 1):
            l = (a * b) // gcd(a, b)
            c = l <= limit
            compat[a][b] = c
            compat[b][a] = c

    # Precompute for each x, the set of y compatible with x
    compat_set = [set() for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(1, n + 1):
            if compat[a][b]:
                compat_set[a].add(b)

    # Precompute bitmask of compatible elements for fast intersection
    # Using Python ints as bitmasks (512 bits fits in Python int)
    compat_mask = [0] * (n + 1)
    for a in range(1, n + 1):
        mask = 0
        for b in range(1, n + 1):
            if compat[a][b]:
                mask |= (1 << b)
        compat_mask[a] = mask

    def is_valid_add(S, x):
        """Check if x can be added to set S using bitmask intersection."""
        # Convert S to bitmask
        S_mask = 0
        for y in S:
            S_mask |= (1 << y)
        # x is compatible with all in S iff (S_mask & ~compat_mask[x]) == 0
        return (S_mask & ~compat_mask[x]) == 0

    def complete(S):
        """Add all compatible elements to S."""
        S = set(S)
        changed = True
        while changed:
            changed = False
            for x in range(1, n + 1):
                if time.time() > deadline - 3:
                    return S
                if x not in S and is_valid_add(S, x):
                    S.add(x)
                    changed = True
        return S

    best_set = set()
    best_size = 0

    # Better seeding strategy: start with small numbers and greedily add
    def greedy_seed(start_elements):
        """Build a seed by starting with given elements and greedily adding compatible ones."""
        S = set(start_elements)
        # Sort remaining by degree (number of compatible elements) descending
        remaining = list(range(1, n + 1))
        remaining.sort(key=lambda x: -len(compat_set[x]))
        for x in remaining:
            if time.time() > deadline - 10:
                break
            if x not in S and is_valid_add(S, x):
                S.add(x)
        return S

    # Multiple diverse seeds
    seed_strategies = [
        # Start with 1 and greedily add
        [1],
        # Start with small primes
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        # Start with powers of 2
        [2, 4, 8, 16, 32, 64, 128, 256],
        # Start with divisors of 240
        [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 30, 40, 48, 60, 80, 120, 240],
        # Start with divisors of 360
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24, 30, 36, 40, 45, 60, 72, 90, 120, 180, 360],
        # Start with numbers up to 16
        list(range(1, 17)),
        # Start with numbers up to 12
        list(range(1, 13)),
    ]

    for seed_elements in seed_strategies:
        if time.time() > deadline - 15:
            break
        S = greedy_seed(seed_elements)
        S = complete(S)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    # Local search: try removing one element and adding the best replacement
    def local_search(S):
        nonlocal best_set, best_size
        S = set(S)
        S = complete(S)
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
        
        improved = True
        while improved and time.time() < deadline - 8:
            improved = False
            # Try 1-for-1 swaps: remove e, add best x
            for e in list(S):
                if time.time() > deadline - 8:
                    break
                S_minus = S - {e}
                # Find all x that can be added to S_minus
                candidates = []
                for x in range(1, n + 1):
                    if x != e and x not in S and is_valid_add(S_minus, x):
                        candidates.append(x)
                
                if not candidates:
                    continue
                
                # Score each candidate by how many more elements it allows
                best_x = None
                best_score = -1
                for x in candidates:
                    if time.time() > deadline - 8:
                        break
                    S_test = S_minus | {x}
                    # Count how many additional elements can be added
                    score = 0
                    for y in range(1, n + 1):
                        if y not in S_test and is_valid_add(S_test, y):
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_x = x
                
                if best_x is not None and best_score > 0:
                    S2 = complete(S_minus | {best_x})
                    if len(S2) > len(S):
                        S = S2
                        improved = True
                        if len(S) > best_size:
                            best_size = len(S)
                            best_set = set(S)
        
        return S

    # Run local search from best found
    if time.time() < deadline - 5:
        local_search(set(best_set))

    # Try a few more diverse starting points with local search
    import random
    rng = random.Random(12345)
    for _ in range(20):
        if time.time() > deadline - 5:
            break
        # Random seed: pick some small numbers randomly
        seed = {1}
        for x in range(2, 30):
            if rng.random() < 0.3:
                if is_valid_add(seed, x):
                    seed.add(x)
        S = greedy_seed(list(seed))
        S = local_search(S)
        if len(S) > len(best_set):
            best_set = S
            best_size = len(S)

    # Final polish
    if time.time() < deadline - 2:
        S = complete(set(best_set))
        if len(S) > len(best_set):
            best_set = S

    return sorted(best_set)
