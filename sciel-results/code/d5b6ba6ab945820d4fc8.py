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

    def is_valid_add(S, x):
        # Fast check: x must be compatible with all elements in S
        for y in S:
            if not compat[x][y]:
                return False
        return True

    def find_addable(S):
        """Return set of all x not in S that can be added to S."""
        addable = set()
        for x in range(1, n + 1):
            if x not in S and is_valid_add(S, x):
                addable.add(x)
        return addable

    def complete(S):
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

    def complete_fast(S):
        """Faster completion by finding all addable elements at once."""
        S = set(S)
        changed = True
        while changed:
            changed = False
            addable = find_addable(S)
            if not addable:
                break
            # Add all addable elements (they're all compatible with S, but may not be with each other)
            # So we need to be careful - add them one by one
            for x in sorted(addable):
                if time.time() > deadline - 3:
                    return S
                if x not in S and is_valid_add(S, x):
                    S.add(x)
                    changed = True
        return S

    def upper_bound(S, candidates):
        """Upper bound on clique size using greedy coloring on candidate subgraph."""
        if not candidates:
            return 0
        # Greedy coloring to get upper bound
        color = {}
        num_colors = 0
        for x in sorted(candidates):
            used_colors = set()
            for y in candidates:
                if y != x and y in color and compat[x][y]:
                    used_colors.add(color[y])
            for c in range(1, num_colors + 1):
                if c not in used_colors:
                    color[x] = c
                    break
            else:
                num_colors += 1
                color[x] = num_colors
        return num_colors

    def branch_and_bound(S, candidates, best_size, deadline):
        """Branch and bound for maximum clique."""
        nonlocal best_set
        if time.time() > deadline:
            return
        S = set(S)
        if len(S) + len(candidates) <= best_size:
            return
        if not candidates:
            if len(S) > best_size:
                best_size = len(S)
                best_set = set(S)
            return
        # Order candidates by degree in candidate subgraph (descending)
        cand_list = list(candidates)
        cand_list.sort(key=lambda x: -len(compat_set[x] & set(candidates)))
        
        for x in cand_list:
            if time.time() > deadline:
                return
            new_S = S | {x}
            new_candidates = candidates & compat_set[x] - {x}
            if len(new_S) + len(new_candidates) <= best_size:
                continue
            branch_and_bound(new_S, new_candidates, best_size, deadline)
            candidates = candidates - {x}
            if len(S) + len(candidates) <= best_size:
                return

    best_set = set()
    best_size = 0

    # Key structural seed: divisors of 504 (most highly composite number <= 512)
    # 504 = 2^3 * 3^2 * 7^1, has 24 divisors, and lcm of any two divisors divides 504 <= 512
    def get_divisors(m):
        divs = set()
        for i in range(1, int(m**0.5) + 1):
            if m % i == 0:
                divs.add(i)
                divs.add(m // i)
        return divs

    # Try divisors of various highly composite numbers, prioritizing 504
    hc_numbers = [504, 480, 420, 360, 336, 288, 240, 210, 180, 168, 144, 120, 96, 72, 60, 48, 36, 24, 12, 6, 4, 2]
    for m in hc_numbers:
        if time.time() > deadline - 15:
            break
        divs = get_divisors(m)
        S = complete(divs)
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)

    # Also try ranges as additional seeds
    for r in [16, 12, 10, 8]:
        if time.time() > deadline - 15:
            break
        S = complete(set(range(1, r + 1)))
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)

    # Branch and bound from best seed
    if time.time() < deadline - 10:
        # Use B&B on the complement of best_set
        remaining = set(range(1, n + 1)) - best_set
        # Filter to only those compatible with all of best_set
        candidates = set()
        for x in remaining:
            if all(compat[x][y] for y in best_set):
                candidates.add(x)
        branch_and_bound(set(best_set), candidates, best_size, deadline - 5)

    # Local search with 3-for-3 swaps
    def local_search(S):
        nonlocal best_set, best_size
        S = set(S)
        S = complete_fast(S)
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
        improved = True
        while improved and time.time() < deadline - 8:
            improved = False
            # 1-for-1 swaps
            for e in list(S):
                if time.time() > deadline - 8:
                    break
                S_minus = S - {e}
                for x in range(1, n + 1):
                    if time.time() > deadline - 8:
                        break
                    if x != e and x not in S and is_valid_add(S_minus, x):
                        S2 = complete_fast(S_minus | {x})
                        if len(S2) > len(S):
                            S = S2
                            improved = True
                            if len(S) > best_size:
                                best_size = len(S)
                                best_set = set(S)
                        break
            # 1-for-2 swaps
            for e in list(S):
                if time.time() > deadline - 8:
                    break
                S_minus = S - {e}
                added = 0
                S2 = set(S_minus)
                for x in range(1, n + 1):
                    if time.time() > deadline - 8:
                        break
                    if x not in S2 and is_valid_add(S2, x):
                        S2.add(x)
                        added += 1
                        if added >= 2:
                            break
                if len(S2) > len(S):
                    S = complete_fast(S2)
                    improved = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_set = set(S)
            # 2-for-2 swaps
            for e1 in list(S):
                if time.time() > deadline - 8:
                    break
                for e2 in list(S):
                    if time.time() > deadline - 8:
                        break
                    if e1 >= e2:
                        continue
                    S_minus = S - {e1, e2}
                    S2 = set(S_minus)
                    added = 0
                    for x in range(1, n + 1):
                        if time.time() > deadline - 8:
                            break
                        if x not in S2 and is_valid_add(S2, x):
                            S2.add(x)
                            added += 1
                            if added >= 2:
                                break
                    if len(S2) > len(S):
                        S = complete_fast(S2)
                        improved = True
                        if len(S) > best_size:
                            best_size = len(S)
                            best_set = set(S)
            # 3-for-3 swaps
            if time.time() < deadline - 15:
                for e1, e2, e3 in combinations(list(S), 3):
                    if time.time() > deadline - 8:
                        break
                    S_minus = S - {e1, e2, e3}
                    S2 = set(S_minus)
                    added = 0
                    for x in range(1, n + 1):
                        if time.time() > deadline - 8:
                            break
                        if x not in S2 and is_valid_add(S2, x):
                            S2.add(x)
                            added += 1
                            if added >= 3:
                                break
                    if len(S2) > len(S):
                        S = complete_fast(S2)
                        improved = True
                        if len(S) > best_size:
                            best_size = len(S)
                            best_set = set(S)
        return S

    # Run local search from best found so far
    if time.time() < deadline - 5:
        local_search(set(best_set))

    # Random restarts with better seed generation
    import random
    rng = random.Random(42)
    for _ in range(30):
        if time.time() > deadline - 5:
            break
        # Generate random seed based on divisors of a random highly composite number
        m = rng.choice(hc_numbers)
        divs = get_divisors(m)
        # Randomly remove some elements to create variation
        seed = set()
        for d in divs:
            if rng.random() < 0.7:
                seed.add(d)
        if not seed:
            seed = {1}
        S = local_search(complete_fast(seed))
        if len(S) > len(best_set):
            best_set = S

    return sorted(best_set)
