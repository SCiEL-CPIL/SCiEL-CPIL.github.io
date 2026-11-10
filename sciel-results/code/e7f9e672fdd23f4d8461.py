def construct():
    import time
    import random
    from math import gcd

    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    n = 128
    all_elements = list(range(1, n + 1))

    # Precompute GCD table
    gcd_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            gcd_table[i][j] = gcd(i, j)

    # Precompute bad pairs for each x:
    # bad_pairs_for_x[x] = set of frozenset({a, b}) such that (a, b, x) is a bad triple
    # i.e., gcd(a,b) == gcd(a,x) == gcd(b,x)
    bad_pairs_for_x = [set() for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            g_ab = gcd_table[a][b]
            for c in range(b + 1, n + 1):
                g_ac = gcd_table[a][c]
                g_bc = gcd_table[b][c]
                if g_ab == g_ac == g_bc:
                    bad_pairs_for_x[c].add(frozenset((a, b)))
                    bad_pairs_for_x[b].add(frozenset((a, c)))
                    bad_pairs_for_x[a].add(frozenset((b, c)))

    def can_add(x, S_set, S_pairs):
        """Check if x can be added to S (represented as set and set of frozenset pairs)."""
        for bp in bad_pairs_for_x[x]:
            if bp in S_pairs:
                return False
        return True

    def make_pairs(S):
        """Create set of frozenset pairs from list S."""
        pairs = set()
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                pairs.add(frozenset((S_list[i], S_list[j])))
        return pairs

    def greedy_build(order, S_set=None, S_pairs=None):
        """Greedily build a set by adding elements in the given order."""
        if S_set is None:
            S_set = set()
        if S_pairs is None:
            S_pairs = set()
        for x in order:
            if x not in S_set and can_add(x, S_set, S_pairs):
                S_set.add(x)
                # Update pairs
                for y in S_set:
                    if y != x:
                        S_pairs.add(frozenset((x, y)))
        return S_set, S_pairs

    def hill_climb(S_set, S_pairs):
        """Try to add any element that can be added, repeatedly."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in range(1, n + 1):
                if x not in S_set and can_add(x, S_set, S_pairs):
                    S_set.add(x)
                    for y in S_set:
                        if y != x:
                            S_pairs.add(frozenset((x, y)))
                    improved = True
        return S_set, S_pairs

    def try_swap(S_set, S_pairs, x_out, x_in):
        """Try to swap x_out for x_in. Returns (new_S_set, new_S_pairs, success)."""
        if x_in in S_set:
            return S_set, S_pairs, False
        # Check if x_in can be added to S_set - {x_out}
        S_temp = S_set - {x_out}
        P_temp = set()
        S_list = list(S_temp)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                P_temp.add(frozenset((S_list[i], S_list[j])))
        if can_add(x_in, S_temp, P_temp):
            # Success: perform the swap
            S_new = S_temp | {x_in}
            P_new = set()
            S_new_list = list(S_new)
            for i in range(len(S_new_list)):
                for j in range(i + 1, len(S_new_list)):
                    P_new.add(frozenset((S_new_list[i], S_new_list[j])))
            return S_new, P_new, True
        return S_set, S_pairs, False

    def local_search(S_set, S_pairs, max_iter=500):
        """Local search with swaps and additions."""
        best_S = S_set
        best_P = S_pairs
        best_size = len(S_set)
        
        for _ in range(max_iter):
            if time.time() >= deadline:
                break
            
            # Try to add elements
            S_set, S_pairs = hill_climb(S_set, S_pairs)
            if len(S_set) > best_size:
                best_S = set(S_set)
                best_P = set(S_pairs)
                best_size = len(S_set)
            
            # Try swaps
            improved = False
            S_list = list(S_set)
            random.shuffle(S_list)
            for x_out in S_list:
                if time.time() >= deadline:
                    break
                # Try to find a good x_in
                candidates = [x for x in range(1, n + 1) if x not in S_set]
                random.shuffle(candidates)
                for x_in in candidates:
                    S_new, P_new, success = try_swap(S_set, S_pairs, x_out, x_in)
                    if success:
                        S_set = S_new
                        S_pairs = P_new
                        improved = True
                        # After a successful swap, try to add more
                        S_set, S_pairs = hill_climb(S_set, S_pairs)
                        if len(S_set) > best_size:
                            best_S = set(S_set)
                            best_P = set(S_pairs)
                            best_size = len(S_set)
                        break
            if not improved:
                break
        
        return best_S, best_P

    # Initial heuristic orders
    rng = random.Random(42)
    orders = [
        all_elements,
        list(reversed(all_elements)),
        [x for x in all_elements if x % 2 == 0],
        [x for x in all_elements if x % 2 == 1],
        sorted(all_elements, key=lambda x: -x.bit_count()),
        sorted(all_elements, key=lambda x: x.bit_count()),
        sorted(all_elements, key=lambda x: len([d for d in range(2, x + 1) if x % d == 0])),
        sorted(all_elements, key=lambda x: -len([d for d in range(2, x + 1) if x % d == 0])),
        rng.sample(all_elements, n),
        rng.sample(all_elements, n),
    ]

    best_S = set()
    best_P = set()
    best_size = 0

    for order in orders:
        if time.time() >= deadline:
            break
        S_set, S_pairs = greedy_build(order)
        S_set, S_pairs = local_search(S_set, S_pairs, max_iter=200)
        if len(S_set) > best_size:
            best_S = S_set
            best_P = S_pairs
            best_size = len(S_set)

    # More random restarts with kicks
    for _ in range(300):
        if time.time() >= deadline:
            break
        
        if best_S:
            # Kick: remove some random elements
            kick_size = min(len(best_S), rng.randint(3, 10))
            to_remove = set(rng.sample(list(best_S), kick_size))
            S_set = best_S - to_remove
            S_pairs = set()
            S_list = list(S_set)
            for i in range(len(S_list)):
                for j in range(i + 1, len(S_list)):
                    S_pairs.add(frozenset((S_list[i], S_list[j])))
            
            S_set, S_pairs = local_search(S_set, S_pairs, max_iter=300)
            if len(S_set) > best_size:
                best_S = S_set
                best_P = S_pairs
                best_size = len(S_set)
        else:
            order = rng.sample(all_elements, n)
            S_set, S_pairs = greedy_build(order)
            S_set, S_pairs = local_search(S_set, S_pairs, max_iter=300)
            if len(S_set) > best_size:
                best_S = S_set
                best_P = S_pairs
                best_size = len(S_set)

    return list(best_S)
