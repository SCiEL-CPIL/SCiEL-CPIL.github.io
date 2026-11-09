"""
Maximize |S| for S ⊆ [1,512] with lcm(a,b) <= 512 for all a,b in S.

Maximum clique on the compatibility graph (edge iff lcm(i,j) <= 512).

Strategy:
1. Precompute adjacency bitmasks.
2. Greedy + fast local search to get a strong initial bound.
3. Tomita's maximum clique with corrected color-bound pruning for exact search.
"""

def construct():
    import math
    import time
    import random
    from math import gcd

    t0 = time.time()
    DEADLINE = t0 + 87  # 87s safety margin (limit is 90s)
    LIMIT = 512
    N = LIMIT

    def time_left():
        return DEADLINE - time.time()

    # Precompute adjacency bitmasks
    adj = [0] * (N + 1)
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            g = gcd(i, j)
            if i * j // g <= LIMIT:
                mask |= (1 << (j - 1))
        adj[i] = mask

    def popcount(m):
        return m.bit_count()

    def list_to_mask(lst):
        mask = 0
        for x in lst:
            mask |= (1 << (x - 1))
        return mask

    # Precompute compatible-with-all check: for a given mask, which nodes are compatible with all in mask
    # This is: adj[x] & mask == mask

    # Fast greedy clique
    def greedy_clique(order):
        clique = []
        clique_mask = 0
        for node in order:
            if time_left() < 3:
                break
            if (adj[node] & clique_mask) == clique_mask:
                clique.append(node)
                clique_mask |= (1 << (node - 1))
        return clique

    # Fast local improvement using bitmask operations
    def local_improve(clique):
        S = list(clique)
        S_set = set(S)
        S_mask = list_to_mask(S)
        improved = True

        while improved and time_left() > 10:
            improved = False

            # Try adding any compatible node (pick one with max degree among compatible)
            best_add = -1
            best_deg = -1
            for x in range(1, N + 1):
                if x in S_set:
                    continue
                if (adj[x] & S_mask) == S_mask:
                    deg = popcount(adj[x])
                    if deg > best_deg:
                        best_deg = deg
                        best_add = x

            if best_add > 0:
                S.append(best_add)
                S_set.add(best_add)
                S_mask |= (1 << (best_add - 1))
                improved = True

            # Try 1-for-2 swap
            if time_left() > 10:
                for remove_node in list(S):
                    if time_left() < 10:
                        break
                    S_trial_mask = S_mask & ~(1 << (remove_node - 1))

                    # Find up to 2 nodes compatible with S_trial_mask and each other
                    added = []
                    for x in range(1, N + 1):
                        if len(added) >= 2:
                            break
                        if x == remove_node:
                            continue
                        if (S_trial_mask >> (x - 1)) & 1:
                            continue
                        if (adj[x] & S_trial_mask) == S_trial_mask:
                            if added:
                                if not ((adj[x] >> (added[0] - 1)) & 1):
                                    continue
                            added.append(x)
                            S_trial_mask |= (1 << (x - 1))

                    if len(added) >= 2:
                        S = [x for x in S if x != remove_node] + added
                        S_set = set(S)
                        S_mask = list_to_mask(S)
                        improved = True

        return S

    # Tomita's maximum clique with corrected color bound
    def max_clique_tomita(adj, N, initial_bound=0):
        best = [initial_bound]
        best_clique = [[]]

        def color_sort(candidates_mask):
            """
            Greedy coloring. Returns (list_of_(color,node_idx), max_color).
            The color of each node is an upper bound on the clique size
            in the subgraph induced by nodes with that color or lower.
            """
            uncolored = candidates_mask
            color = 0
            # We'll build an ordered list: process colors from 1 upward,
            # within each color, add nodes. The final order is reversed for search.
            ordered = []  # list of (color, node_idx) in the order they were colored

            while uncolored:
                color += 1
                colored_this = 0
                remaining = uncolored
                while remaining:
                    lb = remaining & (-remaining)
                    node_idx = lb.bit_length() - 1
                    node = node_idx + 1

                    if (adj[node] & colored_this) == 0:
                        colored_this |= lb
                        ordered.append((color, node_idx))
                        remaining ^= lb

                uncolored &= ~colored_this

            # Reverse so highest color is processed first
            ordered.reverse()
            return ordered, color

        def expand(clique, candidates_mask):
            if time_left() < 1.5:
                return

            if not candidates_mask:
                if len(clique) > best[0]:
                    best[0] = len(clique)
                    best_clique[0] = clique[:]
                return

            ordered, max_color = color_sort(candidates_mask)

            # Process from the end (highest color first)
            i = len(ordered) - 1
            while i >= 0:
                if time_left() < 1.5:
                    return

                color, node_idx = ordered[i]
                node = node_idx + 1

                # Pruning: color is an upper bound on additional clique size
                # from nodes at position i and below (lower colors)
                if len(clique) + color <= best[0]:
                    return

                # Add node
                new_clique = clique + [node]
                new_candidates = candidates_mask & adj[node] & ~(1 << node_idx)

                expand(new_clique, new_candidates)

                # Remove node from candidates
                candidates_mask &= ~(1 << node_idx)
                i -= 1

        full_mask = (1 << N) - 1
        expand([], full_mask)

        return best[0], best_clique[0]

    # Phase 1: Greedy from various orderings to get strong initial bound
    best_clique = []
    best_size = 0

    degrees = [(popcount(adj[i]), i) for i in range(1, N + 1)]
    degrees.sort(reverse=True)
    order_high = [i for _, i in degrees]

    S = greedy_clique(order_high)
    S = local_improve(S)
    if len(S) > best_size:
        best_size = len(S)
        best_clique = S[:]

    order_asc = list(range(1, N + 1))
    S = greedy_clique(order_asc)
    S = local_improve(S)
    if len(S) > best_size:
        best_size = len(S)
        best_clique = S[:]

    # Phase 2: Random restarts (limited, to save time for exact solver)
    random.seed(42)
    n_restarts = 80
    for restart in range(n_restarts):
        if time_left() < 20:
            break

        order = list(range(1, N + 1))
        if restart % 4 == 0:
            random.shuffle(order)
        elif restart % 4 == 1:
            order = order_high[:]
            split = N // 3
            random.shuffle(order[split:])
        elif restart % 4 == 2:
            order_low = [i for _, i in sorted(degrees)]
            order = order_low[:]
            split = N // 3
            random.shuffle(order[:split])
        else:
            random.shuffle(order)

        S = greedy_clique(order)
        S = local_improve(S)
        if len(S) > best_size:
            best_size = len(S)
            best_clique = S[:]

    # Phase 3: Tomita's max clique with strong initial bound
    if time_left() > 15:
        try:
            size, clique = max_clique_tomita(adj, N, initial_bound=best_size)
            if size > best_size:
                best_size = size
                best_clique = clique[:]
        except Exception:
            pass

    # Phase 4: If time remains, try more local search from best
    if time_left() > 8 and best_clique:
        S = list(best_clique)
        S_set = set(S)
        S_mask = list_to_mask(S)

        for remove_node in list(S):
            if time_left() < 5:
                break

            S_trial = [x for x in S if x != remove_node]
            S_trial_set = set(S_trial)
            S_trial_mask = list_to_mask(S_trial)

            changed = True
            while changed and time_left() > 5:
                changed = False
                best_add = -1
                best_deg = -1
                for x in range(1, N + 1):
                    if x in S_trial_set:
                        continue
                    if (adj[x] & S_trial_mask) == S_trial_mask:
                        deg = popcount(adj[x])
                        if deg > best_deg:
                            best_deg = deg
                            best_add = x
                if best_add > 0:
                    S_trial.append(best_add)
                    S_trial_set.add(best_add)
                    S_trial_mask |= (1 << (best_add - 1))
                    changed = True

            if len(S_trial) > len(S):
                S = S_trial
                S_set = S_trial_set
                S_mask = S_trial_mask
                best_clique = S[:]
                best_size = len(S)

    # Final validation
    result = sorted(set(best_clique)) if best_clique else [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    valid = []
    valid_mask = 0
    for a in result:
        if (adj[a] & valid_mask) == valid_mask:
            valid.append(a)
            valid_mask |= (1 << (a - 1))

    if len(valid) < 2:
        valid = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    return valid
