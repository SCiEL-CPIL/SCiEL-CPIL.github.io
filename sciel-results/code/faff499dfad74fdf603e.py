import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    N = 512
    start_time = time.time()
    time_limit = 85.0  # Safety margin before 90s kill

    # 1. Precompute 4-APs
    ap_list = []
    aps_containing = [[] for _ in range(N + 1)]

    for d in range(1, (N - 1) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            d_val = a + 3 * d
            ap_idx = len(ap_list)
            ap_list.append((a, b, c, d_val))
            for v in (a, b, c, d_val):
                aps_containing[v].append(ap_idx)

    num_aps = len(ap_list)

    # Precompute AP membership as a numpy boolean matrix for fast swap checks
    # ap_membership[ap_idx, element] = 1 if element in AP, else 0
    # But this is num_aps x (N+1) which might be large. Let's use a different approach.
    # Instead, precompute for each element, the list of APs it's in (already done)
    # And for swap checking, we need to know if y is in a given AP.
    # Precompute: for each AP, store its 4 elements as a frozenset for O(1) lookup
    ap_elements = [frozenset((a, b, c, d_val)) for (a, b, c, d_val) in ap_list]

    def is_valid_addition(ap_count, x):
        for ap_idx in aps_containing[x]:
            if ap_count[ap_idx] == 3:
                return False
        return True

    def greedy_construct(order):
        S = set()
        ap_count = [0] * num_aps
        for x in order:
            if is_valid_addition(ap_count, x):
                S.add(x)
                for ap_idx in aps_containing[x]:
                    ap_count[ap_idx] += 1
        return S, ap_count

    def local_search(S, ap_count):
        best_S = S.copy()
        best_ap_count = ap_count.copy()
        best_size = len(S)
        all_elements = list(range(1, N + 1))
        not_in_S = [x for x in all_elements if x not in S]

        while time.time() - start_time < time_limit:
            # Phase 1: Try systematic additions
            added = False
            for x in not_in_S:
                if is_valid_addition(ap_count, x):
                    S.add(x)
                    for ap_idx in aps_containing[x]:
                        ap_count[ap_idx] += 1
                    not_in_S.remove(x)
                    added = True
                    if len(S) > best_size:
                        best_S = S.copy()
                        best_ap_count = ap_count.copy()
                        best_size = len(S)

            if added:
                continue  # Try more additions

            # Phase 2: Try swaps - for each x not in S, find if any y in S can be swapped
            swapped = False
            # Build a list of (x, y) candidates to try
            # For efficiency, try random swaps
            for _ in range(200):
                x = random.choice(not_in_S)
                y = random.choice(list(S))

                # Check if S \ {y} + {x} is valid
                valid = True
                for ap_idx in aps_containing[x]:
                    current_count = ap_count[ap_idx]
                    y_in_ap = 1 if y in ap_elements[ap_idx] else 0
                    if current_count - y_in_ap >= 3:
                        valid = False
                        break

                if valid:
                    S.remove(y)
                    for ap_idx in aps_containing[y]:
                        ap_count[ap_idx] -= 1

                    S.add(x)
                    for ap_idx in aps_containing[x]:
                        ap_count[ap_idx] += 1

                    not_in_S.remove(x)
                    not_in_S.append(y)
                    swapped = True
                    break

            if swapped:
                if len(S) > best_size:
                    best_S = S.copy()
                    best_ap_count = ap_count.copy()
                    best_size = len(S)
                continue

            # Phase 3: Try remove-and-refill with a specific element
            if S:
                # Pick the element whose removal frees the most potential additions
                # Simplified: try removing a random element and greedily refill
                y = random.choice(list(S))
                S_old = S.copy()
                ap_count_old = ap_count.copy()

                S.remove(y)
                for ap_idx in aps_containing[y]:
                    ap_count[ap_idx] -= 1

                # Greedily refill from not_in_S (which now includes y)
                for z in list(not_in_S):
                    if is_valid_addition(ap_count, z):
                        S.add(z)
                        for ap_idx in aps_containing[z]:
                            ap_count[ap_idx] += 1
                        not_in_S.remove(z)

                if len(S) > best_size:
                    best_S = S.copy()
                    best_ap_count = ap_count.copy()
                    best_size = len(S)
                else:
                    S = S_old
                    ap_count = ap_count_old
                    not_in_S = [x for x in all_elements if x not in S]

        return best_S, best_ap_count

    def final_polish(S):
        """Systematic final polish: try every element addition, then every swap."""
        ap_count = [0] * num_aps
        for x in S:
            for ap_idx in aps_containing[x]:
                ap_count[ap_idx] += 1

        all_elements = list(range(1, N + 1))
        not_in_S = [x for x in all_elements if x not in S]

        # Repeatedly try all additions until no more can be made
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            for x in list(not_in_S):
                if is_valid_addition(ap_count, x):
                    S.add(x)
                    for ap_idx in aps_containing[x]:
                        ap_count[ap_idx] += 1
                    not_in_S.remove(x)
                    improved = True

        # Try all swaps systematically
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            for x in list(not_in_S):
                for y in list(S):
                    valid = True
                    for ap_idx in aps_containing[x]:
                        current_count = ap_count[ap_idx]
                        y_in_ap = 1 if y in ap_elements[ap_idx] else 0
                        if current_count - y_in_ap >= 3:
                            valid = False
                            break
                    if valid:
                        S.remove(y)
                        for ap_idx in aps_containing[y]:
                            ap_count[ap_idx] -= 1
                        S.add(x)
                        for ap_idx in aps_containing[x]:
                            ap_count[ap_idx] += 1
                        not_in_S.remove(x)
                        not_in_S.append(y)
                        improved = True
                        break
                if improved:
                    break

        return S

    best_S = set()
    best_size = 0

    # Strategy: Use different initial orderings
    # 1. Random shuffles
    # 2. Sorted by some heuristic (e.g., number of APs containing the element)
    # Elements in fewer APs are "safer" to add

    ap_degree = [len(aps_containing[i]) for i in range(N + 1)]

    num_restarts = 0
    max_restarts = 50

    for i in range(max_restarts):
        if time.time() - start_time > time_limit - 10:
            break

        if i % 3 == 0:
            # Random order
            order = list(range(1, N + 1))
            random.shuffle(order)
        elif i % 3 == 1:
            # Sort by AP degree (ascending) - safer elements first
            order = sorted(range(1, N + 1), key=lambda x: ap_degree[x])
            random.shuffle(order)  # Add some randomness
        else:
            # Reverse AP degree - riskier elements first (might block more)
            order = sorted(range(1, N + 1), key=lambda x: -ap_degree[x])
            random.shuffle(order)

        S, ap_count = greedy_construct(order)
        if len(S) > best_size:
            best_S = S
            best_size = len(S)

        # Local search
        if time.time() - start_time < time_limit - 15:
            S_ls, _ = local_search(S, ap_count)
            if len(S_ls) > best_size:
                best_S = S_ls
                best_size = len(S_ls)

        num_restarts += 1

    # Final polish on the best solution
    if time.time() - start_time < time_limit - 5:
        best_S = final_polish(best_S)

    return sorted(best_S)

# EVOLVE-BLOCK-END
