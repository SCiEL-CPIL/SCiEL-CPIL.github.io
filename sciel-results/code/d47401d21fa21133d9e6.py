"""Combined program: bitmask greedy with grouping-based conflict detection and local search."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    n = 16
    k = 4

    # Precompute all 4-subsets as bitmasks
    all_subsets = []
    for row in combinations(range(n), k):
        mask = 0
        for x in row:
            mask |= (1 << x)
        all_subsets.append(mask)

    num_subsets = len(all_subsets)  # C(16,4) = 1820

    # Precompute pair intersections as a flat list for speed
    # pair_int[i][j] = all_subsets[i] & all_subsets[j]
    # Store as a list of lists, but only upper triangle needed
    pair_int = [[0] * num_subsets for _ in range(num_subsets)]
    for i in range(num_subsets):
        mi = all_subsets[i]
        for j in range(i + 1, num_subsets):
            v = mi & all_subsets[j]
            pair_int[i][j] = v
            pair_int[j][i] = v

    def can_add_grouped(mask_new, chosen_indices):
        """
        Check if adding mask_new to chosen creates a sunflower.
        Uses grouping: for each existing set A, compute A ∩ mask_new.
        Group by intersection value. For each group with intersection I,
        check if any pair (A,B) in the group has A ∩ B == I.
        """
        groups = defaultdict(list)
        for idx in chosen_indices:
            inter = all_subsets[idx] & mask_new
            groups[inter].append(idx)

        for inter, members in groups.items():
            if len(members) < 2:
                continue
            # Check all pairs in this group
            L = len(members)
            for a_pos in range(L):
                a = members[a_pos]
                for b_pos in range(a_pos + 1, L):
                    b = members[b_pos]
                    if pair_int[a][b] == inter:
                        return False  # conflict found
        return True  # no conflict

    best_family = []
    best_size = 0

    # Strategy 1: Deterministic Greedy (Lexicographic)
    chosen = []
    for mask_idx in range(num_subsets):
        if time.time() > deadline:
            break
        if can_add_grouped(all_subsets[mask_idx], chosen):
            chosen.append(mask_idx)

    if len(chosen) > best_size:
        best_size = len(chosen)
        best_family = chosen[:]

    # Strategy 2: Randomized Greedy Restarts
    num_restarts = 200
    for restart in range(num_restarts):
        if time.time() > deadline - 3.0:
            break

        random.seed(12345 + restart)
        order = list(range(num_subsets))
        random.shuffle(order)

        chosen = []
        for mask_idx in order:
            if time.time() > deadline - 3.0:
                break
            if can_add_grouped(all_subsets[mask_idx], chosen):
                chosen.append(mask_idx)

        if len(chosen) > best_size:
            best_size = len(chosen)
            best_family = chosen[:]

            if best_size >= 250:
                break

    # Strategy 3: Local search on best family
    if best_size > 0 and time.time() < deadline - 5.0:
        family = best_family[:]
        family_set = set(family)
        improved = True
        max_iterations = 50

        while improved and time.time() < deadline - 5.0 and max_iterations > 0:
            improved = False
            max_iterations -= 1

            # Try adding any set that doesn't conflict
            for idx in range(num_subsets):
                if time.time() > deadline - 5.0:
                    break
                if idx in family_set:
                    continue
                if can_add_grouped(all_subsets[idx], family):
                    family.append(idx)
                    family_set.add(idx)
                    improved = True
                    break

            if improved:
                continue

            # Try swap: remove one, add one
            shuffled_family = family[:]
            random.shuffle(shuffled_family)
            for remove_idx in shuffled_family:
                if time.time() > deadline - 5.0:
                    break

                temp_family = [x for x in family if x != remove_idx]
                temp_set = set(temp_family)

                # Find best candidate to add
                best_add = -1
                for add_idx in range(num_subsets):
                    if add_idx in temp_set:
                        continue
                    if can_add_grouped(all_subsets[add_idx], temp_family):
                        best_add = add_idx
                        break

                if best_add != -1:
                    family = temp_family
                    family.append(best_add)
                    family_set = set(family)
                    improved = True
                    break

        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]

    # Convert back to sorted lists
    result = []
    for idx in best_family:
        mask = all_subsets[idx]
        elems = []
        for i in range(n):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))

    return result
# EVOLVE-BLOCK-END
