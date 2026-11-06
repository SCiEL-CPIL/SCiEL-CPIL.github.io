import time
import random
import numpy as np

def construct():
    start_time = time.time()
    n = 256
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # Pre-compute all 3-APs and 4-APs
    # 3-AP: (a, b, c) where a < b < c and b-a == c-b
    # 4-AP: (a, b, c, d) where a < b < c < d and b-a == c-b == d-c

    # Build lists of APs
    aps_3 = []
    for d in range(1, n // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            if c <= n:
                aps_3.append((a, b, c))

    aps_4 = []
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            d_val = a + 3 * d
            if d_val <= n:
                aps_4.append((a, b, c, d_val))

    num_3 = len(aps_3)
    num_4 = len(aps_4)

    # Map element -> list of AP indices
    elem_to_3aps = {i: [] for i in range(1, n + 1)}
    for idx, ap in enumerate(aps_3):
        for x in ap:
            elem_to_3aps[x].append(idx)

    elem_to_4aps = {i: [] for i in range(1, n + 1)}
    for idx, ap in enumerate(aps_4):
        for x in ap:
            elem_to_4aps[x].append(idx)

    def count_three_aps(selected_set):
        """Count distinct 3-APs in the selected set."""
        count = 0
        for d in range(1, n // 2 + 1):
            for a in range(1, n - 2 * d + 1):
                b = a + d
                c = a + 2 * d
                if a in selected_set and b in selected_set and c in selected_set:
                    count += 1
        return count

    def has_four_ap(selected_set, x):
        """Check if adding x to selected_set would create a 4-AP."""
        for ap_idx in elem_to_4aps[x]:
            ap = aps_4[ap_idx]
            others = [elem for elem in ap if elem != x]
            if all(elem in selected_set for elem in others):
                return True
        return False

    def run_greedy_removal(random_seed=None):
        """Greedy removal: start with full set, remove elements to kill 4-APs while preserving 3-APs."""
        if random_seed is not None:
            random.seed(random_seed)

        in_A = [True] * (n + 1)
        active_3 = [True] * num_3
        active_4 = [True] * num_4

        active_4_count = [0] * (n + 1)
        for idx, ap in enumerate(aps_4):
            for x in ap:
                active_4_count[x] += 1

        active_3_count = [0] * (n + 1)
        for idx, ap in enumerate(aps_3):
            for x in ap:
                active_3_count[x] += 1

        current_total_3 = num_3
        current_active_4s = num_4

        while current_active_4s > 0:
            candidates = []
            for x in range(1, n + 1):
                if in_A[x] and active_4_count[x] > 0:
                    score = active_3_count[x] / active_4_count[x]
                    candidates.append((score, x))

            if not candidates:
                break

            candidates.sort(key=lambda t: t[0])
            top_k = min(5, len(candidates))
            best_x = random.choice(candidates[:top_k])[1]

            in_A[best_x] = False

            for ap_idx in elem_to_3aps[best_x]:
                if active_3[ap_idx]:
                    active_3[ap_idx] = False
                    current_total_3 -= 1
                    ap = aps_3[ap_idx]
                    for other in ap:
                        if other != best_x:
                            active_3_count[other] -= 1

            for ap_idx in elem_to_4aps[best_x]:
                if active_4[ap_idx]:
                    active_4[ap_idx] = False
                    current_active_4s -= 1
                    ap = aps_4[ap_idx]
                    for other in ap:
                        if other != best_x:
                            active_4_count[other] -= 1

        return [i for i in range(1, n + 1) if in_A[i]], current_total_3

    def run_greedy_addition(random_seed=None):
        """Greedy addition: start empty, add elements that maximize 3-APs while avoiding 4-APs."""
        if random_seed is not None:
            random.seed(random_seed)

        selected = set()

        # Order elements by potential 3-AP contribution
        elem_potential = {}
        for x in range(1, n + 1):
            potential = len(elem_to_3aps[x])
            elem_potential[x] = potential

        # Try different ordering strategies
        if random_seed % 3 == 0:
            order = list(range(1, n + 1))
        elif random_seed % 3 == 1:
            order = sorted(range(1, n + 1), key=lambda x: -elem_potential[x])
        else:
            order = list(range(1, n + 1))
            random.shuffle(order)

        for x in order:
            if time.time() > deadline - 2.0:
                break

            # Check if adding x creates a 4-AP
            if has_four_ap(selected, x):
                continue

            # Count new 3-APs created by adding x
            new_3aps = 0
            for ap_idx in elem_to_3aps[x]:
                ap = aps_3[ap_idx]
                others = [elem for elem in ap if elem != x]
                if all(elem in selected for elem in others):
                    new_3aps += 1

            # Only add if it creates new 3-APs or if we're early in the process
            if new_3aps > 0 or len(selected) < n // 4:
                selected.add(x)

        return list(selected), count_three_aps(selected)

    def local_improvement(initial_set, random_seed=None):
        """Local improvement: try to swap elements to improve 3-AP count while maintaining no-4-AP."""
        if random_seed is not None:
            random.seed(random_seed)

        best_set = set(initial_set)
        best_score = count_three_aps(best_set)
        improved = True

        while improved and time.time() < deadline - 2.0:
            improved = False
            for x in list(best_set):
                for y in range(1, n + 1):
                    if y in best_set:
                        continue
                    if time.time() > deadline - 1.0:
                        break

                    # Try swapping x for y
                    new_set = set(best_set)
                    new_set.remove(x)
                    new_set.add(y)

                    # Check 4-AP constraint
                    if has_four_ap(new_set, y):
                        continue

                    new_score = count_three_aps(new_set)
                    if new_score > best_score:
                        best_score = new_score
                        best_set = new_set
                        improved = True
                        break

                if time.time() > deadline - 1.0:
                    break

        return list(best_set), best_score

    best_set = []
    best_score = -1

    # Run multiple strategies
    num_runs = 30
    for i in range(num_runs):
        if time.time() > deadline:
            break

        # Alternate between removal and addition strategies
        if i % 2 == 0:
            s, count = run_greedy_removal(random_seed=i)
        else:
            s, count = run_greedy_addition(random_seed=i)

        if count > best_score:
            best_score = count
            best_set = s

        # Apply local improvement if we have time
        if time.time() < deadline - 10.0:
            s_imp, count_imp = local_improvement(s, random_seed=i)
            if count_imp > best_score:
                best_score = count_imp
                best_set = s_imp

    if not best_set:
        # Fallback to seed if something went wrong or no time
        selected = set()
        for x in range(1, 257):
            if all(not all(x - j * d in selected for j in (1, 2, 3)) for d in range(1, (x - 1) // 3 + 1)):
                selected.add(x)
        best_set = sorted(selected)

    return best_set
