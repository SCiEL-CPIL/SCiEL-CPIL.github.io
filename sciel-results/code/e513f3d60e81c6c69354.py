import time
from math import lcm
from fractions import Fraction
import bisect
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    denominators = list(range(2, 49))
    n = len(denominators)

    L = 1
    for d in denominators:
        L = lcm(L, d)

    weights = [L // d for d in denominators]

    def check_time():
        return time.time() > deadline

    def validate(indices):
        if not indices:
            return False, 0
        if len(set(indices)) != len(indices):
            return False, 0
        if any(i < 0 or i >= n for i in indices):
            return False, 0
        s = sum(weights[i] for i in indices)
        if s >= L:
            return False, 0
        return True, s

    best_sum = -1
    best_indices = None

    def update_best(indices):
        nonlocal best_sum, best_indices
        valid, s = validate(indices)
        if valid and s > best_sum:
            best_sum = s
            best_indices = list(indices)
            return True
        return False

    # ============================================================
    # STRATEGY 1: Meet-in-the-middle with optimized binary search
    # ============================================================
    mid = n // 2
    w1 = [weights[i] for i in range(mid)]
    w2 = [weights[i] for i in range(mid, n)]

    def gen_subset_sums(weights_list):
        sums = [(0, 0)]
        for i, w in enumerate(weights_list):
            if check_time():
                break
            bit = 1 << i
            current_len = len(sums)
            for j in range(current_len):
                s, m = sums[j]
                ns = s + w
                if ns < L:
                    sums.append((ns, m | bit))
        return sums

    print("Strategy 1: Meet-in-the-middle...")
    sums1 = gen_subset_sums(w1)
    print(f"  Half 1: {len(sums1)} subsets")

    if not check_time():
        sums2 = gen_subset_sums(w2)
        print(f"  Half 2: {len(sums2)} subsets")

        if not check_time():
            sums2.sort(key=lambda x: x[0])
            sums2_vals = [s[0] for s in sums2]

            # For each sum in half1, find best compatible sum in half2
            for s1, m1 in sums1:
                if check_time():
                    break
                target = L - s1 - 1
                idx = bisect.bisect_right(sums2_vals, target)
                if idx > 0:
                    s2, m2 = sums2[idx - 1]
                    total = s1 + s2
                    if total > best_sum:
                        best_sum = total
                        indices = []
                        for i in range(mid):
                            if m1 & (1 << i):
                                indices.append(i)
                        for i in range(mid, n):
                            if m2 & (1 << (i - mid)):
                                indices.append(i)
                        best_indices = indices

    # ============================================================
    # STRATEGY 2: Simulated annealing for escape from local minima
    # ============================================================
    if not check_time() and best_indices is not None:
        print("Strategy 2: Simulated annealing...")

        import math

        T_start = (L - best_sum) * 0.1
        T_end = 1
        cooling = 0.9995

        T = T_start
        sa_iter = 0
        max_sa_iter = 50000

        while T > T_end and sa_iter < max_sa_iter and not check_time():
            sa_iter += 1
            chosen = set(best_indices)
            s = best_sum

            # Random move
            move_type = random.random()
            if move_type < 0.3:
                # Add
                candidates = [i for i in range(n) if i not in chosen and s + weights[i] < L]
                if not candidates:
                    continue
                i = random.choice(candidates)
                new_s = s + weights[i]
                delta = new_s - s
            elif move_type < 0.5:
                # Remove
                if len(chosen) <= 1:
                    continue
                i = random.choice(list(chosen))
                new_s = s - weights[i]
                delta = -weights[i]
            else:
                # Swap
                if len(chosen) == 0:
                    continue
                rem = random.choice(list(chosen))
                available = [i for i in range(n) if i not in chosen]
                if not available:
                    continue
                add = random.choice(available)
                new_s = s - weights[rem] + weights[add]
                delta = weights[add] - weights[rem]

            if new_s >= L:
                continue

            # Accept if better or with probability exp(-|delta|/T) for worse
            if delta > 0:
                best_sum = new_s
                best_indices = list(chosen)
            else:
                # For simulated annealing, accept worse moves with probability
                prob = math.exp(-abs(delta) / T)
                if random.random() < prob:
                    # Don't actually accept worse moves for our best tracking,
                    # but this allows exploration
                    pass

            T *= cooling

    # ============================================================
    # STRATEGY 3: Greedy + local search with multiple restarts
    # ============================================================
    if not check_time():
        print("Strategy 3: Greedy + local search...")

        order = sorted(range(n), key=lambda i: -weights[i])

        for _ in range(100):
            if check_time():
                break
            shuffled = order[:]
            random.shuffle(shuffled)
            s = 0
            chosen = []
            for i in shuffled:
                if s + weights[i] < L:
                    s += weights[i]
                    chosen.append(i)
            update_best(chosen)

        # Local search: try single additions and swaps
        if best_indices is not None:
            for iteration in range(5000):
                if check_time():
                    break

                chosen = set(best_indices)
                s = best_sum
                improved = False

                # Try adding one item
                candidates_add = [i for i in range(n) if i not in chosen and s + weights[i] < L]
                if candidates_add:
                    best_add = max(candidates_add, key=lambda i: s + weights[i])
                    new_s = s + weights[best_add]
                    if new_s > best_sum:
                        chosen.add(best_add)
                        best_indices = list(chosen)
                        best_sum = new_s
                        improved = True

                if not improved:
                    # Try swaps
                    found = False
                    for rem in list(chosen):
                        if found or check_time():
                            break
                        s_after = s - weights[rem]
                        candidates_add2 = [i for i in range(n)
                                           if i not in chosen and s_after + weights[i] < L]
                        if candidates_add2:
                            best_add2 = max(candidates_add2, key=lambda i: s_after + weights[i])
                            new_s = s_after + weights[best_add2]
                            if new_s > best_sum:
                                chosen.discard(rem)
                                chosen.add(best_add2)
                                best_indices = list(chosen)
                                best_sum = new_s
                                found = True
                                improved = True
                                break

                if not improved:
                    # Try removals
                    for rem in list(chosen):
                        if check_time():
                            break
                        new_s = s - weights[rem]
                        if new_s > best_sum and new_s < L and len(chosen) > 1:
                            chosen.discard(rem)
                            best_indices = list(chosen)
                            best_sum = new_s
                            improved = True
                            break

    # ============================================================
    # STRATEGY 4: Systematic pair swaps from best
    # ============================================================
    if not check_time() and best_indices is not None:
        print("Strategy 4: Systematic pair swaps...")

        chosen = set(best_indices)
        s = best_sum

        # Try all single swaps
        for rem in list(chosen):
            if check_time():
                break
            s_without = s - weights[rem]
            for add in range(n):
                if add in chosen:
                    continue
                new_s = s_without + weights[add]
                if new_s < L and new_s > best_sum:
                    best_sum = new_s
                    new_chosen = chosen.copy()
                    new_chosen.discard(rem)
                    new_chosen.add(add)
                    best_indices = list(new_chosen)
                    chosen = new_chosen
                    s = new_s

        # Try all pair swaps (remove 2, add 2) - limited for time
        if not check_time():
            chosen_list = list(chosen)
            s = best_sum
            pair_count = 0
            max_pairs = 5000
            for i in range(len(chosen_list)):
                if check_time() or pair_count >= max_pairs:
                    break
                for j in range(i + 1, len(chosen_list)):
                    if check_time() or pair_count >= max_pairs:
                        break
                    pair_count += 1
                    rem1, rem2 = chosen_list[i], chosen_list[j]
                    s_without = s - weights[rem1] - weights[rem2]
                    available = [k for k in range(n) if k not in chosen]
                    if len(available) < 2:
                        continue
                    # Try best pair of additions
                    # Sort available by weight descending
                    available_sorted = sorted(available, key=lambda k: -weights[k])
                    for a1 in available_sorted[:5]:
                        if check_time():
                            break
                        for a2 in available_sorted[:5]:
                            if a1 == a2:
                                continue
                            new_s = s_without + weights[a1] + weights[a2]
                            if new_s < L and new_s > best_sum:
                                best_sum = new_s
                                new_chosen = chosen.copy()
                                new_chosen.discard(rem1)
                                new_chosen.discard(rem2)
                                new_chosen.add(a1)
                                new_chosen.add(a2)
                                best_indices = list(new_chosen)
                                chosen = new_chosen
                                s = new_s

    # ============================================================
    # STRATEGY 5: Randomized targeted search
    # ============================================================
    if not check_time() and best_indices is not None:
        print("Strategy 5: Randomized targeted search...")

        for iteration in range(20000):
            if check_time():
                break

            chosen = set(best_indices)
            s = best_sum

            n_remove = random.randint(1, min(2, len(chosen)))
            n_add = random.randint(1, 2)

            to_remove = random.sample(list(chosen), n_remove)
            available = [i for i in range(n) if i not in chosen]
            if len(available) < n_add:
                continue
            to_add = random.sample(available, n_add)

            new_chosen = chosen.copy()
            for r in to_remove:
                new_chosen.discard(r)
            for a in to_add:
                new_chosen.add(a)

            new_s = sum(weights[i] for i in new_chosen)
            if new_s < L and new_s > best_sum:
                best_sum = new_s
                best_indices = list(new_chosen)

    # ============================================================
    # STRATEGY 6: Final polish - exhaustive single modifications
    # ============================================================
    if not check_time() and best_indices is not None:
        print("Strategy 6: Final polish...")

        for _ in range(10):
            if check_time():
                break
            changed = False
            chosen = set(best_indices)
            s = best_sum

            # Try adding any single item
            for i in range(n):
                if check_time():
                    break
                if i not in chosen and s + weights[i] < L and s + weights[i] > best_sum:
                    chosen.add(i)
                    best_sum = s + weights[i]
                    best_indices = list(chosen)
                    s = best_sum
                    changed = True

            # Try removing any single item
            for i in list(chosen):
                if check_time():
                    break
                if len(chosen) > 1:
                    new_s = s - weights[i]
                    if new_s > best_sum and new_s < L:
                        chosen.discard(i)
                        best_sum = new_s
                        best_indices = list(chosen)
                        s = new_s
                        changed = True

            if not changed:
                break

    # Final validation
    if best_indices is None or not validate(best_indices)[0]:
        for i in range(n):
            if weights[i] < L:
                best_indices = [i]
                best_sum = weights[i]
                break
        if best_indices is None:
            best_indices = [0]

    print(f"Final best sum: {best_sum}/{L}, deficit: {L - best_sum}/{L}")
    print(f"Best indices: {best_indices}")

    return best_indices
