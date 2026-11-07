import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    def get_subset_sums(S):
        """Get all non-empty subset sums of S using iterative DP."""
        sums = set()
        for x in S:
            new_sums = {x + s for s in sums}
            new_sums.add(x)
            sums |= new_sums
        return sums

    def is_valid_fast(S):
        """Check if all non-empty subset sums are divisibility-incomparable."""
        sums = get_subset_sums(S)
        sums_list = sorted(sums)
        n = len(sums_list)
        for i in range(n):
            a = sums_list[i]
            for j in range(i + 1, n):
                b = sums_list[j]
                if b < 2 * a:
                    continue
                if b % a == 0:
                    return False
        return True

    def can_add_fast(current_sums, x):
        """Check if adding x to a set (with current_sums) keeps divisibility-incomparability."""
        new_sums = {x}
        for s in current_sums:
            new_sums.add(x + s)
        all_sums = current_sums | new_sums
        sums_list = sorted(all_sums)
        n = len(sums_list)
        for i in range(n):
            a = sums_list[i]
            for j in range(i + 1, n):
                b = sums_list[j]
                if b < 2 * a:
                    continue
                if b % a == 0:
                    return False
        return True

    best = [128, 129, 131]
    best_size = len(best)

    # Focus on large numbers to minimize divisibility conflicts
    # Numbers in [128, 256] have subset sums in [128, 256*n]
    # Two sums s1, s2 with s2 % s1 == 0 requires s2 >= 2*s1
    # For large base numbers, this is harder to satisfy

    def try_greedy(initial, candidates, deadline):
        """Greedy extend: add candidates if they fit."""
        S = list(initial)
        current_sums = get_subset_sums(S)
        for x in candidates:
            if x in S:
                continue
            if time.time() > deadline:
                break
            if can_add_fast(current_sums, x):
                S.append(x)
                new_sums = {x}
                for s in current_sums:
                    new_sums.add(x + s)
                current_sums = current_sums | new_sums
        return S, current_sums

    # Strategy: use large numbers primarily
    # Candidates from 128 to 256 (129 numbers)
    large_candidates = list(range(128, 257))

    # Try several starting seeds with large numbers
    seed_sets = [
        [128, 129, 131],
        [130, 131, 133],
        [140, 141, 143],
        [150, 151, 153],
        [160, 161, 163],
        [170, 171, 173],
        [180, 181, 183],
        [190, 191, 193],
        [200, 201, 203],
        [210, 211, 213],
        [220, 221, 223],
        [230, 231, 233],
        [240, 241, 243],
        [245, 247, 249],
        [250, 251, 253],
    ]

    for seed in seed_sets:
        if time.time() > deadline - 2.0:
            break
        if not is_valid_fast(seed):
            continue
        # Try descending order (larger first)
        candidates = [x for x in large_candidates if x not in seed]
        candidates.sort(reverse=True)
        S, _ = try_greedy(seed, candidates, deadline - 2.0)
        if len(S) > best_size:
            best = S
            best_size = len(S)

        # Try ascending
        candidates.sort()
        S, _ = try_greedy(seed, candidates, deadline - 2.0)
        if len(S) > best_size:
            best = S
            best_size = len(S)

    # Randomized restarts with large numbers
    random.seed(42)
    for _ in range(500):
        if time.time() > deadline - 3.0:
            break
        seed_size = random.randint(2, 5)
        seed = random.sample(large_candidates, seed_size)
        seed.sort()
        if not is_valid_fast(seed):
            continue
        candidates = [x for x in large_candidates if x not in seed]
        random.shuffle(candidates)
        S, _ = try_greedy(seed, candidates, deadline - 3.0)
        if len(S) > best_size:
            best = S
            best_size = len(S)

    # Also try mixing in some medium numbers
    medium_large = list(range(80, 257))
    for _ in range(200):
        if time.time() > deadline - 3.0:
            break
        seed_size = random.randint(2, 4)
        seed = random.sample(medium_large, seed_size)
        seed.sort()
        if not is_valid_fast(seed):
            continue
        candidates = [x for x in medium_large if x not in seed]
        random.shuffle(candidates)
        S, _ = try_greedy(seed, candidates, deadline - 3.0)
        if len(S) > best_size:
            best = S
            best_size = len(S)

    # Local search: try swaps to enable further growth
    if time.time() < deadline - 5.0:
        current_sums = get_subset_sums(best)
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(len(best)):
                if time.time() > deadline - 5.0:
                    break
                for x in range(128, 257):
                    if x in best:
                        continue
                    new_set = best[:i] + [x] + best[i+1:]
                    if is_valid_fast(new_set):
                        # Try to extend from this swapped set
                        current_sums_new = get_subset_sums(new_set)
                        candidates = [y for y in range(128, 257) if y not in new_set]
                        candidates.sort(reverse=True)
                        S, _ = try_greedy(new_set, candidates, deadline - 5.0)
                        if len(S) > best_size:
                            best = S
                            best_size = len(S)
                            current_sums = get_subset_sums(best)
                            improved = True
                            break
                if improved:
                    break

    # Final verification
    if not is_valid_fast(best):
        best = [128, 129, 131]

    return list(best)
