import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # Compute LCM of 1..22
    L = 1
    for i in range(1, 23):
        L = L * i // math.gcd(L, i)

    # weights[i] = L // i, so 1/i = weights[i] / L
    weights = {i: L // i for i in range(1, 23)}

    # Order elements by weight ascending (larger original values first, smaller weights)
    elements = list(range(1, 23))
    elements.sort(key=lambda x: weights[x])

    best = [1, 2, 3, 5, 7, 11, 13, 17, 19]

    def backtrack(chosen, subset_sums, diff_set, idx):
        nonlocal best
        if time.time() > deadline:
            return
        if len(chosen) > len(best):
            best = chosen[:]
            if len(best) >= 12:
                return
        # Pruning: remaining elements can't beat best
        if len(chosen) + (22 - idx) <= len(best):
            return
        if idx >= 22:
            return

        x = elements[idx]
        w = weights[x]

        # Try including x
        if w not in diff_set:
            new_sums = set(subset_sums)
            for s in subset_sums:
                new_sums.add(s + w)

            new_diffs = set(diff_set)
            new_diffs.add(w)
            for s in subset_sums:
                for t in subset_sums:
                    new_diffs.add(abs(s - t - w))

            backtrack(chosen + [x], new_sums, new_diffs, idx + 1)

        # Try excluding x
        backtrack(chosen, subset_sums, diff_set, idx + 1)

    backtrack([], {0}, {0}, 0)

    return best
