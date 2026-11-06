"""Construct a large 4-AP-free subset of [1,512]."""
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    n = 512

    def is_valid_addition(x, S, n):
        """Check if x can be added to set S (boolean array) without forming a 4-AP."""
        # x is 4th term: x-3d, x-2d, x-d in S
        for d in range(1, (x - 1) // 3 + 1):
            if S[x - d] and S[x - 2 * d] and S[x - 3 * d]:
                return False
        # x is 1st term: x+d, x+2d, x+3d in S
        for d in range(1, (n - x) // 3 + 1):
            if S[x + d] and S[x + 2 * d] and S[x + 3 * d]:
                return False
        # x is 2nd term: x-d, x+d, x+2d in S
        max_d = min(x - 1, (n - x) // 2)
        for d in range(1, max_d + 1):
            if S[x - d] and S[x + d] and S[x + 2 * d]:
                return False
        # x is 3rd term: x-2d, x-d, x+d in S
        max_d = min((x - 1) // 2, n - x)
        for d in range(1, max_d + 1):
            if S[x - 2 * d] and S[x - d] and S[x + d]:
                return False
        return True

    def has_no_two_in_base3(x):
        while x > 0:
            if x % 3 == 2:
                return False
            x //= 3
        return True

    # Build initial candidates: Cantor set and greedy
    S_cantor = np.zeros(n + 1, dtype=bool)
    for i in range(1, n + 1):
        if has_no_two_in_base3(i):
            S_cantor[i] = True

    S_greedy = np.zeros(n + 1, dtype=bool)
    for x in range(1, n + 1):
        if is_valid_addition(x, S_greedy, n):
            S_greedy[x] = True

    cantor_count = int(np.sum(S_cantor))
    greedy_count = int(np.sum(S_greedy))

    if cantor_count >= greedy_count:
        S_best = S_cantor.copy()
        best_count = cantor_count
    else:
        S_best = S_greedy.copy()
        best_count = greedy_count

    # Local search: try to add elements that fit
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, n + 1):
            if not S_best[x] and is_valid_addition(x, S_best, n):
                S_best[x] = True
                best_count += 1
                improved = True

    # Randomized restarts with local search
    for restart in range(10000):
        if time.time() > deadline - 5:
            break

        S_curr = S_best.copy()
        elements_in_set = [i for i in range(1, n + 1) if S_curr[i]]
        if len(elements_in_set) <= 5:
            continue

        # Randomly remove some elements
        num_remove = random.randint(1, 5)
        to_remove = random.sample(elements_in_set, num_remove)
        for elem in to_remove:
            S_curr[elem] = False

        # Try to add back elements in random order
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        for x in candidates:
            if not S_curr[x] and is_valid_addition(x, S_curr, n):
                S_curr[x] = True

        curr_count = int(np.sum(S_curr))
        if curr_count > best_count:
            best_count = curr_count
            S_best = S_curr.copy()

            # Immediate local improvement
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for x in range(1, n + 1):
                    if not S_best[x] and is_valid_addition(x, S_best, n):
                        S_best[x] = True
                        best_count += 1
                        improved = True

    # Final local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, n + 1):
            if not S_best[x] and is_valid_addition(x, S_best, n):
                S_best[x] = True
                best_count += 1
                improved = True

    return [i for i in range(1, n + 1) if S_best[i]]
