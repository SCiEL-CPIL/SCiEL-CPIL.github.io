import time
import random
from math import lcm
from functools import reduce

UNIVERSE = list(range(1, 25))
L = reduce(lcm, UNIVERSE)

def is_valid(S):
    """Check if set S is reciprocal-subsum-free."""
    if not S:
        return True
    S_list = list(S)
    n = len(S_list)
    if n == 0:
        return True

    weights = [L // x for x in S_list]

    for i, x in enumerate(S_list):
        target = weights[i]
        others = weights[:i] + weights[i + 1:]
        if not others:
            continue
        if sum(others) < target:
            continue
        # If a single other weight equals target, conflict (subset of size 1)
        # This is handled by the general check below.

        m = len(others)
        mid = m // 2
        left = others[:mid]
        right = others[mid:]

        # Build all subset sums of left
        left_sums = {0}
        for w in left:
            left_sums = left_sums | {s + w for s in left_sums}

        # Build right sums incrementally and check
        right_sums = {0}
        found = False
        for w in right:
            new_sums = {s + w for s in right_sums}
            for s in new_sums:
                if (target - s) in left_sums:
                    found = True
                    break
            if found:
                break
            right_sums |= new_sums

        if found:
            return False

    return True

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # 8s safety margin before 90s kill

    def now():
        return time.time()

    def in_time():
        return now() < deadline

    random.seed(2024)

    # Fallback
    fallback = list(range(13, 25))
    if is_valid(fallback):
        best = fallback
    else:
        best = [24]
    best_size = len(best)

    def consider(cand):
        nonlocal best, best_size
        if len(cand) > best_size and is_valid(cand):
            best = list(cand)
            best_size = len(best)

    # Multi-start greedy with diverse orderings
    orderings = [
        sorted(UNIVERSE, reverse=True),   # Large first
        sorted(UNIVERSE),                 # Small first
        [UNIVERSE[i] for i in range(0, 24, 2)] + [UNIVERSE[i] for i in range(1, 24, 2)],  # even then odd
        [UNIVERSE[i] for i in range(1, 24, 2)] + [UNIVERSE[i] for i in range(0, 24, 2)],  # odd then even
    ]
    for _ in range(20):
        perm = UNIVERSE[:]
        random.shuffle(perm)
        orderings.append(perm)

    for order in orderings:
        if not in_time():
            break
        S = []
        for x in order:
            if not in_time():
                break
            candidate = S + [x]
            if is_valid(candidate):
                S = candidate
        consider(S)

    # Local search: try adding any missing element (1-for-0)
    improved = True
    while improved and in_time():
        improved = False
        for x in UNIVERSE:
            if not in_time():
                break
            if x not in best:
                candidate = best + [x]
                if is_valid(candidate):
                    best = candidate
                    best_size = len(best)
                    improved = True

    # 1-for-1 swaps: remove one, add one different element
    for _ in range(40):
        if not in_time():
            break
        improved = False
        for i in range(len(best)):
            if not in_time():
                break
            removed = best[i]
            S_no_i = best[:i] + best[i + 1:]
            for x in UNIVERSE:
                if not in_time():
                    break
                if x == removed or x in S_no_i:
                    continue
                candidate = S_no_i + [x]
                if len(candidate) > best_size and is_valid(candidate):
                    best = candidate
                    best_size = len(best)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    # 1-for-2 swaps: remove one, add two different elements
    for _ in range(30):
        if not in_time():
            break
        improved = False
        for i in range(len(best)):
            if not in_time():
                break
            removed = best[i]
            S_no_i = best[:i] + best[i + 1:]
            missing = [x for x in UNIVERSE if x != removed and x not in S_no_i]
            for a in missing:
                if not in_time():
                    break
                for b in missing:
                    if not in_time():
                        break
                    if b <= a:
                        continue
                    candidate = S_no_i + [a, b]
                    if len(candidate) > best_size and is_valid(candidate):
                        best = candidate
                        best_size = len(best)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break

    # Final add pass
    improved = True
    while improved and in_time():
        improved = False
        for x in UNIVERSE:
            if not in_time():
                break
            if x not in best:
                candidate = best + [x]
                if is_valid(candidate):
                    best = candidate
                    best_size = len(best)
                    improved = True

    if not best:
        best = [24]
    return best
