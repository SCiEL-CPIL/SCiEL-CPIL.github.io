import time
import random
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 80

    L = 1
    for i in range(1, 23):
        L = lcm(L, i)

    vals = [L // i for i in range(1, 23)]

    def try_add(sums, sums_set, v):
        for s in sums:
            if (s + v) in sums_set:
                return False, None, None
        new_sums = [s + v for s in sums]
        new_set = sums_set | set(new_sums)
        return True, sums + new_sums, new_set

    def backtrack(remaining, sums, sums_set, current_set, target_size):
        if time.time() > deadline:
            return None
        if len(current_set) == target_size:
            return list(current_set)

        for i in range(len(remaining)):
            idx = remaining[i]
            v = vals[idx]
            valid, new_sums, new_sums_set = try_add(sums, sums_set, v)
            if not valid:
                continue

            current_set.append(idx)
            result = backtrack(remaining[:i] + remaining[i+1:], new_sums, new_sums_set, current_set, target_size)
            current_set.pop()

            if result is not None:
                return result

        return None

    def greedy_build(order):
        current_set = []
        sums = [0]
        sums_set = {0}
        for idx in order:
            if time.time() > deadline:
                break
            v = vals[idx]
            valid, new_sums, new_sums_set = try_add(sums, sums_set, v)
            if valid:
                current_set.append(idx)
                sums = new_sums
                sums_set = new_sums_set
        return current_set

    rng = random.Random(12345)
    orderings = [list(range(22)), list(range(21, -1, -1))]
    for _ in range(30):
        ord_list = list(range(22))
        rng.shuffle(ord_list)
        orderings.append(ord_list)

    best_set = None
    best_size = 0

    for order in orderings:
        if time.time() > deadline:
            break
        s = greedy_build(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    for target in range(12, best_size, -1):
        if time.time() > deadline:
            break
        for order in orderings[:8]:
            if time.time() > deadline:
                break
            result = backtrack(order, [0], {0}, [], target)
            if result is not None:
                best_set = result
                best_size = target
                break
        if best_size >= target:
            break

    if best_set is None or len(best_set) == 0:
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]

    return sorted([idx + 1 for idx in best_set])
