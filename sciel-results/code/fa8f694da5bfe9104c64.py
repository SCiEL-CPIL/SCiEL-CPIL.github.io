import itertools
import time
import random

L = 232792560
VALS = [L // k for k in range(1, 23)]

def is_dissociated(subset):
    if not subset:
        return True
    m = len(subset)
    if m > 20:
        return False
    vals = [VALS[k-1] for k in subset]
    sums = [0]
    for v in vals:
        n = len(sums)
        sums.extend(s + v for s in sums[:n])
    return len(set(sums)) == (1 << m)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin

    def time_ok():
        return time.time() < deadline

    # Phase 1: Greedy with multiple orderings
    def greedy(order):
        current = []
        for k in order:
            if not time_ok():
                break
            new_set = current + [k]
            if is_dissociated(new_set):
                current = new_set
        return current

    best_set = []
    best_size = 0

    # Try ascending, descending, and several random orderings
    orders = [list(range(1, 23)), list(range(22, 0, -1))]
    rng = random.Random(42)
    for _ in range(20):
        o = list(range(1, 23))
        rng.shuffle(o)
        orders.append(o)

    for o in orders:
        if not time_ok():
            break
        s = greedy(o)
        if len(s) > best_size:
            best_set = s
            best_size = len(s)

    # Phase 2: Local search - try to add one more element
    def try_add(current, candidates_list):
        for k in candidates_list:
            if k in current:
                continue
            new_set = current + [k]
            if is_dissociated(new_set):
                return new_set
        return None

    # Try adding elements in descending order (larger k = smaller value, easier to add)
    current = list(best_set)
    improved = True
    while improved and time_ok():
        improved = False
        for k in range(22, 0, -1):
            if k in current:
                continue
            new_set = current + [k]
            if is_dissociated(new_set):
                current = new_set
                improved = True
                if len(current) > best_size:
                    best_set = current
                    best_size = len(current)
                break

    # Phase 3: More random restarts with greedy + local add
    rng2 = random.Random(123)
    num_restarts = 500
    for r in range(num_restarts):
        if not time_ok():
            break
        o = list(range(1, 23))
        rng2.shuffle(o)
        s = greedy(o)
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
        # Try to add one more
        for k in range(22, 0, -1):
            if k in s:
                continue
            ns = s + [k]
            if is_dissociated(ns):
                s = ns
                if len(s) > best_size:
                    best_set = s
                    best_size = len(s)
                break

    # Phase 4: If we have a large set, try swap-based local search
    if len(best_set) >= 10 and time_ok():
        current = list(best_set)
        # Try removing one element and adding a different one to enable further additions
        for remove_k in list(current):
            if not time_ok():
                break
            for add_k in range(22, 0, -1):
                if add_k == remove_k or add_k in current:
                    continue
                new_set = [x for x in current if x != remove_k] + [add_k]
                if is_dissociated(new_set):
                    # Try to add more
                    for k2 in range(22, 0, -1):
                        if k2 in new_set:
                            continue
                        ns2 = new_set + [k2]
                        if is_dissociated(ns2):
                            new_set = ns2
                            break
                    if len(new_set) > len(best_set):
                        best_set = new_set
                        best_size = len(best_set)
                    break

    return best_set
