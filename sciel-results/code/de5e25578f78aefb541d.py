import time
import random
from itertools import combinations

def construct():
    start = time.time()
    DEADLINE = 75.0  # safety margin before 90s kill

    N = 256
    universe = list(range(1, N + 1))

    def conflict(a, b):
        return (a * b) % (a + b) == 0

    def can_add(S_set, x):
        for a in S_set:
            if conflict(a, x):
                return False
        return True

    def greedy_fill(candidates, S_set, S_list):
        # try to add each candidate if possible
        for x in candidates:
            if can_add(S_set, x):
                S_set.add(x)
                S_list.append(x)
        S_list.sort()

    def local_search(S_set, S_list, iters):
        # 1-for-1 swaps and 1-for-0 additions, plus 2-for-1
        improved = True
        while improved and iters > 0:
            improved = False
            iters -= 1
            # try adding any missing element
            for x in universe:
                if x not in S_set and can_add(S_set, x):
                    S_set.add(x)
                    S_list.append(x)
                    improved = True
            if improved:
                continue
            # try 1-for-1 swap: remove a, add x
            for a in list(S_list):
                for x in universe:
                    if x in S_set:
                        continue
                    # can we remove a and add x?
                    S_set.discard(a)
                    if can_add(S_set, x):
                        S_set.add(x)
                        S_list.remove(a)
                        S_list.append(x)
                        improved = True
                        break
                    S_set.add(a)
                    if improved:
                        break
                if improved:
                    break
        return S_set, S_list

    def run_random_greedy(seed):
        rng = random.Random(seed)
        order = universe[:]
        rng.shuffle(order)
        S_set = set()
        S_list = []
        greedy_fill(order, S_set, S_list)
        S_set, S_list = local_search(S_set, S_list, 200)
        return S_set, S_list

    # Start with odd-only seed (known valid baseline of size 128)
    best_set = set(range(1, N + 1, 2))
    best_list = sorted(best_set)
    best_size = len(best_set)

    # Try to improve with local search on the odd seed
    best_set, best_list = local_search(best_set, best_list, 500)
    best_size = len(best_set)

    if time.time() > DEADLINE:
        return best_list

    # Randomized restarts
    seed = 0
    while time.time() < DEADLINE:
        S_set, S_list = run_random_greedy(seed)
        if len(S_set) > best_size:
            best_set, best_list = S_set, S_list
            best_size = len(best_set)
        seed += 1
        # Occasionally do a more aggressive local search on best
        if seed % 5 == 0 and time.time() < DEADLINE - 10:
            S_set2 = set(best_set)
            S_list2 = list(best_list)
            S_set2, S_list2 = local_search(S_set2, S_list2, 300)
            if len(S_set2) > best_size:
                best_set, best_list = S_set2, S_list2
                best_size = len(best_set)

    return best_list
