"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)
    
    def greedy_sum_free(elements, order=None):
        S = []
        S_set = set()
        sums_set = set()
        if order is None:
            order = list(range(len(elements)))
        for idx in order:
            x = elements[idx]
            if x in S_set:
                continue
            if x in sums_set:
                continue
            valid = True
            for s in S:
                if (s - x) in S_set:
                    valid = False
                    break
            if not valid:
                continue
            for a in S:
                sums_set.add(x + a)
            sums_set.add(x + x)
            S.append(x)
            S_set.add(x)
        return S
    
    def local_search_add(S, elements, time_budget=1.5):
        S = list(S)
        S_set = set(S)
        sums_set = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums_set.add(S[i] + S[j])
        candidates = [x for x in elements if x not in S_set]
        deadline = time.time() + time_budget
        while time.time() < deadline:
            added = False
            for i, x in enumerate(candidates):
                if x in S_set or x in sums_set:
                    continue
                valid = True
                for s in S:
                    if (s - x) in S_set:
                        valid = False
                        break
                if not valid:
                    continue
                for a in S:
                    sums_set.add(x + a)
                sums_set.add(x + x)
                S.append(x)
                S_set.add(x)
                candidates.pop(i)
                added = True
                break
            if not added:
                break
        return S
    
    def solve_instance(elements, time_budget=35):
        n = len(elements)
        best = []
        best_size = 0
        odd_subset = [x for x in elements if x % 2 != 0]
        if len(odd_subset) > best_size:
            best = list(odd_subset)
            best_size = len(odd_subset)
        orderings = [list(range(n)), list(range(n-1, -1, -1)),
                     sorted(range(n), key=lambda i: -abs(elements[i])),
                     sorted(range(n), key=lambda i: abs(elements[i]))]
        rng = random.Random(789)
        for _ in range(25):
            order = list(range(n))
            rng.shuffle(order)
            orderings.append(order)
        deadline = start_time + time_budget
        for order in orderings:
            if time.time() > deadline:
                break
            S = greedy_sum_free(elements, order)
            S = local_search_add(S, elements, time_budget=1.0)
            if len(S) > best_size:
                best_size = len(S)
                best = S
        return best
    
    results = []
    for size, seed in [(60, 801), (120, 802), (240, 803)]:
        elements = instance(size, seed)
        S = solve_instance(elements, time_budget=40)
        results.append(S)
    return results

# EVOLVE-BLOCK-END
