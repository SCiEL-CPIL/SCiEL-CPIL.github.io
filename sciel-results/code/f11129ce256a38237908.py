import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 75
    MAX_MOD = 128

    def are_disjoint(r1, m1, r2, m2):
        g = math.gcd(m1, m2)
        return (r1 - r2) % g != 0

    def is_compatible(r, m, existing):
        for (r2, m2) in existing:
            if not are_disjoint(r, m, r2, m2):
                return False
        return True

    all_moduli = list(range(2, MAX_MOD + 1))

    def greedy_assign(moduli_list):
        assigned = []
        for m in moduli_list:
            for r in range(m):
                if is_compatible(r, m, assigned):
                    assigned.append((r, m))
                    break
        return assigned

    # Initial greedy with natural order
    best_set = greedy_assign(all_moduli)
    best_count = len(best_set)

    def local_search(current_dict, moduli_to_try, dl):
        improved = True
        while improved and time.time() < dl:
            improved = False
            for m in moduli_to_try:
                if m in current_dict:
                    continue
                for r in range(m):
                    if is_compatible(r, m, list(current_dict.items())):
                        current_dict[m] = r
                        improved = True
                        break
                if improved:
                    break
        return current_dict

    current = {m: r for (r, m) in best_set}
    current = local_search(current, all_moduli, deadline)
    if len(current) > best_count:
        best_count = len(current)
        best_set = list(current.items())

    # Try many random orderings
    random.seed(42)
    for _ in range(200):
        if time.time() >= deadline:
            break
        shuffled = all_moduli.copy()
        random.shuffle(shuffled)
        result = greedy_assign(shuffled)
        if len(result) > best_count:
            best_count = len(result)
            best_set = result
            current = {m: r for (r, m) in best_set}
            current = local_search(current, all_moduli, deadline)
            if len(current) > best_count:
                best_count = len(current)
                best_set = list(current.items())

    return [[r, m] for (r, m) in best_set]
