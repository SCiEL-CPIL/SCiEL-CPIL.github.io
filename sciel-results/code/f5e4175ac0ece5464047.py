import time
import random
from math import gcd

def construct():
    N = 128
    start_time = time.time()
    deadline = start_time + 85

    # Precompute gcd matrix
    gcd_mat = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    def has_bad_triple_with_new(selected, new_elem):
        n = len(selected)
        gx = [gcd_mat[new_elem][a] for a in selected]
        for i in range(n):
            a = selected[i]
            gax = gx[i]
            for j in range(i + 1, n):
                b = selected[j]
                if gx[j] == gax and gcd_mat[a][b] == gax:
                    return True
        return False

    def greedy_build(order):
        selected = []
        selected_set = set()
        for x in order:
            if x not in selected_set:
                if not has_bad_triple_with_new(selected, x):
                    selected.append(x)
                    selected_set.add(x)
        return selected

    def local_search(set_list, max_iters=3000):
        set_list = list(set_list)
        set_list_set = set(set_list)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1

            # 1. Try adding all possible elements
            candidates = [x for x in range(1, N + 1) if x not in set_list_set]
            random.shuffle(candidates)
            for x in candidates:
                if not has_bad_triple_with_new(set_list, x):
                    set_list.append(x)
                    set_list_set.add(x)
                    improved = True

            # 2. Swaps
            if not improved and time.time() < deadline:
                for idx in range(len(set_list)):
                    if time.time() >= deadline:
                        break
                    removed = set_list[idx]
                    candidate_set = set_list[:idx] + set_list[idx + 1:]
                    for x in range(1, N + 1):
                        if x in set_list_set:
                            continue
                        if not has_bad_triple_with_new(candidate_set, x):
                            set_list = candidate_set + [x]
                            set_list_set.remove(removed)
                            set_list_set.add(x)
                            improved = True
                            break
                    if improved:
                        break

        return set_list

    def perturb(set_list, k):
        set_list = list(set_list)
        set_list_set = set(set_list)
        to_remove = random.sample(set_list, min(k, len(set_list)))
        for r in to_remove:
            set_list.remove(r)
            set_list_set.remove(r)
        candidates = list(to_remove) + [x for x in range(1, N + 1) if x not in set_list_set]
        random.shuffle(candidates)
        for x in candidates:
            if x not in set_list_set:
                if not has_bad_triple_with_new(set_list, x):
                    set_list.append(x)
                    set_list_set.add(x)
        return set_list

    best_set = []

    # Multiple greedy starts
    for order in [range(1, N + 1), range(N, 0, -1)]:
        if time.time() >= deadline - 5:
            break
        s = greedy_build(order)
        s = local_search(s)
        if len(s) > len(best_set):
            best_set = s

    for _ in range(8):
        if time.time() >= deadline - 10:
            break
        random_order = list(range(1, N + 1))
        random.shuffle(random_order)
        s = greedy_build(random_order)
        s = local_search(s)
        if len(s) > len(best_set):
            best_set = s

    # ILS
    current_best = best_set
    current_best_len = len(current_best)
    while time.time() < deadline - 3:
        perturbed = perturb(current_best, random.choice([2, 3, 4, 5]))
        improved_set = local_search(perturbed, max_iters=500)
        if len(improved_set) > current_best_len:
            current_best = improved_set
            current_best_len = len(improved_set)
            if current_best_len > len(best_set):
                best_set = current_best
        if random.random() < 0.2 and time.time() < deadline - 8:
            random_order = list(range(1, N + 1))
            random.shuffle(random_order)
            s_new = greedy_build(random_order)
            s_new = local_search(s_new, max_iters=500)
            if len(s_new) > current_best_len:
                current_best = s_new
                current_best_len = len(s_new)
                if current_best_len > len(best_set):
                    best_set = current_best

    # Final add pass
    final_set = list(best_set)
    final_set_set = set(final_set)
    for x in range(1, N + 1):
        if x not in final_set_set:
            if not has_bad_triple_with_new(final_set, x):
                final_set.append(x)
                final_set_set.add(x)

    if not final_set:
        final_set = [1]
    return final_set
