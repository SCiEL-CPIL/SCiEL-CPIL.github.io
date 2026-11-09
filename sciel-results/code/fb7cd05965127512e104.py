# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s kill

    n = 16
    k = 4
    max_target = 250

    # Generate all 4-subsets as bitmasks (0-indexed internally, 1-indexed in output)
    all_sets = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)

    total = len(all_sets)
    best_family = []
    rng = random.Random(12345)

    for run in range(500):
        if time.time() > deadline:
            break

        order = list(range(total))
        rng.shuffle(order)

        F = []
        # inter_maps[i] maps intersection_mask -> list of indices j in F where F[i] & F[j] == intersection_mask
        inter_maps = []

        for idx in order:
            if time.time() > deadline:
                break

            m = all_sets[idx]
            conflict = False

            for i in range(len(F)):
                s = m & F[i]
                lst = inter_maps[i].get(s)
                if lst:
                    for j in lst:
                        if (m & F[j]) == s:
                            conflict = True
                            break
                if conflict:
                    break

            if not conflict:
                idx_new = len(F)
                F.append(m)
                inter_maps.append({})

                for i in range(idx_new):
                    s = m & F[i]
                    # Update inter_maps[i]
                    if s not in inter_maps[i]:
                        inter_maps[i][s] = []
                    inter_maps[i][s].append(idx_new)
                    # Update inter_maps[idx_new]
                    if s not in inter_maps[idx_new]:
                        inter_maps[idx_new][s] = []
                    inter_maps[idx_new][s].append(i)

                if len(F) >= max_target:
                    break

        if len(F) > len(best_family):
            best_family = F[:]
            if len(best_family) >= max_target:
                break

    # Convert best_family masks to sorted 1-indexed lists
    result = []
    for mask in best_family:
        elems = []
        for i in range(n):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))

    return result
# EVOLVE-BLOCK-END
