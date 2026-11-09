import time

L = 5354228880  # lcm(1..24)

def construct():
    start_time = time.time()
    deadline = start_time + 85

    weights = [0] * 25
    for i in range(1, 25):
        weights[i] = L // i

    elements = list(range(24, 0, -1))
    n = len(elements)

    # Precompute suffix sums of weights for tighter pruning
    suffix_sum = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + weights[elements[i]]

    best_set = set()
    best_size = 0

    def backtrack(idx, selected, sums):
        nonlocal best_set, best_size

        if time.time() > deadline:
            return

        remaining = n - idx
        if len(selected) + remaining <= best_size:
            return

        if len(selected) > best_size:
            best_size = len(selected)
            best_set = set(selected)

        if idx >= n:
            return

        x = elements[idx]
        w = weights[x]

        # Try include x if no conflict
        if w not in sums:
            new_selected = selected + [x]
            new_sums = set(sums)
            for s in sums:
                ns = s + w
                if ns <= suffix_sum[idx + 1]:
                    new_sums.add(ns)
            if w <= suffix_sum[idx + 1]:
                new_sums.add(w)
            if not new_sums:
                new_sums = {0}
            backtrack(idx + 1, new_selected, new_sums)

        # Try exclude x
        backtrack(idx + 1, selected, sums)

    backtrack(0, [], {0})

    # Local improvement: try adding elements one at a time
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for y in range(24, 0, -1):
                if y in best_set:
                    continue
                wy = weights[y]
                cur_w = [weights[z] for z in best_set]

                sums = {0}
                for w in cur_w:
                    new_s = set()
                    for s in sums:
                        ns = s + w
                        if ns <= wy:
                            new_s.add(ns)
                    sums.update(new_s)

                if wy in sums:
                    continue

                valid = True
                for z in best_set:
                    wz = weights[z]
                    others = [weights[zz] for zz in best_set if zz != z] + [wy]
                    sz = {0}
                    for w in others:
                        if w > wz:
                            continue
                        new_sz = set()
                        for s in sz:
                            ns = s + w
                            if ns <= wz:
                                new_sz.add(ns)
                        sz.update(new_sz)
                    if wz in sz:
                        valid = False
                        break

                if valid:
                    best_set.add(y)
                    improved = True
                    break

    # Try 1-for-2 swap: remove one element, add two others
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for z in list(best_set):
                trial_set = best_set - {z}
                for y1 in range(24, 0, -1):
                    if y1 in best_set:
                        continue
                    w1 = weights[y1]
                    cur_w = [weights[zz] for zz in trial_set]
                    sums1 = {0}
                    for w in cur_w:
                        new_s = set()
                        for s in sums1:
                            ns = s + w
                            if ns <= w1:
                                new_s.add(ns)
                        sums1.update(new_s)
                    if w1 in sums1:
                        continue
                    valid1 = True
                    for zz in trial_set:
                        wz = weights[zz]
                        others = [weights[ww] for ww in trial_set if ww != zz] + [w1]
                        sz = {0}
                        for w in others:
                            if w > wz:
                                continue
                            new_sz = set()
                            for s in sz:
                                ns = s + w
                                if ns <= wz:
                                    new_sz.add(ns)
                            sz.update(new_sz)
                        if wz in sz:
                            valid1 = False
                            break
                    if not valid1:
                        continue

                    trial_set2 = trial_set | {y1}
                    for y2 in range(24, 0, -1):
                        if y2 in trial_set2:
                            continue
                        w2 = weights[y2]
                        cur_w2 = [weights[zz] for zz in trial_set2]
                        sums2 = {0}
                        for w in cur_w2:
                            new_s = set()
                            for s in sums2:
                                ns = s + w
                                if ns <= w2:
                                    new_s.add(ns)
                            sums2.update(new_s)
                        if w2 in sums2:
                            continue
                        valid2 = True
                        for zz in trial_set2:
                            wz = weights[zz]
                            others = [weights[ww] for ww in trial_set2 if ww != zz] + [w2]
                            sz = {0}
                            for w in others:
                                if w > wz:
                                    continue
                                new_sz = set()
                                for s in sz:
                                    ns = s + w
                                    if ns <= wz:
                                        new_sz.add(ns)
                                sz.update(new_sz)
                            if wz in sz:
                                valid2 = False
                                break
                        if valid2:
                            best_set = trial_set2 | {y2}
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

    if not best_set:
        best_set = {24}

    return list(best_set)
