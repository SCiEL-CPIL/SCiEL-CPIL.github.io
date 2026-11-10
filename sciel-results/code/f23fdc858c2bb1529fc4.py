"""Combined approach: fast backtracking Sidon evaluation + clustered local search."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def _largest_sidon(s):
    """Exact largest subset of s with all unordered pair sums (incl. diagonals) unique."""
    n = len(s)
    if n == 0:
        return 0
    s = list(s)
    pair_sum = [[s[i] + s[j] for j in range(n)] for i in range(n)]
    best = 0

    def bt(idx, chosen, sumset):
        nonlocal best
        rem = n - idx
        if len(chosen) + rem <= best:
            return
        if idx == n:
            if len(chosen) > best:
                best = len(chosen)
            return
        # skip idx
        bt(idx + 1, chosen, sumset)
        # try to take idx
        v = s[idx]
        new_sums = []
        ok = True
        if (v + v) in sumset:
            ok = False
        else:
            new_sums.append(v + v)
            for j in chosen:
                ss = v + s[j]
                if ss in sumset:
                    ok = False
                    break
                new_sums.append(ss)
        if ok:
            for ns in new_sums:
                sumset.add(ns)
            chosen.append(idx)
            bt(idx + 1, chosen, sumset)
            chosen.pop()
            for ns in new_sums:
                sumset.remove(ns)

    bt(0, [], set())
    return best

def _count_conflicts(s):
    """Count number of 4-element subsets that are NOT Sidon (as a proxy for density of collisions)."""
    n = len(s)
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    subset = [s[i], s[j], s[k], s[l]]
                    sums = set()
                    valid = True
                    for a in range(4):
                        for b in range(a, 4):
                            sm = subset[a] + subset[b]
                            if sm in sums:
                                valid = False
                                break
                            sums.add(sm)
                        if not valid:
                            break
                    if not valid:
                        count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin under 90s

    def try_set(vals):
        nonlocal best_set, best_val
        vals = sorted(set(vals))
        if len(vals) != 12:
            return
        v = _largest_sidon(vals)
        if v < best_val:
            best_val = v
            best_set = list(vals)

    best_set = None
    best_val = 10**9

    # Seed with clustered candidates - dense sets have more sum collisions
    structured = [
        list(range(12)),           # 0-11
        list(range(1, 13)),        # 1-12
        list(range(2, 14)),        # 2-13
        [0,1,2,3,4,5,6,7,8,9,10,20],
        [0,1,2,3,4,5,6,7,8,9,11,12],
        [0,1,2,3,4,5,6,7,8,10,11,12],
        [0,1,2,3,4,5,6,7,9,10,11,12],
        [0,1,2,3,4,5,6,8,9,10,11,12],
        [0,1,2,3,4,5,7,8,9,10,11,12],
        [0,1,2,3,4,6,7,8,9,10,11,12],
        [0,1,2,3,5,6,7,8,9,10,11,12],
        [0,1,2,4,5,6,7,8,9,10,11,12],
        [0,1,3,4,5,6,7,8,9,10,11,12],
        [0,2,3,4,5,6,7,8,9,10,11,12],
        [1,2,3,4,5,6,7,8,9,10,11,12],
        [0,1,2,3,4,5,6,7,8,9,10,11],
        # More clustered variants
        [0,1,2,3,4,5,6,7,8,9,10,15],
        [0,1,2,3,4,5,6,7,8,9,10,14],
        [0,1,2,3,4,5,6,7,8,9,10,13],
        [0,1,2,3,4,5,6,7,8,9,10,12],
        [0,1,2,3,4,5,6,7,8,9,10,11],
        [0,1,2,3,4,5,6,7,8,9,11,13],
        [0,1,2,3,4,5,6,7,8,9,11,14],
        [0,1,2,3,4,5,6,7,8,9,11,15],
        [0,1,2,3,4,5,6,7,8,9,12,13],
        [0,1,2,3,4,5,6,7,8,9,12,14],
        [0,1,2,3,4,5,6,7,8,9,12,15],
        [0,1,2,3,4,5,6,7,8,10,11,13],
        [0,1,2,3,4,5,6,7,8,10,11,14],
        [0,1,2,3,4,5,6,7,8,10,11,15],
        [0,1,2,3,4,5,6,7,8,10,12,13],
        [0,1,2,3,4,5,6,7,8,10,12,14],
        [0,1,2,3,4,5,6,7,8,10,12,15],
        [0,1,2,3,4,5,6,7,8,10,13,14],
        [0,1,2,3,4,5,6,7,8,10,13,15],
        [0,1,2,3,4,5,6,7,8,10,14,15],
    ]

    for sc in structured:
        if time.time() > deadline - 5:
            break
        try_set(sc)

    # Local search: try swaps that keep set dense
    rng = random.Random(42)
    if best_set is None:
        best_set = list(range(12))
        best_val = _largest_sidon(best_set)

    current = list(best_set)
    current_val = best_val

    no_improve_count = 0
    max_no_improve = 200

    while time.time() < deadline - 3:
        improved = False

        # Try single-element replacements with emphasis on nearby values
        for i in range(12):
            if time.time() > deadline - 3:
                break
            old_val = current[i]
            # Try values near the current value to maintain density
            candidates_to_try = []
            # Nearby values (within 5)
            for dv in range(-5, 6):
                nv = old_val + dv
                if 0 <= nv <= 128 and nv != old_val and nv not in current:
                    candidates_to_try.append(nv)
            # Also try some random values but biased toward low range
            for _ in range(10):
                nv = rng.randint(0, min(30, 128))
                if nv != old_val and nv not in current:
                    candidates_to_try.append(nv)

            for nv in candidates_to_try:
                if time.time() > deadline - 3:
                    break
                trial = current.copy()
                trial[i] = nv
                trial.sort()
                tv = _largest_sidon(trial)
                if tv < current_val:
                    current = trial
                    current_val = tv
                    improved = True
                    break
            if improved:
                break

        if not improved:
            # Try random perturbation: swap two elements
            if no_improve_count < max_no_improve:
                for _ in range(50):
                    if time.time() > deadline - 3:
                        break
                    i = rng.randint(0, 11)
                    j = rng.randint(0, 11)
                    if i == j:
                        continue
                    # Try swapping or replacing
                    trial = current.copy()
                    nv = rng.randint(0, 40)
                    if nv in trial:
                        continue
                    trial[i] = nv
                    trial.sort()
                    tv = _largest_sidon(trial)
                    if tv < current_val:
                        current = trial
                        current_val = tv
                        improved = True
                        break
            if not improved:
                no_improve_count += 1
            else:
                no_improve_count = 0

        if current_val <= 4:
            # Can't do much better
            if current_val < best_val:
                best_val = current_val
                best_set = list(current)
            break

        if current_val < best_val:
            best_val = current_val
            best_set = list(current)
            no_improve_count = 0

    if best_set is None:
        return list(range(12))
    return sorted(best_set)
# EVOLVE-BLOCK-END
