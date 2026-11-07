"""Combined program: precomputed comparability bitset + O(1) swap local search."""
# EVOLVE-BLOCK-START
import time

P = {'n': 12, 'm': 40}

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin under 90s wall-clock

    n = P['n']
    m = P['m']
    total = 1 << n  # 4096

    # ---- Precompute comparability bitsets ----
    # comp[a] is a Python int whose bit b is 1 iff a and b are comparable (a != b).
    # Two distinct masks a,b are comparable iff (a & b) == a or (a & b) == b.
    comp = [0] * total
    for a in range(total):
        bits = 0
        for b in range(total):
            if a == b:
                continue
            if (a & b) == a or (a & b) == b:
                bits |= (1 << b)
        comp[a] = bits

    # ---- Initial solution: structured chains, then sort by size for a denser start ----
    # Build several chains so the initial solution already has many comparable pairs.
    used = set()
    init_masks = []
    # 10 chains of 4 (sizes 1,2,3,4) using rotating offsets to keep them distinct
    for i in range(10):
        acc = 0
        for j in range(4):
            acc |= (1 << ((i + j) % 12))
            if acc not in used:
                used.add(acc)
                init_masks.append(acc)
    # Pad with a few extra distinct masks if needed (shouldn't be needed)
    if len(init_masks) < m:
        for k in range(total):
            if k not in used:
                used.add(k)
                init_masks.append(k)
            if len(init_masks) == m:
                break
    init_masks = init_masks[:m]

    # Sort by size (popcount) ascending so chains are aligned; gives a strong baseline.
    init_masks.sort(key=lambda x: x.bit_count())

    selected = init_masks[:]
    selected_mask = 0
    for mk in selected:
        selected_mask |= (1 << mk)

    # ---- Exact score via bitsets ----
    def score_of(sel, sel_mask):
        total_score = 0
        for i in range(m):
            rest = sel_mask & ~(1 << sel[i])
            total_score += (comp[sel[i]] & rest).bit_count()
        # each comparable pair counted twice (once from each endpoint)
        return total_score // 2

    current_score = score_of(selected, selected_mask)
    best_score = current_score
    best_selected = selected[:]

    def try_full_pass():
        """For each slot, scan all candidate masks, take best improving swap."""
        nonlocal selected, selected_mask, current_score, best_score, best_selected
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(m):
                if time.time() >= deadline:
                    return
                old = selected[idx]
                rest = selected_mask & ~(1 << old)
                loss = (comp[old] & rest).bit_count()
                best_new = -1
                best_delta = 0
                for new in range(total):
                    if (selected_mask >> new) & 1:
                        continue
                    gain = (comp[new] & rest).bit_count()
                    delta = gain - loss
                    if delta > best_delta:
                        best_delta = delta
                        best_new = new
                if best_new >= 0 and best_delta > 0:
                    selected[idx] = best_new
                    selected_mask = (selected_mask & ~(1 << old)) | (1 << best_new)
                    current_score += best_delta
                    if current_score > best_score:
                        best_score = current_score
                        best_selected = selected[:]
                    improved = True

    # ---- Local search loop: alternate exhaustive passes and random rebuilds ----
    import random
    random.seed(12345)

    pass_count = 0
    while time.time() < deadline:
        # Exhaustive best-swap pass (strong refinement)
        try_full_pass()
        pass_count += 1

        if time.time() >= deadline:
            break

        # Random single-slot rebuilds to escape local optima (now O(1) each)
        random_budget = min(20000, max(0, int((deadline - time.time()) * 2000)))
        for _ in range(random_budget):
            if time.time() >= deadline:
                break
            idx = random.randrange(m)
            old = selected[idx]
            rest = selected_mask & ~(1 << old)
            loss = (comp[old] & rest).bit_count()
            # pick a random candidate
            new = random.randrange(total)
            if (selected_mask >> new) & 1:
                continue
            gain = (comp[new] & rest).bit_count()
            delta = gain - loss
            if delta > 0:
                selected[idx] = new
                selected_mask = (selected_mask & ~(1 << old)) | (1 << new)
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_selected = selected[:]

    # ---- Convert best solution to lists of ints ----
    result = []
    for mask in best_selected:
        result.append([b for b in range(n) if (mask >> b) & 1])

    final_result = []
    for lst in result:
        final_result.append([int(x) for x in lst])

    return final_result
# EVOLVE-BLOCK-END
