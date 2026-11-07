import time
import random

def construct():
    n = 12
    m = 40
    total_masks = 1 << n  # 4096

    start_time = time.time()
    deadline = start_time + 82.0  # safety margin under 90s wall-clock

    # ---- Precompute comparability table (O(1) lookup in inner loop) ----
    # _comp[a*total_masks + b] = 1 iff a and b are comparable (one strict subset of other)
    _comp = bytearray(total_masks * total_masks)
    for a in range(total_masks):
        base = a * total_masks
        for b in range(total_masks):
            if a != b and ((a & b) == a or (a & b) == b):
                _comp[base + b] = 1

    # ---- Build a strong initial family: main chain + side branches ----
    fam_set = set()
    fam = []

    def add(mask):
        if mask not in fam_set:
            fam_set.add(mask)
            fam.append(mask)

    # Main chain: {}, {0}, {0,1}, ..., {0..11}  (13 sets)
    for i in range(n + 1):
        add((1 << i) - 1)

    # Side branches: for several chain prefixes, add sets that extend the prefix
    # with extra elements, creating additional subset relations.
    # Prefix {0..i-1} plus a "new" element j > i-1, plus maybe more.
    used_prefix = list(range(n + 1))
    random.shuffle(used_prefix)
    for i in used_prefix:
        if len(fam) >= m:
            break
        # prefix mask = {0..i-1}
        prefix = (1 << i) - 1
        # try adding elements beyond i-1
        for j in range(i, n):
            if len(fam) >= m:
                break
            add(prefix | (1 << j))
            # extend one more
            for k in range(j + 1, n):
                if len(fam) >= m:
                    break
                add(prefix | (1 << j) | (1 << k))
                # extend one more
                for ell in range(k + 1, n):
                    if len(fam) >= m:
                        break
                    add(prefix | (1 << j) | (1 << k) | (1 << ell))

    # Fill any remaining slots with random masks
    while len(fam) < m:
        mask = random.randint(0, total_masks - 1)
        if mask not in fam_set:
            fam_set.add(mask)
            fam.append(mask)

    # ---- Fast count of comparable pairs between x and fam, skipping skip_idx ----
    def count_comp(x, fam, skip_idx):
        base = x * total_masks
        c = 0
        for j, y in enumerate(fam):
            if j != skip_idx and _comp[base + y]:
                c += 1
        return c

    # ---- Initial score ----
    current_score = 0
    for i in range(m):
        current_score += count_comp(fam[i], fam, i)
    current_score //= 2

    best_fam = fam[:]
    best_score = current_score

    # ---- Local search: hill-climbing via single-position swaps ----
    no_improve_passes = 0
    max_no_improve = 30

    while time.time() < deadline:
        improved = False
        for i in range(m):
            if time.time() > deadline:
                break
            old = fam[i]
            old_comp = count_comp(old, fam, i)
            # Try several random candidates for position i
            for _ in range(12):
                cand = random.randint(0, total_masks - 1)
                if cand in fam_set:
                    continue
                gain = count_comp(cand, fam, i) - old_comp
                if gain > 0:
                    fam_set.remove(old)
                    fam_set.add(cand)
                    fam[i] = cand
                    current_score += gain
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_fam = fam[:]
                    break  # move to next index i

        if not improved:
            no_improve_passes += 1
            if no_improve_passes >= max_no_improve:
                # Perturb: randomly swap a few elements to escape local optimum
                for _ in range(5):
                    i = random.randrange(m)
                    old = fam[i]
                    for _ in range(20):
                        cand = random.randint(0, total_masks - 1)
                        if cand in fam_set:
                            continue
                        fam_set.remove(old)
                        fam_set.add(cand)
                        fam[i] = cand
                        # recompute score
                        current_score = 0
                        for k in range(m):
                            current_score += count_comp(fam[k], fam, k)
                        current_score //= 2
                        if current_score > best_score:
                            best_score = current_score
                            best_fam = fam[:]
                        break
                no_improve_passes = 0
        else:
            no_improve_passes = 0

    # Use the best family found
    result = []
    for mask in best_fam:
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)

    return result
