import random
import time

N = 256
MAX_SUM = 2 * N  # 512

def _init_masks(sel):
    """Precompute for each element x the bitmask of sums it creates with existing sel."""
    masks = {}
    for x in sel:
        m = 0
        for a in sel:
            m |= 1 << (x + a)
        m |= 1 << (2 * x)
        masks[x] = m
    return masks

def _build_min_max(deadline, seed=None):
    """
    Construct an inclusion-maximal Sidon set by always picking the element
    that blocks the most remaining elements.  This tends to produce a small
    set after saturation because few elements remain addable.
    """
    rng = random.Random(seed)
    sel = []
    sum_mask = 0
    remaining = set(range(1, N + 1))

    # Precompute all pair-sum masks for each element against the full range
    # mask_all[x] = bitmask of sums x+a for a in [1..N] plus 2x
    # We'll compute on the fly to save memory; N=256 is small.

    while remaining and time.time() < deadline:
        best_x = None
        best_block = -1
        # Sample a subset if remaining is large to save time
        if len(remaining) > 80:
            candidates = rng.sample(list(remaining), 80)
        else:
            candidates = list(remaining)

        for x in candidates:
            # Check if x can be added: no sum x+a (a in sel) or 2x already used
            ok = True
            for a in sel:
                if (sum_mask >> (x + a)) & 1:
                    ok = False
                    break
            if not ok:
                continue
            if (sum_mask >> (2 * x)) & 1:
                continue
            # Count how many remaining elements this would block
            block = 0
            for y in remaining:
                if y == x:
                    continue
                # y is blocked if any sum y+a (a in sel) or 2y or x+y is already used
                # After adding x, new sums are x+a for a in sel, and 2x.
                # y is blocked if (y+a) in new_sums for some a in sel, or 2y in new_sums, or x+y in used_or_new
                blocked = False
                for a in sel:
                    if (sum_mask >> (y + a)) & 1:
                        blocked = True
                        break
                    if (sum_mask >> (x + a)) & 1 and (y == a or True):
                        # Actually we need: after adding x, sum y+x becomes available.
                        # y is blocked if y+x is in sum_mask OR y+x == 2x (i.e., y==x, already excluded)
                        # OR y+a is in new sums (x+a) for some a => y+x = x+a => y=a (impossible since y not in sel)
                        # Simpler: y is blocked if 2y is in new_sums or y+x is in sum_mask or y+a in new_sums
                        pass
                # More efficient: y is blocked if (2y in new_sums) or (x+y in sum_mask) or (y+a in new_sums for some a)
                # new_sums = {x+a for a in sel} ∪ {2x}
                if (sum_mask >> (2 * y)) & 1:
                    blocked = True
                elif (sum_mask >> (x + y)) & 1:
                    blocked = True
                else:
                    for a in sel:
                        if (sum_mask >> (y + a)) & 1:
                            blocked = True
                            break
                        # y+a could equal x+a' => y=a' (impossible)
                if blocked:
                    block += 1
            if block > best_block:
                best_block = block
                best_x = x

        if best_x is None:
            break

        # Add best_x
        sel.append(best_x)
        for a in sel[:-1]:
            sum_mask |= 1 << (best_x + a)
        sum_mask |= 1 << (2 * best_x)
        remaining.discard(best_x)

    # Saturate: add all remaining addable elements in ascending order
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in range(1, N + 1):
            if x in sel:
                continue
            ok = True
            for a in sel:
                if (sum_mask >> (x + a)) & 1:
                    ok = False
                    break
            if not ok:
                continue
            if (sum_mask >> (2 * x)) & 1:
                continue
            sel.append(x)
            for a in sel[:-1]:
                sum_mask |= 1 << (x + a)
            sum_mask |= 1 << (2 * x)
            changed = True

    return sel

def _is_valid_sidon(sel):
    sums = set()
    for i in range(len(sel)):
        a = sel[i]
        for j in range(i, len(sel)):
            s = a + sel[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(sel):
    if not _is_valid_sidon(sel):
        return False
    sums = set()
    for i in range(len(sel)):
        a = sel[i]
        for j in range(i, len(sel)):
            sums.add(a + sel[j])
    sel_set = set(sel)
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        ok = True
        for a in sel:
            if x + a in sums:
                ok = False
                break
        if not ok:
            continue
        if 2 * x in sums:
            continue
        # x can be added, so not maximal
        return False
    return True

def _local_search_reduce(sel, deadline, max_iters=50):
    """Try removing elements and re-saturating minimally to reduce size."""
    best = list(sel)
    best_size = len(best)

    for it in range(max_iters):
        if time.time() > deadline:
            break
        improved = False
        # Try removing each element
        for i in range(len(best)):
            if time.time() > deadline:
                break
            candidate = best[:i] + best[i + 1:]
            # Check if candidate is still Sidon
            sums = set()
            valid = True
            for a_idx in range(len(candidate)):
                a = candidate[a_idx]
                for b_idx in range(a_idx, len(candidate)):
                    s = a + candidate[b_idx]
                    if s in sums:
                        valid = False
                        break
                    sums.add(s)
                if not valid:
                    break
            if not valid:
                continue
            # Saturate candidate minimally (ascending order)
            changed = True
            while changed and time.time() < deadline:
                changed = False
                for x in range(1, N + 1):
                    if x in set(candidate):
                        continue
                    ok = True
                    for a in candidate:
                        if x + a in sums:
                            ok = False
                            break
                    if not ok:
                        continue
                    if 2 * x in sums:
                        continue
                    candidate.append(x)
                    for a in candidate[:-1]:
                        sums.add(x + a)
                    sums.add(2 * x)
                    changed = True
            if len(candidate) < best_size:
                best = candidate
                best_size = len(best)
                improved = True
                break
        if not improved:
            # Try swapping: remove one, add another, then saturate
            for i in range(len(best)):
                if time.time() > deadline:
                    break
                for x in range(1, N + 1):
                    if x in set(best):
                        continue
                    candidate = best[:i] + best[i + 1:]
                    # Check if x can be added to candidate
                    sums = set()
                    valid = True
                    for a_idx in range(len(candidate)):
                        a = candidate[a_idx]
                        for b_idx in range(a_idx, len(candidate)):
                            s = a + candidate[b_idx]
                            if s in sums:
                                valid = False
                                break
                            sums.add(s)
                        if not valid:
                            break
                    if not valid:
                        continue
                    ok = True
                    for a in candidate:
                        if x + a in sums:
                            ok = False
                            break
                    if not ok:
                        continue
                    if 2 * x in sums:
                        continue
                    candidate.append(x)
                    for a in candidate[:-1]:
                        sums.add(x + a)
                    sums.add(2 * x)
                    # Saturate
                    changed = True
                    while changed and time.time() < deadline:
                        changed = False
                        for y in range(1, N + 1):
                            if y in set(candidate):
                                continue
                            ok2 = True
                            for a in candidate:
                                if y + a in sums:
                                    ok2 = False
                                    break
                            if not ok2:
                                continue
                            if 2 * y in sums:
                                continue
                            candidate.append(y)
                            for a in candidate[:-1]:
                                sums.add(y + a)
                            sums.add(2 * y)
                            changed = True
                    if len(candidate) < best_size:
                        best = candidate
                        best_size = len(best)
                        improved = True
                        break
                if improved:
                    break
        if not improved:
            break

    return best

def construct():
    start = time.time()
    deadline = start + 55  # safety margin

    best_set = None
    best_size = float('inf')

    # Phase 1: Many randomized min-max constructions
    for trial in range(200):
        if time.time() > deadline:
            break
        sel = _build_min_max(deadline, seed=trial)
        if len(sel) < best_size:
            best_size = len(sel)
            best_set = list(sel)
            # Local search on this
            improved = _local_search_reduce(sel, deadline, max_iters=20)
            if len(improved) < best_size:
                best_size = len(improved)
                best_set = improved

    # Phase 2: More local search on the best
    if best_set is not None and time.time() < deadline - 5:
        improved = _local_search_reduce(best_set, deadline, max_iters=80)
        if len(improved) < best_size:
            best_set = improved

    # Fallback: simple greedy
    if best_set is None:
        sel = []
        sums = set()
        for x in range(1, N + 1):
            ok = True
            for a in sel:
                if x + a in sums:
                    ok = False
                    break
            if not ok:
                continue
            if 2 * x in sums:
                continue
            sel.append(x)
            for a in sel[:-1]:
                sums.add(x + a)
            sums.add(2 * x)
        best_set = sel

    return best_set
