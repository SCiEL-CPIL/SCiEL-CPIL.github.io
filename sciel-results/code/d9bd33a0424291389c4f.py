"""Randomized local search for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time


def _block_mask_list(block, subset_to_idx):
    """Return the list of t-subset indices covered by a block."""
    covered = []
    for c in itertools.combinations(block, 3):
        covered.append(subset_to_idx[frozenset(c)])
    return covered


def construct(v, k, t):
    rng = random.Random(7)
    start_time = time.time()
    time_limit = 117.0  # safety margin under 150s

    def time_left():
        return time_limit - (time.time() - start_time)

    # Precompute all t-subsets and index them
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_to_idx = {frozenset(s): i for i, s in enumerate(all_subsets)}

    # For each point, the set of t-subset indices containing it (for fast gain)
    point_to_subsets = [set() for _ in range(v)]
    for idx, s in enumerate(all_subsets):
        for p in s:
            point_to_subsets[p].add(idx)

    def block_mask(block):
        return _block_mask_list(block, subset_to_idx)

    # ---- Greedy initial covering ----
    uncovered = set(range(n_subsets))
    blocks = []
    block_set = set()

    while uncovered:
        seed_idx = rng.choice(list(uncovered))
        seed = all_subsets[seed_idx]
        block = list(seed)
        rng.shuffle(block)
        block_local = set(block)

        while len(block) < k:
            best_x = None
            best_gain = -1
            cands = [x for x in range(v) if x not in block_local]
            rng.shuffle(cands)
            for x in cands:
                # gain = # uncovered t-subsets that include x and k-1 points of block
                gain = len(point_to_subsets[x] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
            block_local.add(best_x)

        block.sort()
        bt = tuple(block)
        if bt not in block_set:
            block_set.add(bt)
            blocks.append(bt)
            for c in itertools.combinations(block, t):
                uncovered.discard(subset_to_idx[frozenset(c)])

    # ---- State: counts array, block masks ----
    counts = [0] * n_subsets
    block_masks = []
    for b in blocks:
        m = block_mask(b)
        block_masks.append(m)
        for idx in m:
            counts[idx] += 1

    best_blocks = [list(b) for b in blocks]
    best_count = len(blocks)
    no_improve = 0
    max_no_improve = 400
    T = 50.0
    T_min = 0.1
    cooling = 0.9995

    def try_remove_block():
        """Try to remove a redundant block. Returns True if removed."""
        nonlocal blocks, block_masks, counts, best_blocks, best_count, no_improve
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        for i in indices:
            redundant = True
            for idx in block_masks[i]:
                if counts[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                for idx in block_masks[i]:
                    counts[idx] -= 1
                blocks.pop(i)
                block_masks.pop(i)
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = [list(b) for b in blocks]
                no_improve = 0
                return True
        return False

    def try_swap_block():
        """Try to swap one block for a better one. Returns True if swapped."""
        nonlocal blocks, block_masks, counts, no_improve, T
        if not blocks:
            return False
        i = rng.randrange(len(blocks))
        old_block = blocks[i]
        old_mask = block_masks[i]

        critical = [idx for idx in old_mask if counts[idx] == 1]
        if not critical:
            return False

        points_needed = set()
        for idx in critical:
            points_needed.update(all_subsets[idx])
        if len(points_needed) > k:
            return False

        new_set = set(points_needed)
        cands = [x for x in range(v) if x not in new_set]
        rng.shuffle(cands)

        for x in cands:
            if len(new_set) >= k:
                break
            gain = 0
            for rest in itertools.combinations(new_set, t - 1):
                fs = frozenset(list(rest) + [x])
                idx = subset_to_idx[fs]
                if counts[idx] == 0:
                    gain += 10
                elif counts[idx] == 1:
                    gain += 1
            new_set.add(x)

        if len(new_set) != k:
            return False

        new_block = tuple(sorted(new_set))
        if new_block == old_block:
            return False

        new_mask = block_mask(new_block)
        new_set_mask = set(new_mask)
        for idx in old_mask:
            if idx not in new_set_mask and counts[idx] <= 1:
                return False

        for idx in old_mask:
            counts[idx] -= 1
        for idx in new_mask:
            counts[idx] += 1
        blocks[i] = new_block
        block_masks[i] = new_mask
        no_improve = 0
        return True

    def try_two_for_one():
        """Replace two blocks with one that covers their critical union.
        Returns True if successful."""
        nonlocal blocks, block_masks, counts, best_blocks, best_count, no_improve
        if len(blocks) < 2:
            return False
        i, j = rng.sample(range(len(blocks)), 2)
        b1 = blocks[i]
        b2 = blocks[j]
        m1 = block_masks[i]
        m2 = block_masks[j]

        crit1 = [idx for idx in m1 if counts[idx] == 1]
        crit2 = [idx for idx in m2 if counts[idx] == 1]

        points_needed = set()
        for idx in crit1:
            points_needed.update(all_subsets[idx])
        for idx in crit2:
            points_needed.update(all_subsets[idx])

        if len(points_needed) > k:
            return False

        new_set = set(points_needed)
        cands = [x for x in range(v) if x not in new_set]
        rng.shuffle(cands)

        for x in cands:
            if len(new_set) >= k:
                break
            gain = 0
            for rest in itertools.combinations(new_set, t - 1):
                fs = frozenset(list(rest) + [x])
                idx = subset_to_idx[fs]
                if counts[idx] == 0:
                    gain += 10
                elif counts[idx] == 1:
                    gain += 1
            new_set.add(x)

        if len(new_set) != k:
            return False

        new_block = tuple(sorted(new_set))
        if new_block in (b1, b2):
            return False

        new_mask = block_mask(new_block)
        new_mask_set = set(new_mask)

        for idx in crit1:
            if idx not in new_mask_set:
                return False
        for idx in crit2:
            if idx not in new_mask_set:
                return False

        m1_set = set(m1)
        m2_set = set(m2)
        valid = True
        for idx in m1_set:
            if idx not in new_mask_set:
                if idx not in m2_set:
                    valid = False
                    break
        if valid:
            for idx in m2_set:
                if idx not in new_mask_set:
                    if idx not in m1_set:
                        valid = False
                        break

        if not valid:
            return False

        for idx in m1:
            counts[idx] -= 1
        for idx in m2:
            counts[idx] -= 1
        for idx in new_mask:
            counts[idx] += 1

        hi = max(i, j)
        lo = min(i, j)
        blocks.pop(hi)
        block_masks.pop(hi)
        blocks.pop(lo)
        block_masks.pop(lo)
        blocks.append(new_block)
        block_masks.append(new_mask)

        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
        no_improve = 0
        return True

    def perturb():
        """Perturb: remove a random block and greedily re-cover.
        Returns True if valid covering restored."""
        nonlocal blocks, block_masks, counts, no_improve
        if len(blocks) <= 1:
            return False
        old_blocks = list(blocks)
        old_masks = list(block_masks)
        old_counts = list(counts)

        i = rng.randrange(len(blocks))
        b = blocks[i]
        m = block_masks[i]
        for idx in m:
            counts[idx] -= 1
        blocks.pop(i)
        block_masks.pop(i)

        unc_list = [idx for idx in range(n_subsets) if counts[idx] == 0]

        while unc_list:
            seed_idx = rng.choice(unc_list)
            seed = all_subsets[seed_idx]
            new_b = list(seed)
            rng.shuffle(new_b)
            new_local = set(new_b)

            while len(new_b) < k:
                best_x = None
                best_gain = -1
                cands = [x for x in range(v) if x not in new_local]
                rng.shuffle(cands)
                for x in cands:
                    gain = len(point_to_subsets[x] & set(unc_list))
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                new_b.append(best_x)
                new_local.add(best_x)

            new_b.sort()
            new_bt = tuple(new_b)
            if new_bt not in blocks:
                new_m = block_mask(new_bt)
                for idx in new_m:
                    counts[idx] += 1
                blocks.append(new_bt)
                block_masks.append(new_m)
                unc_list = [idx for idx in range(n_subsets) if counts[idx] == 0]

        if not unc_list:
            no_improve = 0
            return True
        else:
            blocks = old_blocks
            block_masks = old_masks
            counts = old_counts
            return False

    # ---- Main search loop ----
    while time_left() > 2.0:
        # Try removals first (cheap and always beneficial)
        if try_remove_block():
            continue

        # Try swaps
        if try_swap_block():
            T = max(T * cooling, T_min)
            continue

        # Try 2-for-1 replacements (key move for escaping local minima)
        if try_two_for_one():
            T = max(T * cooling, T_min)
            continue

        # If stuck, perturb
        no_improve += 1
        if no_improve > max_no_improve:
            perturb()
            no_improve = 0
        else:
            # Small random perturbation: swap a block with a random one
            if blocks and rng.random() < 0.3:
                i = rng.randrange(len(blocks))
                old_block = blocks[i]
                old_mask = block_masks[i]
                new_block = tuple(sorted(rng.sample(range(v), k)))
                if new_block != old_block:
                    new_mask = block_mask(new_block)
                    valid = True
                    new_mask_set = set(new_mask)
                    for idx in old_mask:
                        if idx not in new_mask_set and counts[idx] <= 1:
                            valid = False
                            break
                    if valid:
                        for idx in old_mask:
                            counts[idx] -= 1
                        for idx in new_mask:
                            counts[idx] += 1
                        blocks[i] = new_block
                        block_masks[i] = new_mask
                    else:
                        no_improve += 1

    # ---- Final high-precision polish pass ----
    # 1) Repeated exact redundancy sweep to a fixed point
    for _ in range(200):
        if time_left() < 1.0:
            break
        removed = True
        while removed:
            removed = try_remove_block()

    # 2) Exact 2-for-1 scan: for every pair of blocks, if their critical
    #    points fit in one block and a covering block exists, replace.
    for _ in range(50):
        if time_left() < 1.0:
            break
        progress = False
        n = len(blocks)
        for i in range(n):
            if time_left() < 1.0:
                break
            m1 = block_masks[i]
            crit1 = [idx for idx in m1 if counts[idx] == 1]
            if not crit1:
                continue
            pts1 = set()
            for idx in crit1:
                pts1.update(all_subsets[idx])
            if len(pts1) > k:
                continue
            for j in range(i + 1, n):
                m2 = block_masks[j]
                crit2 = [idx for idx in m2 if counts[idx] == 1]
                pts = set(pts1)
                for idx in crit2:
                    pts.update(all_subsets[idx])
                if len(pts) > k:
                    continue
                # Try to find a single block covering all of pts and both
                # critical sets, without breaking any other count.
                new_set = set(pts)
                cands = [x for x in range(v) if x not in new_set]
                # deterministic fill to maximize low-count coverage
                cands_sorted = sorted(cands, key=lambda x: -len(point_to_subsets[x]))
                for x in cands_sorted:
                    if len(new_set) >= k:
                        break
                    new_set.add(x)
                if len(new_set) != k:
                    continue
                new_block = tuple(sorted(new_set))
                if new_block in (blocks[i], blocks[j]):
                    continue
                new_mask = block_mask(new_block)
                new_mask_set = set(new_mask)
                ok = True
                for idx in crit1:
                    if idx not in new_mask_set:
                        ok = False
                        break
                if ok:
                    for idx in crit2:
                        if idx not in new_mask_set:
                            ok = False
                            break
                if not ok:
                    continue
                m1_set = set(m1)
                m2_set = set(m2)
                for idx in m1_set:
                    if idx not in new_mask_set and idx not in m2_set:
                        ok = False
                        break
                if ok:
                    for idx in m2_set:
                        if idx not in new_mask_set and idx not in m1_set:
                            ok = False
                            break
                if not ok:
                    continue
                # Apply
                for idx in m1:
                    counts[idx] -= 1
                for idx in m2:
                    counts[idx] -= 1
                for idx in new_mask:
                    counts[idx] += 1
                blocks.pop(j)
                block_masks.pop(j)
                blocks.pop(i)
                block_masks.pop(i)
                blocks.append(new_block)
                block_masks.append(new_mask)
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = [list(b) for b in blocks]
                progress = True
                break
            if progress:
                break
        if not progress:
            break

    # Final cleanup: try removals one more time
    while True:
        if not try_remove_block():
            break

    if len(blocks) < best_count:
        best_count = len(blocks)
        best_blocks = [list(b) for b in blocks]

    return best_blocks
# EVOLVE-BLOCK-END
