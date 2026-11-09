"""Greedy covering with bitset counting + local-search refinement for C(v,k,t)."""
# EVOLVE-BLOCK-START
import time
import itertools


def _comb_index(n, r, c):
    # Combinadic (colex) rank of combination c (strictly increasing, 0-based).
    idx = 0
    for j in range(r):
        idx += (1 << (c[j] - 1 - j))
    return idx


def construct(v, k, t):
    start = time.time()
    deadline = start + 140.0          # safety margin vs the 150s wall-clock limit

    # Enumerate all t-subsets and assign each a bit position.
    tsubs = list(itertools.combinations(range(v), t))
    num_t = len(tsubs)
    tsub_index = {c: i for i, c in enumerate(tsubs)}

    # For every point, the set (as a bitmask) of t-subsets containing it.
    point_mask = [0] * v
    for c in tsubs:
        i = tsub_index[c]
        bit = 1 << i
        for x in c:
            point_mask[x] |= bit

    # For every k-subset, the bitmask of t-subsets it covers.
    ksubs = list(itertools.combinations(range(v), k))
    ksub_mask = []
    for c in ksubs:
        m = 0
        for x in c:
            m |= point_mask[x]
        ksub_mask.append(m)
    ksub_index = {c: i for i, c in enumerate(ksubs)}

    def mask_of_block(block):
        m = 0
        for x in block:
            m |= point_mask[x]
        return m

    # ---------- Greedy construction ----------
    uncovered = (1 << num_t) - 1
    blocks = []
    while uncovered:
        # Start the block on the rarest uncovered t-subset (fewest covering k-blocks).
        first = None
        best_count = None
        for i in range(num_t):
            if (uncovered >> i) & 1:
                c = tsubs[i]
                cnt = 0
                for x in c:
                    cnt |= point_mask[x]
                cnt = cnt.bit_count()
                if best_count is None or cnt < best_count:
                    best_count = cnt
                    first = c
                    if cnt == 1:
                        break
        if first is None:
            break
        block = list(first)
        block_set = set(block)
        while len(block) < k:
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                gain_mask = 0
                for rest in itertools.combinations(block, t - 1):
                    gain_mask |= point_mask[x] & (1 << tsub_index[rest])
                g = (gain_mask & uncovered).bit_count()
                if g > best_gain:
                    best_gain = g
                    best_x = x
            if best_x < 0:
                break
            block.append(best_x)
            block_set.add(best_x)
        block.sort()
        blocks.append(block)
        uncovered &= ~mask_of_block(block)

    # ---------- Local-search refinement ----------
    def uncovered_count(blks):
        u = 0
        for b in blks:
            u &= ~mask_of_block(b)
        return u.bit_count()

    def best_gain_block(uncovered_mask, exclude):
        best_i = -1
        best_g = -1
        for i in range(len(ksubs)):
            if exclude is not None and i == exclude:
                continue
            g = (ksub_mask[i] & uncovered_mask).bit_count()
            if g > best_g:
                best_g = g
                best_i = i
        return best_i, best_g

    best_blocks = list(blocks)
    best_unc = uncovered_count(best_blocks)
    if best_unc == 0:
        return [list(b) for b in best_blocks]

    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Work with a copy so we can roll back.
        cur = list(best_blocks)
        cur_unc = best_unc
        for idx in range(len(cur)):
            if time.time() >= deadline:
                break
            # Mask of t-subsets covered ONLY by this block.
            others = 0
            for j in range(len(cur)):
                if j != idx:
                    others |= mask_of_block(cur[j])
            only = mask_of_block(cur[idx]) & ~others
            # If we remove this block, what becomes uncovered?
            new_uncovered = cur_unc & ~others
            new_uncovered_mask = 0
            # Build the mask of newly uncovered t-subsets (those only this block covered).
            new_uncovered_mask = only
            # Find a replacement block covering the most of new_uncovered_mask.
            bi, g = best_gain_block(new_uncovered_mask, exclude=ksub_index.get(tuple(cur[idx]), -1))
            if bi < 0:
                continue
            if g >= new_uncovered_mask.bit_count():
                # Replacement covers everything the old block uniquely covered.
                new_unc = cur_unc - new_uncovered_mask.bit_count() + (new_uncovered_mask.bit_count() - g)
                # new_unc = cur_unc - g  (we drop old block, add new covering g of the lost)
                new_unc = cur_unc - g
                if new_unc < cur_unc:
                    cur[idx] = list(ksubs[bi])
                    cur_unc = new_unc
                    improved = True
                    break  # restart scan with updated state

        if improved:
            best_blocks = cur
            best_unc = cur_unc
            if best_unc == 0:
                break

    if time.time() < deadline and best_unc > 0:
        # Final pass: drop any block that is now redundant.
        for idx in range(len(best_blocks)):
            if time.time() >= deadline:
                break
            others = 0
            for j in range(len(best_blocks)):
                if j != idx:
                    others |= mask_of_block(best_blocks[j])
            if (mask_of_block(best_blocks[idx]) & ~others) == 0:
                best_blocks.pop(idx)
                break

    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
