"""Randomized greedy covering for C(v,k,t) with bitmask bookkeeping,
randomized restarts, a final redundancy-removal pass, and a deadline guard."""
import time
import random
import itertools


def construct(v, k, t):
    t0 = time.time()
    DEADLINE = 146.0          # wall-clock safety margin before the 150s kill
    MARGIN = 4.0              # stop searching this far before the deadline

    # ---- enumerate all t-subsets and map them to bit indices ----
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    comb_idx = {c: i for i, c in enumerate(all_t)}

    # ---- precompute per-point coverage masks (bit i set iff point in t-subset i) ----
    sub_mask = [0] * v
    for i, c in enumerate(all_t):
        bit = 1 << i
        for p in c:
            sub_mask[p] |= bit

    full = (1 << n_t) - 1

    def block_mask(block):
        m = 0
        for p in block:
            m |= sub_mask[p]
        return m

    def greedy(shuffle_order, rng):
        """Build one covering. Returns (blocks, covered_mask)."""
        if shuffle_order:
            order = list(range(v))
            rng.shuffle(order)
        else:
            order = list(range(v))
        uncovered = full
        blocks = []
        bmasks = []
        while uncovered:
            # pick the first (in order) uncovered t-subset to anchor the block
            anchor = None
            for p in order:
                if (uncovered & sub_mask[p]) != 0:
                    anchor = p
                    break
            if anchor is None:
                break
            block = [anchor]
            used = 1 << anchor
            bm = sub_mask[anchor]
            while len(block) < k:
                best = -1
                best_score = -1
                for p in order:
                    if (used >> p) & 1:
                        continue
                    s = (uncovered & sub_mask[p]).bit_count()
                    if s > best_score:
                        best_score = s
                        best = p
                if best < 0:
                    break
                block.append(best)
                used |= 1 << best
                bm |= sub_mask[best]
            block.sort()
            blocks.append(block)
            bmasks.append(bm)
            uncovered &= ~bm
        return blocks, bmasks, uncovered

    def is_valid(bmasks):
        cov = 0
        for bm in bmasks:
            cov |= bm
        return cov == full

    def remove_redundant(blocks, bmasks):
        """Drop blocks whose coverage is fully contained in the rest. Returns (blocks, bmasks)."""
        if len(bmasks) < 2:
            return blocks, bmasks
        keep = list(bmasks)
        changed = True
        while changed:
            changed = False
            total = 0
            for bm in keep:
                total |= bm
            for i in range(len(keep) - 1, -1, -1):
                rest = total & ~keep[i]
                if (rest | keep[i]) == rest:
                    # keep[i] adds nothing new -> redundant
                    del keep[i]
                    del blocks[i]
                    changed = True
        return blocks, keep

    best_blocks = None
    best_bmasks = None
    best_n = None

    def consider(blocks, bmasks):
        nonlocal best_blocks, best_bmasks, best_n
        if not is_valid(bmasks):
            return
        if best_n is None or len(blocks) < best_n:
            best_blocks = blocks
            best_bmasks = bmasks
            best_n = len(blocks)

    # deterministic baseline first (guarantees a valid result)
    b, bm, unc = greedy(False, None)
    if unc == 0:
        consider(b, bm)

    # randomized restarts
    rng = random.Random(12345)
    while time.time() + MARGIN < DEADLINE:
        b, bm, unc = greedy(True, rng)
        if unc == 0:
            consider(b, bm)
        # occasional local improvement on the current best
        if best_blocks is not None and rng.random() < 0.3:
            b2, bm2 = remove_redundant(list(best_blocks), list(best_bmasks))
            consider(b2, bm2)

    # final cleanup on the best result
    if best_blocks is not None:
        b2, bm2 = remove_redundant(list(best_blocks), list(best_bmasks))
        consider(b2, bm2)

    return best_blocks
