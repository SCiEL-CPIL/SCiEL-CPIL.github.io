"""Covering design C(v,k,t) builder: fast bitset greedy + iterative local search."""
import time
import itertools
import random
import numpy as np

_DEADLINE = 150.0
_MARGIN = 5.0


def _all_t_masks(v, t):
    """Return list of integer masks, one per t-subset, in lex order."""
    masks = []
    for combo in itertools.combinations(range(v), t):
        m = 0
        for x in combo:
            m |= 1 << x
        masks.append(m)
    return masks


def _blocks_to_set(blocks):
    return set(tuple(b) for b in blocks)


def _cover_mask(blocks, masks, v, t):
    """Union (OR) of masks of all t-subsets contained in the given blocks."""
    cm = 0
    for b in blocks:
        bs = set(b)
        for m in masks:
            if not (m & ~bs):
                cm |= m
    return cm


def _gain(block_set, cand, masks, v, t):
    """Cheap estimate: number of t-subsets that contain cand and whose remaining
    t-1 points are a subset of block_set."""
    bs = block_set
    cnt = 0
    for m in masks:
        if (m >> cand) & 1:
            rest = m & ~(1 << cand)
            if not (rest & ~bs):
                cnt += 1
    return cnt


def _greedy_construct(v, k, t, masks, rng, time_limit):
    """Greedy set-cover construction using bitset coverage. Returns a valid cover."""
    all_mask = (1 << (1 << t)) - 1
    covered = 0
    blocks = []
    while covered != all_mask:
        if time.time() > time_limit:
            break
        # find first uncovered t-subset (in lex order)
        idx = 0
        for i in range(len(masks)):
            if not (covered >> i) & 1:
                idx = i
                break
        block = [i for i in range(v) if (masks[idx] >> i) & 1]
        block_set = set(block)
        while len(block) < k:
            best_x, best_g = None, -1
            for x in range(v):
                if x in block_set:
                    continue
                g = _gain(block_set, x, masks, v, t)
                if g > best_g:
                    best_g, best_x = g, x
            block.append(best_x)
            block_set.add(best_x)
        block.sort()
        blocks.append(block)
        # update covered
        bs = set(block)
        for i, m in enumerate(masks):
            if not (covered >> i) & 1 and not (m & ~bs):
                covered |= 1 << i
    return blocks, covered


def _is_valid(blocks, masks, v, t):
    cm = 0
    for b in blocks:
        bs = set(b)
        for m in masks:
            cm |= m
            if not (m & ~bs):
                pass
    # recompute properly: OR all t-subsets contained in any block
    cm = 0
    for b in blocks:
        bs = set(b)
        for m in masks:
            if not (m & ~bs):
                cm |= m
    return cm == ((1 << len(masks)) - 1)


def _reoptimize_block(block, blocks, masks, v, t, rng, unique_mask, covered):
    """Try to replace `block` with a same-size block covering its unique t-subsets
    while keeping overall coverage valid. Return new block or None."""
    bs = set(block)
    # unique t-subsets: covered by this block but by no other block
    others_covered = 0
    for b in blocks:
        if b is block:
            continue
        o = set(b)
        for m in masks:
            if not (m & ~o):
                others_covered |= m
    unique = 0
    for m in masks:
        if not (m & ~bs) and not ((others_covered >> m.bit_length()) & 1) if False else False:
            pass
    # compute unique by index
    unique_idx = []
    for i, m in enumerate(masks):
        if not (m & ~bs):
            # in this block
            in_other = False
            for b in blocks:
                if b is block:
                    continue
                o = set(b)
                if not (m & ~o):
                    in_other = True
                    break
            if not in_other:
                unique_idx.append(i)
    if not unique_idx:
        return None
    # candidate block must contain all points appearing in unique t-subsets
    required = 0
    for i in unique_idx:
        required |= masks[i]
    # choose k points including required, optimizing coverage of remaining uniques
    req_list = [x for x in range(v) if (required >> x) & 1]
    if len(req_list) > k:
        return None
    # build candidate: start with required, fill rest greedily
    cand = list(req_list)
    cand_set = set(cand)
    # target: cover as many of the unique_idx t-subsets as possible
    # but they are all already in required? ensure all unique are covered
    # fill remaining slots to maximize coverage of unique_idx
    while len(cand) < k:
        best_x, best_g = None, -1
        for x in range(v):
            if x in cand_set:
                continue
            g = 0
            for i in unique_idx:
                m = masks[i]
                if not (m & ~cand_set):
                    continue
                # does adding x help cover this unique?
                if (m >> x) & 1 and not ((m & ~(1 << x)) & ~cand_set):
                    g += 1
            if g > best_g:
                best_g, best_x = g, x
        if best_x is None:
            break
        cand.append(best_x)
        cand_set.add(best_x)
    # verify candidate covers all unique_idx
    for i in unique_idx:
        if (masks[i] & ~cand_set):
            return None
    # verify overall coverage stays valid (no t-subset becomes uncovered)
    # i.e. every t-subset covered by old block is covered by candidate or others
    for i, m in enumerate(masks):
        if (m & ~bs):
            continue
        # was covered by old block; must be covered by candidate or an other block
        if not (m & ~cand_set):
            continue
        ok = False
        for b in blocks:
            if b is block:
                continue
            o = set(b)
            if not (m & ~o):
                ok = True
                break
        if not ok:
            return None
    cand.sort()
    return cand


def construct(v, k, t):
    start = time.time()
    deadline = start + _DEADLINE - _MARGIN
    rng = random.Random(12345)

    masks = _all_t_masks(v, t)
    n_t = len(masks)
    full_mask = (1 << n_t) - 1

    # ---- Phase 1: greedy construction ----
    blocks, covered = _greedy_construct(v, k, t, masks, rng, deadline)
    if covered != full_mask:
        # incomplete; do a final brute completion to guarantee validity
        covered_set = set()
        for b in blocks:
            bs = set(b)
            for m in masks:
                if not (m & ~bs):
                    covered_set.add(m)
        for m in masks:
            if m in covered_set:
                continue
            # build a block containing this t-subset
            block = [x for x in range(v) if (m >> x) & 1]
            block_set = set(block)
            while len(block) < k:
                x = rng.randrange(v)
                if x not in block_set:
                    block.append(x)
                    block_set.add(x)
            block.sort()
            blocks.append(block)
            bs = set(block)
            for mm in masks:
                if not (mm & ~bs):
                    covered_set.add(mm)

    blocks = _blocks_to_set(blocks)
    best_blocks = sorted(blocks)
    best_len = len(best_blocks)

    # ---- Phase 2: iterative local search (redundancy removal + re-optimization) ----
    while time.time() < deadline:
        progress = False

        # 2a: remove redundant blocks
        cur = sorted(blocks)
        n = len(cur)
        if n > 0:
            # coverage mask per block
            cov = [0] * n
            for i, b in enumerate(cur):
                bs = set(b)
                m = 0
                for j, mm in enumerate(masks):
                    if not (mm & ~bs):
                        m |= 1 << j
                cov[i] = m
            total = 0
            for i in range(n):
                total |= cov[i]
            for i in range(n):
                if total == 0:
                    break
                without = 0
                for j in range(n):
                    if j != i:
                        without |= cov[j]
                if without == total:
                    cur.pop(i)
                    cov.pop(i)
                    total = without
                    n -= 1
                    progress = True
                    i = -1
                    continue
                i += 1
            blocks = _blocks_to_set(cur)

        # 2b: re-optimize blocks that cover unique t-subsets
        cur = sorted(blocks)
        n = len(cur)
        if n > 0:
            cov = [0] * n
            for i, b in enumerate(cur):
                bs = set(b)
                m = 0
                for j, mm in enumerate(masks):
                    if not (mm & ~bs):
                        m |= 1 << j
                cov[i] = m
            total = 0
            for i in range(n):
                total |= cov[i]
            # find blocks with unique contribution
            for i in range(n):
                if time.time() > deadline:
                    break
                without = 0
                for j in range(n):
                    if j != i:
                        without |= cov[j]
                unique = total & ~without
                if unique == 0:
                    continue
                new_block = _reoptimize_block(cur[i], cur, masks, v, t, rng, unique, total)
                if new_block is not None:
                    cur[i] = new_block
                    # recompute coverage
                    bs = set(new_block)
                    cov[i] = 0
                    for j, mm in enumerate(masks):
                        if not (mm & ~bs):
                            cov[i] |= 1 << j
                    total = 0
                    for j in range(n):
                        total |= cov[j]
                    progress = True
            blocks = _blocks_to_set(cur)

        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = sorted(blocks)

        if not progress:
            break

    # ensure validity of best
    if not _is_valid(best_blocks, masks, v, t):
        # fallback to full greedy (recompute)
        blocks2, covered2 = _greedy_construct(v, k, t, masks, rng, deadline)
        if covered2 == full_mask:
            best_blocks = sorted(_blocks_to_set(blocks2))
        else:
            best_blocks = sorted(_blocks_to_set(best_blocks))

    return best_blocks
