import time
import random
import itertools
import numpy as np


def _greedy_init(v, k, t, seed=0):
    rng = random.Random(seed)
    points = np.arange(v)
    subs = np.array(list(itertools.combinations(range(v), t)))
    nsub = len(subs)
    sub_to_idx = {tuple(r): i for i, r in enumerate(subs)}
    covered = np.zeros(nsub, dtype=np.uint8)
    blocks = []
    used = [False] * nsub
    while True:
        unc = np.nonzero(covered == 0)[0]
        if len(unc) == 0:
            break
        # pick the rarest uncovered 4-subset as seed of the block
        idx = int(rng.choice(unc))
        block = set(subs[idx].tolist())
        while len(block) < k:
            cands = [x for x in points if x not in block]
            best = -1
            bestx = cands[0]
            for x in cands:
                g = 0
                for rest in itertools.combinations(sorted(block), t - 1):
                    key = tuple(sorted(rest + (x,)))
                    j = sub_to_idx.get(key)
                    if j is not None and covered[j] == 0:
                        g += 1
                if g > best:
                    best = g
                    bestx = x
            block.add(bestx)
        blk = sorted(block)
        blocks.append(blk)
        for c in itertools.combinations(blk, t):
            j = sub_to_idx[c]
            covered[j] = 1
    return blocks, covered, sub_to_idx, subs


def _coverage_stats(blocks, sub_to_idx):
    nsub = len(sub_to_idx)
    covered = np.zeros(nsub, dtype=np.uint8)
    for blk in blocks:
        for c in itertools.combinations(blk, 4):
            j = sub_to_idx[c]
            covered[j] = 1
    return covered


def construct(v, k, t):
    start = time.time()
    deadline = start + 110.0

    points = np.arange(v)
    subs = np.array(list(itertools.combinations(range(v), t)))
    nsub = len(subs)
    sub_to_idx = {tuple(r): i for i, r in enumerate(subs)}

    # Precompute: for each point, the list of 4-subset indices that contain it.
    pt_subsets = [[] for _ in range(v)]
    for i, r in enumerate(subs):
        for x in r:
            pt_subsets[x].append(i)
    pt_subsets = [np.array(a, dtype=np.int64) for a in pt_subsets]

    # Greedy initialization with a few seeds; keep the best.
    best_blocks = None
    best_covered = None
    for s in range(6):
        if time.time() > deadline - 10:
            break
        blocks, covered, _, _ = _greedy_init(v, k, t, seed=s)
        if best_blocks is None or len(blocks) < len(best_blocks):
            best_blocks = blocks
            best_covered = covered

    if best_blocks is None:
        return []

    blocks = best_blocks
    covered = best_covered.copy()
    uncovered_count = int(np.count_nonzero(covered == 0))

    def _apply_add(blk):
        nonlocal uncovered_count
        for c in itertools.combinations(blk, t):
            j = sub_to_idx[c]
            if covered[j] == 0:
                covered[j] = 1
                uncovered_count -= 1

    def _apply_remove(blk):
        nonlocal uncovered_count
        for c in itertools.combinations(blk, t):
            j = sub_to_idx[c]
            if covered[j] == 1:
                covered[j] = 0
                uncovered_count += 1

    # Local search loop
    improved = True
    while uncovered_count > 0 and improved and time.time() < deadline - 20:
        improved = False

        # Pass 1: drop redundant blocks (ones whose removal keeps coverage complete)
        for bi in range(len(blocks) - 1, -1, -1):
            if uncovered_count > 0:
                break
            # only try dropping if the block is "redundant": every 4-subset it
            # covers is also covered by another block. Quick check:
            redundant = True
            for c in itertools.combinations(blocks[bi], t):
                j = sub_to_idx[c]
                if covered[j] == 1:
                    # count how many blocks cover j
                    cnt = 0
                    for bj in blocks:
                        if bj == blocks[bi]:
                            continue
                        if all(x in bj for x in c):
                            cnt += 1
                            if cnt >= 1:
                                break
                    if cnt == 0:
                        redundant = False
                        break
            if redundant:
                _apply_remove(blocks[bi])
                blocks.pop(bi)
                improved = True

        if uncovered_count == 0:
            break

        # Pass 2: swap a block for a better one (targeting rarest uncovered)
        if time.time() > deadline - 20:
            break
        # rarest uncovered 4-subset
        unc_idx = np.nonzero(covered == 0)[0]
        if len(unc_idx) == 0:
            break
        # pick a random uncovered 4-subset
        target = subs[int(unc_idx[random.randrange(len(unc_idx))])]
        # build a candidate block containing target
        base = set(target.tolist())
        # choose remaining points by gain
        rem = [x for x in points if x not in base]
        random.shuffle(rem)
        # greedy fill
        blk = set(base)
        while len(blk) < k:
            bestg = -1
            bestx = rem[0]
            for x in rem:
                if x in blk:
                    continue
                g = 0
                for rest in itertools.combinations(sorted(blk), t - 1):
                    key = tuple(sorted(rest + (x,)))
                    j = sub_to_idx.get(key)
                    if j is not None and covered[j] == 0:
                        g += 1
                if g > bestg:
                    bestg = g
                    bestx = x
            blk.add(bestx)
        cand = sorted(blk)
        # gain of adding cand
        gain_add = 0
        for c in itertools.combinations(cand, t):
            j = sub_to_idx[c]
            if covered[j] == 0:
                gain_add += 1
        if gain_add > 0:
            # find a block to drop: the one with the least unique coverage
            # (i.e., whose 4-subsets are all covered by others after adding cand)
            # Simpler: drop the block that currently has the smallest "unique" count.
            # Compute unique coverage per block (4-subsets covered only by it).
            # For speed, approximate: drop the block with the most overlap.
            # We'll just try dropping each block and measure net change; pick best.
            best_net = -1
            best_drop = -1
            for bi in range(len(blocks)):
                # simulate: remove bi, add cand
                # unique count of bi among current covered
                # net change in uncovered = (new uncovered after remove) - (covered by cand)
                # approximate by counting how many of bi's 4-subsets become uncovered
                lost = 0
                for c in itertools.combinations(blocks[bi], t):
                    j = sub_to_idx[c]
                    if covered[j] == 1:
                        # is it covered by another block?
                        cnt = 0
                        for bj in blocks:
                            if bj == blocks[bi]:
                                continue
                            if all(x in bj for x in c):
                                cnt += 1
                                break
                        if cnt == 0:
                            lost += 1
                net = gain_add - lost
                if net > best_net:
                    best_net = net
                    best_drop = bi
            if best_drop >= 0 and best_net > 0:
                _apply_remove(blocks[best_drop])
                blocks.pop(best_drop)
                blocks.append(cand)
                _apply_add(cand)
                improved = True

    # Final cleanup: drop any redundant blocks
    for bi in range(len(blocks) - 1, -1, -1):
        if uncovered_count > 0:
            break
        redundant = True
        for c in itertools.combinations(blocks[bi], t):
            j = sub_to_idx[c]
            if covered[j] == 1:
                cnt = 0
                for bj in blocks:
                    if bj == blocks[bi]:
                        continue
                    if all(x in bj for x in c):
                        cnt += 1
                        break
                if cnt == 0:
                    redundant = False
                    break
        if redundant:
            blocks.pop(bi)

    # Validate
    final_covered = np.zeros(nsub, dtype=np.uint8)
    for blk in blocks:
        for c in itertools.combinations(blk, t):
            final_covered[sub_to_idx[c]] = 1
    if np.count_nonzero(final_covered == 0) > 0:
        # invalid; fall back to greedy result which was valid
        return best_blocks

    return blocks
