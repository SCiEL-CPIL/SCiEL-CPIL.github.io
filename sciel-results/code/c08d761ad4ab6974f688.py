import time
import random
import itertools
import numpy as np
from collections import defaultdict

def _greedy(v, k, t, deadline, rng):
    n_t = int(np.math.comb(v, t))
    all_t = list(itertools.combinations(range(v), t))
    t_to_idx = {sub: i for i, sub in enumerate(all_t)}
    t_arr = np.array(all_t, dtype=np.int32)  # (n_t, t)

    # For each point p, and each t-subset, whether p is in it (bool matrix)
    in_mat = np.zeros((v, n_t), dtype=np.bool_)
    for p in range(v):
        in_mat[p] = np.any(t_arr == p, axis=1)

    covered = np.zeros(n_t, dtype=np.bool_)
    blocks = []
    seen_blocks = set()
    t_comb = list(itertools.combinations(range(k), t))

    def add_block(blk):
        idxs = [t_to_idx[tuple(sorted(blk[j] for j in c))] for c in t_comb]
        covered[idxs] = True
        blocks.append(blk)
        seen_blocks.add(blk)

    while True:
        if time.time() > deadline - 8:
            break
        # find first uncovered t-subset
        unc = np.flatnonzero(~covered)
        if len(unc) == 0:
            break
        seed = tuple(all_t[unc[0]])
        blk = list(seed)
        bset = set(blk)
        while len(blk) < k:
            if time.time() > deadline - 8:
                break
            # gain for each candidate point
            gains = np.zeros(v, dtype=np.int32)
            for x in range(v):
                if x in bset:
                    continue
                # t-1 combos of current block
                g = 0
                for c in itertools.combinations(blk, t - 1):
                    sub = tuple(sorted(c + (x,)))
                    idx = t_to_idx[sub]
                    if not covered[idx]:
                        g += 1
                gains[x] = g
            nz = np.flatnonzero(gains > 0)
            if len(nz) == 0:
                # pick any unused point
                for x in range(v):
                    if x not in bset:
                        blk.append(x)
                        bset.add(x)
                        break
                else:
                    break
            else:
                # pick argmax; tie-break randomly
                best_g = gains[nz].max()
                cands = nz[gains[nz] == best_g]
                x = int(rng.choice(cands))
                blk.append(x)
                bset.add(x)
        blk = tuple(sorted(blk))
        if blk in seen_blocks:
            # still add? it covers nothing new; skip to avoid dup
            continue
        add_block(blk)

    return blocks, covered, t_to_idx, all_t


def _build_cover_map(blocks, t):
    cm = defaultdict(set)
    for i, b in enumerate(blocks):
        for ts in itertools.combinations(b, t):
            cm[ts].add(i)
    return cm


def _remove_redundant(blocks, t, deadline):
    if not blocks:
        return blocks
    cm = _build_cover_map(blocks, t)
    changed = True
    while changed and time.time() < deadline - 10:
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() > deadline - 10:
                break
            redundant = True
            for ts in itertools.combinations(blocks[i], t):
                if len(cm[ts]) <= 1:
                    redundant = False
                    break
            if redundant:
                rem = blocks.pop(i)
                for ts in itertools.combinations(rem, t):
                    cm[ts].discard(i)
                # reindex
                for ts in cm:
                    cm[ts] = { (j-1 if j > i else j) for j in cm[ts] }
                changed = True
                break
    return blocks


def _merge_blocks(blocks, t, k, v, deadline):
    """Try to replace two blocks by one block (size k) that covers all t-sets
    uniquely covered by either, reducing count by 1."""
    if len(blocks) < 2:
        return blocks
    cm = _build_cover_map(blocks, t)
    improved = True
    while improved and time.time() < deadline - 12:
        improved = False
        n = len(blocks)
        for i in range(n):
            if time.time() > deadline - 12:
                break
            for j in range(i + 1, n):
                if time.time() > deadline - 12:
                    break
                bi = blocks[i]
                bj = blocks[j]
                # union of points
                union_pts = set(bi) | set(bj)
                if len(union_pts) > k:
                    continue
                # t-sets uniquely covered by i or j
                must = []
                for ts in itertools.combinations(bi, t):
                    if len(cm[ts]) == 1 and i in cm[ts]:
                        must.append(ts)
                for ts in itertools.combinations(bj, t):
                    if len(cm[ts]) == 1 and j in cm[ts]:
                        must.append(ts)
                # required points = union of points in must
                req = set()
                for ts in must:
                    req.update(ts)
                if len(req) > k:
                    continue
                # build candidate: req + fill from union_pts (or any) to size k
                cand = set(req)
                # prefer points already in bi or bj to maximize overlap
                pool = list(union_pts - cand)
                random.shuffle(pool)
                for p in pool:
                    if len(cand) == k:
                        break
                    cand.add(p)
                if len(cand) < k:
                    for p in range(v):
                        if len(cand) == k:
                            break
                        if p not in cand:
                            cand.add(p)
                if len(cand) != k:
                    continue
                newb = tuple(sorted(cand))
                # verify covers all must
                ok = True
                for ts in must:
                    if not all(p in cand for p in ts):
                        ok = False
                        break
                if not ok:
                    continue
                # also ensure new block doesn't break coverage of others (it only adds)
                # remove i and j, add new
                # find higher index first for removal
                hi, lo = (j, i) if j > i else (i, j)
                for idx in (hi, lo):
                    b = blocks[idx]
                    for ts in itertools.combinations(b, t):
                        cm[ts].discard(idx)
                blocks.pop(hi)
                blocks.pop(lo)
                new_idx = len(blocks)
                blocks.append(newb)
                for ts in itertools.combinations(newb, t):
                    cm[ts].add(new_idx)
                # reindex all
                for ts in cm:
                    s = cm[ts]
                    cm[ts] = set()
                    for x in s:
                        if x > hi:
                            x -= 2
                        elif x == hi:
                            continue
                        if x > lo:
                            x -= 1
                        elif x == lo:
                            continue
                        cm[ts].add(x)
                improved = True
                break
            if improved:
                break
    return blocks


def _local_replace(blocks, t, k, v, deadline, rng):
    """Try to replace a single block with another that covers all its unique
    t-sets and additionally covers more uncovered t-sets (or at least same),
    possibly enabling later removals. Only accept if it strictly helps a
    secondary objective: number of t-sets with coverage==1 (smaller is better)."""
    if not blocks:
        return blocks
    cm = _build_cover_map(blocks, t)

    def unique_count():
        return sum(1 for ts, s in cm.items() if len(s) == 1)

    base = unique_count()
    n = len(blocks)
    for _ in range(400):
        if time.time() > deadline - 12:
            break
        i = rng.randrange(n)
        b = blocks[i]
        must = [ts for ts in itertools.combinations(b, t) if len(cm[ts]) == 1]
        req = set()
        for ts in must:
            req.update(ts)
        if len(req) > k:
            continue
        # build candidate: req + random fill
        cand = set(req)
        remaining = [p for p in range(v) if p not in cand]
        rng.shuffle(remaining)
        for p in remaining:
            if len(cand) == k:
                break
            cand.add(p)
        if len(cand) != k:
            continue
        newb = tuple(sorted(cand))
        if newb == b:
            continue
        ok = all(all(p in cand for p in ts) for ts in must)
        if not ok:
            continue
        # simulate: remove i, add newb at same index
        old_unique = len(must)
        # count how many t-sets of old block would lose coverage (those unique to i)
        # that's old_unique. New block adds coverage to its t-sets.
        # net unique change = (unique t-sets of newb not covered by others) - old_unique
        new_unique = 0
        for ts in itertools.combinations(newb, t):
            s = cm[ts]
            if i in s:
                s2 = {x for x in s if x != i}
            else:
                s2 = s
            if len(s2) == 0:
                new_unique += 1
        if new_unique < old_unique:
            # apply
            for ts in itertools.combinations(b, t):
                cm[ts].discard(i)
            blocks[i] = newb
            for ts in itertools.combinations(newb, t):
                cm[ts].add(i)
    return blocks


def construct(v, k, t):
    deadline = time.time() + 115
    rng = random.Random(12345)

    best = None
    # multiple greedy restarts with different seeds
    n_restarts = 3
    for r in range(n_restarts):
        if time.time() > deadline - 20:
            break
        rng_r = random.Random(1000 + r)
        blocks, covered, t_to_idx, all_t = _greedy(v, k, t, deadline - 15, rng_r)
        # local improvement
        blocks = _remove_redundant(blocks, t, deadline - 12)
        blocks = _merge_blocks(blocks, t, k, v, deadline - 12)
        blocks = _local_replace(blocks, t, k, v, deadline - 10, rng_r)
        blocks = _remove_redundant(blocks, t, deadline - 8)
        if best is None or len(blocks) < len(best):
            best = blocks
    if best is None:
        # fallback single greedy
        blocks, _, _, _ = _greedy(v, k, t, deadline - 5, rng)
        best = blocks
    return [list(b) for b in best]
