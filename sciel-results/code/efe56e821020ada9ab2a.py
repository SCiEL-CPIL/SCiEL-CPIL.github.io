"""Covering design C(v,k,t): minimize number of k-blocks covering all t-subsets.
Bitset-coverage greedy + randomized restarts + deletion/swap local search with a
hard deadline guard. Standard library + numpy only."""

import time
import random
import itertools
import numpy as np


def _precompute(v, k, t):
    """Precompute the t-subset id map and, for each point, the list of t-subset ids
    containing it. Also the full list of t-subsets as tuples (for min/anchor)."""
    tsub_ids = {}
    tsub_list = []
    for c in itertools.combinations(range(v), t):
        tsub_ids[c] = len(tsub_list)
        tsub_list.append(c)
    n_tsub = len(tsub_list)
    point_tsubs = [[] for _ in range(v)]
    for cid, c in enumerate(tsub_list):
        for x in c:
            point_tsubs[x].append(cid)
    return tsub_ids, tsub_list, point_tsubs, n_tsub


def _block_tsubs(block, t, tsub_ids):
    """List of t-subset ids contained in a k-block."""
    return [tsub_ids[tuple(c)] for c in itertools.combinations(block, t)]


def _greedy_build(v, k, t, tsub_ids, point_tsubs, n_tsub, rng, deadline,
                  max_blocks, randomize, anchor_weight=1.0):
    """Greedy: repeatedly build a block anchored on a (random-ish) uncovered t-subset,
    adding points by marginal uncovered coverage. Returns list of blocks or None on timeout."""
    cov = np.zeros(n_tsub, dtype=np.int32)   # coverage count per t-subset
    blocks = []
    while True:
        if time.time() > deadline:
            return None
        if len(blocks) >= max_blocks:
            return None
        # find uncovered
        uncov = np.flatnonzero(cov == 0)
        if uncov.size == 0:
            break
        # anchor: pick an uncovered t-subset (rarest = any, all uncovered have cov 0)
        # choose one; randomize for diversity
        if randomize and uncov.size > 1:
            anchor_id = int(uncov[rng.randrange(uncov.size)])
        else:
            anchor_id = int(uncov[0])
        anchor = tsub_ids_inv = None
        # reconstruct anchor tuple from id: we stored tsub_list; pass it via closure
        # (tsub_list passed as point_tsubs? no) -> use global-ish: we recompute from id
        # We'll pass tsub_list through; here use a module-level trick is bad. Instead
        # compute anchor tuple from a lookup passed in.
        raise RuntimeError("need tsub_list")
    return blocks


def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 145.0          # safety margin before 150s kill
    rng = random.Random(12345)

    tsub_ids, tsub_list, point_tsubs, n_tsub = _precompute(v, k, t)

    # lower-ish bound: at least ceil(C(v,t) / C(k,t)) blocks
    from math import comb
    lb = (comb(v, t) + comb(k, t) - 1) // comb(k, t)

    best = None
    best_len = 10 ** 18

    # budget for greedy restarts
    max_blocks = 2000000

    def run_greedy(randomize):
        nonlocal best, best_len
        cov = np.zeros(n_tsub, dtype=np.int32)
        blocks = []
        while True:
            if time.time() > deadline:
                return
            if len(blocks) >= max_blocks:
                return
            uncov = np.flatnonzero(cov == 0)
            if uncov.size == 0:
                break
            if randomize and uncov.size > 1:
                anchor_id = int(uncov[rng.randrange(uncov.size)])
            else:
                anchor_id = int(uncov[0])
            anchor = tsub_list[anchor_id]
            block = list(anchor)
            # add points by marginal uncovered coverage
            while len(block) < k:
                if time.time() > deadline:
                    return
                # candidate points not in block
                cand = [x for x in range(v) if x not in block]
                # compute marginal gain for each candidate: number of uncovered tsubs
                # that would be covered by adding x to current block
                # a tsub is covered by block+{x} iff it contains x and (block & tsub) size == t-1
                # i.e. tsub \ {x} subset of block
                best_x = None
                best_g = -1
                # precompute: for each candidate x, count uncovered tsubs containing x
                # where all other t-1 elements are in block.
                # Efficient: build set of uncovered ids, then for each x iterate its tsubs.
                uncov_set = set(uncov.tolist())
                for x in cand:
                    g = 0
                    for cid in point_tsubs[x]:
                        if cid in uncov_set:
                            # check other elements in block
                            tt = tsub_list[cid]
                            ok = True
                            for y in tt:
                                if y == x:
                                    continue
                                if y not in block:
                                    ok = False
                                    break
                            if ok:
                                g += 1
                    if g > best_g:
                        best_g = g
                        best_x = x
                        if best_g == 0:
                            # all zero; just pick first for speed (break)
                            pass
                if best_x is None:
                    best_x = cand[0]
                block.append(best_x)
            block.sort()
            # update coverage
            for cid in _block_tsubs(block, t, tsub_ids):
                cov[cid] += 1
            blocks.append(block)
        # evaluate
        if len(blocks) < best_len:
            best_len = len(blocks)
            best = [list(b) for b in blocks]

    # Phase 1: a couple of greedy restarts (deterministic first, then randomized)
    run_greedy(randomize=False)
    run_greedy(randomize=True)
    run_greedy(randomize=True)

    if best is None:
        # fallback: pure greedy deterministic (should not happen)
        return []

    # Phase 2: local search on best (deletion + swap)
    _local_search(best, v, k, t, tsub_ids, tsub_list, point_tsubs, n_tsub,
                  deadline, rng)

    return best


def _local_search(blocks, v, k, t, tsub_ids, tsub_list, point_tsubs, n_tsub,
                  deadline, rng):
    """Deletion + swap local search. Modifies blocks in place (shrinks)."""
    n = len(blocks)
    if n == 0:
        return
    # coverage counts
    cov = np.zeros(n_tsub, dtype=np.int32)
    block_ids = [None] * n
    for i, b in enumerate(blocks):
        ids = _block_tsubs(b, t, tsub_ids)
        block_ids[i] = ids
        for cid in ids:
            cov[cid] += 1

    def try_delete():
        nonlocal n
        changed = True
        while changed:
            changed = False
            if time.time() > deadline:
                return
            # find a removable block: one whose removal keeps all tsubs covered
            # a block is removable iff for every tsub it covers, cov[tsub] >= 2
            # equivalently, no tsub it covers has cov==1
            for i in range(n):
                if time.time() > deadline:
                    return
                removable = True
                for cid in block_ids[i]:
                    if cov[cid] == 1:
                        removable = False
                        break
                if removable:
                    # remove block i
                    for cid in block_ids[i]:
                        cov[cid] -= 1
                    blocks.pop(i)
                    block_ids.pop(i)
                    n -= 1
                    changed = True
                    break

    def try_swap_pass():
        nonlocal n
        # try replacing some blocks with a better block that covers more uncovered-after-removal
        # simple heuristic: for each block i, temporarily remove it, then build a new block
        # anchored on the most-exposed tsub (cov==1 after removal), add it, and if it
        # covers at least as many as old and allows deletion, keep.
        if time.time() > deadline:
            return
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                return
            for i in range(n):
                if time.time() > deadline:
                    return
                # temporary remove
                old_ids = block_ids[i]
                for cid in old_ids:
                    cov[cid] -= 1
                # build replacement anchored on a cov==1 tsub (most constrained)
                exposed = np.flatnonzero(cov == 1)
                if exposed.size == 0:
                    exposed = np.flatnonzero(cov == 0)
                if exposed.size == 0:
                    # nothing exposed; skip
                    for cid in old_ids:
                        cov[cid] += 1
                    continue
                anchor_id = int(exposed[rng.randrange(exposed.size)])
                anchor = tsub_list[anchor_id]
                block = list(anchor)
                while len(block) < k:
                    cand = [x for x in range(v) if x not in block]
                    # gain: cover exposed (cov<=1) tsubs
                    best_x, best_g = None, -1
                    for x in cand:
                        g = 0
                        for cid in point_tsubs[x]:
                            if cov[cid] <= 1:
                                tt = tsub_list[cid]
                                ok = True
                                for y in tt:
                                    if y == x:
                                        continue
                                    if y not in block:
                                        ok = False
                                        break
                                if ok:
                                    g += 1
                        if g > best_g:
                            best_g, best_x = g, x
                    if best_x is None:
                        best_x = cand[0]
                    block.append(best_x)
                block.sort()
                new_ids = _block_tsubs(block, t, tsub_ids)
                # does new block cover all tsubs that old block uniquely covered (cov==1 after removal)?
                # i.e. for every cid in old_ids, cov[cid]+1 must be covered by new or remain >=1
                # After removal cov[cid] for old_ids are <= original-1. We need final cov>=1.
                # final cov[cid] = cov[cid] + (1 if cid in new_ids else 0). Need >=1.
                ok = True
                new_set = set(new_ids)
                for cid in old_ids:
                    if cov[cid] == 0 and cid not in new_set:
                        ok = False
                        break
                if ok:
                    # accept swap
                    for cid in new_ids:
                        cov[cid] += 1
                    blocks[i] = block
                    block_ids[i] = new_ids
                    improved = True
                    # re-run deletion to clean up
                    try_delete()
                    break
                else:
                    # revert
                    for cid in old_ids:
                        cov[cid] += 1

    try_delete()
    # a few swap passes
    for _ in range(3):
        if time.time() > deadline:
            break
        try_swap_pass()
        try_delete()
    # final deletion
    try_delete()


if __name__ == "__main__":
    res = construct(19, 8, 5)
    print("blocks:", len(res))
