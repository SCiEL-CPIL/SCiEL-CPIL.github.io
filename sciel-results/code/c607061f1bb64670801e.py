"""Covering design C(v,k,t) via fast bitmasked greedy + aggressive local search.

Strategy:
  - Precompute, for every point x, a bitset of the t-subsets that x participates in.
  - Greedy: maintain a global uncovered bitset; each step picks the point with the
    largest marginal gain, using numpy bitwise ops for speed.
  - Local search: (a) drop any redundant block, (b) 1-exchange (swap a point in a
    block for another) that increases the number of newly-covered subsets.
"""
import time
import itertools
import numpy as np


def construct(v, k, t):
    start = time.time()
    deadline = start + 147.0  # safety margin under 150s kill

    # --- Precompute t-subsets and per-point membership bitsets ---
    t_subs = list(itertools.combinations(range(v), t))
    n_t = len(t_subs)
    sub_to_idx = {s: i for i, s in enumerate(t_subs)}

    # words: number of 64-bit words needed to hold n_t bits
    words = (n_t + 63) // 64
    # point_mask[x]: uint64 array of length `words`, bit i set iff x in t_subs[i]
    point_mask = np.zeros((v, words), dtype=np.uint64)
    for x in range(v):
        mask = point_mask[x]
        for i, s in enumerate(t_subs):
            if x in s:
                mask[i >> 6] |= (1 << (i & 63))

    # Global uncovered bitset (start: everything uncovered)
    uncov = np.ones(words, dtype=np.uint64)
    # per-word count of uncovered bits
    uncov_wcnt = uncov.copy()
    # maintain a running total of uncovered bits
    total_uncov = int(n_t)

    def time_ok():
        return time.time() < deadline

    def popcount_u64(arr):
        return int(np.sum(arr.view(np.uint64)))  # arr is uint64 already

    # --- Greedy construction ---
    blocks = []
    while total_uncov > 0 and time_ok():
        # find the first uncovered t-subset to seed the block
        seed_idx = None
        for w in range(words):
            if uncov[w]:
                # find lowest set bit
                word = uncov[w]
                bit = (word & -word).bit_length() - 1
                seed_idx = (w << 6) + bit
                break
        if seed_idx is None:
            break

        seed = t_subs[seed_idx]
        block = list(seed)
        in_block = set(block)

        # coverage of current block (bitset of t-subsets it covers)
        block_cov = np.zeros(words, dtype=np.uint64)
        for x in block:
            block_cov |= point_mask[x]
        # what this block newly covers
        new_cov = block_cov & uncov
        # we need to grow block to size k; current covers `new_cov`
        # We'll add (k - t) more points to maximize additional coverage.

        remaining_to_add = k - t
        for _ in range(remaining_to_add):
            if not time_ok():
                break
            best_x = -1
            best_gain = -1
            # candidates
            for x in range(v):
                if x in in_block:
                    continue
                # new coverage if we add x: (point_mask[x] & uncov) minus what's
                # already covered by block_cov
                cand = (point_mask[x] & uncov) & ~block_cov
                gain = int(np.sum(cand))
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x < 0:
                # fill arbitrarily
                for x in range(v):
                    if x not in in_block:
                        block.append(x)
                        in_block.add(x)
                        block_cov |= point_mask[x]
                        if len(block) >= k:
                            break
                break
            block.append(best_x)
            in_block.add(best_x)
            block_cov |= point_mask[best_x]

        block.sort()
        blocks.append(block)

        # update global uncovered
        newly = block_cov & uncov
        newly_cnt = int(np.sum(newly))
        uncov &= ~block_cov
        total_uncov -= newly_cnt

    if total_uncov > 0:
        # couldn't finish; return what we have (may be invalid)
        return blocks

    # --- Local search ---
    # Helper: coverage count of a block's t-subsets that are covered by OTHER blocks
    def block_uncovered_count(block, exclude_idx):
        """Number of t-subsets of `block` not covered by any block other than exclude_idx."""
        cnt = 0
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx[sub]
            covered_elsewhere = False
            for j, b in enumerate(blocks):
                if j == exclude_idx:
                    continue
                if all(p in b for p in sub):
                    covered_elsewhere = True
                    break
            if not covered_elsewhere:
                cnt += 1
        return cnt

    # Phase A: remove redundant blocks
    changed = True
    while changed and time_ok():
        changed = False
        for i in range(len(blocks)):
            if not time_ok():
                break
            if block_uncovered_count(blocks[i], i) == 0:
                blocks.pop(i)
                changed = True
                break

    # Phase B: 1-exchange local search to shrink / improve
    improved = True
    while improved and time_ok():
        improved = False
        for i in range(len(blocks)):
            if not time_ok():
                break
            bi = blocks[i]
            base_uncov = block_uncovered_count(bi, i)
            # try swapping one point for another
            for pos in range(k):
                if not time_ok():
                    break
                old_pt = bi[pos]
                for new_pt in range(v):
                    if new_pt == old_pt or new_pt in bi:
                        continue
                    cand = bi[:]
                    cand[pos] = new_pt
                    cand.sort()
                    new_uncov = block_uncovered_count(cand, i)
                    if new_uncov < base_uncov:
                        blocks[i] = cand
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    return blocks
