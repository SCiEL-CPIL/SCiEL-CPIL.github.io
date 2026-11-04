import itertools
import time
import random
import sys
import math
import numpy as np


def construct(v, k, t):
    """
    Find a covering of t-subsets of v points using k-subsets (blocks).
    Returns a list of blocks (each a sorted list of k distinct ints in 0..v-1).
    Every t-subset must be contained in at least one block.
    """
    start_time = time.time()
    deadline = start_time + 145.0

    def get_elapsed():
        return time.time() - start_time

    # --- Precomputation ---
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}

    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # block_tsubs[i] = list of t-subset indices in block i
    block_tsubs = []
    for b in all_blocks:
        tsubs_in_b = [tsub_index[ts] for ts in itertools.combinations(b, t)]
        block_tsubs.append(tsubs_in_b)

    # tsub_blocks[j] = list of block indices covering t-subset j
    tsub_blocks = [[] for _ in range(num_tsubs)]
    for i, b_tsubs in enumerate(block_tsubs):
        for j in b_tsubs:
            tsub_blocks[j].append(i)

    # Bitmasks: block_mask[b] has bit j set iff block b covers t-subset j
    # Use numpy uint64 arrays. num_tsubs=11628 -> ~185 uint64 words.
    num_words = (num_tsubs + 63) // 64
    block_mask = np.zeros((num_blocks, num_words), dtype=np.uint64)
    for i, b_tsubs in enumerate(block_tsubs):
        for j in b_tsubs:
            block_mask[i, j // 64] |= (1 << (j % 64))

    # popcount helper for a single uint64
    def _popcount(x):
        x = x - ((x >> 1) & 0x5555555555555555)
        x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
        x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
        return int((x * 0x0101010101010101) >> 56)

    # Full uncovered mask (all bits set for valid t-subsets)
    full_mask = np.zeros(num_words, dtype=np.uint64)
    for w in range(num_words):
        full_mask[w] = 0xFFFFFFFFFFFFFFFF
    # Clear bits beyond num_tsubs in the last word
    extra = num_words * 64 - num_tsubs
    if extra > 0:
        full_mask[-1] &= (1 << (64 - extra)) - 1

    def popcount_mask(mask):
        return int(np.sum([_popcount(x) for x in mask]))

    # --- Bitmask Greedy ---
    def greedy_construct(seed=None, randomize_ties=True):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random

        uncovered = full_mask.copy()
        uncovered_count = num_tsubs
        selected = []

        while uncovered_count > 0:
            if get_elapsed() > deadline - 12.0:
                return None  # signal timeout

            # Pick a random uncovered t-subset
            if uncovered_count < 200:
                uncov = []
                for w in range(num_words):
                    m = uncovered[w]
                    while m:
                        low = m & (-m)
                        bit = low.bit_length() - 1
                        idx = w * 64 + bit
                        if idx < num_tsubs:
                            uncov.append(idx)
                        m ^= low
                if not uncov:
                    break
                s_idx = rng.choice(uncov)
            else:
                while True:
                    s_idx = rng.randint(0, num_tsubs - 1)
                    if (uncovered[s_idx // 64] >> (s_idx % 64)) & 1:
                        break

            candidates = tsub_blocks[s_idx]

            # Vectorized gain computation over all candidates
            cand_arr = np.array(candidates, dtype=np.int64)
            cand_masks = block_mask[cand_arr]  # shape (len(candidates), num_words)
            gains = np.zeros(len(candidates), dtype=np.int64)
            for w in range(num_words):
                gains += np.bitwise_and(cand_masks[:, w], uncovered[w])

            # popcount each gain
            gain_vals = np.array([_popcount(g) for g in gains], dtype=np.int64)

            best_gain = int(gain_vals.max())
            if best_gain <= 0:
                break

            # Indices with max gain
            max_idx = np.where(gain_vals == best_gain)[0]
            if randomize_ties and len(max_idx) > 1:
                chosen = rng.choice(max_idx.tolist())
            else:
                chosen = int(max_idx[0])

            best_block_idx = int(cand_arr[chosen])
            selected.append(best_block_idx)

            # Update uncovered
            bm = block_mask[best_block_idx]
            for w in range(num_words):
                new_uncovered = uncovered[w] & ~bm[w]
                delta = _popcount(uncovered[w]) - _popcount(new_uncovered)
                uncovered_count -= delta
                uncovered[w] = new_uncovered

        if uncovered_count > 0:
            return None  # incomplete
        return selected

    # --- Local Search: redundancy removal + swaps ---
    def local_search(block_indices):
        block_indices = list(set(block_indices))

        def compute_cover_counts(blist):
            cover_count = [0] * num_tsubs
            for b_idx in blist:
                for j in block_tsubs[b_idx]:
                    cover_count[j] += 1
            return cover_count

        # Phase 1: iterative redundancy removal
        changed = True
        while changed:
            changed = False
            if get_elapsed() > deadline - 8.0:
                break
            cover_count = compute_cover_counts(block_indices)
            new_list = []
            for b_idx in block_indices:
                essential = False
                for j in block_tsubs[b_idx]:
                    if cover_count[j] == 1:
                        essential = True
                        break
                if essential:
                    new_list.append(b_idx)
                else:
                    changed = True
            block_indices = new_list

        # Phase 2: swap-based local search
        # Try replacing block A with block B to enable further removals.
        if get_elapsed() < deadline - 15.0:
            improved = True
            while improved:
                improved = False
                if get_elapsed() > deadline - 8.0:
                    break
                cover_count = compute_cover_counts(block_indices)
                # For each block, find its "exclusive" t-subsets (covered only by it)
                for i, b_idx in enumerate(block_indices):
                    if get_elapsed() > deadline - 8.0:
                        break
                    exclusive = [j for j in block_tsubs[b_idx] if cover_count[j] == 1]
                    if not exclusive:
                        # This block is redundant, should have been removed; skip
                        continue
                    # Find a replacement block B that covers all exclusive t-subsets
                    # and is not already in the list
                    # Candidate blocks: those covering the most exclusive t-subsets
                    # Use the first exclusive t-subset to get candidates
                    cands = tsub_blocks[exclusive[0]]
                    best_b = -1
                    best_score = -1
                    for c in cands:
                        if c in block_indices:
                            continue
                        # Check if c covers all exclusive
                        c_tsubs = set(block_tsubs[c])
                        if all(j in c_tsubs for j in exclusive):
                            # Score: number of t-subsets c covers that are currently
                            # covered only by b_idx (i.e., would become uncovered if b_idx
                            # removed but c added). Actually we just need coverage.
                            # Prefer blocks that also cover many other t-subsets
                            # to create more redundancy.
                            score = 0
                            for j in block_tsubs[c]:
                                if cover_count[j] == 1:
                                    score += 1
                            if score > best_score:
                                best_score = score
                                best_b = c
                    if best_b != -1:
                        # Try the swap
                        new_list = [b for b in block_indices if b != b_idx]
                        new_list.append(best_b)
                        # Check validity: every t-subset still covered
                        # Quick check: recompute cover counts
                        new_cover = compute_cover_counts(new_list)
                        if all(new_cover[j] >= 1 for j in range(num_tsubs)):
                            block_indices = new_list
                            improved = True
                            # After swap, do another removal pass
                            changed2 = True
                            while changed2:
                                changed2 = False
                                if get_elapsed() > deadline - 8.0:
                                    break
                                cc = compute_cover_counts(block_indices)
                                nl = []
                                for b in block_indices:
                                    ess = any(cc[j] == 1 for j in block_tsubs[b])
                                    if ess:
                                        nl.append(b)
                                    else:
                                        changed2 = True
                                block_indices = nl
                            break  # restart outer loop
        return block_indices

    # --- Deterministic fallback (guaranteed valid) ---
    def deterministic_greedy():
        rng = random.Random(42)
        uncovered = full_mask.copy()
        uncovered_count = num_tsubs
        selected = []
        while uncovered_count > 0:
            # Pick the uncovered t-subset with fewest covering blocks (most constrained)
            best_s = -1
            best_cand_count = float('inf')
            for s_idx in range(num_tsubs):
                if (uncovered[s_idx // 64] >> (s_idx % 64)) & 1:
                    cc = len(tsub_blocks[s_idx])
                    if cc < best_cand_count:
                        best_cand_count = cc
                        best_s = s_idx
                    if best_cand_count <= 1:
                        break
            if best_s == -1:
                break
            candidates = tsub_blocks[best_s]
            cand_arr = np.array(candidates, dtype=np.int64)
            cand_masks = block_mask[cand_arr]
            gains = np.zeros(len(candidates), dtype=np.int64)
            for w in range(num_words):
                gains += np.bitwise_and(cand_masks[:, w], uncovered[w])
            gain_vals = np.array([_popcount(g) for g in gains], dtype=np.int64)
            best_idx = int(gain_vals.argmax())
            best_block_idx = int(cand_arr[best_idx])
            selected.append(best_block_idx)
            bm = block_mask[best_block_idx]
            for w in range(num_words):
                new_uncovered = uncovered[w] & ~bm[w]
                delta = _popcount(uncovered[w]) - _popcount(new_uncovered)
                uncovered_count -= delta
                uncovered[w] = new_uncovered
        if uncovered_count > 0:
            # Should not happen; return what we have (may be invalid, but try)
            pass
        return selected

    # --- Main search loop ---
    best_len = float('inf')
    best_block_list = None

    # First: deterministic greedy as a safe baseline
    det_blocks = deterministic_greedy()
    if det_blocks is not None and len(det_blocks) > 0:
        det_optimized = local_search(det_blocks)
        best_len = len(det_optimized)
        best_block_list = [all_blocks[i] for i in det_optimized]

    # Random restarts with bitmask greedy
    restart = 0
    while get_elapsed() < deadline - 15.0:
        restart += 1
        current = greedy_construct(seed=None, randomize_ties=True)
        if current is None:
            continue
        optimized = local_search(current)
        if len(optimized) < best_len:
            best_len = len(optimized)
            best_block_list = [all_blocks[i] for i in optimized]
            if best_len <= 320:
                break

    # Fallback: if no valid solution found (shouldn't happen), use deterministic
    if best_block_list is None:
        best_block_list = [all_blocks[i] for i in det_blocks]

    # Final: ensure distinct blocks (they should be, but just in case)
    seen = set()
    final_blocks = []
    for b in best_block_list:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            final_blocks.append(list(b))

    return final_blocks
