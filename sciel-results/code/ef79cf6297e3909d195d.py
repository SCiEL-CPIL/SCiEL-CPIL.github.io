"""Combined: Bitset-accelerated greedy + 1-for-1 swap local search + pair-replacement refinement.
Uses Python big-int bitsets for O(1) coverage checks and maximal independent set packing lower bound.
"""
import time
import random
import itertools
import heapq
import numpy as np

_RNG = random.Random(20250101)


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118.0

    # 1. Precomputation
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)

    all_tsubsets = list(itertools.combinations(points, t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    # Data structures
    block_tsubs = [[] for _ in range(n_blocks)]
    tsub_blocks = [[] for _ in range(n_tsubsets)]
    
    # Bitsets for fast set operations
    tsub_mask = [0] * n_blocks       # tsub_mask[b] has bits for tsubs covered by block b
    tsub_blocks_mask = [0] * n_tsubsets # tsub_blocks_mask[ts] has bits for blocks covering ts
    block_tsub_sets = [None] * n_blocks # Sets for local search intersection
    tsub_block_sets = [None] * n_tsubsets

    for b_idx, block in enumerate(all_blocks):
        mask = 0
        tsub_set = set()
        for ts in itertools.combinations(block, t):
            idx = tsub_idx[ts]
            mask |= (1 << idx)
            tsub_blocks_mask[idx] |= (1 << b_idx)
            block_tsubs[b_idx].append(idx)
            tsub_set.add(idx)
        tsub_mask[b_idx] = mask
        block_tsub_sets[b_idx] = tsub_set

    for i in range(n_tsubsets):
        tsub_block_sets[i] = set()
        m = tsub_blocks_mask[i]
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            tsub_block_sets[i].add(idx)
            m ^= lsb

    full_tsub_mask = (1 << n_tsubsets) - 1
    
    # Precompute conflict masks for Packing LB (independent sets)
    # Two t-subsets conflict if they are covered by the same block
    tsub_conflict_mask = [0] * n_tsubsets
    for b_idx in range(n_blocks):
        mask = tsub_mask[b_idx]
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            tsub_conflict_mask[idx] |= mask
            m ^= lsb

    def packing_lb(uncovered_mask):
        """
        Calculate lower bound using maximal independent set of uncovered t-subsets.
        Two t-subsets are independent if no single block covers both.
        """
        count = 0
        remaining = uncovered_mask
        while remaining:
            lsb = remaining & -remaining
            idx = lsb.bit_length() - 1
            count += 1
            remaining &= ~tsub_conflict_mask[idx]
        return count

    def greedy_cover(seed):
        """
        Standard greedy using heap for gain calculation.
        """
        rng = random.Random(seed)
        uncovered_mask = full_tsub_mask
        selected = []
        selected_mask = 0
        
        # Initial gains: number of tsubs covered by each block
        gains = [0] * n_blocks
        for b_idx in range(n_blocks):
            gains[b_idx] = tsub_mask[b_idx].bit_count()
            
        heap = [(-gains[b_idx], b_idx) for b_idx in range(n_blocks)]
        heapq.heapify(heap)
        
        while uncovered_mask:
            if time.time() > deadline - 10.0:
                break

            best_block = -1
            best_gain = 0

            # Pop stale entries
            while heap:
                neg_g, b_idx = heapq.heappop(heap)
                g = -neg_g
                if g == gains[b_idx] and not (selected_mask & (1 << b_idx)):
                    best_block = b_idx
                    best_gain = g
                    break
                # Stale or already selected, discard

            if best_block == -1:
                break

            selected.append(best_block)
            selected_mask |= (1 << best_block)
            gains[best_block] = -1 # Mark as selected
            
            # Update gains for blocks covering newly covered tsubs
            new_covered = tsub_mask[best_block] & uncovered_mask
            uncovered_mask &= ~tsub_mask[best_block]
            
            m = new_covered
            while m:
                lsb = m & -m
                ts_idx = lsb.bit_length() - 1
                # For every block covering this ts_idx, decrease its gain
                cm = tsub_blocks_mask[ts_idx]
                while cm:
                    lsb2 = cm & -cm
                    b_idx2 = lsb2.bit_length() - 1
                    if not (selected_mask & (1 << b_idx2)):
                        gains[b_idx2] -= 1
                        heapq.heappush(heap, (-gains[b_idx2], b_idx2))
                    cm ^= lsb2
                m ^= lsb

        return selected

    def local_search(blocks, seed):
        """
        Local search with swaps and deletions.
        """
        rng = random.Random(seed)
        if not blocks:
            return blocks

        block_set = set(blocks)

        cover_count = [0] * n_tsubsets
        for b in block_set:
            for ts_idx in block_tsubs[b]:
                cover_count[ts_idx] += 1

        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False

            # Phase 1: Remove redundant blocks
            to_remove = []
            for b in block_set:
                redundant = True
                for ts_idx in block_tsubs[b]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(b)

            if to_remove:
                for b in to_remove:
                    block_set.remove(b)
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1
                improved = True
                continue

            # Phase 2: Try 1-for-1 swaps
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)

            for b in current_blocks:
                if time.time() > deadline - 5.0:
                    break
                if b not in block_set:
                    continue

                unique_tsubs = []
                for ts_idx in block_tsubs[b]:
                    if cover_count[ts_idx] == 1:
                        unique_tsubs.append(ts_idx)

                if not unique_tsubs:
                    block_set.remove(b)
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1
                    improved = True
                    break

                # Find candidates using set intersection
                # Start with the most constrained unique tsub
                min_ts = min(unique_tsubs, key=lambda x: len(tsub_block_sets[x]))
                candidates = set(tsub_block_sets[min_ts])
                
                for ts_idx in unique_tsubs[1:]:
                    if not candidates:
                        break
                    candidates &= tsub_block_sets[ts_idx]

                if not candidates:
                    continue

                candidates.discard(b)
                if not candidates:
                    continue

                cand_list = list(candidates)
                rng.shuffle(cand_list)
                
                # Limit candidates to avoid too much work per swap attempt
                if len(cand_list) > 20:
                    cand_list = cand_list[:20]

                swapped = False
                for r in cand_list:
                    if r in block_set:
                        continue

                    block_set.remove(b)
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1

                    block_set.add(r)
                    for ts_idx in block_tsubs[r]:
                        cover_count[ts_idx] += 1

                    improved = True
                    swapped = True
                    break

                if swapped:
                    break

        return list(block_set)

    # Run multiple greedy restarts, keep best
    best_greedy = None
    best_greedy_size = n_blocks + 1
    for seed in range(12):
        if time.time() > deadline - 50.0:
            break
        blocks = greedy_cover(seed)
        if len(blocks) < best_greedy_size:
            best_greedy_size = len(blocks)
            best_greedy = blocks
        if len(blocks) <= 135:  # Good enough
            break

    if best_greedy is None:
        best_greedy = greedy_cover(0)

    selected = set(best_greedy)

    # Build cover_count
    cover_count = [0] * n_tsubsets
    for bi in selected:
        for ts_idx in block_tsubs[bi]:
            cover_count[ts_idx] += 1

    # ---------------- Deletion reduction ----------------
    def reduce_blocks(selected_set, cc):
        changed = True
        while changed:
            changed = False
            for bi in list(selected_set):
                if time.time() > deadline - 20.0:
                    return selected_set, cc
                cc[block_tsubs[bi]]
                # Check if any t-subset would become uncovered
                redundant = True
                for ts_idx in block_tsubs[bi]:
                    if cc[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    for ts_idx in block_tsubs[bi]:
                        cc[ts_idx] -= 1
                    selected_set.discard(bi)
                    changed = True
        return selected_set, cc

    selected, cover_count = reduce_blocks(selected, cover_count)

    # ---------------- Pair replacement: try to replace 2 blocks with 1 ----------------
    def pair_replace(selected_set, cc, max_trials=800):
        rng = _RNG
        n_sel = len(selected_set)
        if n_sel < 2:
            return selected_set, cc
        sel_list = list(selected_set)
        for trial in range(max_trials):
            if time.time() > deadline - 30.0:
                break
            i, j = rng.sample(range(n_sel), 2)
            bi, bj = sel_list[i], sel_list[j]
            # Union of t-subsets covered by bi and bj
            union_t = set(block_tsubs[bi]) | set(block_tsubs[bj])
            if len(union_t) > 35:  # Can't be covered by one block
                continue
            # Check if any single block covers all t-subsets in union_t
            # For each t-subset in union_t, find common blocks
            common = None
            for ti in union_t:
                blocks_covering = set(tsub_blocks[ti])
                if common is None:
                    common = blocks_covering
                else:
                    common &= blocks_covering
                if not common:
                    break
            if not common:
                continue
            # Try each candidate block
            for cand in common:
                if cand == bi or cand == bj:
                    continue
                # Check if replacing bi, bj with cand maintains coverage
                # Temporarily remove bi, bj
                for ts_idx in block_tsubs[bi]:
                    cc[ts_idx] -= 1
                for ts_idx in block_tsubs[bj]:
                    cc[ts_idx] -= 1
                for ts_idx in block_tsubs[cand]:
                    cc[ts_idx] += 1
                # Check if all still covered
                valid = True
                for ts_idx in block_tsubs[bi]:
                    if cc[ts_idx] == 0:
                        valid = False
                        break
                if not valid:
                    for ts_idx in block_tsubs[bi]:
                        cc[ts_idx] += 1
                    for ts_idx in block_tsubs[bj]:
                        cc[ts_idx] += 1
                    for ts_idx in block_tsubs[cand]:
                        cc[ts_idx] -= 1
                    continue
                for ts_idx in block_tsubs[bj]:
                    if cc[ts_idx] == 0:
                        valid = False
                        break
                if not valid:
                    for ts_idx in block_tsubs[bi]:
                        cc[ts_idx] += 1
                    for ts_idx in block_tsubs[bj]:
                        cc[ts_idx] += 1
                    for ts_idx in block_tsubs[cand]:
                        cc[ts_idx] -= 1
                    continue
                # Valid replacement
                selected_set.discard(bi)
                selected_set.discard(bj)
                selected_set.add(cand)
                n_sel = len(selected_set)
                sel_list = list(selected_set)
                # Reduce again
                selected_set, cc = reduce_blocks(selected_set, cc)
                n_sel = len(selected_set)
                sel_list = list(selected_set)
                break
        return selected_set, cc

    selected, cover_count = pair_replace(selected, cover_count)

    # ---------------- Local search with swaps ----------------
    selected_list = local_search(list(selected), seed=42)
    selected = set(selected_list)

    # Rebuild cover_count
    cover_count = [0] * n_tsubsets
    for bi in selected:
        for ts_idx in block_tsubs[bi]:
            cover_count[ts_idx] += 1

    # Final reduction
    selected, cover_count = reduce_blocks(selected, cover_count)

    # Second pair replacement pass after final reduction
    if time.time() < deadline - 10.0:
        selected, cover_count = pair_replace(selected, cover_count, max_trials=200)
        if selected:
            cc = [0] * n_tsubsets
            for bi in selected:
                for ts_idx in block_tsubs[bi]:
                    cc[ts_idx] += 1
            selected, cover_count = reduce_blocks(selected, cc)

    # Build output
    blocks = [list(all_blocks[bi]) for bi in sorted(selected)]

    # Final validation
    if blocks:
        cc_final = [0] * n_tsubsets
        for bi in selected:
            for ts_idx in block_tsubs[bi]:
                cc_final[ts_idx] += 1
        if any(c == 0 for c in cc_final):
            # Fallback: rebuild from best_greedy
            selected = set(best_greedy)
            blocks = [list(all_blocks[bi]) for bi in sorted(selected)]

    return blocks
