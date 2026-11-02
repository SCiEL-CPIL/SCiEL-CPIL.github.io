"""Covering design C(19,8,5) - Global Bitmask Greedy with Enhanced Local Search."""
import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin (evaluator kills at 150s)

    # 1. Generate all t-subsets and assign indices
    all_t_subs = list(itertools.combinations(range(v), t))
    n_subs = len(all_t_subs)
    
    # Map subset tuple to index
    sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # 2. Generate all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # 3. Precompute bitmask for each block
    block_masks = [0] * n_blocks
    
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks[b_idx] = mask

    # Precompute for each block: list of subset indices it covers (for fast gap computation)
    block_cover_lists = []
    for b_idx in range(n_blocks):
        mask = block_masks[b_idx]
        covers = []
        m = mask
        while m:
            idx = (m & -m).bit_length() - 1
            covers.append(idx)
            m &= m - 1
        block_cover_lists.append(covers)

    best_blocks = None
    best_count = float('inf')

    rng = random.Random(42)
    max_restarts = 300

    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        uncovered_mask = (1 << n_subs) - 1
        selected_blocks = []
        
        active_blocks = list(range(n_blocks))
        
        while uncovered_mask:
            if time.time() > deadline:
                break

            best_gain = 0
            best_block_candidates = []

            new_active_blocks = []
            for b_idx in active_blocks:
                mask = block_masks[b_idx]
                if not (mask & uncovered_mask):
                    continue
                gain = (mask & uncovered_mask).bit_count()
                if gain > 0:
                    new_active_blocks.append(b_idx)
                    if gain > best_gain:
                        best_gain = gain
                        best_block_candidates = [b_idx]
                    elif gain == best_gain:
                        best_block_candidates.append(b_idx)

            if not best_block_candidates:
                break

            chosen_block = rng.choice(best_block_candidates)
            selected_blocks.append(chosen_block)
            uncovered_mask &= ~block_masks[chosen_block]
            active_blocks = new_active_blocks

            if not active_blocks:
                break

        if uncovered_mask == 0:
            # Local Search: Delete redundant blocks
            cover_counts = np.zeros(n_subs, dtype=np.int32)
            for b_idx in selected_blocks:
                for s in block_cover_lists[b_idx]:
                    cover_counts[s] += 1
            
            changed = True
            while changed:
                changed = False
                for i in range(len(selected_blocks)):
                    b_idx = selected_blocks[i]
                    is_redundant = True
                    for s in block_cover_lists[b_idx]:
                        if cover_counts[s] == 1:
                            is_redundant = False
                            break
                    if is_redundant:
                        for s in block_cover_lists[b_idx]:
                            cover_counts[s] -= 1
                        selected_blocks.pop(i)
                        changed = True
                        break

            # 1-exchange: remove one block, re-cover gap with one block
            improved_1x = True
            while improved_1x and time.time() < deadline:
                improved_1x = False
                n_sel = len(selected_blocks)
                if n_sel < 2:
                    break
                n_tries = min(200, n_sel)
                for _ in range(n_tries):
                    if time.time() > deadline:
                        break
                    i = rng.randrange(n_sel)
                    b_idx = selected_blocks[i]
                    gap_mask = 0
                    for s in block_cover_lists[b_idx]:
                        if cover_counts[s] == 1:
                            gap_mask |= (1 << s)
                    if gap_mask == 0:
                        for s in block_cover_lists[b_idx]:
                            cover_counts[s] -= 1
                        selected_blocks.pop(i)
                        improved_1x = True
                        break
                    if gap_mask.bit_count() > 20:
                        continue
                    for attempt in range(100):
                        cand = rng.randrange(n_blocks)
                        if cand == b_idx:
                            continue
                        if (block_masks[cand] & gap_mask) == gap_mask:
                            if cand in selected_blocks:
                                continue
                            for s in block_cover_lists[b_idx]:
                                cover_counts[s] -= 1
                            for s in block_cover_lists[cand]:
                                cover_counts[s] += 1
                            selected_blocks[i] = cand
                            improved_1x = True
                            break
                    if improved_1x:
                        break

            # 2-exchange: remove two blocks, re-cover gap with one block
            if len(selected_blocks) >= 3:
                improved_2x = True
                while improved_2x and time.time() < deadline:
                    improved_2x = False
                    n_sel = len(selected_blocks)
                    n_pairs = min(300, n_sel * (n_sel - 1) // 2)
                    for _ in range(n_pairs):
                        if time.time() > deadline:
                            break
                        i, j = rng.sample(range(n_sel), 2)
                        bi = selected_blocks[i]
                        bj = selected_blocks[j]
                        
                        gap_mask = 0
                        for s in block_cover_lists[bi]:
                            if cover_counts[s] == 1:
                                gap_mask |= (1 << s)
                        for s in block_cover_lists[bj]:
                            if cover_counts[s] == 1:
                                gap_mask |= (1 << s)
                        
                        # Add subsets in intersection with count==2
                        inter = block_masks[bi] & block_masks[bj]
                        m = inter
                        while m:
                            idx = (m & -m).bit_length() - 1
                            if cover_counts[idx] == 2:
                                gap_mask |= (1 << idx)
                            m &= m - 1

                        if gap_mask == 0:
                            for s in block_cover_lists[bi]:
                                cover_counts[s] -= 1
                            for s in block_cover_lists[bj]:
                                cover_counts[s] -= 1
                            if j > i:
                                selected_blocks.pop(j)
                                selected_blocks.pop(i)
                            else:
                                selected_blocks.pop(i)
                                selected_blocks.pop(j)
                            improved_2x = True
                            break

                        if gap_mask.bit_count() > 25:
                            continue
                        for attempt in range(100):
                            cand = rng.randrange(n_blocks)
                            if cand == bi or cand == bj:
                                continue
                            if cand in selected_blocks:
                                continue
                            if (block_masks[cand] & gap_mask) == gap_mask:
                                for s in block_cover_lists[bi]:
                                    cover_counts[s] -= 1
                                for s in block_cover_lists[bj]:
                                    cover_counts[s] -= 1
                                for s in block_cover_lists[cand]:
                                    cover_counts[s] += 1
                                if j > i:
                                    selected_blocks.pop(j)
                                    selected_blocks.pop(i)
                                else:
                                    selected_blocks.pop(i)
                                    selected_blocks.pop(j)
                                selected_blocks.append(cand)
                                improved_2x = True
                                break
                        if improved_2x:
                            break

            # 3-exchange: remove three blocks, re-cover gap with one or two blocks
            if len(selected_blocks) >= 4:
                improved_3x = True
                while improved_3x and time.time() < deadline:
                    improved_3x = False
                    n_sel = len(selected_blocks)
                    n_triples = min(200, n_sel * (n_sel - 1) * (n_sel - 2) // 6)
                    for _ in range(n_triples):
                        if time.time() > deadline:
                            break
                        i, j, l = rng.sample(range(n_sel), 3)
                        bi = selected_blocks[i]
                        bj = selected_blocks[j]
                        bl = selected_blocks[l]
                        
                        # Compute gap after removing all three
                        gap_mask = 0
                        for s in block_cover_lists[bi]:
                            if cover_counts[s] == 1:
                                gap_mask |= (1 << s)
                        for s in block_cover_lists[bj]:
                            if cover_counts[s] == 1:
                                gap_mask |= (1 << s)
                        for s in block_cover_lists[bl]:
                            if cover_counts[s] == 1:
                                gap_mask |= (1 << s)
                        
                        # Add subsets in pairwise intersections with count==2
                        inter_ij = block_masks[bi] & block_masks[bj]
                        m = inter_ij
                        while m:
                            idx = (m & -m).bit_length() - 1
                            if cover_counts[idx] == 2:
                                gap_mask |= (1 << idx)
                            m &= m - 1
                        
                        inter_il = block_masks[bi] & block_masks[bl]
                        m = inter_il
                        while m:
                            idx = (m & -m).bit_length() - 1
                            if cover_counts[idx] == 2:
                                gap_mask |= (1 << idx)
                            m &= m - 1
                        
                        inter_jl = block_masks[bj] & block_masks[bl]
                        m = inter_jl
                        while m:
                            idx = (m & -m).bit_length() - 1
                            if cover_counts[idx] == 2:
                                gap_mask |= (1 << idx)
                            m &= m - 1
                        
                        # Add subsets in triple intersection with count==3
                        inter_ijk = block_masks[bi] & block_masks[bj] & block_masks[bl]
                        m = inter_ijk
                        while m:
                            idx = (m & -m).bit_length() - 1
                            if cover_counts[idx] == 3:
                                gap_mask |= (1 << idx)
                            m &= m - 1

                        if gap_mask == 0:
                            # All three redundant together
                            for s in block_cover_lists[bi]:
                                cover_counts[s] -= 1
                            for s in block_cover_lists[bj]:
                                cover_counts[s] -= 1
                            for s in block_cover_lists[bl]:
                                cover_counts[s] -= 1
                            # Remove in reverse order of indices
                            idxs = sorted([i, j, l], reverse=True)
                            for idx in idxs:
                                selected_blocks.pop(idx)
                            improved_3x = True
                            break

                        # Try to re-cover gap with one new block
                        if gap_mask.bit_count() > 30:
                            continue
                        for attempt in range(150):
                            cand = rng.randrange(n_blocks)
                            if cand == bi or cand == bj or cand == bl:
                                continue
                            if cand in selected_blocks:
                                continue
                            if (block_masks[cand] & gap_mask) == gap_mask:
                                # Perform swap: remove bi, bj, bl, add cand
                                for s in block_cover_lists[bi]:
                                    cover_counts[s] -= 1
                                for s in block_cover_lists[bj]:
                                    cover_counts[s] -= 1
                                for s in block_cover_lists[bl]:
                                    cover_counts[s] -= 1
                                for s in block_cover_lists[cand]:
                                    cover_counts[s] += 1
                                idxs = sorted([i, j, l], reverse=True)
                                for idx in idxs:
                                    selected_blocks.pop(idx)
                                selected_blocks.append(cand)
                                improved_3x = True
                                break
                        if improved_3x:
                            break

            if len(selected_blocks) < best_count:
                best_count = len(selected_blocks)
                best_blocks = [list(all_blocks[i]) for i in selected_blocks]
                if best_count <= 335:
                    break

    if best_blocks is None:
        return []
        
    return best_blocks
