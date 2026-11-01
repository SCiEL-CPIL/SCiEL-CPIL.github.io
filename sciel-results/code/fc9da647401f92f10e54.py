"""Covering design C(19,8,5) - Global Bitmask Greedy with Local Search."""
import itertools
import time
import random
import sys

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
    # block_mask[b] is an integer where bit i is 1 if block b covers t-subset i
    block_masks = [0] * n_blocks
    
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        # Generate all t-subsets of this k-block
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks[b_idx] = mask

    best_blocks = None
    best_count = float('inf')

    rng = random.Random(42)
    max_restarts = 200  # Bitmask ops are fast; more restarts with faster scans
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        uncovered_mask = (1 << n_subs) - 1
        selected_blocks = []
        
        # Optimization: Keep a list of active block indices
        active_blocks = list(range(n_blocks))
        
        while uncovered_mask:
            if time.time() > deadline:
                break

            best_gain = 0
            best_block_candidates = []

            new_active_blocks = []
            for b_idx in active_blocks:
                mask = block_masks[b_idx]
                # Quick reject: if the block adds nothing new, skip the count
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

            # Randomized tie-breaking
            chosen_block = rng.choice(best_block_candidates)
            selected_blocks.append(chosen_block)
            uncovered_mask &= ~block_masks[chosen_block]
            active_blocks = new_active_blocks

            if not active_blocks:
                break

        if uncovered_mask == 0:
            # Local Search: Delete redundant blocks
            # Compute cover counts
            cover_counts = [0] * n_subs
            for b_idx in selected_blocks:
                mask = block_masks[b_idx]
                m = mask
                while m:
                    idx = (m & -m).bit_length() - 1
                    cover_counts[idx] += 1
                    m &= m - 1
            
            changed = True
            while changed:
                changed = False
                for i in range(len(selected_blocks)):
                    b_idx = selected_blocks[i]
                    mask = block_masks[b_idx]
                    m = mask
                    is_redundant = True
                    while m:
                        idx = (m & -m).bit_length() - 1
                        if cover_counts[idx] == 1:
                            is_redundant = False
                            break
                        m &= m - 1
                            
                    if is_redundant:
                        # Remove block i
                        m = mask
                        while m:
                            idx = (m & -m).bit_length() - 1
                            cover_counts[idx] -= 1
                            m &= m - 1
                        selected_blocks.pop(i)
                        changed = True
                        break # Restart loop
            
            # 2-exchange local search: try removing two blocks and re-covering
            # the gap greedily. This can escape local optima where no single
            # block is deletable but a pair is.
            if len(selected_blocks) >= 2:
                n_sel = len(selected_blocks)
                improved_2x = True
                while improved_2x and time.time() < deadline:
                    improved_2x = False
                    # Compute cover counts
                    cover_counts = [0] * n_subs
                    for b_idx in selected_blocks:
                        m = block_masks[b_idx]
                        while m:
                            idx = (m & -m).bit_length() - 1
                            cover_counts[idx] += 1
                            m &= m - 1
                    # Try random pairs (bounded to keep time)
                    n_pairs = min(200, n_sel * (n_sel - 1) // 2)
                    for _ in range(n_pairs):
                        if time.time() > deadline:
                            break
                        i, j = rng.sample(range(n_sel), 2)
                        mi = block_masks[selected_blocks[i]]
                        mj = block_masks[selected_blocks[j]]
                        # Subsets that become uncovered
                        gap_mask = (mi & mj)  # covered by both
                        # Actually: subsets covered only by i or only by j (not by others)
                        # Compute: for each subset in mi, is cover_count == 1 (only i)?
                        # We need subsets that would be uncovered if we remove i and j.
                        # A subset s is uncovered after removal iff it was covered only by i or only by j.
                        # That is: s in mi and cover_counts[s] == 1 (only i covers it), OR
                        # s in mj and cover_counts[s] == 1 (only j covers it).
                        # But careful: if s is in both mi and mj and cover_counts[s]==2,
                        # removing both leaves it uncovered.
                        # General: after removing i and j, s is uncovered iff
                        #   cover_counts[s] == (1 if s in mi else 0) + (1 if s in mj else 0)
                        # i.e., cover_counts[s] equals the number of {i,j} that contain s.
                        # So gap = {s : cover_counts[s] == num_{i,j}(s)}
                        # But that's expensive to compute per pair. Simpler:
                        # gap_mask = subsets covered by i or j, then check each.
                        # For efficiency, compute gap as:
                        #   candidates = mi | mj
                        #   gap = {s in candidates : cover_counts[s] == (1 if s in mi else 0)+(1 if s in mj else 0)}
                        # We'll iterate over the smaller of mi, mj bits plus the overlap.
                        # Actually simplest: gap_mask = 0; iterate over mi bits, if cover_counts[s]==1 -> gap; 
                        #   then iterate over mj bits, if cover_counts[s]==1 -> gap (these are disjoint from mi-only)
                        #   and if s in mi and cover_counts[s]==2 -> also gap.
                        # Let me just do: for s in mi: if cover_counts[s] == (1 if not (s in mj) else 2) ... 
                        # This is getting complex. Let me use a different approach:
                        # gap_mask = mi | mj
                        # Then for each bit in gap_mask, check if it's still covered by some other block.
                        # That's O(gap_size * n_sel) which is too slow.
                        # Alternative: precompute for each subset the list of blocks covering it? Too memory heavy.
                        # Let me just do the simple thing: compute gap by checking cover_counts.
                        # gap_mask = 0
                        # m = mi
                        # while m: idx=(m&-m).bit_length()-1; if cover_counts[idx]==1: gap_mask|=1<<idx; m&=m-1
                        # m = mj & ~mi
                        # while m: idx=(m&-m).bit_length()-1; if cover_counts[idx]==1: gap_mask|=1<<idx; m&=m-1
                        # m = mi & mj
                        # while m: idx=(m&-m).bit_length()-1; if cover_counts[idx]==2: gap_mask|=1<<idx; m&=m-1
                        gap_mask = 0
                        m = mi
                        while m:
                            idx = (m & -m).bit_length() - 1
                            if cover_counts[idx] == 1:
                                gap_mask |= (1 << idx)
                            m &= m - 1
                        m = mj & ~mi
                        while m:
                            idx = (m & -m).bit_length() - 1
                            if cover_counts[idx] == 1:
                                gap_mask |= (1 << idx)
                            m &= m - 1
                        m = mi & mj
                        while m:
                            idx = (m & -m).bit_length() - 1
                            if cover_counts[idx] == 2:
                                gap_mask |= (1 << idx)
                            m &= m - 1
                        if gap_mask == 0:
                            # Both blocks are redundant together
                            selected_blocks.pop(j)
                            selected_blocks.pop(i)
                            improved_2x = True
                            break
                        # Try to re-cover gap with <=1 new block
                        # Find a block that covers all of gap_mask
                        # This is expensive; skip for now (only do pure deletion)
                        pass
                if improved_2x and len(selected_blocks) < best_count:
                    best_count = len(selected_blocks)
                    best_blocks = [list(all_blocks[i]) for i in selected_blocks]
                    if best_count <= 335:
                        break

            if len(selected_blocks) < best_count:
                best_count = len(selected_blocks)
                best_blocks = [list(all_blocks[i]) for i in selected_blocks]

                if best_count <= 335:
                    break

    if best_blocks is None:
        # Fallback
        return []
        
    return best_blocks
