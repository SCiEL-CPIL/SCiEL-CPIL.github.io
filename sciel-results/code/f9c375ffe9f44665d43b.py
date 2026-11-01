import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Precomputation
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_to_idx = {tset: i for i, tset in enumerate(all_tsets)}

    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    t_subsets_per_block = len(list(itertools.combinations(range(k), t)))
    BlockTsets = np.zeros((num_blocks, t_subsets_per_block), dtype=np.int32)

    for i, block in enumerate(all_blocks):
        count = 0
        for sub in itertools.combinations(block, t):
            BlockTsets[i, count] = tset_to_idx[sub]
            count += 1

    # Convert to a list of frozensets for faster set operations if needed
    # Actually, for intersection size, we can use the precomputed BlockTsets and check overlap in t-set space?
    # No, intersection size of blocks is |A ∩ B|.
    # We can precompute a matrix of intersection sizes? 75582 x 75582 is too big.
    # We can compute on the fly.
    # Since k=8, intersection size is small.
    # We can just compute len(set(a) & set(b)).

    def intersection_size(b1, b2):
        return len(set(all_blocks[b1]) & set(all_blocks[b2]))

    best_blocks = None
    best_len = float('inf')
    best_cover_count = None

    def polish(blocks):
        if len(blocks) >= best_len:
            return blocks
        
        # Build coverage count
        cover_count = np.zeros(num_tsets, dtype=np.int32)
        for b in blocks:
            cover_count[BlockTsets[b]] += 1
        
        # Remove redundant blocks (multiple passes)
        # Pass 1: Reverse greedy removal
        i = len(blocks) - 1
        while i >= 0:
            b = blocks[i]
            can_remove = True
            for ts in BlockTsets[b]:
                if cover_count[ts] == 1:
                    can_remove = False
                    break
            if can_remove:
                for ts in BlockTsets[b]:
                    cover_count[ts] -= 1
                blocks.pop(i)
            else:
                i -= 1
        
        # Pass 2: Random removals
        for _ in range(min(20, len(blocks))):
            if time.time() > deadline:
                break
            idx = np.random.randint(len(blocks))
            b = blocks[idx]
            can_remove = True
            for ts in BlockTsets[b]:
                if cover_count[ts] == 1:
                    can_remove = False
                    break
            if can_remove:
                for ts in BlockTsets[b]:
                    cover_count[ts] -= 1
                blocks.pop(idx)
        
        # Pass 3: Pairwise merge local search
        # If two blocks' uniquely-covered t-sets are disjoint, we can try to replace them with one block
        # that covers their union of uniquely-covered t-sets, if such a block exists.
        # This is complex. Let's stick to simpler improvements.
        # Actually, the "pairwise merge" idea from inspiration 3 is:
        # If two blocks have a large intersection, their coverage is redundant.
        # We can try to remove one if the other covers all its unique parts.
        # But that's just the redundancy removal.
        
        # Let's try a swap-based local search.
        # If we can replace one block with another that reduces the total count? No, swap keeps count same.
        # We want to reduce count.
        
        # The main improvement here is the intersection constraint during greedy.
        
        if len(blocks) < best_len:
            return blocks
        return blocks

    # Greedy Search with Intersection Constraint
    while time.time() < deadline:
        uncovered = np.ones(num_tsets, dtype=bool)
        blocks = []
        selected_blocks_set = set()
        
        while uncovered.any() and time.time() < deadline:
            # Calculate gain for all blocks
            gains = uncovered[BlockTsets].sum(axis=1)
            
            max_gain = gains.max()
            if max_gain == 0:
                break
                
            # Find all blocks with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Filter candidates by intersection constraint
            # We want blocks that intersect existing blocks in at most k-t points?
            # Actually, if a new block intersects an existing block in > k-t points,
            # it means they share > k-t points, so their t-subsets overlap significantly.
            # Specifically, if |A ∩ B| >= t, they share t-subsets.
            # If |A ∩ B| > k-t, then the number of shared t-subsets is large.
            # Let's enforce: intersection with ANY existing block must be <= k-t.
            # This is a strong constraint.
            
            valid_candidates = []
            for c in candidates:
                if c in selected_blocks_set:
                    continue
                valid = True
                for sb in selected_blocks_set:
                    if intersection_size(c, sb) > k - t:
                        valid = False
                        break
                if valid:
                    valid_candidates.append(c)
            
            if not valid_candidates:
                # Relax constraint: just pick a random candidate
                best_block_idx = candidates[np.random.randint(len(candidates))]
            else:
                # Among valid candidates, pick one with minimum average intersection with existing blocks
                # to maximize diversity
                min_avg_inter = float('inf')
                best_candidate = valid_candidates[0]
                for vc in valid_candidates:
                    if len(selected_blocks_set) == 0:
                        avg_inter = 0
                    else:
                        total_inter = sum(intersection_size(vc, sb) for sb in selected_blocks_set)
                        avg_inter = total_inter / len(selected_blocks_set)
                    if avg_inter < min_avg_inter:
                        min_avg_inter = avg_inter
                        best_candidate = vc
                
                best_block_idx = best_candidate
            
            blocks.append(best_block_idx)
            selected_blocks_set.add(best_block_idx)
            
            # Update uncovered
            uncovered[BlockTsets[best_block_idx]] = False
        
        # Polish
        polished_blocks = polish(blocks)
        
        if len(polished_blocks) < best_len:
            best_len = len(polished_blocks)
            best_blocks = polished_blocks[:]

    if best_blocks is None:
        return []

    result = [list(all_blocks[b]) for b in best_blocks]
    return result
