import itertools
import time

def construct(v, k, t):
        # Precomputation
        points = range(v)
        all_t_subsets = list(itertools.combinations(points, t))
        t_sub_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
        
        # We need to quickly find blocks containing a given t-subset.
        # A block is a k-subset. If it contains t-subset S, the other k-t points are from v-t.
        
        uncovered = set(range(len(all_t_subsets)))
        used_blocks = set()
        blocks = []
        
        # Count of available blocks for each t-subset
        # Initially, for any t-subset, there are C(v-t, k-t) blocks containing it.
        num_blocks_per_subset = 1
        for i in range(k - t):
            num_blocks_per_subset *= (v - t - i)
            num_blocks_per_subset //= (i + 1)
            
        subset_avail = [num_blocks_per_subset] * len(all_t_subsets)
        
        start_time = time.time()
        deadline = start_time + 110 # Safety margin
        
        while uncovered and time.time() < deadline:
            # 1. Select the most constrained uncovered t-subset
            # Find index in uncovered with min subset_avail
            best_idx = -1
            min_avail = float('inf')
            for idx in uncovered:
                if subset_avail[idx] < min_avail:
                    min_avail = subset_avail[idx]
                    best_idx = idx
            
            if best_idx == -1:
                break
                
            S = all_t_subsets[best_idx]
            
            # 2. Generate all blocks containing S
            # Remaining points
            S_set = set(S)
            remaining_points = [p for p in points if p not in S_set]
            
            best_block = None
            max_gain = -1
            
            # Iterate over combinations of k-t points from remaining_points
            for comb in itertools.combinations(remaining_points, k - t):
                block_tuple = tuple(sorted(S + comb))
                if block_tuple in used_blocks:
                    continue
                
                # Calculate gain: how many uncovered t-subsets does this block cover?
                # A block covers C(k, t) t-subsets.
                gain = 0
                # We need to check if each t-subset of the block is in uncovered
                # Precomputed list of t-subsets for this block? No, generate on fly.
                # C(8, 5) = 56. This is small.
                for t_sub in itertools.combinations(block_tuple, t):
                    idx = t_sub_to_idx[t_sub]
                    if idx in uncovered:
                        gain += 1
                
                if gain > max_gain:
                    max_gain = gain
                    best_block = block_tuple
            
            if best_block is None:
                # No valid block found? Should not happen if uncovered is not empty
                break
                
            # 3. Add block
            blocks.append(list(best_block))
            used_blocks.add(best_block)
            
            # 4. Update uncovered and subset_avail
            for t_sub in itertools.combinations(best_block, t):
                idx = t_sub_to_idx[t_sub]
                if idx in uncovered:
                    uncovered.remove(idx)
                # Decrement availability for this subset
                # Note: The block we just used is no longer available for other subsets
                # But subset_avail tracks how many *potential* blocks (from the full set of C(v,k))
                # are still "available" in the sense of not being used?
                # Actually, subset_avail should reflect how many blocks *could still be chosen* to cover it.
                # Since we are picking blocks from the full space, and we mark used_blocks,
                # the number of *remaining* blocks containing this subset decreases by 1 for every block we pick that contains it.
                # So yes, decrement subset_avail[idx] for every t-subset in the chosen block.
                subset_avail[idx] -= 1

        return blocks
