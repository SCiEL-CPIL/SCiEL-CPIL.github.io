import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin

    # Precompute all t-subsets and map to IDs
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    
    def get_id(triplet_tuple):
        return triplet_to_id[triplet_tuple]

    # Fast greedy construction with incremental block building
    def greedy_construct(seed=0):
        rng = random.Random(seed)
        is_uncovered = np.ones(n_triplets, dtype=bool)
        uncovered_list = list(range(n_triplets))
        pos = np.arange(n_triplets)
        blocks = []
        
        while uncovered_list:
            if time.time() > deadline - 60:
                break
                
            # Pick random uncovered triplet to seed the block
            idx = rng.randrange(len(uncovered_list))
            start_tid = uncovered_list[idx]
            start_triplet = all_triplets[start_tid]
            
            block = list(start_triplet)
            block_set = set(block)
            
            while len(block) < k:
                best_point = -1
                max_gain = -1
                candidates = [p for p in range(v) if p not in block_set]
                
                # Compute (t-1)-subsets of current block
                pairs_in_block = list(itertools.combinations(block, t - 1))
                
                for p in candidates:
                    gain = 0
                    for pair in pairs_in_block:
                        triplet = tuple(sorted(pair + (p,)))
                        tid = get_id(triplet)
                        if is_uncovered[tid]:
                            gain += 1
                            
                    if gain > max_gain:
                        max_gain = gain
                        best_point = p
                        if max_gain == len(pairs_in_block):
                            break
                            
                if max_gain <= 0:
                    best_point = rng.choice(candidates)

                block.append(best_point)
                block_set.add(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered: mark all triplets in this block as covered
            for triplet in itertools.combinations(block, t):
                tid = get_id(triplet)
                if is_uncovered[tid]:
                    is_uncovered[tid] = False
                    # Swap-remove from uncovered_list
                    cur_idx = pos[tid]
                    last_idx = len(uncovered_list) - 1
                    if cur_idx != last_idx:
                        last_tid = uncovered_list[last_idx]
                        uncovered_list[cur_idx] = last_tid
                        pos[last_tid] = cur_idx
                    uncovered_list.pop()
            
        return blocks

    # Run multiple greedy restarts, keep the best
    best_blocks = None
    best_count = float('inf')
    
    # Budget for greedy: use up to 30 seconds for multiple restarts
    greedy_deadline = start_time + 30
    seeds = list(range(20))
    random.shuffle(seeds)
    
    for seed in seeds:
        if time.time() > greedy_deadline:
            break
        blocks = greedy_construct(seed)
        # Verify coverage
        covered = set()
        for b in blocks:
            for triplet in itertools.combinations(b, t):
                covered.add(triplet)
        if len(covered) == n_triplets:
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
    
    if best_blocks is None:
        # Fallback: single greedy
        best_blocks = greedy_construct(0)
    
    blocks = best_blocks
    
    # Local Search
    # Maintain count of how many times each triplet is covered
    count = np.zeros(n_triplets, dtype=np.int32)
    
    def update_counts(blocks_list, delta=1):
        for b in blocks_list:
            for triplet in itertools.combinations(b, t):
                tid = triplet_to_id[triplet]
                count[tid] += delta

    # Initialize counts
    update_counts(blocks, 1)
    
    def is_redundant(block):
        for triplet in itertools.combinations(block, t):
            tid = triplet_to_id[triplet]
            if count[tid] == 1:
                return False
        return True

    def try_remove_redundant(blocks_list):
        removed = False
        new_blocks = []
        for b in blocks_list:
            if is_redundant(b):
                update_counts([b], -1)
                removed = True
            else:
                new_blocks.append(b)
        return new_blocks, removed

    def get_unique_triplets(block):
        """Return list of triplet IDs that are only covered by this block."""
        unique = []
        for triplet in itertools.combinations(block, t):
            tid = triplet_to_id[triplet]
            if count[tid] == 1:
                unique.append(tid)
        return unique

    def try_merge_2_for_1(blocks_list):
        n = len(blocks_list)
        if n < 2:
            return blocks_list, False
            
        # Precompute unique triplets for each block
        block_unique = [get_unique_triplets(b) for b in blocks_list]
        
        for i in range(n):
            if time.time() > deadline - 5:
                return blocks_list, False
            if not block_unique[i]:
                continue
                
            for j in range(i + 1, n):
                if time.time() > deadline - 5:
                    return blocks_list, False
                if not block_unique[j]:
                    continue
                
                # Union of unique triplets
                all_unique = set(block_unique[i]) | set(block_unique[j])
                
                # Check if there is a block that covers all these unique triplets
                # The union of points in these triplets must be <= k
                points_needed = set()
                for tid in all_unique:
                    points_needed.update(all_triplets[tid])
                    
                if len(points_needed) <= k:
                    # We can form a block containing points_needed
                    # Fill the rest with arbitrary points to make size k
                    new_block = list(points_needed)
                    while len(new_block) < k:
                        p = random.randint(0, v - 1)
                        if p not in new_block:
                            new_block.append(p)
                    new_block.sort()
                    
                    # Update counts
                    update_counts([blocks_list[i], blocks_list[j]], -1)
                    update_counts([new_block], 1)
                    
                    new_blocks = [b for idx, b in enumerate(blocks_list) if idx != i and idx != j]
                    new_blocks.append(new_block)
                    
                    return new_blocks, True
                    
        return blocks_list, False

    def try_swap_1_for_1(blocks_list):
        """Try to replace one block with another that covers more unique triplets."""
        n = len(blocks_list)
        if n < 1:
            return blocks_list, False
            
        # For each block, find its unique triplets
        # Try to find a replacement block that covers all unique triplets of the original
        # and also covers some additional triplets
        for i in range(n):
            if time.time() > deadline - 5:
                return blocks_list, False
                
            unique_i = get_unique_triplets(blocks_list[i])
            if not unique_i:
                continue
                
            # Points that must be in the replacement block
            points_needed = set()
            for tid in unique_i:
                points_needed.update(all_triplets[tid])
            
            if len(points_needed) > k:
                continue
                
            # Try to find a block containing points_needed that covers more triplets
            # than the original block
            orig_mask = 0
            for triplet in itertools.combinations(blocks_list[i], t):
                tid = triplet_to_id[triplet]
                orig_mask |= (1 << tid)
            
            best_new_block = None
            best_gain = 0
            
            # Try a few random completions
            for _ in range(10):
                new_block = list(points_needed)
                remaining = [p for p in range(v) if p not in points_needed]
                random.shuffle(remaining)
                new_block.extend(remaining[:k - len(new_block)])
                new_block.sort()
                
                # Compute coverage gain
                new_mask = 0
                for triplet in itertools.combinations(new_block, t):
                    tid = triplet_to_id[triplet]
                    new_mask |= (1 << tid)
                
                # Gain: triplets covered by new but not by old, minus those only covered by old
                # Actually, we need: new block must cover all unique_i, and cover more overall
                # Check that new covers all unique_i
                covers_unique = True
                for tid in unique_i:
                    if not (new_mask >> tid) & 1:
                        covers_unique = False
                        break
                if not covers_unique:
                    continue
                
                # Count how many triplets are only covered by old block
                # and would become uncovered
                # This is complex; simple heuristic: if new covers all unique_i, accept
                # if it covers at least as many total triplets
                new_count = new_mask.bit_count()
                if new_count > orig_mask.bit_count():
                    if best_new_block is None or new_count > best_gain:
                        best_gain = new_count
                        best_new_block = new_block
            
            if best_new_block is not None:
                # Replace block i with best_new_block
                update_counts([blocks_list[i]], -1)
                update_counts([best_new_block], 1)
                blocks_list[i] = best_new_block
                return blocks_list, True
        
        return blocks_list, False

    # Iterative local search
    progress_made = True
    iteration = 0
    while progress_made and time.time() < deadline - 10:
        progress_made = False
        iteration += 1
        
        # Try to remove redundant blocks
        blocks, removed = try_remove_redundant(blocks)
        if removed:
            progress_made = True
            
        if not progress_made:
            # Try 2-for-1 merges
            blocks, merged = try_merge_2_for_1(blocks)
            if merged:
                progress_made = True
            
        if not progress_made:
            # Try 1-for-1 swaps
            blocks, swapped = try_swap_1_for_1(blocks)
            if swapped:
                progress_made = True

    return blocks

if __name__ == "__main__":
    pass
