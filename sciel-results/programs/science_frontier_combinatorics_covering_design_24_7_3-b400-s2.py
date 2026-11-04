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
    
    # Precompute bitsets for all possible k-subsets (blocks)
    # There are C(24, 7) = 346,104 blocks
    # This might take a bit of time/memory but is doable
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # bitset[block_id] is an integer where bit i is 1 if block covers triplet i
    block_triplets = [0] * n_blocks
    
    # Mapping from block tuple to index for quick lookup if needed
    # Not strictly needed if we just return the list of blocks
    
    # Precomputation is O(C(v,k) * C(k,t))
    # 346,104 * 35 ~ 12M operations. This is fast in Python if optimized?
    # Let's try.
    
    for idx, block in enumerate(all_blocks):
        mask = 0
        for triplet in itertools.combinations(block, t):
            tid = triplet_to_id[triplet]
            mask |= (1 << tid)
        block_triplets[idx] = mask
        
    # Greedy Construction
    covered = 0
    all_mask = (1 << n_triplets) - 1
    uncovered = all_mask
    blocks = []
    block_indices = []
    
    while uncovered:
        if time.time() > deadline - 30:
            break
            
        best_idx = -1
        best_gain = 0
        
        # Find block with max gain
        # Iterate over all blocks
        for i in range(n_blocks):
            gain = (block_triplets[i] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                if gain == n_triplets: # Perfect cover? Unlikely
                    break
        
        if best_idx == -1:
            break # Should not happen if uncovered != 0
            
        blocks.append(all_blocks[best_idx])
        block_indices.append(best_idx)
        uncovered &= ~block_triplets[best_idx]
        
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
                        # Pick a point not in new_block
                        # Simple random or sequential
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

    progress_made = True
    while progress_made and time.time() < deadline - 10:
        progress_made = False
        
        blocks, removed = try_remove_redundant(blocks)
        if removed:
            progress_made = True
            
        if not progress_made:
            blocks, merged = try_merge_2_for_1(blocks)
            if merged:
                progress_made = True

    return blocks

if __name__ == "__main__":
    pass
