import itertools
import random
import time
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Precompute all t-subsets and map them to indices
    all_tsets = list(itertools.combinations(range(v), t))
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
    num_tsets = len(all_tsets)
    
    # Precompute all k-blocks and their t-subsets
    # Note: Generating all k-blocks might be memory heavy if v is large, but for v=19, k=8, it's ~75k blocks.
    # Each block has C(8,5)=56 t-subsets. 75k * 56 = 4.2M entries. Manageable.
    all_kblocks = list(itertools.combinations(range(v), k))
    block_tsets = []
    for b in all_kblocks:
        ts = [tset_to_idx[comb] for comb in itertools.combinations(b, t)]
        block_tsets.append(ts)
        
    # Map t-subset index to list of block indices containing it
    tset_to_blocks = [[] for _ in range(num_tsets)]
    for bi, ts in enumerate(block_tsets):
        for ti in ts:
            tset_to_blocks[ti].append(bi)
            
    def greedy_construct(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered = [True] * num_tsets
        num_uncovered = num_tsets
        blocks = []
        
        while num_uncovered > 0:
            # Find an uncovered t-subset
            # To add variety, we can pick randomly from uncovered ones
            # But picking the one with fewest covering blocks is better for greedy
            # Let's pick the first uncovered one for speed, or random
            # Random selection from uncovered list
            uncovered_indices = [i for i in range(num_tsets) if uncovered[i]]
            if not uncovered_indices:
                break
            target_tset = random.choice(uncovered_indices)
            
            # Find best block covering this t-subset
            candidates = tset_to_blocks[target_tset]
            best_block = -1
            best_gain = -1
            
            for bi in candidates:
                gain = 0
                for ti in block_tsets[bi]:
                    if uncovered[ti]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = bi
                    
            if best_block == -1:
                break # Should not happen if covering exists
                
            blocks.append(all_kblocks[best_block])
            for ti in block_tsets[best_block]:
                if uncovered[ti]:
                    uncovered[ti] = False
                    num_uncovered -= 1
                    
        return blocks

    def remove_redundant(blocks, cover_count):
        """
        Removes redundant blocks.
        cover_count: list of length num_tsets, count of blocks covering each t-set.
        Returns new list of blocks and updated cover_count.
        """
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b in blocks:
                # Check if b is redundant
                # Find index of b in all_kblocks? 
                # This is slow. We should store block indices.
                pass
            
            # This implementation is too slow if we don't use indices.
            # Let's rewrite to use block indices.
            pass

    # Revised approach using indices for speed
    def greedy_construct_idx(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered = [True] * num_tsets
        num_uncovered = num_tsets
        block_indices = []
        
        while num_uncovered > 0:
            uncovered_indices = [i for i in range(num_tsets) if uncovered[i]]
            if not uncovered_indices:
                break
            target_tset = random.choice(uncovered_indices)
            
            candidates = tset_to_blocks[target_tset]
            best_block_idx = -1
            best_gain = -1
            
            for bi in candidates:
                gain = 0
                for ti in block_tsets[bi]:
                    if uncovered[ti]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = bi
                    
            if best_block_idx == -1:
                break
                
            block_indices.append(best_block_idx)
            for ti in block_tsets[best_block_idx]:
                if uncovered[ti]:
                    uncovered[ti] = False
                    num_uncovered -= 1
                    
        return block_indices

    def local_search(block_indices, time_limit):
        """
        Performs local search on block_indices.
        Returns best block_indices found.
        """
        best_indices = list(block_indices)
        best_size = len(best_indices)
        
        # Initialize cover counts
        cover_count = [0] * num_tsets
        for bi in best_indices:
            for ti in block_tsets[bi]:
                cover_count[ti] += 1
                
        # Precompute set of blocks for faster random selection?
        # We will just pick random indices from the current list.
        
        while time.time() < time_limit:
            # Try to remove redundant blocks
            changed = True
            while changed:
                changed = False
                new_indices = []
                for bi in best_indices:
                    is_redundant = True
                    for ti in block_tsets[bi]:
                        if cover_count[ti] == 1:
                            is_redundant = False
                            break
                    if is_redundant:
                        # Remove bi
                        for ti in block_tsets[bi]:
                            cover_count[ti] -= 1
                        changed = True
                    else:
                        new_indices.append(bi)
                if changed:
                    best_indices = new_indices
                    best_size = len(best_indices)
            
            if len(best_indices) == 0:
                break
                
            # Try a swap move
            # Pick a random block to replace
            idx_to_replace = random.randrange(len(best_indices))
            bi_old = best_indices[idx_to_replace]
            
            # Temporarily remove bi_old
            for ti in block_tsets[bi_old]:
                cover_count[ti] -= 1
                
            # Find critical t-sets (count == 0)
            critical_tsets = []
            for ti in block_tsets[bi_old]:
                if cover_count[ti] == 0:
                    critical_tsets.append(ti)
                    
            if not critical_tsets:
                # bi_old was redundant, just remove it
                best_indices.pop(idx_to_replace)
                best_size -= 1
                continue # Loop back to try more removals
                
            # Need to find a new block that covers all critical_tsets
            # Pick one critical t-set to constrain the search
            target_ti = critical_tsets[0]
            candidates = tset_to_blocks[target_ti]
            
            found_new = False
            for bi_new in candidates:
                if bi_new == bi_old:
                    continue
                # Check if bi_new covers all critical_tsets
                valid = True
                # Optimization: only check critical_tsets
                # We need to ensure all ti in critical_tsets are in block_tsets[bi_new]
                # Since block_tsets[bi_new] is a list, we can convert to set for O(1) lookup?
                # Or just loop. critical_tsets is usually small.
                set_new_tsets = set(block_tsets[bi_new])
                for ti in critical_tsets:
                    if ti not in set_new_tsets:
                        valid = False
                        break
                if valid:
                    # Found a valid replacement
                    # Add bi_new
                    for ti in block_tsets[bi_new]:
                        cover_count[ti] += 1
                    best_indices[idx_to_replace] = bi_new
                    found_new = True
                    break
                    
            if not found_new:
                # Revert
                for ti in block_tsets[bi_old]:
                    cover_count[ti] += 1
                    
        return best_indices

    best_result = None
    best_size = float('inf')
    
    # Run multiple greedy restarts
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        indices = greedy_construct_idx(random_seed=i)
        size = len(indices)
        
        if size < best_size:
            best_size = size
            best_result = list(indices)
            
    # If we have a result, try to improve it with local search
    if best_result is not None:
        # Allocate remaining time to local search
        time_left = deadline - time.time()
        if time_left > 5:
            # Run local search on the best result
            # We might want to run it on a few of the best results
            improved = local_search(best_result, time_left)
            if len(improved) < best_size:
                best_result = improved
                best_size = len(best_result)
                
    # Convert indices back to blocks
    if best_result is None:
        # Fallback: return empty or minimal if something went wrong
        return []
        
    blocks = [list(all_kblocks[bi]) for bi in best_result]
    
    # Final validation (optional, but good for debugging)
    # covered = set()
    # for b in blocks:
    #     for ts in itertools.combinations(b, t):
    #         covered.add(ts)
    # if len(covered) != num_tsets:
    #     print("Warning: Invalid cover generated")
        
    return blocks
