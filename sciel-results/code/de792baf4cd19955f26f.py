import itertools
import random
import time
import sys

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115  # Leave margin
    
    all_points = list(range(v))
    all_tsubsets = list(itertools.combinations(all_points, t))
    tsubset_index = {ts: i for i, ts in enumerate(all_tsubsets)}
    n_tsubsets = len(all_tsubsets)
    
    # Precompute all k-blocks and their covered t-subsets
    all_blocks = list(itertools.combinations(all_points, k))
    block_to_tsubs = {}
    tsub_to_blocks = {i: [] for i in range(n_tsubsets)}
    
    for b in all_blocks:
        covered_tsubs = []
        for ts in itertools.combinations(b, t):
            idx = tsubset_index[ts]
            covered_tsubs.append(idx)
            tsub_to_blocks[idx].append(b)
        block_to_tsubs[b] = covered_tsubs
        
    def greedy_init():
        uncovered = [True] * n_tsubsets
        n_uncovered = n_tsubsets
        blocks = []
        block_set = set()
        
        while n_uncovered > 0:
            best_block = None
            best_gain = -1
            # To speed up, we can just pick a random uncovered t-subset and find a block covering it,
            # but standard greedy picks the block covering most uncovered.
            # With precomputed block_to_tsubs, we can iterate blocks.
            # C(19,7) is 50k. Iterating 50k blocks * 150 steps is 7.5M ops. Fast enough.
            
            # Optimization: Pick a random uncovered t-subset, get its candidate blocks,
            # and pick the one covering most uncovered among those.
            # This is much faster than scanning all blocks.
            
            # Find an uncovered t-subset
            # Instead of scanning all, maintain a list of uncovered indices?
            # For simplicity and speed, let's scan. 
            # Actually, let's use the tsub_to_blocks.
            
            # Pick a random uncovered t-subset
            # We need to know which are uncovered.
            # Let's just iterate all blocks and calculate gain.
            # 50k blocks * 35 tsubs = 1.75M checks per step.
            # 150 steps * 1.75M = 262M checks. Might be too slow in Python.
            
            # Better Greedy:
            # 1. Pick a random uncovered t-subset index `u_idx`.
            # 2. Get all blocks covering `u_idx` from `tsub_to_blocks[u_idx]`.
            # 3. For each candidate block, count how many *other* uncovered t-subsets it covers.
            # 4. Pick the best one.
            
            # We need to efficiently find an uncovered t-subset.
            # Maintain a list `uncovered_list`.
            
            pass # Placeholder, will implement in final code

    # Let's implement the faster greedy
    def greedy_init_fast():
        uncovered = [True] * n_tsubsets
        n_uncovered = n_tsubsets
        blocks = []
        block_set = set()
        uncovered_indices = list(range(n_tsubsets))
        
        while n_uncovered > 0:
            # Pick a random uncovered t-subset
            rand_idx_in_list = random.randint(0, len(uncovered_indices) - 1)
            u_idx = uncovered_indices[rand_idx_in_list]
            
            # Get candidate blocks covering this t-subset
            candidates = tsub_to_blocks[u_idx]
            
            best_block = None
            best_gain = -1
            
            for b in candidates:
                if b in block_set:
                    continue
                gain = 0
                for ts_idx in block_to_tsubs[b]:
                    if uncovered[ts_idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
            
            if best_block is None:
                # Should not happen if covering exists
                break
                
            blocks.append(best_block)
            block_set.add(best_block)
            
            # Update uncovered
            for ts_idx in block_to_tsubs[best_block]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    n_uncovered -= 1
            
            # Remove newly covered from uncovered_indices?
            # It's O(N) to filter. Let's just let it contain stale entries and check `uncovered` flag.
            # Or, we can just rebuild uncovered_indices occasionally?
            # For now, just rely on `uncovered` array.
            # To pick random uncovered, we need to handle the fact that `uncovered_indices` has stale entries.
            # We can just pick random from `uncovered_indices` and check if it's still uncovered.
            # If not, pick again.
            
            # Actually, simpler: just scan for first uncovered? No, random is better for diversity.
            # Let's just pick random index from 0 to n_tsubsets-1 and check if uncovered.
            # This is O(1) expected time if density is high.
            
        return blocks

    # ILS
    def ils(blocks_init):
        blocks = list(blocks_init)
        block_set = set(blocks)
        
        # Compute coverage counts
        coverage = [0] * n_tsubsets
        for b in blocks:
            for ts_idx in block_to_tsubs[b]:
                coverage[ts_idx] += 1
        
        # Check validity
        # All should be > 0
        
        best_blocks = blocks[:]
        best_size = len(blocks)
        
        current_time = time.time()
        
        while (current_time - start_time) < time_limit:
            # Pick a random block to remove/swap
            idx = random.randint(0, len(blocks) - 1)
            b_old = blocks[idx]
            
            # Temporarily remove
            for ts_idx in block_to_tsubs[b_old]:
                coverage[ts_idx] -= 1
            
            # Find uncovered t-subsets (holes)
            holes = []
            for ts_idx in block_to_tsubs[b_old]:
                if coverage[ts_idx] == 0:
                    holes.append(ts_idx)
            
            if not holes:
                # Valid removal
                blocks.pop(idx)
                block_set.remove(b_old)
                if len(blocks) < best_size:
                    best_size = len(blocks)
                    best_blocks = blocks[:]
            else:
                # Need to replace b_old with a block that covers all holes
                # Find intersection of tsub_to_blocks[h] for all h in holes
                # Since holes are small (max 35), we can intersect lists.
                
                # Optimize: Pick the hole with fewest candidate blocks
                holes_sorted = sorted(holes, key=lambda h: len(tsub_to_blocks[h]))
                candidates = set(tsub_to_blocks[holes_sorted[0]])
                
                for h in holes_sorted[1:]:
                    candidates.intersection_update(tsub_to_blocks[h])
                    if not candidates:
                        break
                
                if candidates:
                    # Pick a random candidate or the one that covers most other low-coverage tsubs
                    # For now, pick random to explore
                    b_new = random.choice(list(candidates))
                    
                    # If b_new is already in block_set (and not b_old), we can't add it (duplicate blocks not allowed? 
                    # "blocks must be distinct").
                    # If b_new is in block_set, we can't use it as a replacement for b_old if it's the same block?
                    # No, b_old is removed from set. If b_new is another existing block, we can't add it.
                    # So we must pick b_new NOT in block_set.
                    
                    valid_candidates = [b for b in candidates if b not in block_set]
                    
                    if valid_candidates:
                        b_new = random.choice(valid_candidates)
                        
                        # Replace
                        blocks[idx] = b_new
                        block_set.add(b_new)
                        
                        # Update coverage
                        for ts_idx in block_to_tsubs[b_new]:
                            coverage[ts_idx] += 1
                    else:
                        # Revert
                        for ts_idx in block_to_tsubs[b_old]:
                            coverage[ts_idx] += 1
                else:
                    # Revert
                    for ts_idx in block_to_tsubs[b_old]:
                        coverage[ts_idx] += 1
            
            # Periodic kick: if stuck, randomize
            # Simple kick: remove 5 random blocks, add 5 random valid ones?
            # Too complex. Let's just run.
            
            current_time = time.time()
            
        return best_blocks

    # Run
    init_blocks = greedy_init_fast()
    result_blocks = ils(init_blocks)
    
    return [list(b) for b in result_blocks]
