import time
import itertools
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Precomputation
    all_points = list(range(v))
    all_triples = list(itertools.combinations(all_points, t))
    n_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    # Generate all possible k-blocks
    all_blocks = list(itertools.combinations(all_points, k))
    n_blocks = len(all_blocks)
    
    # Precompute masks for each block
    # mask is an integer where bit i is 1 if block covers triple i
    block_masks = [0] * n_blocks
    # Map triple index to list of block indices that cover it
    blocks_by_triple = [[] for _ in range(n_triples)]
    
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for sub in itertools.combinations(block, t):
            t_idx = triple_to_idx[sub]
            mask |= (1 << t_idx)
            blocks_by_triple[t_idx].append(b_idx)
        block_masks[b_idx] = mask
        
    # Helper: Greedy Set Cover using bitsets
    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered_mask = (1 << n_triples) - 1
        selected_indices = set()
        selected_blocks = []
        
        while uncovered_mask:
            # Pick a random uncovered triple
            while True:
                t_idx = rng.randint(0, n_triples - 1)
                if (uncovered_mask >> t_idx) & 1:
                    break
            
            candidates = blocks_by_triple[t_idx]
            
            best_gain = -1
            best_block_idx = -1
            
            # Evaluate candidates
            for b_idx in candidates:
                if b_idx in selected_indices:
                    continue
                mask = block_masks[b_idx]
                gain = (mask & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
                elif gain == best_gain and gain > 0 and rng.random() < 0.5:
                    # Random tie-breaking to explore different paths
                    best_block_idx = b_idx
                    
            if best_block_idx == -1:
                # Fallback: pick any valid candidate
                for b_idx in candidates:
                    if b_idx not in selected_indices:
                        best_block_idx = b_idx
                        break
                if best_block_idx == -1:
                    break # Should not happen
            
            selected_blocks.append(all_blocks[best_block_idx])
            selected_indices.add(best_block_idx)
            uncovered_mask &= ~block_masks[best_block_idx]
            
        return selected_blocks

    def local_search(blocks, seed, time_limit=None):
        if not blocks:
            return blocks
        
        if time_limit is None:
            time_limit = deadline - 5
            
        rng = random.Random(seed)
        blocks = [list(b) for b in blocks]
        
        # Precompute triple indices for each block in current solution
        def get_block_triple_indices(block):
            return [triple_to_idx[sub] for sub in itertools.combinations(block, t)]
        
        # Build coverage counts
        cov_count = np.zeros(n_triples, dtype=np.int32)
        block_triple_indices = []
        for b in blocks:
            tris = get_block_triple_indices(b)
            block_triple_indices.append(tris)
            for ti in tris:
                cov_count[ti] += 1
                
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # 1. Remove redundant blocks
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() > time_limit:
                    break
                redundant = True
                for ti in block_triple_indices[i]:
                    if cov_count[ti] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block i
                    for ti in block_triple_indices[i]:
                        cov_count[ti] -= 1
                    blocks.pop(i)
                    block_triple_indices.pop(i)
                    improved = True
                    break # Restart loop as indices changed
            
            if not improved:
                # 2. Try swaps to create redundancy
                # Pick a random block and a random point to swap out
                if len(blocks) > 0:
                    # Increase swap attempts per pass for better exploration
                    for _ in range(50):
                        if time.time() > time_limit:
                            break
                        idx = rng.randint(0, len(blocks) - 1)
                        old_x = rng.choice(blocks[idx])
                        new_x = rng.randint(0, v - 1)
                        
                        if new_x == old_x or new_x in blocks[idx]:
                            continue
                        
                        # Calculate effect of swap
                        old_tris = block_triple_indices[idx]
                        new_block = blocks[idx][:]
                        new_block[new_block.index(old_x)] = new_x
                        new_block.sort()
                        new_tris = get_block_triple_indices(new_block)
                        
                        # Check if swap is valid (doesn't leave any triple uncovered)
                        old_set = set(old_tris)
                        new_set = set(new_tris)
                        
                        safe = True
                        for ti in old_set - new_set:
                            if cov_count[ti] == 1:
                                safe = False
                                break
                                
                        if safe:
                            # Apply swap
                            # Update cov_count
                            for ti in old_set - new_set:
                                cov_count[ti] -= 1
                            for ti in new_set - old_set:
                                cov_count[ti] += 1
                            
                            blocks[idx] = new_block
                            block_triple_indices[idx] = new_tris
                            improved = True
                            break # Restart main loop to try removing blocks again
        
        return blocks

    best_blocks = None
    best_len = float('inf')
    
    # Main loop: Multiple restarts
    seed_counter = 0
    while time.time() < deadline - 10:
        # Generate initial covering
        blocks = greedy_cover(seed_counter)
        
        # Refine with local search
        # Use a different seed for local search to ensure variety
        refined_blocks = local_search(blocks, seed_counter + 1000)
        
        if len(refined_blocks) < best_len:
            best_len = len(refined_blocks)
            best_blocks = refined_blocks
        
        seed_counter += 1
        
        # If we have time, do more local search on the best solution found so far
        if best_blocks is not None and time.time() < deadline - 20:
            # Try to improve the best solution further
            current_best_copy = [list(b) for b in best_blocks]
            improved_best = local_search(current_best_copy, seed_counter + 2000)
            if len(improved_best) < best_len:
                best_len = len(improved_best)
                best_blocks = improved_best

    return best_blocks if best_blocks is not None else []
