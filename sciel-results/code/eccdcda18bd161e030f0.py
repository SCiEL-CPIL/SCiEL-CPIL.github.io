import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    # Precompute all t-subsets and map to indices
    all_ts = list(itertools.combinations(range(v), t))
    n_ts = len(all_ts)
    ts_index = {s: i for i, s in enumerate(all_ts)}

    # Precompute all k-subsets
    all_ks = list(itertools.combinations(range(v), k))
    n_ks = len(all_ks)
    
    # Number of t-subsets in a k-set
    t_count = int(np.prod([k - i for i in range(t)]) / np.prod([i + 1 for i in range(t)]))
    
    # Precompute for each k-subset, the indices of the t-subsets it contains
    # block_t_indices[i, j] is the index of the j-th t-subset in the i-th k-subset
    block_t_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    for i, block in enumerate(all_ks):
        subs = itertools.combinations(block, t)
        for j, sub in enumerate(subs):
            block_t_indices[i, j] = ts_index[sub]

    def greedy_randomized(seed, top_k=10):
        """
        Randomized greedy set cover.
        Picks a block from the top 'top_k' gainers randomly.
        """
        rng = np.random.RandomState(seed)
        covered = np.zeros(n_ts, dtype=bool)
        selected_blocks = []
        
        while True:
            if time.time() > deadline - 5.0:
                return None
            
            # Calculate gains: number of uncovered triples covered by each block
            # covered[block_t_indices] gets the coverage status for all triples in all blocks
            # ~ inverts it to get "uncovered" status
            # .sum(axis=1) counts them per block
            gains = (~covered[block_t_indices]).sum(axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                # If max gain is 0, we are stuck or done. 
                # If not all covered, this is a failure for this run.
                if not np.all(covered):
                    return None
                else:
                    break
            
            # Find indices of blocks with max gain
            # To add randomness and escape local optima, we pick from the top K
            # But strictly speaking, standard greedy picks max. 
            # "Randomized Top-K" picks from blocks with gain close to max.
            
            # Let's pick from blocks with gain == max_gain first. 
            # If there are many, pick randomly.
            # To be more aggressive in escaping, we can allow gain == max_gain - 1 if top_k is large?
            # The prompt suggests "weighted sum". 
            # Let's stick to: Pick randomly from candidates with max_gain.
            # If we want to explore more, we can pick from top 'top_k' gainers.
            
            # Get sorted gains? No, that's expensive.
            # Just find where gains == max_gain
            candidates = np.where(gains == max_gain)[0]
            
            if len(candidates) > top_k:
                # Sample from top candidates
                # Actually, if there are thousands of max gainers, any is fine.
                # If we want to diversify, we might want to pick from a broader set.
                # Let's just pick randomly from all max gainers.
                # This is effectively "randomized greedy".
                pass
            
            best_idx = candidates[rng.randint(len(candidates))]
            
            # Add block
            selected_blocks.append(best_idx)
            
            # Update covered
            covered[block_t_indices[best_idx]] = True
            
            if np.all(covered):
                break

        return selected_blocks

    def reduce_blocks(block_indices):
        """
        Remove redundant blocks.
        A block is redundant if every t-subset in it is covered by at least one other block.
        """
        if not block_indices:
            return []
            
        current_blocks = list(block_indices)
        
        # We iterate until no more blocks can be removed
        while True:
            if time.time() > deadline - 2.0:
                break
                
            # Count coverage for each t-subset
            # coverage_count[j] = number of blocks in current_blocks that cover t-subset j
            coverage_count = np.zeros(n_ts, dtype=np.int32)
            for b_idx in current_blocks:
                coverage_count[block_t_indices[b_idx]] += 1
            
            new_blocks = []
            removed_any = False
            
            for b_idx in current_blocks:
                # Check if this block is redundant
                # Get the t-subsets in this block
                subs_in_block = block_t_indices[b_idx]
                
                # If all these t-subsets have coverage_count > 1, the block is redundant
                # Note: coverage_count includes this block itself, so we need > 1
                if np.all(coverage_count[subs_in_block] > 1):
                    # Redundant: do not add to new_blocks
                    # Decrement coverage for these subsets
                    coverage_count[subs_in_block] -= 1
                    removed_any = True
                else:
                    # Keep it
                    new_blocks.append(b_idx)
            
            current_blocks = new_blocks
            
            if not removed_any:
                break
                
        return current_blocks

    best_blocks = None
    best_size = float('inf')
    
    # Run multiple greedy attempts
    # We can afford many runs due to speed of NumPy gain calculation
    num_runs = 0
    while time.time() < deadline - 10.0:
        seed = num_runs
        blocks = greedy_randomized(seed)
        
        if blocks is not None:
            # Reduce redundancy
            reduced = reduce_blocks(blocks)
            
            if len(reduced) < best_size:
                best_size = len(reduced)
                best_blocks = reduced
                
                # Early stop if we hit theoretical lower bound or known best
                if best_size <= 76:
                    # We could try to improve further, but 76 is the known best.
                    # Let's run a few more to see if we can get 75? Unlikely.
                    # But maybe we can get a consistent 76 faster?
                    pass
        
        num_runs += 1
        
        # If we've run enough, break
        if num_runs > 200: 
             break

    if best_blocks is not None:
        # Convert indices back to lists of points
        return [list(all_ks[i]) for i in best_blocks]
    
    # Fallback if no valid covering found (should not happen)
    # Return a valid covering using pure greedy (no randomness) if possible
    # Or just return empty list which will be invalid.
    # Let's try one deterministic run.
    blocks = greedy_randomized(0)
    if blocks:
        return [list(all_ks[i]) for i in blocks]
        
    return []
