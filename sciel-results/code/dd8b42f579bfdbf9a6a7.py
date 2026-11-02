import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin for 120s limit
    
    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    
    # Map each t-subset to an index for fast lookup
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # Precompute for each t-subset, the list of k-blocks that contain it
    # We need to iterate over all k-blocks? No, we can generate them on the fly or precompute.
    # For v=19, k=7, total blocks is 50388.
    # Let's precompute the incidence: block -> list of t-subset indices it covers.
    # And reverse: t-subset -> list of block indices (or just the blocks themselves) that cover it.
    
    # To save memory and time, we will generate blocks on the fly if needed, 
    # but precomputing the reverse map is better for the greedy step.
    
    # Let's precompute all k-blocks and their covered t-subsets.
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)
    
    # block_to_t_subs: list of lists of indices
    block_to_t_subs = [None] * n_k_blocks
    # t_sub_to_blocks: list of lists of block indices
    t_sub_to_blocks = [[] for _ in range(n_t_subs)]
    
    for i, block in enumerate(all_k_blocks):
        covered = []
        for t_sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[t_sub]
            covered.append(idx)
            t_sub_to_blocks[idx].append(i)
        block_to_t_subs[i] = covered
        
    # State: which t-subsets are still uncovered
    # Using a boolean list or set. Set is good for "is it covered?" check.
    # But we need to iterate to find an uncovered one.
    # Let's use a list of booleans for fast access, and a set of uncovered indices for iteration.
    is_covered = [False] * n_t_subs
    uncovered_set = set(range(n_t_subs))
    
    selected_blocks = []
    
    while uncovered_set:
        if time.time() > deadline:
            break
            
        # Pick an arbitrary uncovered t-subset (e.g., the first one)
        # Heuristic: picking the one with the fewest covering blocks might be better (most constrained),
        # but for covering designs, random or first is often fine. 
        # Let's pick the one with the smallest index for determinism, or just pop one.
        # Actually, a better heuristic is to pick the uncovered t-subset that is covered by the fewest 
        # *potentially useful* blocks, but that's complex. 
        # Let's just pick the first uncovered one.
        target_t_sub_idx = min(uncovered_set)
        
        # Get all blocks that cover this target t-subset
        candidate_block_indices = t_sub_to_blocks[target_t_sub_idx]
        
        best_block_idx = -1
        best_gain = -1
        
        # Evaluate each candidate block
        for b_idx in candidate_block_indices:
            gain = 0
            for t_idx in block_to_t_subs[b_idx]:
                if not is_covered[t_idx]:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block_idx = b_idx
                
        # Select the best block
        if best_block_idx != -1:
            selected_blocks.append(all_k_blocks[best_block_idx])
            
            # Update coverage
            for t_idx in block_to_t_subs[best_block_idx]:
                if not is_covered[t_idx]:
                    is_covered[t_idx] = True
                    uncovered_set.discard(t_idx)
            
            # Optimization: If we have covered everything, break
            if not uncovered_set:
                break

    return selected_blocks
