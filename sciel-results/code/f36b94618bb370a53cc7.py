import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115
    
    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    
    # Precompute which blocks cover which t-subsets
    # For efficiency, we'll use a different approach
    # Precompute all possible blocks (k-subsets)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, compute the set of t-subsets it covers
    block_covers = []
    for block in all_blocks:
        covers = set()
        for ts in itertools.combinations(block, t):
            covers.add(ts)
        block_covers.append(covers)
    
    # Initialize uncovered count for each t-subset
    uncovered = [0] * n_t
    for ts in all_t_subs:
        uncovered[all_t_subs.index(ts)] = 1  # This is O(n^2), let's use a dict instead
    
    # Better: use a dictionary mapping t-subset -> index
    ts_to_idx = {ts: i for i, ts in enumerate(all_t_subs)}
    
    # For each block, store the list of t-subset indices it covers
    block_cover_indices = []
    for covers in block_covers:
        indices = [ts_to_idx[ts] for ts in covers]
        block_cover_indices.append(indices)
    
    # Track which t-subsets are still uncovered
    is_uncovered = [True] * n_t
    
    # Greedy set cover approach with multiple restarts
    best_solution = None
    best_count = float('inf')
    
    # Number of greedy restarts to try
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        if restart == 0:
            # First restart: standard greedy
            rng = None
        else:
            # Subsequent restarts: randomized greedy
            rng = random.Random(restart * 7919)
        
        # Track remaining uncovered count for each t-subset
        rem = [1] * n_t
        
        # For each block, count how many currently-uncovered t-subsets it covers
        # We'll update this as we add blocks
        
        selected_blocks = []
        
        while True:
            # Find the number of remaining uncovered t-subsets
            remaining = sum(rem)
            if remaining == 0:
                break
            
            if time.time() > deadline - 2:
                # Out of time, break and use partial solution
                break
            
            # Find the best block (or a randomly selected good block for diversity)
            best_block = None
            best_gain = -1
            candidates = []
            
            for bi in range(n_blocks):
                gain = 0
                for idx in block_cover_indices[bi]:
                    if rem[idx] > 0:
                        gain += 1
                
                if gain > 0:
                    candidates.append((gain, bi))
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = bi
            
            if best_block is None or best_gain == 0:
                # No block can cover any remaining t-subset (shouldn't happen)
                break
            
            # Select a block
            if rng is None or best_gain <= 2:
                # Use the best block
                chosen_block = best_block
            else:
                # For diversity, sometimes pick from top candidates
                if len(candidates) > 0:
                    candidates.sort(key=lambda x: -x[0])
                    top_k = min(10, len(candidates))
                    # Pick randomly from top candidates, biased towards better
                    weights = [c[0] ** 2 for c in candidates[:top_k]]
                    total_w = sum(weights)
                    r = rng.random() * total_w
                    cum = 0
                    chosen_idx = 0
                    for i, w in enumerate(weights):
                        cum += w
                        if r <= cum:
                            chosen_idx = i
                            break
                    chosen_block = candidates[chosen_idx][1]
                else:
                    chosen_block = best_block
            
            selected_blocks.append(chosen_block)
            
            # Update remaining counts
            for idx in block_cover_indices[chosen_block]:
                if rem[idx] > 0:
                    rem[idx] -= 1
        
        if len(selected_blocks) < best_count:
            best_count = len(selected_blocks)
            best_solution = selected_blocks.copy()
        
        # Check if we have a valid solution (all t-subsets covered)
        if sum(rem) == 0:
            # Valid solution found
            break
    
    # Convert selected block indices to actual blocks
    if best_solution is None:
        # Fallback: use a simple greedy
        uncovered_set = set(all_t_subs)
        blocks = []
        while uncovered_set and time.time() < deadline - 1:
            # Find the first uncovered t-subset
            target = min(uncovered_set)
            # Find the block that covers target and maximizes new coverage
            best_b = None
            best_cov = -1
            for bi in range(n_blocks):
                if target in block_covers[bi]:
                    new_cov = len(block_covers[bi] & uncovered_set)
                    if new_cov > best_cov:
                        best_cov = new_cov
                        best_b = bi
            if best_b is None:
                break
            blocks.append(best_b)
            uncovered_set -= block_covers[best_b]
        
        best_solution = blocks
        best_count = len(blocks)
    
    # Return the blocks as lists
    result = [list(all_blocks[bi]) for bi in best_solution]
    return result
