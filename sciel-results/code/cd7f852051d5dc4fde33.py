import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # 1. Map t-subsets to IDs
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_id = {subset: i for i, subset in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # 2. Precompute k-subsets (blocks) and their covered t-subset IDs
    all_k_subsets = list(itertools.combinations(range(v), k))
    block_covered_ids = []
    for block in all_k_subsets:
        covered = set()
        for t_sub in itertools.combinations(block, t):
            covered.add(t_subset_to_id[t_sub])
        block_covered_ids.append(covered)
        
    # 3. Greedy with Random Restarts
    best_blocks = None
    best_count = float('inf')
    
    # We will run the greedy algorithm multiple times.
    # Each run is deterministic except for tie-breaking.
    # We can also randomize the order of blocks considered if gains are equal.
    
    while time.time() < deadline:
        uncovered = set(range(num_t_subsets))
        current_blocks = []
        
        while uncovered:
            max_gain = -1
            best_candidates = []
            
            # Find block with max gain
            # To speed up, we can iterate through all blocks
            for i, covered in enumerate(block_covered_ids):
                # Intersection size
                # If covered is a set and uncovered is a set
                gain = len(uncovered & covered)
                
                if gain > max_gain:
                    max_gain = gain
                    best_candidates = [i]
                elif gain == max_gain and gain > 0:
                    best_candidates.append(i)
            
            if not best_candidates:
                # Should not happen if we have a valid covering design space
                break
                
            # Pick a candidate
            # Random tie-breaking
            chosen_idx = random.choice(best_candidates)
            
            # Add block
            current_blocks.append(all_k_subsets[chosen_idx])
            
            # Update uncovered
            uncovered.difference_update(block_covered_ids[chosen_idx])
            
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break
            
        if len(current_blocks) < best_count:
            best_count = len(current_blocks)
            best_blocks = [list(b) for b in current_blocks]
            
            # Early termination if we hit a known lower bound or very good number?
            # Lower bound is 131. If we get 131, we can stop.
            if best_count <= 131:
                break

    if best_blocks is None:
        # Fallback: return something valid if we timed out before finishing a run
        # This shouldn't happen if the first run finishes.
        # But if the first run is incomplete, we can't return it.
        # We must ensure we return a valid covering.
        # If we haven't completed a single run, we are in trouble.
        # Given the speed estimate (~1s/run), we should finish at least one.
        # To be safe, if best_blocks is None, run a deterministic greedy without time check?
        # No, that might time out.
        # Let's assume the first run completes.
        
        # Actually, if the first run is interrupted, best_blocks is None.
        # We need a valid output.
        # We can reconstruct the last complete run? No, we didn't save it.
        # Let's just run one deterministic greedy at the start to have a fallback.
        
        pass

    # If best_blocks is still None, run a fast deterministic greedy
    if best_blocks is None:
        uncovered = set(range(num_t_subsets))
        current_blocks = []
        while uncovered:
            max_gain = -1
            chosen_idx = -1
            for i, covered in enumerate(block_covered_ids):
                gain = len(uncovered & covered)
                if gain > max_gain:
                    max_gain = gain
                    chosen_idx = i
            if chosen_idx == -1: break
            current_blocks.append(all_k_subsets[chosen_idx])
            uncovered.difference_update(block_covered_ids[chosen_idx])
        best_blocks = [list(b) for b in current_blocks]

    return best_blocks
