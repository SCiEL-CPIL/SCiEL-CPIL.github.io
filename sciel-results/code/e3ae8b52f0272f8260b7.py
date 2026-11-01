import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    timeout = 115.0 # Leave 5s margin for return/eval
    
    # 1. Precomputation
    # Generate all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Generate all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map t-subsets to indices
    t_subset_to_idx = {subset: i for i, subset in enumerate(all_t_subsets)}
    
    # For each block, store the indices of t-subsets it covers
    block_t_indices = []
    for block in all_blocks:
        indices = [t_subset_to_idx[sub] for sub in itertools.combinations(block, t)]
        block_t_indices.append(indices)
        
    best_blocks = None
    best_count = float('inf')
    
    # 2. Main Search Loop
    while time.time() - start_time < timeout:
        # Randomized Greedy Construction
        uncovered = set(range(num_t_subsets))
        used_blocks = []
        
        # To break ties randomly, we can shuffle the list of block indices 
        # or use a random threshold. 
        # A simple way: when finding the max gain, if there are ties, pick one randomly.
        # Or, add a small random noise to the gain.
        
        while uncovered:
            best_gain = -1
            best_block_idx = -1
            candidates = []
            
            # Iterate through all blocks to find the one covering most uncovered t-subsets
            for i in range(num_blocks):
                # Calculate gain
                # Optimization: if len(uncovered) is small, this is fast.
                # We can skip if max possible gain (len of block_t_indices[i]) <= best_gain
                if len(block_t_indices[i]) <= best_gain:
                    continue
                    
                gain = 0
                # Count how many of this block's t-subsets are in uncovered
                # Using a set intersection is efficient
                gain = len(set(block_t_indices[i]) & uncovered)
                
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
                    candidates = [i]
                elif gain == best_gain and gain > 0:
                    candidates.append(i)
            
            if best_gain == 0:
                # This should not happen if uncovered is not empty and we have all blocks
                # But if it does, we break to avoid infinite loop
                break
                
            # Randomly pick from candidates to diversify
            selected_idx = random.choice(candidates)
            used_blocks.append(selected_idx)
            
            # Update uncovered
            uncovered.difference_update(block_t_indices[selected_idx])
            
            # Check time
            if time.time() - start_time > timeout - 5:
                break
                
        if time.time() - start_time > timeout - 5:
            break
            
        # Pruning Phase
        # Count coverage for each t-subset
        coverage_counts = np.zeros(num_t_subsets, dtype=int)
        for b_idx in used_blocks:
            for t_idx in block_t_indices[b_idx]:
                coverage_counts[t_idx] += 1
                
        # Try to remove redundant blocks
        # A block is redundant if all its t-subsets have coverage count >= 2
        # We iterate through used_blocks and remove if possible
        # We need to update coverage_counts as we remove
        # To be safe and fast, we do a single pass or a few passes
        # Single pass:
        new_used_blocks = []
        for b_idx in used_blocks:
            # Check if all t-subsets in this block are covered by others
            # i.e., coverage_counts[t_idx] >= 2 for all t_idx in block_t_indices[b_idx]
            is_redundant = True
            for t_idx in block_t_indices[b_idx]:
                if coverage_counts[t_idx] < 2:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove it: decrement coverage counts
                for t_idx in block_t_indices[b_idx]:
                    coverage_counts[t_idx] -= 1
            else:
                new_used_blocks.append(b_idx)
                
        used_blocks = new_used_blocks
        
        # If we found a better solution
        if len(used_blocks) < best_count:
            best_count = len(used_blocks)
            best_blocks = [all_blocks[i] for i in used_blocks]
            
    if best_blocks is None:
        # Fallback: return a valid (though potentially large) covering
        # If we ran out of time before completing even one, we need a valid set.
        # The greedy loop above might have been interrupted.
        # Let's just do one fast, non-randomized greedy to ensure validity.
        uncovered = set(range(num_t_subsets))
        used_blocks = []
        while uncovered:
            best_gain = -1
            best_block_idx = -1
            for i in range(num_blocks):
                gain = len(set(block_t_indices[i]) & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
            used_blocks.append(best_block_idx)
            uncovered.difference_update(block_t_indices[best_block_idx])
        best_blocks = [all_blocks[i] for i in used_blocks]
            
    return best_blocks
