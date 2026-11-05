import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = 120.0
    
    # 1. Pre-computation
    # Generate all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(t_subsets)
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    
    # Generate all k-subsets and their coverage bitmasks
    k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(k_subsets)
    
    # Pre-compute bitmask for each k-subset
    # mask[i] is an integer where bit j is 1 if t_subsets[j] is a subset of k_subsets[i]
    masks = [0] * num_k_subsets
    for i, k_sub in enumerate(k_subsets):
        m = 0
        for t_sub in itertools.combinations(k_sub, t):
            m |= 1 << t_subset_index[t_sub]
        masks[i] = m

    # 2. Greedy Set Cover
    # uncovered is a bitmask of all t-subsets
    uncovered = (1 << num_t_subsets) - 1
    selected_indices = []
    
    # We can optimize the search for the best block by not re-scanning all 
    # if we track which are still useful, but with 75k blocks, full scan is fine.
    
    while uncovered:
        if time.time() > deadline - 10:
            break
            
        best_idx = -1
        best_gain = 0
        
        # Find block with max overlap with uncovered
        # Optimization: If we have many blocks, this loop is the bottleneck.
        # But 75k * ~400 iterations = 30M operations. In Python, this might be slow.
        # Let's try to speed it up.
        
        # Alternative: Group by gain? No, just iterate.
        for i in range(num_k_subsets):
            gain = (masks[i] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        if best_gain == 0:
            break
            
        selected_indices.append(best_idx)
        uncovered &= ~masks[best_idx]

    # 3. Reduction Pass
    # We have selected_indices. Let's convert to list of blocks.
    # We want to remove redundant blocks.
    # Strategy: Maintain coverage counts for each t-subset.
    
    # Initialize counts
    counts = [0] * num_t_subsets
    
    # Add all selected blocks
    for idx in selected_indices:
        m = masks[idx]
        # Iterate through set bits in m
        while m:
            # Get index of lowest set bit
            lsb = m & -m
            idx_t = lsb.bit_length() - 1
            counts[idx_t] += 1
            m ^= lsb
            
    # Now try to remove blocks
    # We iterate through the list of selected indices.
    # If we remove one, we must update counts.
    # We can do a single pass removal, or repeat until stable.
    # Single pass is often enough, but repeating is safer.
    
    # To make removal efficient, we need to know which t-subsets are in each block.
    # We can re-derive them from the mask or pre-compute list of t-subset indices per k-subset.
    
    # Pre-compute list of t-subset indices for each k-subset for fast removal
    k_sub_t_indices = []
    for i in range(num_k_subsets):
        m = masks[i]
        indices = []
        while m:
            lsb = m & -m
            idx_t = lsb.bit_length() - 1
            indices.append(idx_t)
            m ^= lsb
        k_sub_t_indices.append(indices)

    # Reduction
    # We keep a list of active block indices
    active_blocks = list(selected_indices)
    
    # We will iterate through active_blocks and try to remove them.
    # If removed, we don't re-check it.
    # This is a "first-fit" style reduction.
    
    # To be more thorough, we can repeat the reduction pass multiple times.
    # Given time constraints, 1-2 passes should be sufficient.
    
    for _ in range(2):
        if time.time() > deadline - 5:
            break
            
        new_active_blocks = []
        for idx in active_blocks:
            # Check if this block is redundant
            # It is redundant if for all t-subsets in it, counts[t] > 1
            redundant = True
            for t_idx in k_sub_t_indices[idx]:
                if counts[t_idx] <= 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove block: decrement counts
                for t_idx in k_sub_t_indices[idx]:
                    counts[t_idx] -= 1
            else:
                new_active_blocks.append(idx)
        
        active_blocks = new_active_blocks

    # 4. Construct Result
    result = [list(k_subsets[idx]) for idx in active_blocks]
    
    return result
