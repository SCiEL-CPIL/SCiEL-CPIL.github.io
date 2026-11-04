import itertools
import random
import time

def construct(v, k, t):
    # Pre-compute all t-subsets as bitmasks
    all_points = range(v)
    t_subsets = list(itertools.combinations(all_points, t))
    
    # Map bitmask to index for fast counting
    def to_mask(comb):
        mask = 0
        for p in comb:
            mask |= (1 << p)
        return mask

    subset_masks = [to_mask(c) for c in t_subsets]
    mask_to_idx = {m: i for i, m in enumerate(subset_masks)}
    num_t_subsets = len(subset_masks)
    
    # Pre-compute for each k-subset, the list of t-subset indices it covers
    # This is expensive to pre-compute all, but we can generate on the fly or pre-compute for all k-subsets?
    # C(19,8) = 75582. Each has C(8,5)=56 t-subsets.
    # 75582 * 56 ~ 4.2M entries.
    # Storing list of lists might be memory heavy but okay.
    # However, we don't need all k-subsets, only the ones we generate.
    # Generating t-subsets for a specific k-subset is fast.
    
    def get_t_indices(k_subset):
        # k_subset is a tuple/list of points
        mask_k = 0
        for p in k_subset:
            mask_k |= (1 << p)
        # We need to find all t-subsets of k_subset
        # We can iterate combinations of k_subset
        inds = []
        for t_comb in itertools.combinations(k_subset, t):
            m = 0
            for p in t_comb:
                m |= (1 << p)
            inds.append(mask_to_idx[m])
        return inds

    def greedy_construct(seed):
        random.seed(seed)
        # uncovered is a set of t-subset indices
        uncovered = set(range(num_t_subsets))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered t-subset to start the block
            start_idx = random.choice(list(uncovered))
            start_mask = subset_masks[start_idx]
            
            # Convert mask to list of points
            # We need the actual points to build the block
            # We can store the points for each t-subset index
            # t_subsets[start_idx] gives the points
            
            block_points = list(t_subsets[start_idx])
            
            while len(block_points) < k:
                # Calculate gain for each candidate point
                best_gain = -1
                best_points = []
                
                # Current block mask
                # We just check if adding x covers new t-subsets
                for x in all_points:
                    if x in block_points:
                        continue
                    
                    # Potential new t-subsets are those containing x and t-1 points from block_points
                    # Number of such subsets is C(len(block_points), t-1)
                    gain = 0
                    # Iterate t-1 combinations of block_points
                    for rest in itertools.combinations(block_points, t - 1):
                        # Form the t-subset: rest + (x,)
                        # Create mask
                        m = 1 << x
                        for p in rest:
                            m |= (1 << p)
                        idx = mask_to_idx.get(m)
                        if idx is not None and idx in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_points = [x]
                    elif gain == best_gain:
                        best_points.append(x)
                
                # Pick randomly from best_points
                next_point = random.choice(best_points)
                block_points.append(next_point)
            
            # Sort block for consistency (optional, but good for distinctness check if we used a set)
            block_points.sort()
            blocks.append(block_points)
            
            # Update uncovered
            # Get all t-subsets of this block
            block_t_indices = get_t_indices(block_points)
            for idx in block_t_indices:
                if idx in uncovered:
                    uncovered.remove(idx)
                    
        return blocks

    def deletion_pass(blocks):
        # Count coverage
        counts = [0] * num_t_subsets
        for b in blocks:
            indices = get_t_indices(b)
            for idx in indices:
                counts[idx] += 1
        
        # Remove redundant blocks
        # A block is redundant if all its t-subsets have count >= 2
        # We iterate until no more removals
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b in blocks:
                indices = get_t_indices(b)
                redundant = True
                for idx in indices:
                    if counts[idx] < 2:
                        redundant = False
                        break
                
                if redundant:
                    # Remove it
                    for idx in indices:
                        counts[idx] -= 1
                    changed = True
                else:
                    new_blocks.append(b)
            blocks = new_blocks
            
        return blocks

    start_time = time.time()
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple restarts
    seed = 42
    while time.time() - start_time < 140:
        blocks = greedy_construct(seed)
        blocks = deletion_pass(blocks)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
        
        seed += 1
        
        # Optional: If we get very good, stop early?
        # Theoretical lower bound is 264.
        # If we reach 264, we can stop.
        if best_count <= 264:
            break

    return best_blocks
