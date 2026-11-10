import itertools
import random
import time

def construct(v, k, t):
    # Precomputation
    # 1. Map triplets to indices
    triplets = list(itertools.combinations(range(v), t))
    triplet_index = {trip: i for i, trip in enumerate(triplets)}
    n_triplets = len(triplets)
    
    # 2. Generate all blocks and their masks
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)
    block_masks = [0] * n_blocks
    
    for j, block in enumerate(blocks):
        mask = 0
        for triple in itertools.combinations(block, t):
            idx = triplet_index[triple]
            mask |= (1 << idx)
        block_masks[j] = mask
            
    # Greedy Search with Randomization + Local Search Polish
    best_covering = None
    best_size = float('inf')
    
    # Time limit check
    start_time = time.time()
    max_time = 110  # Leave buffer for evaluation overhead
    
    def greedy_once():
        """Run one greedy pass with random tie-breaking."""
        uncovered = (1 << n_triplets) - 1
        selected = []
        
        while uncovered:
            best_gain = -1
            best_candidates = []
            
            for j in range(n_blocks):
                inter = block_masks[j] & uncovered
                if inter:
                    gain = inter.bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [j]
                    elif gain == best_gain:
                        best_candidates.append(j)
            
            if best_gain <= 0:
                break
                
            chosen_j = random.choice(best_candidates)
            selected.append(chosen_j)
            uncovered &= ~block_masks[chosen_j]
            
        return selected
    
    def polish(selected_indices, time_limit):
        """Local search: try removing redundant blocks and 1-for-1 swaps."""
        selected = list(selected_indices)
        
        # Build coverage count for each triplet
        coverage = [0] * n_triplets
        for idx in selected:
            mask = block_masks[idx]
            # Count bits set in mask
            m = mask
            while m:
                lb = m & (-m)
                coverage[lb.bit_length() - 1] += 1
                m ^= lb
        
        def try_remove():
            """Try to remove a block that doesn't uncover any triplet."""
            for i in range(len(selected)):
                idx = selected[i]
                mask = block_masks[idx]
                # Check if any triplet in this block has coverage == 1
                m = mask
                redundant = True
                while m:
                    lb = m & (-m)
                    pos = lb.bit_length() - 1
                    if coverage[pos] == 1:
                        redundant = False
                        break
                    m ^= lb
                if redundant:
                    # Remove it
                    m = block_masks[idx]
                    while m:
                        lb = m & (-m)
                        pos = lb.bit_length() - 1
                        coverage[pos] -= 1
                        m ^= lb
                    selected.pop(i)
                    return True
            return False
        
        def try_swap():
            """Try to swap a block with another that covers more uncovered triples."""
            for i in range(len(selected)):
                idx = selected[i]
                mask_i = block_masks[idx]
                # Find triples only covered by this block
                m = mask_i
                exclusive = 0
                while m:
                    lb = m & (-m)
                    pos = lb.bit_length() - 1
                    if coverage[pos] == 1:
                        exclusive |= lb
                    m ^= lb
                
                if exclusive == 0:
                    continue
                
                # Find a block that covers all exclusive triples and more
                best_gain = 0
                best_j = -1
                for j in range(n_blocks):
                    if j == idx:
                        continue
                    mask_j = block_masks[j]
                    if (exclusive & ~mask_j) != 0:
                        continue  # Doesn't cover all exclusive
                    # Gain: number of new triples covered
                    new_cover = mask_j & ~mask_i
                    gain = new_cover.bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_j = j
                
                if best_j >= 0 and best_gain > 0:
                    # Perform swap
                    # Remove old
                    m = mask_i
                    while m:
                        lb = m & (-m)
                        pos = lb.bit_length() - 1
                        coverage[pos] -= 1
                        m ^= lb
                    # Add new
                    m = block_masks[best_j]
                    while m:
                        lb = m & (-m)
                        pos = lb.bit_length() - 1
                        coverage[pos] += 1
                        m ^= lb
                    selected[i] = best_j
                    return True
            
            return False
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try removals
            if try_remove():
                improved = True
            # Try swaps
            if try_swap():
                improved = True
        
        return selected
    
    while time.time() - start_time < max_time:
        # Phase 1: Greedy construction
        selected = greedy_once()
        
        # Phase 2: Local search polish
        remaining_time = max_time - (time.time() - start_time)
        polish_time = max(0.5, min(5.0, remaining_time * 0.3))
        polished = polish(selected, time.time() + polish_time)
        
        if len(polished) < best_size:
            best_size = len(polished)
            best_covering = [blocks[i] for i in polished]
            
        # Early termination if we hit the theoretical lower bound
        if best_size <= 69:
            break

    # Format output: list of lists of ints
    if best_covering is None:
        return []
        
    return [list(b) for b in best_covering]
