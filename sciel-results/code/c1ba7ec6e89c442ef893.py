import time
import random
import itertools

def construct(v, k, t):
    deadline = time.time() + 138.0  # safety margin below 150s kill
    start_time = time.time()

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    
    # For fast lookup: map each t-subset to an index
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute for each block (k-subset) the list of t-subset indices it covers
    # This is expensive for all C(24,7), so we'll compute on the fly or cache smartly.
    # Actually, for v=24, k=7, C(24,7) is 346104. That's manageable to precompute if needed,
    # but let's see if we can avoid it.
    # A block covers C(k,t) = C(7,3) = 35 triples.
    
    # Helper to get the list of covered triple indices for a given block (tuple of sorted ints)
    # We'll use a simple cache or just compute. Computing C(7,3)=35 lookups is fast.
    
    def get_covered_indices(block_tuple):
        indices = []
        for sub in itertools.combinations(block_tuple, t):
            indices.append(subset_to_idx[sub])
        return indices

    # --- Strategy: Batched Greedy Construction ---
    def batched_greedy(seed, batch_size):
        rng = random.Random(seed)
        uncovered_mask = [1] * n_t_subsets  # 1 if uncovered, 0 if covered
        covered_count = 0
        blocks = []
        
        while covered_count < n_t_subsets and time.time() < deadline:
            # Select a batch of blocks
            current_batch = []
            used_points = set()
            
            for _ in range(batch_size):
                if time.time() >= deadline:
                    break
                if len(used_points) >= v:
                    break
                
                # Start with a random point not used
                available = [x for x in range(v) if x not in used_points]
                if not available:
                    break
                start_point = rng.choice(available)
                block = [start_point]
                used_points.add(start_point)
                
                # Greedily add points to maximize new coverage
                for _ in range(k - 1):
                    best_gain = -1
                    best_point = -1
                    candidates = [x for x in range(v) if x not in used_points]
                    
                    # Current block's (t-1) subsets
                    if len(block) >= t - 1:
                        block_subsets = list(itertools.combinations(block, t - 1))
                    else:
                        block_subsets = []
                        
                    for x in candidates:
                        # Calculate gain: number of NEW triples covered
                        # A new triple is one that contains x and (t-1) points from block
                        # AND is currently uncovered.
                        gain = 0
                        for sub in block_subsets:
                            new_triple = tuple(sorted(sub + (x,)))
                            idx = subset_to_idx[new_triple]
                            if uncovered_mask[idx]:
                                gain += 1
                            
                        # Tie-breaking: prefer points that are "central" or random?
                        # Let's add a small random factor for diversity
                        gain += rng.random() * 0.001
                        
                        if gain > best_gain:
                            best_gain = gain
                            best_point = x
                            
                    if best_point == -1:
                        # No candidates or no gain? Should not happen if v > k
                        # If gain is 0, we still need to fill the block.
                        # Pick a random candidate.
                        if candidates:
                            best_point = rng.choice(candidates)
                        else:
                            break
                            
                    block.append(best_point)
                    used_points.add(best_point)
                
                if len(block) == k:
                    block.sort()
                    block_tuple = tuple(block)
                    current_batch.append(block_tuple)
                    # Update coverage mask
                    for idx in get_covered_indices(block_tuple):
                        if uncovered_mask[idx]:
                            uncovered_mask[idx] = 0
                            covered_count += 1
            
            # Add the whole batch to the list of blocks
            blocks.extend(current_batch)
            
            # If we ran out of time, break
            if time.time() >= deadline:
                break
        
        return blocks

    # --- Strategy: Local Search Refinement ---
    def refine(blocks, seed, max_iterations=5000):
        rng = random.Random(seed)
        current_blocks = list(blocks)
        
        # Precompute coverage for current blocks
        # coverage[i] = number of blocks covering triple i
        coverage = [0] * n_t_subsets
        block_covers = [] # list of lists of indices covered by each block
        
        for b in current_blocks:
            idxs = get_covered_indices(b)
            block_covers.append(idxs)
            for idx in idxs:
                coverage[idx] += 1
        
        # Check if valid
        if any(c == 0 for c in coverage):
            return None # Invalid starting point
            
        best_count = len(current_blocks)
        
        for _ in range(max_iterations):
            if time.time() >= deadline:
                break
            
            # Random move: try to remove a redundant block
            # A block is redundant if removing it doesn't uncover any triple (i.e., all its triples are covered by others)
            # Or try to replace a block with a smaller one? No, k is fixed.
            # Or try to swap a block with another that covers more unique triples?
            
            # Simplest effective move: Remove a redundant block.
            # Identify candidates for removal: blocks where every triple they cover has coverage > 1.
            # To speed up, we can check a random block.
            
            if current_blocks:
                idx = rng.randrange(len(current_blocks))
                b = current_blocks[idx]
                idxs = block_covers[idx]
                
                # Check if redundant
                redundant = True
                for i in idxs:
                    if coverage[i] == 1:
                        redundant = False
                        break
                        
                if redundant:
                    # Remove it
                    for i in idxs:
                        coverage[i] -= 1
                    current_blocks.pop(idx)
                    block_covers.pop(idx)
                    best_count = len(current_blocks)
                    # Loop continues to try more removals
                else:
                    # Try a "swap": replace this block with a random new block that might cover the unique triples better?
                    # This is complex. Let's stick to removal first.
                    # If no removals found in a while, we are stuck.
                    pass
            else:
                break
                
            # Early termination if we hit a good number?
            # The theoretical lower bound is 69. If we get to 76, that's good.
            # If we get to 70, that's amazing.
            
        return current_blocks

    # --- Main Execution ---
    best_blocks = None
    best_count = float('inf')
    
    # 1. Try a few batched greedy variants
    for seed in range(20):
        if time.time() >= deadline - 10: # Leave time for refinement
            break
        for batch_size in [1, 2, 3]:
            blocks = batched_greedy(seed * 100 + batch_size, batch_size)
            if not blocks:
                continue
            
            # Quick validity check
            covered = set()
            for b in blocks:
                for sub in itertools.combinations(b, t):
                    covered.add(sub)
            if len(covered) == n_t_subsets:
                if len(blocks) < best_count:
                    best_blocks = blocks
                    best_count = len(blocks)
                    
    # 2. Refine the best found
    if best_blocks is not None:
        # Try refining with a few different seeds
        for seed in range(5):
            if time.time() >= deadline - 2:
                break
            refined = refine(best_blocks, seed)
            if refined is not None and len(refined) < best_count:
                best_blocks = refined
                best_count = len(refined)
                
    # 3. Fallback if nothing found (should not happen)
    if best_blocks is None:
        # Simple greedy fallback
        uncovered = set(all_t_subsets)
        best_blocks = []
        while uncovered and time.time() < deadline:
            start = min(uncovered)
            block = list(start)
            while len(block) < k:
                best_gain = -1
                best_point = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for sub in itertools.combinations(block, t-1):
                        if tuple(sorted(sub + (x,))) in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point == -1:
                    break
                block.append(best_point)
            block.sort()
            best_blocks.append(block)
            for sub in itertools.combinations(block, t):
                uncovered.discard(sub)
                
    return best_blocks
