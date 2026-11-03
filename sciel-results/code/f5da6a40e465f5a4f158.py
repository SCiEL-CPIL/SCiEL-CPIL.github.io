import itertools
import random
import time

def construct(v, k, t):
    # Parameters for the specific task C(24, 7, 3)
    # But the function signature is generic, so we must handle v, k, t.
    # However, the task is specific. We can hardcode optimizations for v=24, k=7, t=3
    # but must be careful if the evaluator passes different params.
    # The prompt says: "The evaluator passes this task's parameters v, k and t."
    # So we assume v=24, k=7, t=3. But we should be somewhat generic or at least robust.
    # Given the 120s limit, generic slow algorithms are bad.
    # We will specialize for v=24, k=7, t=3 but include a fallback.
    
    if v != 24 or k != 7 or t != 3:
        # Fallback to simple greedy for other parameters
        return _generic_greedy(v, k, t)
        
    # Specialized for C(24, 7, 3)
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    # Precomputation
    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    idx_to_triple = all_triples
    num_triples = len(all_triples)
    all_mask = (1 << num_triples) - 1
    
    # Precompute for each point, the list of triple indices containing it?
    # Might help in candidate generation?
    # Actually, for a block B, we need the mask of triples in B.
    # We can compute this by iterating combinations(B, 3) and looking up idx.
    
    def get_block_mask(block):
        mask = 0
        for triple in itertools.combinations(block, t):
            mask |= 1 << triple_to_idx[triple]
        return mask

    def greedy_cover(seed=0):
        random.seed(seed)
        covered_mask = 0
        blocks = []
        counts = [0] * num_triples # For deletion phase later, or just use mask for greedy
        
        while covered_mask != all_mask:
            # Find an uncovered triple
            # Instead of scanning all, pick a random one from the remaining?
            # Or just find the first one.
            # To be deterministic-ish but varied, we can shuffle the order of checking candidates.
            
            # Pick a random uncovered triple to anchor the candidate set
            # We need to find an index where bit is 0.
            # This is O(N). N=2024. Fast.
            uncovered_indices = [i for i in range(num_triples) if not (covered_mask >> i) & 1]
            if not uncovered_indices:
                break
            anchor_idx = random.choice(uncovered_indices)
            anchor_triple = idx_to_triple[anchor_idx]
            
            # Generate candidates containing anchor_triple
            # Points in anchor: a, b, c
            # Remaining points: R
            # Choose k - t points from R
            anchor_set = set(anchor_triple)
            remaining_points = [p for p in points if p not in anchor_set]
            
            best_block = None
            best_gain = -1
            best_mask = 0
            
            # Iterate over all combinations of (k-t) points from remaining
            for combo in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(anchor_triple + combo))
                # Calculate mask
                mask = get_block_mask(block)
                gain = mask & ~covered_mask
                gain_count = gain.bit_count()
                
                if gain_count > best_gain:
                    best_gain = gain_count
                    best_block = block
                    best_mask = mask
            
            if best_block is not None:
                blocks.append(best_block)
                covered_mask |= best_mask
                # Update counts if we were tracking them, but we aren't in this phase.
            
        return blocks, covered_mask

    def delete_redundant(blocks):
        # counts[i] = number of blocks covering triple i
        counts = [0] * num_triples
        for b in blocks:
            for triple in itertools.combinations(b, t):
                counts[triple_to_idx[triple]] += 1
        
        # Try to remove blocks
        # We need to check if a block is redundant: all its triples have count > 1
        # If we remove it, we decrement counts.
        # If any count becomes 0, it's not redundant.
        
        # To make this faster, we can iterate and try to remove.
        # If removal is valid, do it and restart? Or continue?
        # Restarting is safer for finding more deletions.
        
        improved = True
        while improved:
            improved = False
            for i in range(len(blocks)):
                b = blocks[i]
                # Check if redundant
                redundant = True
                for triple in itertools.combinations(b, t):
                    if counts[triple_to_idx[triple]] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove it
                    for triple in itertools.combinations(b, t):
                        counts[triple_to_idx[triple]] -= 1
                    blocks.pop(i)
                    improved = True
                    break # Restart the loop
        return blocks

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple greedy instances with different seeds
    # How many? 120s limit.
    # Greedy takes ~1-2s?
    # Let's estimate.
    # Steps ~ 80.
    # Candidates per step ~ 6000.
    # Ops per candidate ~ 35 lookups + bit ops.
    # 80 * 6000 * 35 = 16.8M ops.
    # Python does ~10-50M simple ops/sec. So ~0.5s - 1s per run.
    # We can run 20-50 restarts.
    
    num_restarts = 30
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        blocks, _ = greedy_cover(seed=i)
        # Deletion phase
        blocks = delete_redundant(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    if best_blocks is None:
        # Fallback
        blocks, _ = greedy_cover(seed=0)
        best_blocks = delete_redundant(blocks)
        
    return best_blocks

def _generic_greedy(v, k, t):
    # Simple fallback
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        block = list(min(uncovered))
        while len(block) < k:
            def gain(x):
                return sum(1 for rest in itertools.combinations(block, t - 1)
                           if tuple(sorted(rest + (x,))) in uncovered)
            block.append(max((x for x in range(v) if x not in block), key=gain))
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    return blocks
