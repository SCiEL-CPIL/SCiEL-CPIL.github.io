import itertools
import time
import random

def construct(v, k, t):
    # Parameters for the task: v=24, k=7, t=3
    # But we should use the passed arguments.
    
    # Precompute triples
    triples = list(itertools.combinations(range(v), t))
    num_triples = len(triples)
    
    # Map triple to index
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # Precompute 3D index lookup for speed: idx_3d[a][b][c] = index of triple (a,b,c)
    # -1 if not a valid triple (i.e., a==b etc, though we ensure distinct in logic)
    idx_3d = [[[-1]*v for _ in range(v)] for _ in range(v)]
    for a in range(v):
        for b in range(v):
            for c in range(v):
                if a == b or a == c or b == c:
                    continue
                t1, t2, t3 = sorted((a, b, c))
                idx_3d[a][b][c] = triple_to_idx[(t1, t2, t3)]
        
    def get_triple_idx(a, b, c):
        return idx_3d[a][b][c]
        
    # Precompute bitmasks for pairs of points.
    # pair_masks_2d[u][v] is an integer where the m-th bit is 1 if triple m contains u and v.
    pair_masks_2d = [[0]*v for _ in range(v)]
    for u in range(v):
        for w in range(v):
            if u == w: continue
            mask = 0
            for w2 in range(v):
                if w2 == u or w2 == w: continue
                t1, t2, t3 = sorted((u, w, w2))
                idx = triple_to_idx[(t1, t2, t3)]
                mask |= (1 << idx)
            pair_masks_2d[u][w] = mask
            pair_masks_2d[w][u] = mask
            
    # Helper to get gain
    def calc_gain(x, block, uncovered_mask):
        # block is a list of points
        # We want triples (x, u, v) where u, v in block
        # This is union of pair_masks_2d[x][u] for u in block
        # Then AND with uncovered_mask and popcount
        union_mask = 0
        for u in block:
            if u == x: continue
            union_mask |= pair_masks_2d[x][u]
        # Intersect with uncovered
        common = union_mask & uncovered_mask
        # Popcount
        return bin(common).count('1')
        
    def greedy_cover(random_state):
        rng = random.Random(random_state)
        # Initialize uncovered mask: all bits 0 to num_triples-1 are 1
        # (1 << num_triples) - 1
        uncovered_mask = (1 << num_triples) - 1
        num_uncovered = num_triples
        blocks = []
        
        points = list(range(v))
        
        while num_uncovered > 0:
            # Find an uncovered triple
            if uncovered_mask == 0:
                break
                
            lowbit = uncovered_mask & -uncovered_mask
            target_idx = lowbit.bit_length() - 1
            target_triple = triples[target_idx]
            
            block = list(target_triple)
            
            # Remove this triple from uncovered
            uncovered_mask &= ~(1 << target_idx)
            num_uncovered -= 1
            
            # Add remaining k - t points
            while len(block) < k:
                best_gain = -1
                best_point = -1
                candidates = [x for x in points if x not in block]
                
                # Shuffle for diversity
                rng.shuffle(candidates)
                
                for x in candidates:
                    gain = calc_gain(x, block, uncovered_mask)
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                        # Early exit if gain is max possible?
                        # Max possible is C(len(block), 2)
                        if gain == len(block)*(len(block)-1)//2:
                            break
                            
                if best_point == -1:
                    break # Should not happen
                
                block.append(best_point)
                
                # Update uncovered mask
                # We need to cover all triples (best_point, u, v) where u, v in block (excluding best_point)
                # This is the union of pair_masks_2d[best_point][u] for u in block (u != best_point)
                cover_mask = 0
                for u in block:
                    if u == best_point: continue
                    cover_mask |= pair_masks_2d[best_point][u]
                    
                # Only cover those that were uncovered
                newly_covered = cover_mask & uncovered_mask
                # Update count
                num_uncovered -= bin(newly_covered).count('1')
                
                # Remove from uncovered
                uncovered_mask &= ~newly_covered
                
            blocks.append(tuple(sorted(block)))
            
        return blocks

    def trim_blocks(blocks):
        # Count frequency of each triple
        # Using a list of counts is faster than dict for dense access
        counts = [0] * num_triples
        
        for block in blocks:
            # Iterate all triples in the block
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[triple]
                counts[idx] += 1
                
        active_blocks = [True] * len(blocks)
        changed = True
        while changed:
            changed = False
            for i, block in enumerate(blocks):
                if not active_blocks[i]:
                    continue
                # Check if any triple in this block has count 1
                essential = False
                for triple in itertools.combinations(block, t):
                    idx = triple_to_idx[triple]
                    if counts[idx] == 1:
                        essential = True
                        break
                if not essential:
                    # Remove block
                    active_blocks[i] = False
                    changed = True
                    for triple in itertools.combinations(block, t):
                        idx = triple_to_idx[triple]
                        counts[idx] -= 1
                        
        return [blocks[i] for i in range(len(blocks)) if active_blocks[i]]

    def local_search(blocks, random_state, max_iters=100):
        # Try to remove blocks by replacing them or finding redundant ones
        # A simple local search: try to remove a block. If it breaks coverage, 
        # try to find another block that can cover the "lost" triples, or add a new block?
        # Actually, standard approach: 
        # 1. Try to remove a block. If invalid, try to swap it with a different block that covers the same essential triples but is "better" in some way?
        # 2. Or just try to find a block that is redundant (covered by others). trim_blocks does this.
        # 3. More advanced: try to replace a block with a different one that allows another block to be removed.
        
        # Given time constraints, let's implement a simple "swap" heuristic.
        # Pick a block B. Try to replace it with a different block B'.
        # If the new set of blocks is valid, keep it if it doesn't increase size (it doesn't, 1 for 1).
        # But we want to *reduce* size. 
        # So: Try to remove a block. If invalid, identify the "witness" triples that become uncovered.
        # Try to find a new block that covers these witness triples AND allows another block to be removed?
        # This is complex.
        
        # Alternative: "Greedy with Backtracking" or "Simulated Annealing".
        # Let's stick to a simpler improvement: 
        # After greedy, we have a set of blocks.
        # We can try to "shrink" blocks? No, k is fixed.
        
        # Let's implement a random restart with a slight modification:
        # Instead of just greedy, we can try to add a random block and then trim?
        # Or try to replace a random block with a random block and trim.
        
        # Let's try: 
        # 1. Copy blocks.
        # 2. Pick a random block index i.
        # 3. Generate a random block B_new.
        # 4. Replace blocks[i] with B_new.
        # 5. Trim.
        # 6. If len(new_blocks) < len(old_blocks), accept.
        # 7. Else reject.
        
        # This is essentially a random search in the space of block sets, guided by trim.
        
        rng = random.Random(random_state)
        current_blocks = list(blocks)
        current_len = len(current_blocks)
        
        for _ in range(max_iters):
            # Pick a block to replace
            i = rng.randint(0, len(current_blocks) - 1)
            
            # Generate a random block
            new_block = tuple(sorted(rng.sample(range(v), k)))
            
            # Replace
            new_blocks = list(current_blocks)
            new_blocks[i] = new_block
            
            # Trim
            trimmed = trim_blocks(new_blocks)
            
            if len(trimmed) < len(current_blocks):
                current_blocks = trimmed
                current_len = len(current_blocks)
            elif len(trimmed) == len(current_blocks):
                # Accept with some probability to escape local minima?
                # Or just keep current? 
                # If we accept neutral moves, we might drift.
                # Let's accept neutral moves with 50% probability.
                if rng.random() < 0.5:
                    current_blocks = trimmed
        
        return current_blocks

    # Main loop
    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    deadline = start_time + 110 # 10s buffer
    
    seed_counter = 0
    while time.time() < deadline:
        # Greedy
        blocks = greedy_cover(seed_counter)
        # Trim
        blocks = trim_blocks(blocks)
        
        # Local Search
        # Only do local search if we have time
        if time.time() < deadline - 10:
             blocks = local_search(blocks, seed_counter, max_iters=50)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        # Check lower bound
        if best_len == 69:
            break
            
        seed_counter += 1
        
    # Convert tuples to lists for output
    if best_blocks:
        return [list(b) for b in best_blocks]
    else:
        return []
