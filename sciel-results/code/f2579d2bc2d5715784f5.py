import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    # Parameters for the task: v=24, k=7, t=3
    # But we should use the passed arguments.
    
    # Precompute triples
    triples = list(itertools.combinations(range(v), t))
    num_triples = len(triples)
    
    # Map triple to index
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # Precompute 3D index lookup for speed
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
        
    def greedy_cover(random_state):
        rng = random.Random(random_state)
        uncovered = np.ones(num_triples, dtype=np.uint8)
        num_uncovered = num_triples
        blocks = []
        
        # List of points
        points = list(range(v))
        
        while num_uncovered > 0:
            # Find first uncovered triple
            start_idx = np.argmax(uncovered)
            if uncovered[start_idx] == 0:
                break
                
            # Get the triple
            target_triple = triples[start_idx]
            block = list(target_triple)
            
            # Mark as covered
            uncovered[start_idx] = 0
            num_uncovered -= 1
            
            # Add remaining k - t points
            while len(block) < k:
                best_gain = -1
                best_point = -1
                candidates = [x for x in points if x not in block]
                
                # Shuffle candidates for random tie-breaking
                rng.shuffle(candidates)
                
                for x in candidates:
                    gain = 0
                    # Count triples (x, u, v) where u, v in block
                    for i in range(len(block)):
                        for j in range(i+1, len(block)):
                            p1 = block[i]
                            p2 = block[j]
                            idx = get_triple_idx(x, p1, p2)
                            if uncovered[idx]:
                                gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                    
                if best_point == -1:
                    break
                
                block.append(best_point)
                
                # Update uncovered status for all triples covered by the new point
                for i in range(len(block)):
                    if block[i] == best_point:
                        continue
                    for j in range(i+1, len(block)):
                        if block[j] == best_point:
                            continue
                        p1 = block[i]
                        p2 = block[j]
                        idx = get_triple_idx(best_point, p1, p2)
                        if uncovered[idx]:
                            uncovered[idx] = 0
                            num_uncovered -= 1
                            
            blocks.append(tuple(sorted(block)))
            
        return blocks

    def trim_blocks(blocks):
        # Count frequency of each triple
        counts = [0] * num_triples
        
        for block in blocks:
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
                essential = False
                for triple in itertools.combinations(block, t):
                    idx = triple_to_idx[triple]
                    if counts[idx] == 1:
                        essential = True
                        break
                if not essential:
                    active_blocks[i] = False
                    changed = True
                    for triple in itertools.combinations(block, t):
                        idx = triple_to_idx[triple]
                        counts[idx] -= 1
                        
        return [blocks[i] for i in range(len(blocks)) if active_blocks[i]]

    def simulated_annealing_trim(blocks, random_state, time_limit=3.0):
        """
        Simulated annealing to minimize block count.
        Moves: 
        1. Remove a non-essential block
        2. Swap a block with another that maintains coverage
        3. Add a new block and remove one or more existing blocks
        """
        rng = random.Random(random_state)
        blocks = list(blocks)
        n = len(blocks)
        start_time = time.time()
        
        # Precompute triple counts
        counts = [0] * num_triples
        for block in blocks:
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[triple]
                counts[idx] += 1
        
        # Precompute all possible blocks
        all_blocks = list(itertools.combinations(range(v), k))
        
        # Temperature schedule
        T0 = 10.0
        T_end = 0.1
        cooling_rate = 0.98
        
        T = T0
        best_count = n
        best_blocks = list(blocks)
        
        while time.time() - start_time < time_limit and T > T_end:
            T *= cooling_rate
            
            # Try different moves
            move_type = rng.random()
            
            if move_type < 0.4:
                # Try to remove a block
                removable = [i for i in range(n) if all(
                    counts[triple_to_idx[triple]] > 1 
                    for triple in itertools.combinations(blocks[i], t)
                )]
                if not removable:
                    continue
                i = rng.choice(removable)
                old_block = blocks[i]
                # Remove it
                for triple in itertools.combinations(old_block, t):
                    idx = triple_to_idx[triple]
                    counts[idx] -= 1
                blocks.pop(i)
                n -= 1
                if n < best_count:
                    best_count = n
                    best_blocks = list(blocks)
            elif move_type < 0.7:
                # Try to swap a block with another
                if n < 2:
                    continue
                i = rng.randrange(n)
                old_block = blocks[i]
                # Find a replacement that covers all essential triples of old_block
                essential = [triple_to_idx[triple] for triple in itertools.combinations(old_block, t) 
                           if counts[triple_to_idx[triple]] == 1]
                if not essential:
                    # Can just remove
                    for triple in itertools.combinations(old_block, t):
                        idx = triple_to_idx[triple]
                        counts[idx] -= 1
                    blocks.pop(i)
                    n -= 1
                    if n < best_count:
                        best_count = n
                        best_blocks = list(blocks)
                    continue
                
                # Sample candidates
                candidates = rng.sample(all_blocks, min(100, len(all_blocks)))
                found = False
                for cand in candidates:
                    cand_set = set(cand)
                    if all(x in cand_set for idx in essential for x in triples[idx]):
                        # Accept swap with probability based on cost
                        # Cost: number of new uncovered triples this would create
                        new_uncovered = 0
                        for triple in itertools.combinations(old_block, t):
                            idx = triple_to_idx[triple]
                            if idx not in essential and counts[idx] == 1:
                                # This triple would become uncovered if not in cand
                                if not all(x in cand_set for x in triple):
                                    new_uncovered += 1
                        
                        # Compute benefit: how many other blocks could be removed
                        # Simplified: accept if it doesn't increase uncovered count
                        if new_uncovered == 0 or rng.random() < np.exp(-new_uncovered / T):
                            # Perform swap
                            for triple in itertools.combinations(old_block, t):
                                idx = triple_to_idx[triple]
                                counts[idx] -= 1
                            for triple in itertools.combinations(cand, t):
                                idx = triple_to_idx[triple]
                                counts[idx] += 1
                            blocks[i] = cand
                            found = True
                            break
                
                if not found:
                    # Try to add a new block to cover essentials, then remove old
                    # This is more complex; skip for now
                    pass
            else:
                # Try to add a block and remove one or two existing blocks
                if n < 1:
                    continue
                
                # Pick a candidate new block
                cand = rng.choice(all_blocks)
                cand_set = set(cand)
                
                # Find blocks that are mostly covered by cand
                removable_candidates = []
                for i in range(n):
                    block_i = blocks[i]
                    # Count how many triples of block_i are covered by cand
                    covered = sum(1 for triple in itertools.combinations(block_i, t) 
                                 if all(x in cand_set for x in triple))
                    total = len(list(itertools.combinations(block_i, t)))
                    if covered >= total * 0.7:
                        removable_candidates.append(i)
                
                if not removable_candidates:
                    continue
                
                # Try removing one or two of them
                to_remove = [rng.choice(removable_candidates)]
                if len(removable_candidates) > 1 and rng.random() < 0.3:
                    other = [x for x in removable_candidates if x != to_remove[0]]
                    if other:
                        to_remove.append(rng.choice(other))
                
                # Check if removing these blocks leaves all triples covered
                # (including those only covered by the removed blocks)
                temp_counts = list(counts)
                for i in to_remove:
                    for triple in itertools.combinations(blocks[i], t):
                        idx = triple_to_idx[triple]
                        temp_counts[idx] -= 1
                
                # Check no triple becomes uncovered
                if all(temp_counts[idx] >= 0 for idx in range(num_triples)):
                    # Check all triples still covered
                    if all(temp_counts[idx] >= 1 or counts[idx] >= 1 for idx in range(num_triples)):
                        # Accept with probability
                        delta = len(to_remove) - 1  # net change in block count
                        if delta < 0 or rng.random() < np.exp(-max(0, -delta) / T):
                            # Perform the move
                            for i in to_remove:
                                for triple in itertools.combinations(blocks[i], t):
                                    idx = triple_to_idx[triple]
                                    counts[idx] -= 1
                            for triple in itertools.combinations(cand, t):
                                idx = triple_to_idx[triple]
                                counts[idx] += 1
                            # Remove blocks in reverse order to keep indices valid
                            for i in sorted(to_remove, reverse=True):
                                blocks.pop(i)
                            blocks.append(cand)
                            n = len(blocks)
                            if n < best_count:
                                best_count = n
                                best_blocks = list(blocks)
        
        return best_blocks

    def local_search_trim(blocks, random_state, time_limit=2.0):
        """
        Attempt to further reduce blocks by random reordering and deletion.
        """
        rng = random.Random(random_state)
        blocks = list(blocks)
        start_time = time.time()
        
        # Precompute triple counts for current blocks
        counts = [0] * num_triples
        for block in blocks:
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[triple]
                counts[idx] += 1
                
        active = [True] * len(blocks)
        
        while time.time() - start_time < time_limit:
            active_indices = [i for i in range(len(blocks)) if active[i]]
            rng.shuffle(active_indices)
            
            found_removal = False
            for i in active_indices:
                if not active[i]:
                    continue
                essential = False
                for triple in itertools.combinations(blocks[i], t):
                    idx = triple_to_idx[triple]
                    if counts[idx] == 1:
                        essential = True
                        break
                if not essential:
                    active[i] = False
                    found_removal = True
                    for triple in itertools.combinations(blocks[i], t):
                        idx = triple_to_idx[triple]
                        counts[idx] -= 1
            if not found_removal:
                break
                
        return [blocks[i] for i in range(len(blocks)) if active[i]]

    def swap_local_search(blocks, random_state, time_limit=2.0):
        """
        Try to replace a block with another block that covers the same essential
        triples while potentially allowing removal of other blocks.
        """
        rng = random.Random(random_state)
        blocks = list(blocks)
        n_blocks = len(blocks)
        start_time = time.time()
        
        # Precompute triple counts
        counts = [0] * num_triples
        for block in blocks:
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[triple]
                counts[idx] += 1
        
        active = [True] * n_blocks
        
        # Precompute all possible blocks (7-subsets of v)
        all_blocks = list(itertools.combinations(range(v), k))
        
        while time.time() - start_time < time_limit:
            # Try to find a removable block
            active_indices = [i for i in range(n_blocks) if active[i]]
            if not active_indices:
                break
            
            rng.shuffle(active_indices)
            made_change = False
            
            for i in active_indices:
                if not active[i]:
                    continue
                
                # Check if block i is removable
                essential_triples = []
                for triple in itertools.combinations(blocks[i], t):
                    idx = triple_to_idx[triple]
                    if counts[idx] == 1:
                        essential_triples.append(idx)
                
                if not essential_triples:
                    # Block is removable
                    active[i] = False
                    made_change = True
                    for triple in itertools.combinations(blocks[i], t):
                        idx = triple_to_idx[triple]
                        counts[idx] -= 1
                    continue
                
                # Block has essential triples; try to find a replacement block
                # that covers all essential triples and has minimal overlap with others
                best_replacement = None
                best_score = float('inf')
                
                # Sample some candidate blocks
                num_candidates = min(50, len(all_blocks))
                candidates = rng.sample(all_blocks, num_candidates)
                
                for cand in candidates:
                    cand_set = set(cand)
                    # Check if cand covers all essential triples
                    covers_essential = True
                    for idx in essential_triples:
                        triple = triples[idx]
                        if not all(x in cand_set for x in triple):
                            covers_essential = False
                            break
                    
                    if not covers_essential:
                        continue
                    
                    # Score: number of new triples covered (lower is better for replacement)
                    new_triples = 0
                    for triple in itertools.combinations(cand, t):
                        idx = triple_to_idx[triple]
                        if counts[idx] == 0:
                            new_triples += 1
                    
                    # Prefer blocks with fewer new triples (less redundant)
                    if new_triples < best_score:
                        best_score = new_triples
                        best_replacement = cand
                
                if best_replacement is not None:
                    # Replace block i with best_replacement
                    old_block = blocks[i]
                    blocks[i] = best_replacement
                    
                    # Update counts
                    for triple in itertools.combinations(old_block, t):
                        idx = triple_to_idx[triple]
                        counts[idx] -= 1
                    for triple in itertools.combinations(best_replacement, t):
                        idx = triple_to_idx[triple]
                        counts[idx] += 1
                    
                    made_change = True
                    # After replacement, try to remove more blocks
                    break
            
            if not made_change:
                break
        
        return [blocks[i] for i in range(n_blocks) if active[i]]

    # Main loop
    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    deadline = start_time + 140  # 10s buffer for safety
    
    seed_counter = 0
    while time.time() < deadline:
        blocks = greedy_cover(seed_counter)
        blocks = trim_blocks(blocks)
        blocks = local_search_trim(blocks, seed_counter, time_limit=0.5)
        blocks = swap_local_search(blocks, seed_counter, time_limit=1.0)
        blocks = simulated_annealing_trim(blocks, seed_counter, time_limit=2.0)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)
            
        if best_len == 69:
            break
            
        seed_counter += 1
    
    if best_blocks:
        return [list(b) for b in best_blocks]
    else:
        return []
