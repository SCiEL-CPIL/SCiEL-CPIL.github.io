import itertools
import time
import random
import sys
import numpy as np
from collections import defaultdict

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

    # Precompute pair_to_thirds: for each pair (a,b), a list of (c, triple_idx)
    # This is useful for local search and potentially faster gain calculation if we
    # can vectorize the "is uncovered" check.
    pair_to_thirds = {}
    for a in range(v):
        for b in range(a+1, v):
            lst = []
            for c in range(v):
                if c != a and c != b:
                    lst.append((c, get_triple_idx(a, b, c)))
            pair_to_thirds[(a, b)] = lst

    def greedy_cover(random_state):
        rng = random.Random(random_state)
        uncovered = np.ones(num_triples, dtype=np.uint8)
        num_uncovered = num_triples
        blocks = []
        
        # List of points
        points = list(range(v))
        
        while num_uncovered > 0:
            # Find first uncovered triple
            # Using argmax on uint8 array is fast
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
                
                # Vectorized gain calculation
                # For each candidate x, gain is sum of uncovered[x, u, v] for u,v in block
                # We can build a matrix of indices for all (x, u, v) where x in candidates, u,v in block
                # But block size is small (<=6), so pairs are <=15. Candidates <=21.
                # Total checks <= 315. NumPy overhead might be high for such small numbers,
                # but let's try to batch it.
                
                # Alternative: Pre-compute the indices for the current block's pairs
                block_pairs = [(block[i], block[j]) for i in range(len(block)) for j in range(i+1, len(block))]
                
                # For each candidate, calculate gain
                # We can map candidates to indices and use advanced indexing if we build the array
                # But simple loop is probably faster for < 300 iterations.
                # Let's stick to the loop but optimize the inner part.
                
                for x in candidates:
                    gain = 0
                    for (p1, p2) in block_pairs:
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
        counts = np.zeros(num_triples, dtype=np.int32)
        
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

    def local_search_trim(blocks, random_state, time_limit=5.0):
        """
        Attempt to further reduce blocks by random reordering and deletion.
        """
        rng = random.Random(random_state)
        blocks = list(blocks)
        start_time = time.time()
        
        # Precompute triple counts for current blocks
        counts = np.zeros(num_triples, dtype=np.int32)
        for block in blocks:
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[triple]
                counts[idx] += 1
                
        active = [True] * len(blocks)
        
        while time.time() - start_time < time_limit:
            active_indices = [i for i in range(len(blocks)) if active[i]]
            if not active_indices:
                break
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

    def swap_local_search(blocks, random_state, time_limit=3.0):
        """
        Try to replace a block with another block that covers the same essential
        triples while potentially allowing removal of other blocks.
        """
        rng = random.Random(random_state)
        blocks = list(blocks)
        n_blocks = len(blocks)
        start_time = time.time()
        
        # Precompute triple counts
        counts = np.zeros(num_triples, dtype=np.int32)
        for block in blocks:
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[triple]
                counts[idx] += 1
        
        active = [True] * n_blocks
        
        # Precompute all possible blocks (7-subsets of v)
        # For v=24, k=7, this is C(24,7) = 346104. This is large to precompute lists of triples for all.
        # We will generate candidates on the fly or use a smaller pool.
        
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
                # To cover essential triples, the new block must contain the union of points in essential triples?
                # No, it must contain each essential triple.
                # Let's find the intersection of all blocks that cover the essential triples?
                # Simpler: Randomly sample blocks and check.
                
                num_candidates = 20
                for _ in range(num_candidates):
                    cand = tuple(sorted(rng.sample(range(v), k)))
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
                    
                    # Score: number of new triples covered (lower is better for replacement? 
                    # Actually, we want to minimize redundancy. 
                    # If we replace block i, we lose its coverage and gain cand's coverage.
                    # We want cand to cover the essential triples (which we already know it does)
                    # and ideally cover other triples that are currently covered by other blocks (redundant)
                    # or triples that are uncovered? No, we want to keep the covering valid.
                    # The goal is to allow *other* blocks to be removed.
                    # So we want cand to cover triples that are currently covered by exactly 1 block (other than i)?
                    # No, if cand covers a triple that was only covered by block j, then block j becomes removable.
                    # So we want cand to cover triples with count 1 (after removing i) or count 2 (if one is i).
                    
                    # Let's compute the "removal potential".
                    # Temporarily remove i's coverage, add cand's coverage.
                    # Count how many triples become count 0 (invalid) or count 1 (removable).
                    # We must not create count 0.
                    
                    # This is expensive. Let's use a heuristic:
                    # Prefer blocks that share many points with i? No.
                    # Prefer blocks that cover many triples currently covered by i?
                    
                    # Let's just count how many triples in cand have count > 1 (redundant coverage)
                    # vs count == 1 (essential for others).
                    # We want to avoid covering essential triples of others if possible?
                    # No, covering essential triples of others makes those others removable.
                    
                    # Let's try: Score = - (number of triples in cand that are currently covered by exactly 1 other block)
                    # This means cand takes over the essential load of other blocks.
                    
                    potential_removals = 0
                    for triple in itertools.combinations(cand, t):
                        idx = triple_to_idx[triple]
                        # counts[idx] includes block i.
                        # If we remove i, count becomes counts[idx] - 1.
                        # If counts[idx] == 1, then after removing i, count is 0. Invalid unless cand covers it (which it does).
                        # Wait, if counts[idx] == 1, that means only block i covers it.
                        # If cand covers it, then after replacement, count is 1 (from cand). Valid.
                        # If counts[idx] == 2, one is i, one is j.
                        # If cand covers it, new count is 1 (from cand). j becomes removable.
                        
                        # So we want to maximize the number of triples where counts[idx] == 2 (and i is one of them).
                        # This allows j to be removed.
                        
                        if counts[idx] == 2:
                            # Check if i covers it? Yes, because we are iterating triples of i's essential set?
                            # No, we are iterating triples of cand.
                            # We need to check if block i covers this triple.
                            if all(x in blocks[i] for x in triple):
                                potential_removals += 1
                                
                    # Also, we must ensure we don't break coverage.
                    # Since cand covers essential_triples (which are the only ones with count 1 for i),
                    # and we are replacing i with cand, the essential ones are still covered by cand.
                    # Other triples covered by i:
                    # If count was 1, it must be in essential_triples.
                    # If count was >1, removing i drops count by 1. Adding cand adds 1 for triples in cand.
                    # Net change for triple in cand: -1 (remove i) + 1 (add cand) = 0.
                    # Net change for triple NOT in cand: -1 (remove i).
                    # So we must ensure that no triple NOT in cand has count 1.
                    # But we already ensured essential_triples (count 1) are in cand.
                    # So it is safe.
                    
                    score = -potential_removals
                    if score < best_score:
                        best_score = score
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

    def final_polish(blocks, random_state, time_limit=3.0):
        """
        High-precision polish: repeatedly scan all blocks in random order
        and remove any that are not essential.
        """
        rng = random.Random(random_state)
        blocks = list(blocks)
        start_time = time.time()
        
        # Precompute triple counts
        counts = np.zeros(num_triples, dtype=np.int32)
        for block in blocks:
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[triple]
                counts[idx] += 1
        
        active = [True] * len(blocks)
        
        # Multiple passes until no more removals or time runs out
        while time.time() - start_time < time_limit:
            active_indices = [i for i in range(len(blocks)) if active[i]]
            if not active_indices:
                break
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
    
    # Main loop
    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    deadline = start_time + 140  # 10s buffer for safety
    
    seed_counter = 0
    while time.time() < deadline:
        elapsed = time.time() - start_time
        remaining = deadline - time.time()
        
        # Scale time limits based on remaining time
        if remaining > 10:
            trim_t = 1.0
            swap_t = 2.0
            polish_t = 2.0
        elif remaining > 5:
            trim_t = 0.5
            swap_t = 1.0
            polish_t = 1.0
        else:
            trim_t = 0.2
            swap_t = 0.5
            polish_t = 0.5
        
        blocks = greedy_cover(seed_counter)
        blocks = trim_blocks(blocks)
        blocks = local_search_trim(blocks, seed_counter, time_limit=trim_t)
        blocks = swap_local_search(blocks, seed_counter, time_limit=swap_t)
        blocks = final_polish(blocks, seed_counter, time_limit=polish_t)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)
            
        if best_len == 69:
            break
            
        seed_counter += 1
    
    # Final polish on the best result with whatever time is left
    if best_blocks:
        remaining_time = deadline - (time.time() - start_time)
        if remaining_time > 0.5:
            best_blocks = final_polish(best_blocks, 9999, time_limit=min(remaining_time, 5.0))
    
    if best_blocks:
        return [list(b) for b in best_blocks]
    else:
        return []

