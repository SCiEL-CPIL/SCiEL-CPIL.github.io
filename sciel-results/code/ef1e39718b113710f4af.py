import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    points = list(range(v))
    triples = list(itertools.combinations(points, t))
    triple_to_id = {tr: i for i, tr in enumerate(triples)}
    num_triples = len(triples)
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Map block index to list of triple IDs it covers
    # Using lists of numpy arrays for fast access
    block_triples = []
    for b in all_blocks:
        ts = [triple_to_id[tup] for tup in itertools.combinations(b, t)]
        block_triples.append(np.array(ts, dtype=np.int32))
        
    # Map triple ID to list of block indices covering it
    triple_blocks = [[] for _ in range(num_triples)]
    for b_idx, ts in enumerate(block_triples):
        for t_idx in ts:
            triple_blocks[t_idx].append(b_idx)
    triple_blocks = [np.array(lst, dtype=np.int32) for lst in triple_blocks]
    
    def greedy_construction(seed=0, time_budget=deadline):
        rng = random.Random(seed)
        # gain[b_idx] is the number of currently uncovered triples covered by block b_idx
        gain = np.array([len(ts) for ts in block_triples], dtype=np.int32)
        is_uncovered = np.ones(num_triples, dtype=np.bool_)
        
        selected_blocks = []
        selected_set = set()
        
        while is_uncovered.any():
            if time.time() > time_budget:
                break
            # Find max gain
            max_gain = np.max(gain)
            if max_gain == 0:
                break
            
            # Find indices with max gain
            candidates = np.where(gain == max_gain)[0]
            
            # Random tie-breaking
            b_idx = rng.choice(candidates.tolist())
            
            selected_blocks.append(b_idx)
            selected_set.add(b_idx)
            
            # Update gain and is_uncovered
            for t_idx in block_triples[b_idx]:
                if is_uncovered[t_idx]:
                    is_uncovered[t_idx] = False
                    # For every block that covers this triple, decrease its gain
                    # Only if it's not the one we just added? No, gain is for "future" blocks.
                    # If we just added b_idx, its gain doesn't matter for selection anymore, 
                    # but other blocks covering this triple lose this triple from their gain.
                    gain[triple_blocks[t_idx]] -= 1
                    # Note: gain[b_idx] also decreases, but we don't care about b_idx's gain anymore 
                    # unless we remove it later. For now, it's in selected_set.
                    
        return selected_blocks

    def complete_cover(selected, deadline):
        """Ensure all triples are covered. If greedy stopped early, add more blocks."""
        if not selected:
            return selected
            
        cover_count = np.zeros(num_triples, dtype=np.int32)
        selected_set = set(selected)
        for b_idx in selected:
            cover_count[block_triples[b_idx]] += 1
            
        while True:
            if time.time() > deadline:
                break
            uncovered_mask = (cover_count == 0)
            if not np.any(uncovered_mask):
                break
                
            # Find a block that covers the most uncovered triples
            unc_idx = np.where(uncovered_mask)[0]
            if len(unc_idx) == 0:
                break
                
            # Pick a random uncovered triple to anchor the search
            target_tid = unc_idx[random.randint(0, len(unc_idx)-1)]
            
            # Candidates are blocks covering this target
            candidates = triple_blocks[target_tid]
            
            best_b = -1
            best_gain = -1
            for b_idx in candidates:
                if b_idx in selected_set:
                    continue
                # Gain is number of uncovered triples this block covers
                g = np.sum(uncovered_mask[block_triples[b_idx]])
                if g > best_gain:
                    best_gain = g
                    best_b = b_idx
                    
            if best_b == -1:
                break
                
            selected.append(best_b)
            selected_set.add(best_b)
            cover_count[block_triples[best_b]] += 1
            
        return selected

    def reduce_cover(selected, deadline):
        """Remove redundant blocks iteratively."""
        if not selected:
            return selected
            
        cover_count = np.zeros(num_triples, dtype=np.int32)
        for b_idx in selected:
            cover_count[block_triples[b_idx]] += 1
            
        active = [True] * len(selected)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Randomize order to avoid bias
            indices = list(range(len(selected)))
            random.shuffle(indices)
            
            for i in indices:
                if not active[i]:
                    continue
                b_idx = selected[i]
                # Check if redundant: min cover count of its triples > 1
                if np.min(cover_count[block_triples[b_idx]]) > 1:
                    active[i] = False
                    cover_count[block_triples[b_idx]] -= 1
                    changed = True
                    
        return [selected[i] for i in range(len(selected)) if active[i]]

    def local_search_ilS(selected, deadline):
        """
        Iterated Local Search.
        1. Reduce.
        2. Perturb: Remove a random block, add best replacement.
        3. Reduce.
        Repeat.
        """
        selected = list(selected)
        best_selected = selected[:]
        best_size = len(selected)
        
        cover_count = np.zeros(num_triples, dtype=np.int32)
        for b_idx in selected:
            cover_count[block_triples[b_idx]] += 1
            
        rng = random.Random(42)
        
        # Number of iterations based on time
        # Each iteration is fast. Let's run as many as possible.
        
        while time.time() < deadline:
            # 1. Reduce
            selected = reduce_cover(selected, deadline)
            
            if len(selected) < best_size:
                best_size = len(selected)
                best_selected = selected[:]
                
            if not selected:
                break
                
            # Recalculate cover_count for the reduced set
            cover_count = np.zeros(num_triples, dtype=np.int32)
            selected_set = set(selected)
            for b_idx in selected:
                cover_count[block_triples[b_idx]] += 1
            
            # 2. Perturb
            # Pick a random block to remove
            idx = rng.randint(0, len(selected)-1)
            b_old = selected[idx]
            
            # Remove it
            cover_count[block_triples[b_old]] -= 1
            selected.pop(idx)
            selected_set.discard(b_old)
            
            # Check if valid
            if np.min(cover_count) >= 0 and np.any(cover_count == 0):
                # Invalid, need to add a block
                uncovered_mask = (cover_count == 0)
                unc_idx = np.where(uncovered_mask)[0]
                
                # Find best replacement
                # Use the first uncovered triple as anchor
                target_tid = unc_idx[0]
                candidates = triple_blocks[target_tid]
                
                best_b = -1
                best_gain = -1
                for b_idx in candidates:
                    if b_idx in selected_set:
                        continue
                    g = np.sum(uncovered_mask[block_triples[b_idx]])
                    if g > best_gain:
                        best_gain = g
                        best_b = b_idx
                        
                if best_b != -1:
                    selected.append(best_b)
                    selected_set.add(best_b)
                    cover_count[block_triples[best_b]] += 1
                else:
                    # Should not happen if we started with a valid cover and just removed one block
                    # because the removed block covered those triples.
                    # Wait, if we removed b_old, the uncovered triples are those ONLY covered by b_old.
                    # b_old is a candidate for itself? No, we discarded it.
                    # But other blocks might cover them? No, by definition they were only covered by b_old (count was 1).
                    # So we MUST add a block that covers them.
                    # If no candidate found, it means no block covers that specific triple? Impossible.
                    # Ah, candidates are blocks covering target_tid. b_old is one of them.
                    # If b_old was the ONLY block covering target_tid, and we removed it,
                    # then no other block covers target_tid.
                    # But we are looking for a block to ADD.
                    # The set of all blocks includes b_old.
                    # We skipped b_old because it's not in selected_set?
                    # Wait, `selected_set` was updated. b_old is NOT in selected_set.
                    # So b_old IS a candidate.
                    # Why did we skip it? `if b_idx in selected_set: continue`.
                    # b_old is NOT in selected_set. So it is a candidate.
                    # So best_b will at least be b_old.
                    # If best_b is b_old, we add it back. No change.
                    pass
            else:
                # Valid! We removed a block and didn't need to add one.
                # This means the block was redundant, but reduce_cover should have caught it.
                # Unless reduce_cover didn't run fully?
                # Or maybe removing it made it valid, but it wasn't redundant before?
                # Redundant means min count > 1. If min count was 1, removing it makes it 0 (invalid).
                # So this case (valid after removal) implies it WAS redundant.
                # So reduce_cover should have removed it.
                # This branch is likely rarely taken if reduce_cover is thorough.
                pass

        return best_selected

    # Main loop
    best_cover = None
    best_size = float('inf')
    
    # Phase 1: Greedy Restarts
    seed = 0
    phase1_deadline = deadline - 20.0
    
    while time.time() < phase1_deadline:
        blocks = greedy_construction(seed, phase1_deadline - 1.0)
        blocks = complete_cover(blocks, phase1_deadline - 1.0)
        blocks = reduce_cover(blocks, phase1_deadline - 1.0)
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_cover = blocks[:]
            
        if best_size <= 70: # Lower bound is 69, 70 is great
            break
            
        seed += 1
        
        # Occasionally run ILS on the best
        if seed % 2 == 0 and best_cover:
             ls_deadline = min(deadline, time.time() + 5.0)
             improved = local_search_ilS(best_cover[:], ls_deadline)
             if len(improved) < best_size:
                 best_size = len(improved)
                 best_cover = improved[:]
                 
    # Phase 2: Final ILS on the Best
    if best_cover:
        final_ls_deadline = deadline - 2.0
        best_cover = local_search_ilS(best_cover, final_ls_deadline)
        best_size = len(best_cover)
        
        # One more reduction
        best_cover = reduce_cover(best_cover, deadline - 1.0)
        best_size = len(best_cover)

    if not best_cover:
        blocks = greedy_construction(0, deadline - 5.0)
        blocks = complete_cover(blocks, deadline - 2.0)
        blocks = reduce_cover(blocks, deadline - 1.0)
        best_cover = blocks

    return [list(all_blocks[b_idx]) for b_idx in best_cover]
