import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin

    # 1. Precomputation
    points = range(v)
    all_t_subsets = list(itertools.combinations(points, t))
    N_t = len(all_t_subsets)
    t_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    N_b = len(all_blocks)
    
    # block_cover[b] = list of indices of t-subsets covered by block b
    block_cover = []
    for b in all_blocks:
        covered = [t_to_idx[s] for s in itertools.combinations(b, t)]
        block_cover.append(covered)
        
    # t_to_blocks[ti] = list of block indices covering t-subset ti
    t_to_blocks = [[] for _ in range(N_t)]
    for bi, covered in enumerate(block_cover):
        for ti in covered:
            t_to_blocks[ti].append(bi)

    def fast_greedy(rng):
        """
        Greedy set cover with dynamic weights.
        Weight of block b = sum(1 / (current_count[t] + 1)) for t in b
        """
        counts = np.zeros(N_t, dtype=np.int32)
        uncovered_mask = np.ones(N_t, dtype=bool)
        num_uncovered = N_t
        selected = []
        
        # Precompute list of blocks for faster access? 
        # We need to score all blocks or a subset.
        # Scoring all 75k blocks every step (approx 300 steps) is 75k * 300 * 56 ops = 1.2B ops. Too slow in Python.
        
        # Optimization: 
        # Maintain a priority queue? No, weights change.
        # Use "lazy" evaluation or sampling?
        # Standard approach for large instances:
        # 1. Find a random uncovered t-subset.
        # 2. Consider only blocks covering it.
        # 3. Among those, pick the one with highest gain (covers most other uncovered t-subsets).
        # This is "Randomized Greedy" or "Approximate Greedy".
        
        while num_uncovered > 0:
            # Pick a random uncovered t-subset
            # Instead of random, pick one with fewest covering blocks (hardest to cover)?
            # Or just random for speed and diversity.
            
            # To pick a random uncovered index:
            # We can keep a list of uncovered indices and update it.
            # Or use numpy.
            
            # Let's try the "candidate block" approach.
            # Pick a random uncovered t-subset.
            uncovered_indices = np.where(uncovered_mask)[0]
            if len(uncovered_indices) == 0:
                break
                
            ti_target = rng.choice(uncovered_indices.tolist())
            
            # Candidates are blocks covering ti_target
            candidates = t_to_blocks[ti_target]
            
            best_block = -1
            best_gain = -1
            
            # Evaluate candidates
            # Gain = number of *newly* covered t-subsets
            for bi in candidates:
                gain = 0
                for ti in block_cover[bi]:
                    if uncovered_mask[ti]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = bi
            
            # Add best_block
            selected.append(best_block)
            for ti in block_cover[best_block]:
                if uncovered_mask[ti]:
                    uncovered_mask[ti] = False
                    num_uncovered -= 1
                counts[ti] += 1
                
        return selected

    def prune(blocks):
        """
        Remove redundant blocks.
        """
        if not blocks:
            return []
            
        counts = np.zeros(N_t, dtype=np.int32)
        for bi in blocks:
            for ti in block_cover[bi]:
                counts[ti] += 1
                
        current_blocks = list(blocks)
        changed = True
        while changed:
            changed = False
            # Find blocks with no unique coverage
            # A block is redundant if for all ti in block_cover[b], counts[ti] > 1
            # OR if it's the only block, it's not redundant (unless it covers nothing, which is impossible)
            
            # We can iterate through blocks and check.
            # To speed up, check blocks in random order?
            
            new_blocks = []
            for bi in current_blocks:
                is_redundant = True
                for ti in block_cover[bi]:
                    if counts[ti] == 1:
                        is_redundant = False
                        break
                if not is_redundant:
                    new_blocks.append(bi)
                else:
                    # Remove bi
                    for ti in block_cover[bi]:
                        counts[ti] -= 1
                    changed = True
                    
            current_blocks = new_blocks
        return current_blocks

    def local_search(blocks, rng, time_limit):
        """
        Local search with drop/swap moves.
        """
        current_blocks = list(blocks)
        if not current_blocks:
            return current_blocks
            
        counts = np.zeros(N_t, dtype=np.int32)
        for bi in current_blocks:
            for ti in block_cover[bi]:
                counts[ti] += 1
                
        # Map block index to position in current_blocks for O(1) access? 
        # List is fine for ~300 items.
        
        best_blocks = list(current_blocks)
        best_size = len(current_blocks)
        
        while time.time() < time_limit:
            # Try to drop a random block
            if not current_blocks:
                break
            
            idx = rng.randrange(len(current_blocks))
            bi = current_blocks[idx]
            
            # Check if bi is redundant
            redundant = True
            for ti in block_cover[bi]:
                if counts[ti] == 1:
                    redundant = False
                    break
                    
            if redundant:
                # Drop it
                current_blocks.pop(idx)
                for ti in block_cover[bi]:
                    counts[ti] -= 1
                if len(current_blocks) < best_size:
                    best_size = len(current_blocks)
                    best_blocks = list(current_blocks)
                continue
                
            # Try Swap
            # Find unique t-subsets covered by bi
            unique_ts = [ti for ti in block_cover[bi] if counts[ti] == 1]
            
            if not unique_ts:
                # Should have been caught by redundant check
                continue
                
            # Find candidates for swap
            # Candidates must cover all t in unique_ts
            # Start with candidates for the first unique_t
            candidates = set(t_to_blocks[unique_ts[0]])
            for ti in unique_ts[1:]:
                candidates &= set(t_to_blocks[ti])
                
            candidates.discard(bi)
            
            if not candidates:
                continue
                
            # Score candidates
            # Score = number of t-subsets t in candidate such that counts[t] == 1 
            # AND the unique block covering t is NOT bi.
            # Why? Because if we add candidate, counts[t] becomes 2.
            # If the unique block was 'other', 'other' loses a unique t.
            # This might make 'other' redundant later.
            
            best_cand = -1
            best_score = -1
            
            # Optimization: If candidates is large, sample?
            # Usually unique_ts is small (1-3). Intersection is small.
            
            for c in candidates:
                score = 0
                for ti in block_cover[c]:
                    if counts[ti] == 1:
                        # Who is the unique block?
                        # We need to know which block uniquely covers ti.
                        # We don't have a direct map ti -> unique_block.
                        # But we know bi is one of the blocks covering ti? No, ti is in unique_ts of bi? 
                        # No, ti here is from block_cover[c].
                        # If counts[ti] == 1, there is exactly one block covering ti.
                        # Is it bi?
                        # If ti is in unique_ts of bi, then yes.
                        # But ti is in block_cover[c].
                        # If c != bi, and counts[ti] == 1, then the unique block is some 'other'.
                        # We don't need to know which one, just that it's not bi.
                        # Wait, if ti is in unique_ts of bi, then bi covers ti and counts[ti]==1.
                        # If c also covers ti, then counts[ti] would be at least 2? 
                        # No, counts[ti] is the count in the CURRENT set.
                        # bi is in the current set.
                        # If c is a candidate, c is NOT in the current set.
                        # So if counts[ti] == 1, the unique block is bi (if bi covers ti) or some other block.
                        # If bi covers ti, and c covers ti, and we swap bi->c, 
                        # then ti is still covered (by c).
                        # The count of ti remains 1 (bi removed, c added).
                        # So ti remains unique to c.
                        # This does NOT help other blocks.
                        
                        # We want to find t such that:
                        # 1. t is covered by c.
                        # 2. counts[t] == 1 in current set.
                        # 3. The unique block covering t is NOT bi.
                        # If this condition holds, then after swap, t is covered by c AND the other block.
                        # counts[t] becomes 2.
                        # The "other" block loses its unique coverage of t.
                        
                        # How to check if unique block is bi?
                        # If ti is in block_cover[bi], then bi covers ti.
                        # If counts[ti] == 1 and bi covers ti, then bi is the unique block.
                        # So we want ti NOT in block_cover[bi].
                        
                        if ti not in block_cover[bi]: # This check is O(56) per ti, too slow?
                            # Better: precompute a set of t's covered by bi?
                            pass
                pass
            
            # Refinement: 
            # We can precompute a boolean array or set for t's covered by bi.
            # But bi changes.
            # Let's just check `if ti not in bi_cover_set`.
            # To speed up, we can approximate or skip this specific optimization if it's too slow.
            # Alternative: Just pick a random candidate?
            # Or pick the candidate that covers the most t-subsets with count > 1?
            # Let's stick to a simpler score: 
            # Score = number of t in c such that counts[t] > 1.
            # This ensures we are covering "safe" t-subsets, potentially allowing bi's unique t's to be handled by c, 
            # and consolidating coverage.
            
            # Actually, the "potential to enable drops" score is better.
            # Let's implement it efficiently.
            # Create a set of t's in bi.
            bi_t_set = set(block_cover[bi])
            
            for c in candidates:
                score = 0
                for ti in block_cover[c]:
                    if counts[ti] == 1 and ti not in bi_t_set:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_cand = c
            
            if best_cand == -1 or best_score == 0:
                # No good swap found. Try a random candidate to explore?
                # Or just skip.
                # If we skip, we might be stuck.
                # Let's pick a random candidate if any exist.
                if candidates:
                    best_cand = rng.choice(list(candidates))
                else:
                    continue
                    
            # Perform Swap
            # Remove bi
            for ti in block_cover[bi]:
                counts[ti] -= 1
            # Add best_cand
            for ti in block_cover[best_cand]:
                counts[ti] += 1
            current_blocks[idx] = best_cand
            
            # Check for new drops
            # We only need to check blocks that might have become redundant.
            # A block b becomes redundant if all its unique t's are now covered by other blocks.
            # This happens if counts[t] increases for some t in b? No, counts[t] increases for t in best_cand.
            # If t in b and t in best_cand, and counts[t] was 1, now it is 2.
            # So b lost a unique t.
            # Check blocks that share t's with best_cand.
            
            affected = set()
            for ti in block_cover[best_cand]:
                # Who covers ti?
                # We need to find which current blocks cover ti.
                # We don't have a dynamic index.
                # We can just check all current blocks? 300 * 56 = 16k ops. Fast.
                pass
            
            # Simpler: Just re-run prune? 
            # Prune is O(N_blocks * 56). 300 * 56 = 16k.
            # But we want to keep counts updated.
            # Let's just do a quick drop check on blocks that intersect best_cand's coverage.
            
            # Actually, just calling a lightweight prune is safer.
            # But we must update counts if we drop.
            
            # Let's do a simple check:
            # For each block in current_blocks, check if it's redundant.
            # This is O(300 * 56).
            
            new_blocks = []
            drop_occurred = False
            for b in current_blocks:
                is_redundant = True
                for ti in block_cover[b]:
                    if counts[ti] == 1:
                        is_redundant = False
                        break
                if not is_redundant:
                    new_blocks.append(b)
                else:
                    for ti in block_cover[b]:
                        counts[ti] -= 1
                    drop_occurred = True
                    
            if drop_occurred:
                current_blocks = new_blocks
                if len(current_blocks) < best_size:
                    best_size = len(current_blocks)
                    best_blocks = list(current_blocks)
            else:
                current_blocks = new_blocks # Just to normalize list (it's the same)
                
        return best_blocks

    # Main Execution
    best_solution = None
    best_len = float('inf')
    
    # Phase 1: Multi-start Greedy
    num_starts = 10
    for i in range(num_starts):
        if time.time() > deadline - 30.0:
            break
        rng = random.Random(123 + i)
        blocks = fast_greedy(rng)
        blocks = prune(blocks)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_solution = blocks
            
    # Phase 2: Local Search on Best
    if best_solution:
        rng = random.Random(999)
        optimized = local_search(best_solution, rng, deadline - 5.0)
        if len(optimized) < best_len:
            best_len = len(optimized)
            best_solution = optimized

    if not best_solution:
        # Fallback
        rng = random.Random(1)
        best_solution = fast_greedy(rng)
        best_solution = prune(best_solution)

    return [list(all_blocks[bi]) for bi in best_solution]
