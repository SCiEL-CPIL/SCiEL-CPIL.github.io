import numpy as np
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (evaluator kills at 150s)
    
    # Precompute all triples
    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # Precompute all blocks and the triples they cover
    # C(24, 7) = 346,104 blocks. This is manageable if done efficiently.
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # c(k,t) = number of t-subsets in a k-subset
    c_k_t = 1
    for i in range(1, t+1):
        c_k_t = c_k_t * (k - i + 1) // i
        
    # We need to map triples to blocks efficiently for local search.
    # Instead of storing a full matrix, we store for each block its triples.
    # And for each triple, a list of block indices that contain it.
    # This is memory heavy but necessary for fast lookups in local search.
    
    block_triples = []
    triple_to_blocks = defaultdict(list)
    
    # Optimization: Use a list of tuples for block_triples to save memory and speed up iteration
    for i, block in enumerate(all_blocks):
        bt = tuple(triple_to_idx[triple] for triple in itertools.combinations(block, t))
        block_triples.append(bt)
        for triple_idx in bt:
            triple_to_blocks[triple_idx].append(i)
            
    if time.time() > deadline - 10:
        # Fallback if precomputation took too long (unlikely but safe)
        return [list(b) for b in all_blocks[:100]]

    def greedy_cover(uncovered_init, use_weighted=True, random_tie=False, seed=None):
        """Run greedy set cover."""
        if seed is not None:
            np.random.seed(seed)
        uncovered = uncovered_init.copy()
        selected = []
        selected_set = set()
        
        # Count how many times each triple is covered by the *potential* blocks? 
        # No, standard greedy: pick block that covers most uncovered triples.
        # To speed this up, we can maintain a list of uncovered triples.
        # However, scanning 346k blocks every step is slow if many steps are needed.
        # C(24,7,3) covering number is ~76. 76 * 346k = 26M ops. This is fine in numpy/python if optimized.
        
        # Better approach: 
        # 1. Find an uncovered triple.
        # 2. Look at all blocks covering it.
        # 3. Pick the one that covers the most *other* uncovered triples.
        
        uncovered_set = set(np.where(uncovered)[0])
        
        while uncovered_set:
            if time.time() > deadline:
                break
                
            # Pick a random uncovered triple to start (or the one with fewest covering blocks to be safe)
            # Random is often better for diversity in restarts
            target_triple = random.choice(list(uncovered_set))
            
            # Get all blocks covering this triple
            candidates = triple_to_blocks[target_triple]
            
            best_block = -1
            best_gain = -1
            best_weighted_gain = -1
            
            # Filter candidates: must not be already selected
            # Calculate gain for each candidate
            # Gain = number of uncovered triples in the block
            
            # Optimization: If candidates list is small, iterate.
            for b_idx in candidates:
                if b_idx in selected_set:
                    continue
                    
                gain = 0
                w_gain = 0.0
                for t_idx in block_triples[b_idx]:
                    if t_idx in uncovered_set:
                        gain += 1
                        # Weighted gain: 1/freq or similar. 
                        # For simplicity in this repair, let's stick to unweighted or simple weighted.
                        # Let's use a static weight based on global frequency to break ties?
                        # Or just unweighted for speed.
                        w_gain += 1.0 # Placeholder for weighted logic if needed later
                        
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
                    best_weighted_gain = w_gain
                elif gain == best_gain and random_tie:
                    # Random tie breaking
                    if random.random() < 0.5:
                        best_block = b_idx
                        best_weighted_gain = w_gain

            if best_block == -1 or best_gain == 0:
                break # No progress possible or no candidates
            
            selected.append(best_block)
            selected_set.add(best_block)
            
            # Update uncovered set
            for t_idx in block_triples[best_block]:
                if t_idx in uncovered_set:
                    uncovered_set.remove(t_idx)
                    
        return selected

    def prune_blocks(selected_indices):
        """Remove redundant blocks from a covering."""
        if not selected_indices:
            return selected_indices
            
        remaining = list(selected_indices)
        
        # Build coverage counts: for each triple, how many remaining blocks cover it
        triple_cover_count = defaultdict(int)
        
        for block_idx in remaining:
            for triple_idx in block_triples[block_idx]:
                triple_cover_count[triple_idx] += 1
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_remaining = []
            
            for block_idx in remaining:
                if time.time() > deadline:
                    new_remaining.append(block_idx)
                    continue
                    
                # Check if this block is the only one covering any triple
                can_remove = True
                for triple_idx in block_triples[block_idx]:
                    if triple_cover_count[triple_idx] == 1:
                        can_remove = False
                        break
                        
                if can_remove:
                    # Remove this block
                    changed = True
                    for triple_idx in block_triples[block_idx]:
                        triple_cover_count[triple_idx] -= 1
                else:
                    new_remaining.append(block_idx)
                    
            remaining = new_remaining
            
        return remaining

    def local_search_swap(selected_indices):
        """Try to reduce block count by swapping blocks."""
        if len(selected_indices) <= 1:
            return selected_indices
            
        remaining = list(selected_indices)
        selected_set = set(remaining)
        
        # Build coverage counts
        triple_cover_count = defaultdict(int)
        for block_idx in remaining:
            for triple_idx in block_triples[block_idx]:
                triple_cover_count[triple_idx] += 1
        
        improved = True
        # Limit iterations to prevent infinite loops or excessive time
        max_iterations = 1000 
        iter_count = 0
        
        while improved and time.time() < deadline - 5 and iter_count < max_iterations:
            improved = False
            iter_count += 1
            
            # Try removing each block and see if we can cover the lost triples with one new block
            # We iterate over a copy of indices to avoid modification during iteration issues
            indices_to_try = list(range(len(remaining)))
            random.shuffle(indices_to_try)
            
            for i in indices_to_try:
                if time.time() > deadline - 5:
                    break
                    
                block_idx = remaining[i]
                
                # Temporarily remove this block
                lost_triples = []
                for triple_idx in block_triples[block_idx]:
                    if triple_cover_count[triple_idx] == 1:
                        lost_triples.append(triple_idx)
                    
                if not lost_triples:
                    # Can remove without adding
                    for triple_idx in block_triples[block_idx]:
                        triple_cover_count[triple_idx] -= 1
                    remaining.pop(i)
                    selected_set.remove(block_idx)
                    improved = True
                    break
                    
                # Try to find a single block that covers all lost_triples
                # and is not already in remaining
                found = False
                # Candidates are blocks covering the first lost triple
                # We can filter candidates to those that cover ALL lost triples
                # This is a subset of triple_to_blocks[lost_triples[0]]
                
                # Optimization: If len(lost_triples) is large, it's unlikely to find a swap.
                # But let's check.
                for candidate in triple_to_blocks[lost_triples[0]]:
                    if candidate in selected_set:
                        continue
                    
                    # Check if candidate covers all lost_triples
                    covers_all = True
                    # Use set for O(1) lookup? block_triples is a tuple. 
                    # Converting to set for each candidate is expensive.
                    # Let's just iterate. Lost triples is usually small.
                    c_set = set(block_triples[candidate])
                    for triple_idx in lost_triples:
                        if triple_idx not in c_set:
                            covers_all = False
                            break
                            
                    if covers_all:
                        # Swap: remove block_idx, add candidate
                        for triple_idx in block_triples[block_idx]:
                            triple_cover_count[triple_idx] -= 1
                        for triple_idx in block_triples[candidate]:
                            triple_cover_count[triple_idx] += 1
                        remaining[i] = candidate
                        selected_set.discard(block_idx)
                        selected_set.add(candidate)
                        improved = True
                        found = True
                        break
                        
                if found:
                    break
                    
        return remaining

    def local_search_2swap(selected_indices):
        """Try to remove 2 blocks and add 1 block."""
        if len(selected_indices) <= 2:
            return selected_indices
            
        remaining = list(selected_indices)
        selected_set = set(remaining)
        
        # Build coverage counts
        triple_cover_count = defaultdict(int)
        for block_idx in remaining:
            for triple_idx in block_triples[block_idx]:
                triple_cover_count[triple_idx] += 1
        
        improved = True
        max_iterations = 500
        iter_count = 0
        
        while improved and time.time() < deadline - 5 and iter_count < max_iterations:
            improved = False
            iter_count += 1
            
            n = len(remaining)
            # Try pairs of blocks to remove
            # To save time, we don't try all pairs, just a random sample or sequential
            # Let's try sequential pairs but break early on time
            
            pairs_tried = 0
            for i in range(n):
                if time.time() > deadline - 5:
                    break
                for j in range(i+1, n):
                    if time.time() > deadline - 5:
                        break
                    if pairs_tried > 50: # Limit pairs per iteration to ensure progress
                        break
                    pairs_tried += 1
                    
                    block_i = remaining[i]
                    block_j = remaining[j]
                    
                    # Calculate lost triples if we remove both
                    # A triple is lost if it was covered ONLY by block_i or ONLY by block_j (or both, but count becomes 0)
                    # Actually, if a triple is covered by block_i and block_j, count is 2. Removing both makes it 0.
                    # If covered by block_i and block_k, count is 2. Removing i makes it 1 (safe).
                    
                    # Let's simulate removal
                    # We need to know which triples drop to 0.
                    
                    # Efficient check:
                    # For each triple in block_i:
                    #   if count == 1: lost (unless covered by block_j? no, if count==1 it's only block_i)
                    #   if count == 2: check if block_j is the other one. If yes, lost. If no, safe.
                    
                    lost_triples = set()
                    
                    for t_idx in block_triples[block_i]:
                        if triple_cover_count[t_idx] == 1:
                            lost_triples.add(t_idx)
                        elif triple_cover_count[t_idx] == 2:
                            # Is block_j the other cover?
                            # We can check if t_idx is in block_triples[block_j]
                            # Since block_triples is a tuple, this is O(len). 
                            # Better: precompute intersection? 
                            # For now, linear search in tuple is okay for small k,t
                            if t_idx in block_triples[block_j]:
                                lost_triples.add(t_idx)
                                # It's covered by both, so removing both loses it.
                                # But wait, if count is 2, and both are removed, it becomes 0.
                                # If count is 2, and only one is removed, it becomes 1.
                                # So yes, if t is in both, and count is 2, it is lost.
                    
                    # We also need to check triples in block_j that are NOT in block_i
                    for t_idx in block_triples[block_j]:
                        if t_idx in lost_triples:
                            continue # Already identified
                        if triple_cover_count[t_idx] == 1:
                            # Only covered by block_j
                            lost_triples.add(t_idx)
                        elif triple_cover_count[t_idx] == 2:
                            # Covered by block_j and one other. 
                            # Since t_idx is not in block_i (checked above implicitly? No, we skipped if in lost_triples)
                            # Wait, if t_idx is in block_i, it would have been processed in the first loop.
                            # If it's not in block_i, then the "other" cover is not block_i.
                            # So removing block_j leaves it with count 1. Safe.
                            pass

                    if not lost_triples:
                        # Can remove both
                        for t_idx in block_triples[block_i]:
                            triple_cover_count[t_idx] -= 1
                        for t_idx in block_triples[block_j]:
                            triple_cover_count[t_idx] -= 1
                        # Remove from remaining (handle indices carefully)
                        remaining.pop(j)
                        remaining.pop(i)
                        selected_set.remove(block_i)
                        selected_set.remove(block_j)
                        improved = True
                        break
                    
                    # Try to find a single block that covers all lost_triples
                    if len(lost_triples) > 0:
                        # Find candidates covering the first lost triple
                        first_lost = next(iter(lost_triples))
                        for candidate in triple_to_blocks[first_lost]:
                            if candidate in selected_set:
                                continue
                            
                            covers_all = True
                            c_set = set(block_triples[candidate])
                            for t_idx in lost_triples:
                                if t_idx not in c_set:
                                    covers_all = False
                                    break
                            
                            if covers_all:
                                # Swap: remove i, j, add candidate
                                for t_idx in block_triples[block_i]:
                                    triple_cover_count[t_idx] -= 1
                                for t_idx in block_triples[block_j]:
                                    triple_cover_count[t_idx] -= 1
                                for t_idx in block_triples[candidate]:
                                    triple_cover_count[t_idx] += 1
                                
                                # Update remaining list
                                # Remove j then i (since j > i)
                                remaining.pop(j)
                                remaining.pop(i)
                                remaining.append(candidate)
                                
                                selected_set.remove(block_i)
                                selected_set.remove(block_j)
                                selected_set.add(candidate)
                                
                                improved = True
                                break
                        if improved:
                            break
            if improved:
                continue # Restart outer loop
                
        return remaining

    # Run multiple strategies
    best_solution = None
    best_size = float('inf')
    
    def update_best(selected):
        nonlocal best_solution, best_size
        if selected:
            pruned = prune_blocks(selected)
            # Further local search
            swapped = local_search_swap(pruned)
            swapped2 = local_search_2swap(swapped)
            
            if len(swapped2) < best_size:
                best_size = len(swapped2)
                best_solution = [list(all_blocks[i]) for i in swapped2]

    # Strategy 1: Standard greedy
    selected = greedy_cover(np.ones(n_triples, dtype=bool), use_weighted=False, random_tie=False, seed=42)
    if selected:
        update_best(selected)
        
    # Strategy 2: Greedy with random tie breaking
    for restart in range(10):
        if time.time() > deadline - 10:
            break
        seed = restart * 12345 + 67890
        selected = greedy_cover(np.ones(n_triples, dtype=bool), use_weighted=False, random_tie=True, seed=seed)
        if selected:
            update_best(selected)
            
    # If we still have time, try to improve the best solution further with more local search
    if best_solution is not None and time.time() < deadline - 5:
        # Reconstruct indices
        block_to_idx = {tuple(b): i for i, b in enumerate(all_blocks)}
        current_indices = [block_to_idx[tuple(b)] for b in best_solution]
        
        # Run more local search
        improved = local_search_swap(current_indices)
        improved2 = local_search_2swap(improved)
        improved3 = local_search_swap(improved2)
        
        if len(improved3) < best_size:
            best_size = len(improved3)
            best_solution = [list(all_blocks[i]) for i in improved3]

    # Final validation and return
    if best_solution is None:
        # Fallback: return a valid covering
        # Greedy should have produced something. If not, return all blocks (too many) or a subset.
        # To be safe, return the result of a quick greedy if it exists, else a dummy.
        # But greedy always returns a cover if it runs to completion.
        # If it broke due to time, it might be invalid.
        # Let's do a quick check.
        # For safety, if best_solution is None, we return a known valid structure if possible, 
        # or just the greedy result before pruning if it was valid.
        # Given the constraints, if greedy broke, it's likely invalid.
        # We should ensure we return *a* valid cover.
        # Let's force a final greedy run with strict time check.
        
        # Actually, the first greedy run (seed 42) is most likely to complete or be close.
        # If best_solution is None, it means no valid pruned/swapped set was found.
        # This is unlikely. 
        # As a last resort, return the raw greedy output if it covers all.
        
        # Check raw greedy from seed 42
        raw_selected = greedy_cover(np.ones(n_triples, dtype=bool), use_weighted=False, random_tie=False, seed=42)
        # Verify coverage
        covered = np.zeros(n_triples, dtype=bool)
        for b_idx in raw_selected:
            covered[block_triples[b_idx]] = True
        if np.all(covered):
            return [list(all_blocks[i]) for i in raw_selected]
            
        # If even that fails, return empty list (will score 0) or a small invalid set?
        # Contract says "Valid when every t-subset... lies in at least one block".
        # If we can't find one, we return whatever we have. 
        # But we must return a list of blocks.
        return [list(b) for b in all_blocks[:10]]

    return best_solution
