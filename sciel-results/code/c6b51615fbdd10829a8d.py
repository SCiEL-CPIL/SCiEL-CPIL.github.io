import numpy as np
import itertools
import time
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Precomputation
    # Points are 0..v-1
    points = list(range(v))
    
    # All t-subsets (triples for t=3)
    triples = list(itertools.combinations(points, t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # All k-subsets (blocks)
    blocks = list(itertools.combinations(points, k))
    n_blocks = len(blocks)
    
    # Map each block to the list of triple indices it covers
    # block_triples[i] is a list of triple indices
    block_triples = []
    for block in blocks:
        t_idx = []
        for triple in itertools.combinations(block, t):
            t_idx.append(triple_to_idx[triple])
        block_triples.append(t_idx)
        
    # Inverted index: triple_to_blocks[j] is a list of block indices covering triple j
    triple_to_blocks = [[] for _ in range(n_triples)]
    for i, t_list in enumerate(block_triples):
        for t_idx in t_list:
            triple_to_blocks[t_idx].append(i)
            
    # Convert to numpy arrays for faster access if needed, 
    # but lists of lists are often faster for small sparse data in Python.
    # Let's stick to lists for triple_to_blocks as it's a lookup.
    
    def get_uncovered_count(block_idx, selected_set):
        """
        Returns the number of triples covered by block_idx that are 
        NOT covered by any other block in selected_set.
        """
        if block_idx in selected_set:
            return 0
        count = 0
        for t_idx in block_triples[block_idx]:
            # Check if t_idx is covered by any other block in selected_set
            # This is slow. We need a better way.
            pass
        return count

    # Better approach for local search:
    # Maintain a coverage count array.
    # coverage[t_idx] = number of blocks in selected_set covering t_idx.
    
    def greedy_cover():
        uncovered = np.ones(n_triples, dtype=bool)
        selected = set()
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
            
            best_block = -1
            best_count = -1
            best_weight = -1
            
            # For each block, count how many uncovered triples it covers
            # We can precompute for each block the list of triples it covers.
            # To speed this up, we can iterate through blocks and check overlap with uncovered.
            
            # Optimization: 
            # Instead of checking all blocks every time, we can prioritize blocks 
            # that cover "rare" uncovered triples.
            
            # Simple greedy: pick block with max uncovered coverage.
            # To speed up, we can sample or use a heuristic.
            # Given n_blocks ~ 346k, iterating all is ~346k * 35 (avg triples in block) = 12M ops per step.
            # With ~70 steps, that's ~800M ops. Might be too slow in Python.
            
            # Let's use a weighted greedy that is faster.
            # We can maintain a list of "active" blocks? 
            # Or just use a simple heuristic: 
            # Pick a random uncovered triple, and pick a random block covering it? 
            # No, that's random.
            
            # Let's try to optimize the greedy step using NumPy.
            # uncovered is a boolean array of size n_triples.
            # block_triples is a list of lists.
            
            # We can create a matrix? No, too big (346k x 2024).
            
            # Let's stick to a Python loop but optimize it.
            # We can break early if we find a block covering all remaining? Unlikely.
            
            # Alternative: 
            # Precompute for each triple, the list of blocks.
            # Pick an uncovered triple t.
            # For each block b covering t, count how many OTHER uncovered triples it covers.
            # Pick the best b.
            # This reduces the search space from 346k to ~100-200 blocks per triple.
            
            # Find an uncovered triple
            # To be efficient, we can just pick the first one or a random one.
            # Let's pick the uncovered triple with the fewest covering blocks (most constrained).
            # We need to know which triples are uncovered.
            
            # This requires updating counts.
            # Let's use a simpler greedy first:
            # Iterate through all blocks, count uncovered.
            # This is O(N_blocks * C(k,t)).
            # 346104 * 35 = 12 million operations per step.
            # 70 steps * 12M = 840M operations.
            # In Python, 100M ops/sec is optimistic, maybe 10-20M.
            # So 840M / 20M = 42 seconds. This is borderline.
            
            # We need a faster greedy.
            
            # Heuristic: 
            # Maintain a list of uncovered triples.
            # Pick a random uncovered triple t.
            # Get all blocks covering t.
            # For each such block, count how many uncovered triples it covers.
            # Pick the best block.
            # Remove its triples from uncovered.
            
            uncovered_list = list(np.where(uncovered)[0])
            while uncovered_list:
                if time.time() > deadline:
                    break
                
                # Pick a random uncovered triple to guide the search
                # Or pick the one with fewest blocks? 
                # Let's pick random for now, it's fast.
                t_idx = uncovered_list[np.random.randint(len(uncovered_list))]
                
                # Get blocks covering t_idx
                candidate_blocks = triple_to_blocks[t_idx]
                
                best_block = -1
                best_count = -1
                
                for b_idx in candidate_blocks:
                    if b_idx in selected:
                        continue
                    
                    # Count how many uncovered triples this block covers
                    count = 0
                    for tt in block_triples[b_idx]:
                        if uncovered[tt]:
                            count += 1
                    
                    if count > best_count:
                        best_count = count
                        best_block = b_idx
                
                if best_block == -1:
                    # Should not happen if t_idx is uncovered
                    break
                    
                selected.add(best_block)
                
                # Update uncovered
                for tt in block_triples[best_block]:
                    if uncovered[tt]:
                        uncovered[tt] = False
                        # Remove from uncovered_list? 
                        # It's a list, removal is O(N). 
                        # Better to just leave it and filter when needed, 
                        # or use a set for uncovered.
                        
                # To avoid O(N) removal, let's use a set for uncovered indices.
                # But we need to pick random.
                # Let's just rebuild uncovered_list periodically or use a different structure.
                # Actually, checking `if uncovered[tt]` is fast.
                # The bottleneck is the inner loop `for tt in block_triples[b_idx]`.
                
                # We can speed up the count by using numpy?
                # block_triples[b_idx] is a list. 
                # We can convert to array once?
                # Let's precompute block_triples as numpy arrays.
                
        return list(selected)
    
    # Let's re-implement with NumPy arrays for block_triples to speed up the count.
    block_triples_np = np.array(block_triples, dtype=np.int32)
    # Note: This creates a jagged array if lengths differ, but they all have C(7,3)=35.
    # So it's a regular 2D array (n_blocks, 35).
    
    def greedy_cover_fast():
        uncovered = np.ones(n_triples, dtype=bool)
        selected = set()
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
            
            # Pick a random uncovered triple
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            t_idx = uncovered_indices[np.random.randint(len(uncovered_indices))]
            
            candidate_blocks = triple_to_blocks[t_idx]
            
            best_block = -1
            best_count = -1
            
            # Vectorize the count?
            # For each candidate block, we need sum(uncovered[block_triples_np[b_idx]])
            # We can do this in a loop, but use numpy for the sum.
            
            for b_idx in candidate_blocks:
                if b_idx in selected:
                    continue
                
                # Count uncovered triples in this block
                # block_triples_np[b_idx] is an array of 35 triple indices
                count = np.sum(uncovered[block_triples_np[b_idx]])
                
                if count > best_count:
                    best_count = count
                    best_block = b_idx
            
            if best_block == -1:
                break
                
            selected.add(best_block)
            
            # Update uncovered
            uncovered[block_triples_np[best_block]] = False
            
        return list(selected)
    
    def remove_redundant(selected_list):
        """
        Removes redundant blocks from selected_list.
        A block is redundant if all its triples are covered by other blocks.
        """
        selected_set = set(selected_list)
        if not selected_set:
            return []
        
        # Calculate coverage counts
        coverage = np.zeros(n_triples, dtype=np.int32)
        for b_idx in selected_set:
            coverage[block_triples_np[b_idx]] += 1
            
        removed = []
        changed = True
        while changed:
            changed = False
            for b_idx in list(selected_set):
                if time.time() > deadline:
                    break
                
                # Check if any triple in this block has coverage == 1
                # If all have coverage > 1, it's redundant.
                block_triples = block_triples_np[b_idx]
                if np.all(coverage[block_triples] > 1):
                    # Remove
                    coverage[block_triples] -= 1
                    selected_set.remove(b_idx)
                    changed = True
                    break # Restart loop to handle index changes and new redundancies
        
        return list(selected_set)
    
    def local_search(selected_list):
        """
        Performs local search to minimize the number of blocks.
        1. Remove redundancies.
        2. Try to swap blocks to enable further removals.
        """
        current = remove_redundant(selected_list)
        
        while True:
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try to find a swap that allows removal
            # For each block B in current:
            #   Let U_B be the set of triples covered ONLY by B.
            #   If U_B is empty, B is redundant (should have been removed).
            #   Find a block B' not in current such that U_B is a subset of B'.
            #   If found, swap B with B'.
            #   Then run remove_redundant again.
            
            current_set = set(current)
            coverage = np.zeros(n_triples, dtype=np.int32)
            for b_idx in current_set:
                coverage[block_triples_np[b_idx]] += 1
                
            for b_idx in list(current_set):
                if time.time() > deadline:
                    break
                
                block_triples = block_triples_np[b_idx]
                # Find triples covered only by b_idx
                critical_mask = (coverage[block_triples] == 1)
                if not np.any(critical_mask):
                    # This block is redundant, remove_redundant should have caught it
                    # But let's handle it just in case
                    coverage[block_triples] -= 1
                    current_set.remove(b_idx)
                    improved = True
                    break
                
                critical_triples = block_triples[critical_mask]
                
                # We need a block B' that covers all critical_triples.
                # Pick the rarest critical triple to limit candidates.
                # Count how many blocks cover each critical triple (excluding b_idx? No, total).
                # Actually, we just need B' to cover them.
                
                # Find the critical triple with the fewest covering blocks
                min_count = float('inf')
                min_triple = -1
                for t_idx in critical_triples:
                    count = len(triple_to_blocks[t_idx])
                    if count < min_count:
                        min_count = count
                        min_triple = t_idx
                        
                # Get candidates for min_triple
                candidates = triple_to_blocks[min_triple]
                
                valid_candidate = -1
                for cand in candidates:
                    if cand == b_idx or cand in current_set:
                        continue
                    
                    # Check if cand covers all other critical triples
                    # We can do this by checking intersection
                    # If all critical_triples are in block_triples_np[cand], it's valid.
                    # We can use set intersection or numpy isin.
                    # np.all(np.isin(critical_triples, block_triples_np[cand]))
                    
                    if np.all(np.isin(critical_triples, block_triples_np[cand])):
                        valid_candidate = cand
                        break
                        
                if valid_candidate != -1:
                    # Perform swap
                    coverage[block_triples] -= 1
                    coverage[block_triples_np[valid_candidate]] += 1
                    current_set.remove(b_idx)
                    current_set.add(valid_candidate)
                    improved = True
                    break
            
            if not improved:
                # No simple swap found. Try to remove redundancies again.
                new_current = remove_redundant(list(current_set))
                if len(new_current) < len(current):
                    current = new_current
                    improved = True
                else:
                    break # Stuck
            
            # If improved, loop continues. If not, we break.
            # Wait, if we swapped, we should check for new redundancies.
            # The loop restarts and does remove_redundant at the start of next iteration?
            # No, remove_redundant is only called at the start of local_search.
            # We should call it after a swap.
            
        return current
    
    # Run greedy to get initial solution
    initial_solution = greedy_cover_fast()
    
    # Refine
    best_solution = local_search(initial_solution)
    
    # Try a few more random restarts if time permits
    for i in range(10):
        if time.time() > deadline - 10:
            break
        
        # Slight variation in greedy: use different random seed or strategy
        # For now, just run greedy again
        sol = greedy_cover_fast()
        sol = local_search(sol)
        
        if len(sol) < len(best_solution):
            best_solution = sol
            
    return [list(blocks[i]) for i in best_solution]
