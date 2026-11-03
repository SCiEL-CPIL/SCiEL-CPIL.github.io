import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin for 150s limit (120s eval + buffer)

    # 1. Precomputation
    points = list(range(v))
    triples = list(itertools.combinations(points, t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    blocks = list(itertools.combinations(points, k))
    n_blocks = len(blocks)
    
    # C(k, t)
    c_k_t = 1
    for i in range(1, t+1):
        c_k_t = c_k_t * (k - i + 1) // i
        
    # block_triples[i] contains the indices of the t-subsets covered by blocks[i]
    # Shape: (n_blocks, c_k_t)
    block_triples = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    
    # Also compute frequency of each triple to use for weighting
    triple_freq = np.zeros(n_triples, dtype=np.int32)
    
    for i, block in enumerate(blocks):
        idx = 0
        for triple in itertools.combinations(block, t):
            t_idx = triple_to_idx[triple]
            block_triples[i, idx] = t_idx
            triple_freq[t_idx] += 1
            idx += 1
            
    # Weights: 1 / frequency. Higher weight for rarer triples.
    weights = 1.0 / np.maximum(triple_freq, 1)
    
    best_solution = None
    best_size = float('inf')
    
    # Local search polish: try to remove redundant blocks
    def local_search_polish(selected_indices, block_triples, n_triples, c_k_t):
        """Try to reduce solution size by removing redundant blocks."""
        if len(selected_indices) <= 1:
            return selected_indices
        
        current = list(selected_indices)
        improved = True
        max_iterations = 100  # Increased iterations
        
        while improved and max_iterations > 0:
            improved = False
            max_iterations -= 1
            
            # Compute cover counts
            cover_count = np.zeros(n_triples, dtype=np.int32)
            for b_idx in current:
                cover_count[block_triples[b_idx]] += 1
            
            # Try removing each block if redundant
            for i in range(len(current)):
                b_idx = current[i]
                b_triples = block_triples[b_idx]
                if np.min(cover_count[b_triples]) > 1:
                    # Remove this block
                    cover_count[b_triples] -= 1
                    current.pop(i)
                    improved = True
                    break
        
        return current

    # Weighted greedy with randomization
    def weighted_greedy_cover(weights, block_triples, n_triples, temperature=0.0):
        uncovered = np.ones(n_triples, dtype=bool)
        selected = []
        
        while np.any(uncovered):
            weighted_uncovered = np.where(uncovered, weights, 0.0)
            gains = np.sum(weighted_uncovered[block_triples], axis=1)
            
            if temperature > 0:
                # Softmax sampling for diversity
                max_gain = np.max(gains)
                if max_gain > 0:
                    # Scale temperature to gain magnitude for stability
                    exp_gains = np.exp((gains - max_gain) / temperature)
                    probs = exp_gains / np.sum(exp_gains)
                    best_idx = np.random.choice(n_blocks, p=probs)
                else:
                    best_idx = np.argmax(gains)
            else:
                best_idx = np.argmax(gains)
            
            if gains[best_idx] == 0:
                break
            
            selected.append(best_idx)
            uncovered[block_triples[best_idx]] = False
        
        return selected

    # New: Simulated Annealing / Local Search for block swaps
    # This takes a valid covering and tries to swap blocks to reduce size or maintain size while allowing future removals
    # We focus on "swap" moves: replace one block with another. If the new block covers all triples of the old block (redundancy check), we can potentially remove the old one if it becomes redundant? 
    # Actually, a simpler move is: Remove a block. If coverage breaks, we must add a block that covers the missing triples. 
    # Let's implement a "1-1 swap" where we remove a block and add a different block, aiming to reduce the count or improve "slack".
    # However, reducing count directly is hard. 
    # Better strategy from Inspiration 2: Start with a valid cover. Try to remove blocks. If invalid, try to swap.
    
    def sa_refine(initial_selected, block_triples, n_triples, c_k_t, time_limit):
        """
        Refine a selected list of block indices using simulated annealing.
        Moves:
        1. Remove a block (if valid).
        2. Swap a block: Remove A, Add B. Valid if coverage is maintained.
        Objective: Minimize len(selected).
        """
        current = list(initial_selected)
        current_size = len(current)
        best = list(current)
        best_size = current_size
        
        # Precompute cover counts
        cover_count = np.zeros(n_triples, dtype=np.int32)
        for b_idx in current:
            cover_count[block_triples[b_idx]] += 1
            
        temperature = 1.0
        cooling = 0.99
        min_temp = 0.01
        no_improve = 0
        max_no_improve = 200
        
        start_sa = time.time()
        
        while time.time() - start_sa < time_limit and temperature > min_temp:
            # Pick a random block to remove
            i = np.random.randint(0, len(current))
            b_remove = current[i]
            
            # Simulate removal
            missing = block_triples[b_remove][cover_count[block_triples[b_remove]] == 1]
            
            if len(missing) == 0:
                # Removal is valid
                cover_count[block_triples[b_remove]] -= 1
                current.pop(i)
                current_size -= 1
                
                if current_size < best_size:
                    best_size = current_size
                    best = list(current)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Need to find a block that covers all 'missing' triples
                # Heuristic: Find block with max overlap with missing
                # missing is an array of triple indices
                # For each block, count how many missing triples it covers
                # This is expensive if done naively. 
                # Optimization: Only consider blocks that contain at least one point from the removed block? No, any block.
                
                # We can use the precomputed block_triples.
                # gains_for_missing[j] = sum( (block_triples[j] is in missing_set) )
                
                # To speed up, we can sample a few candidate blocks or use a heuristic.
                # Let's try a few random blocks and the one with max overall gain.
                
                # Find the block that covers the most missing triples
                # We can do this by checking intersection with missing set.
                # Since missing is small (usually), we can check candidates.
                
                # Candidate 1: The block that covers the most total triples (precomputed? no, dynamic)
                # Let's just pick a random block and see if it covers missing.
                # Or better: pick the block that shares the most points with the removed block?
                
                # Efficient check:
                # Create a boolean mask for missing triples
                missing_mask = np.zeros(n_triples, dtype=bool)
                missing_mask[missing] = True
                
                # Count missing covered by each block
                # This is (n_blocks, c_k_t) boolean sum. Expensive.
                # We can approximate by checking only blocks that contain a point from the removed block.
                # Or just try a few random blocks.
                
                found_swap = False
                # Try 10 random blocks
                for _ in range(10):
                    b_add = np.random.randint(0, n_blocks)
                    if b_add == b_remove:
                        continue
                    
                    # Check if b_add covers all missing
                    add_triples = block_triples[b_add]
                    # Check if all missing are in add_triples
                    # np.isin(missing, add_triples) - expensive
                    # Better: create a set of add_triples? No, numpy.
                    # Count how many missing are in add_triples
                    # We can use a precomputed mapping? 
                    # Let's use a simpler heuristic: 
                    # If the removed block is unique for some triples, we need a block that covers them.
                    
                    # Fast check: 
                    # If missing is empty, handled above.
                    # Check if add_triples covers missing.
                    # We can do this by checking if the intersection size equals len(missing).
                    # Since missing is small, we can check each missing triple.
                    
                    valid_swap = True
                    for m in missing:
                        # Check if m is in add_triples
                        # This is O(c_k_t) per missing triple. c_k_t is 35. missing is usually small.
                        if not np.any(add_triples == m):
                            valid_swap = False
                            break
                    
                    if valid_swap:
                        # Perform swap
                        # Remove b_remove
                        cover_count[block_triples[b_remove]] -= 1
                        current.pop(i)
                        
                        # Add b_add
                        # Check if b_add is already in current? 
                        # If yes, we just removed one, so it's not there. 
                        # But what if b_add was already in current? 
                        # If b_add was in current, removing b_remove didn't affect b_add.
                        # So we can just add it.
                        current.append(b_add)
                        cover_count[block_triples[b_add]] += 1
                        
                        found_swap = True
                        break
                
                if not found_swap:
                    # Try a greedy best swap
                    # Find the block that covers the most missing triples
                    # We can compute this for all blocks if time permits, but let's just skip this iteration
                    # Or use a simpler heuristic: pick a block that covers the "hardest" missing triple
                    # Let's just pass for now to save time.
                    pass
            
            temperature *= cooling
            
            if no_improve > max_no_improve:
                break
                
        return best

    # Run multiple greedy passes with different weight perturbations
    n_restarts = 0
    while time.time() < deadline - 10.0:
        n_restarts += 1
        
        # Perturb weights slightly for diversity
        if n_restarts == 1:
            perturbed_weights = weights.copy()
        else:
            noise = np.random.normal(0, 0.1, size=n_triples)
            perturbed_weights = weights * (1 + noise)
            perturbed_weights = np.maximum(perturbed_weights, 0.01)
        
        # Run greedy
        selected = weighted_greedy_cover(perturbed_weights, block_triples, n_triples)
        
        # Polish
        selected = local_search_polish(selected, block_triples, n_triples, c_k_t)
        
        # SA Refine if time permits
        if time.time() < deadline - 15.0:
            time_left_for_sa = min(5.0, deadline - 15.0 - time.time())
            if time_left_for_sa > 1.0:
                selected = sa_refine(selected, block_triples, n_triples, c_k_t, time_left_for_sa)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_solution = [list(blocks[i]) for i in selected]
        
        # Early exit if we hit the lower bound (69)
        if best_size <= 69:
            break
    
    if best_solution is None:
        # Fallback: return a valid covering
        selected = weighted_greedy_cover(weights, block_triples, n_triples)
        selected = local_search_polish(selected, block_triples, n_triples, c_k_t)
        best_solution = [list(blocks[i]) for i in selected]
        
    return best_solution
