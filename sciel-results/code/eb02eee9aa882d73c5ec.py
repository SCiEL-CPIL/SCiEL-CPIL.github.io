"""Improved probabilistic construction: Johnson-scheme-inspired Greedy + Optimized Local Search."""
import itertools
import random
import numpy as np
import time

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t).
    Strategy:
    1. Precompute all k-subsets and their covered t-subsets as bitsets.
    2. Greedy construction using bitset popcount for fast gain calculation, biased by random point weights (Johnson-scheme inspired).
    3. Local search: 
       a. Remove redundant blocks.
       b. Replace blocks with lower-overlap alternatives covering exclusive triples.
       c. Attempt 2-for-1 merges (replace 2 blocks with 1) for pairs with high overlap.
    """
    start_time = time.time()
    timeout = 110 # Leave 10s buffer
    
    # Precomputation
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Generate all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute bitsets for each block
    # bitset[i] is an integer where bit j is 1 if tsub j is covered by block i
    block_bitsets = [0] * n_blocks
    # Precompute list of tsub indices for each block
    block_tsubs = []
    
    # Map tsub index to list of block indices that cover it
    triple_to_blocks = [[] for _ in range(n_tsubs)]
    
    # Precompute point weights for bias
    random.seed(42)
    point_weights = np.random.rand(v)
    
    block_weight_sums = [0.0] * n_blocks
    
    for i, block in enumerate(all_blocks):
        bitset = 0
        tsubs_in_block = []
        weight_sum = 0.0
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[ts]
            bitset |= (1 << idx)
            tsubs_in_block.append(idx)
            triple_to_blocks[idx].append(i)
        # Calculate weight sum for the block's points
        for p in block:
            weight_sum += point_weights[p]
            
        block_bitsets[i] = bitset
        block_tsubs.append(tsubs_in_block)
        block_weight_sums[i] = weight_sum
        
    # Phase 1: Greedy Construction with Bias
    uncovered_mask = (1 << n_tsubs) - 1
    solution = []
    alpha = 0.5 # Bias factor
    
    while uncovered_mask:
        best_block_idx = -1
        best_score = -1
        best_candidates = []
        
        for i in range(n_blocks):
            gain = (block_bitsets[i] & uncovered_mask).bit_count()
            if gain == 0:
                continue
            
            # Score: Gain + Alpha * Weight
            # We want to maximize this
            score = gain + alpha * block_weight_sums[i]
            
            if score > best_score:
                best_score = score
                best_candidates = [i]
            elif score == best_score:
                best_candidates.append(i)
                
        if not best_candidates:
            break # Should not happen
            
        # Pick randomly from best candidates to add diversity
        chosen_idx = random.choice(best_candidates)
        solution.append(chosen_idx)
        uncovered_mask &= ~block_bitsets[chosen_idx]
        
    # Phase 2: Local Search Minimization
    # Maintain coverage counts
    coverage_counts = np.zeros(n_tsubs, dtype=np.int32)
    for b_idx in solution:
        for t_idx in block_tsubs[b_idx]:
            coverage_counts[t_idx] += 1
            
    # Helper function to check if we should continue
    def time_check():
        return time.time() - start_time < timeout

    max_iterations = 50
    for _ in range(max_iterations):
        if not time_check():
            break
            
        improved = False
        
        # --- Step 1: Remove Redundant Blocks ---
        i = 0
        while i < len(solution):
            if not time_check():
                break
            b_idx = solution[i]
            
            # Temporarily remove block b_idx
            for t_idx in block_tsubs[b_idx]:
                coverage_counts[t_idx] -= 1
            
            # Find triples that are now uncovered (count == 0)
            newly_uncovered = []
            for t_idx in block_tsubs[b_idx]:
                if coverage_counts[t_idx] == 0:
                    newly_uncovered.append(t_idx)
            
            if not newly_uncovered:
                # Block was redundant, remove it permanently
                solution.pop(i)
                improved = True
                # Do not increment i, check the next element which shifted into this position
            else:
                # Restore original
                for t_idx in block_tsubs[b_idx]:
                    coverage_counts[t_idx] += 1
                i += 1
                
        # --- Step 2: 1-for-1 Replacement (Optimize Overlap) ---
        for i in range(len(solution)):
            if not time_check():
                break
            b_idx = solution[i]
            
            # Temporarily remove
            for t_idx in block_tsubs[b_idx]:
                coverage_counts[t_idx] -= 1
            
            newly_uncovered = []
            for t_idx in block_tsubs[b_idx]:
                if coverage_counts[t_idx] == 0:
                    newly_uncovered.append(t_idx)
            
            if newly_uncovered:
                # Find candidates covering all newly_uncovered
                candidates = set(triple_to_blocks[newly_uncovered[0]])
                for t_idx in newly_uncovered[1:]:
                    candidates.intersection_update(triple_to_blocks[t_idx])
                candidates.discard(b_idx)
                
                if candidates:
                    # Find candidate with minimum overlap (sum of coverage_counts)
                    best_candidate = -1
                    best_cost = float('inf')
                    
                    # Calculate current block's cost (overlap with others)
                    cur_cost = 0
                    for t_idx in block_tsubs[b_idx]:
                        cur_cost += coverage_counts[t_idx]
                        
                    for c_idx in candidates:
                        cost = 0
                        for t_idx in block_tsubs[c_idx]:
                            cost += coverage_counts[t_idx]
                        if cost < best_cost:
                            best_cost = cost
                            best_candidate = c_idx
                            
                    if best_candidate != -1 and best_cost < cur_cost:
                        # Replace
                        solution[i] = best_candidate
                        for t_idx in block_tsubs[best_candidate]:
                            coverage_counts[t_idx] += 1
                        improved = True
                    else:
                        # Restore
                        for t_idx in block_tsubs[b_idx]:
                            coverage_counts[t_idx] += 1
                else:
                    # No candidate, restore
                    for t_idx in block_tsubs[b_idx]:
                        coverage_counts[t_idx] += 1
            else:
                # Redundant, remove
                solution.pop(i)
                improved = True
                # Restart loop as indices changed
                break
                
        # --- Step 3: 2-for-1 Merge (Limited) ---
        # Try to merge pairs of blocks that have high overlap
        # We only try this if we have time and the solution is not too large
        if time_check() and len(solution) > 5:
            # Sort blocks by size of their exclusive part? No, just random pairs
            # To be efficient, we only check pairs that share at least 1 triple
            # Build a map of triples to blocks in current solution
            sol_set = set(solution)
            
            # For each block, find other blocks that share a triple
            # This is expensive to build from scratch each iteration.
            # We can just pick random pairs and check.
            
            # Let's try a few random pairs
            for _ in range(50): # Limit attempts per iteration
                if not time_check():
                    break
                if len(solution) < 2:
                    break
                i, j = random.sample(range(len(solution)), 2)
                b1 = solution[i]
                b2 = solution[j]
                
                # Calculate exclusive triples for b1 and b2
                # Exclusive to b1: covered by b1, not by any other block in solution
                # Exclusive to b2: covered by b2, not by any other block in solution
                # Union of exclusive: E = (E1 U E2)
                
                # Efficient way:
                # Temporarily remove b1 and b2
                for t_idx in block_tsubs[b1]:
                    coverage_counts[t_idx] -= 1
                for t_idx in block_tsubs[b2]:
                    coverage_counts[t_idx] -= 1
                    
                # Find triples that are now uncovered
                newly_uncovered = set()
                for t_idx in block_tsubs[b1]:
                    if coverage_counts[t_idx] == 0:
                        newly_uncovered.add(t_idx)
                for t_idx in block_tsubs[b2]:
                    if coverage_counts[t_idx] == 0:
                        newly_uncovered.add(t_idx)
                        
                if newly_uncovered:
                    # Find a single block that covers all these
                    # Use the smallest list of blocks for the first triple
                    first_t = next(iter(newly_uncovered))
                    candidates = set(triple_to_blocks[first_t])
                    for t_idx in list(newly_uncovered)[1:]:
                        candidates.intersection_update(triple_to_blocks[t_idx])
                    
                    # Exclude b1 and b2
                    candidates.discard(b1)
                    candidates.discard(b2)
                    
                    if candidates:
                        # Pick the one with lowest overlap
                        best_candidate = -1
                        best_cost = float('inf')
                        
                        # Cost of merging: 0 (we remove 2, add 1)
                        # But we want to minimize the overlap of the new block
                        for c_idx in candidates:
                            cost = 0
                            for t_idx in block_tsubs[c_idx]:
                                cost += coverage_counts[t_idx]
                            if cost < best_cost:
                                best_cost = cost
                                best_candidate = c_idx
                                
                        if best_candidate != -1:
                            # Replace b1, b2 with best_candidate
                            # Remove b1, b2 from solution
                            # We need to be careful with indices
                            # Let's just mark them and rebuild? 
                            # Or swap with last and pop?
                            
                            # Simple approach:
                            # Remove b1 and b2
                            # If i > j, adjust i
                            if i > j:
                                i -= 1
                            solution.pop(i)
                            solution.pop(j)
                            solution.append(best_candidate)
                            
                            # Update coverage counts
                            for t_idx in block_tsubs[best_candidate]:
                                coverage_counts[t_idx] += 1
                            
                            improved = True
                            break # Restart the pair search loop or break out?
                            # Let's break out of the random pair loop and restart the main iteration
                        else:
                            # Restore
                            for t_idx in block_tsubs[b1]:
                                coverage_counts[t_idx] += 1
                            for t_idx in block_tsubs[b2]:
                                coverage_counts[t_idx] += 1
                    else:
                        # Restore
                        for t_idx in block_tsubs[b1]:
                            coverage_counts[t_idx] += 1
                        for t_idx in block_tsubs[b2]:
                            coverage_counts[t_idx] += 1
                else:
                    # Both blocks were redundant? Should have been caught in Step 1
                    # But maybe they became redundant due to each other?
                    # If newly_uncovered is empty, it means b1 and b2 were covering triples
                    # that are all covered by other blocks.
                    # So we can just remove both.
                    if i > j:
                        i -= 1
                    solution.pop(i)
                    solution.pop(j)
                    improved = True
                    break
            if improved:
                # Restart the main iteration loop to handle index changes and new states
                continue
                
        if not improved:
            break
            
    # Convert block indices back to lists of points
    return [list(all_blocks[i]) for i in solution]
