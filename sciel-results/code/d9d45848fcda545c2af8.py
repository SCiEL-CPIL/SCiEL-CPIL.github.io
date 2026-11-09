"""Improved probabilistic construction: Multi-start Greedy + Aggressive Local Search (Redundancy, 1-for-1, 2-for-1)."""
import itertools
import random
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t).
    Strategy:
    1. Precompute all k-subsets and their covered t-subsets as bitsets.
    2. Run multiple Greedy constructions with different random seeds.
    3. Apply aggressive Local Search to the best candidates:
       - Remove redundant blocks.
       - 1-for-1 block replacement to reduce overlap.
       - 2-for-1 block merging to reduce block count.
    """
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
    # Precompute list of tsub indices for each block (for local search cost calculation)
    block_tsubs = []
    
    # Map tsub index to list of block indices that cover it
    triple_to_blocks = [[] for _ in range(n_tsubs)]
    
    for i, block in enumerate(all_blocks):
        bitset = 0
        tsubs_in_block = []
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[ts]
            bitset |= (1 << idx)
            tsubs_in_block.append(idx)
            triple_to_blocks[idx].append(i)
        block_bitsets[i] = bitset
        block_tsubs.append(tsubs_in_block)

    def run_greedy(seed):
        random.seed(seed)
        uncovered_mask = (1 << n_tsubs) - 1
        solution = []
        
        while uncovered_mask:
            best_block_idx = -1
            best_gain = -1
            best_candidates = []
            
            for i in range(n_blocks):
                gain = (block_bitsets[i] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)
            
            if not best_candidates:
                break # Should not happen
                
            # Pick randomly from best candidates to add diversity
            chosen_idx = random.choice(best_candidates)
            solution.append(chosen_idx)
            uncovered_mask &= ~block_bitsets[chosen_idx]
            
        return solution

    def local_search(solution):
        # Maintain coverage counts
        coverage_counts = np.zeros(n_tsubs, dtype=np.int32)
        for b_idx in solution:
            for t_idx in block_tsubs[b_idx]:
                coverage_counts[t_idx] += 1
        
        max_iterations = 100
        for _ in range(max_iterations):
            improved = False
            # Shuffle solution order to avoid bias
            random.shuffle(solution)
            
            # Phase 2a: Redundancy Removal and 1-for-1 Swap
            for i in range(len(solution) - 1, -1, -1):
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
                    continue
                else:
                    # Find replacement block
                    # Must cover all triples in newly_uncovered
                    candidates = set(triple_to_blocks[newly_uncovered[0]])
                    for t_idx in newly_uncovered[1:]:
                        candidates.intersection_update(triple_to_blocks[t_idx])
                    
                    # Exclude the block we just removed
                    candidates.discard(b_idx)
                    
                    if not candidates:
                        # No single block can cover all newly uncovered triples
                        # Restore original
                        for t_idx in block_tsubs[b_idx]:
                            coverage_counts[t_idx] += 1
                        continue
                    
                    # Find candidate with minimum overlap with current coverage
                    best_candidate = -1
                    best_cost = float('inf')
                    
                    for c_idx in candidates:
                        cost = 0
                        for t_idx in block_tsubs[c_idx]:
                            cost += coverage_counts[t_idx]
                        if cost < best_cost:
                            best_cost = cost
                            best_candidate = c_idx
                    
                    if best_candidate != -1:
                        # Replace b_idx with best_candidate
                        solution[i] = best_candidate
                        for t_idx in block_tsubs[best_candidate]:
                            coverage_counts[t_idx] += 1
                        improved = True
                    else:
                        # Restore original
                        for t_idx in block_tsubs[b_idx]:
                            coverage_counts[t_idx] += 1

            # Phase 2b: 2-for-1 Merge
            # Try to remove two blocks and replace with one
            n_sol = len(solution)
            if n_sol > 2:
                # To avoid O(N^2) full scan every iteration if N is large, 
                # but N is ~70-80, so N^2 ~ 6400. This is fine.
                for i in range(n_sol):
                    for j in range(i + 1, n_sol):
                        b1 = solution[i]
                        b2 = solution[j]
                        
                        # Temporarily remove both
                        for t_idx in block_tsubs[b1]:
                            coverage_counts[t_idx] -= 1
                        for t_idx in block_tsubs[b2]:
                            coverage_counts[t_idx] -= 1
                        
                        # Find triples that are now uncovered
                        newly_uncovered = []
                        # Union of tsubs from b1 and b2
                        # We can just check all tsubs in b1 and b2
                        seen = set()
                        for t_idx in block_tsubs[b1]:
                            if t_idx not in seen:
                                seen.add(t_idx)
                                if coverage_counts[t_idx] == 0:
                                    newly_uncovered.append(t_idx)
                        for t_idx in block_tsubs[b2]:
                            if t_idx not in seen:
                                seen.add(t_idx)
                                if coverage_counts[t_idx] == 0:
                                    newly_uncovered.append(t_idx)
                        
                        if not newly_uncovered:
                            # Both blocks were redundant? 
                            # This implies they were covering the same things and nothing else needed them.
                            # Remove both.
                            # Note: solution indices change. 
                            # It's safer to mark for removal and process after loop, 
                            # or just pop in reverse order.
                            # Since we are in a nested loop, popping is tricky.
                            # Let's just do a swap/merge if we find a replacement.
                            # If newly_uncovered is empty, we can remove both.
                            # But we need to restore counts if we don't add a replacement.
                            # Actually, if newly_uncovered is empty, we can just remove both.
                            # But we are in the middle of the loop.
                            # Let's stick to finding a replacement block for the union.
                            # If no replacement is needed (i.e., union is covered by others), 
                            # we can remove both.
                            
                            # To keep it simple and safe, let's only do the "Replace with 1" case.
                            # If newly_uncovered is empty, we can remove both.
                            # We'll handle removal by setting them to a dummy value and filtering later?
                            # No, let's just try to find a replacement block for the union.
                            # If newly_uncovered is empty, the "replacement" is nothing.
                            
                            # Let's check if we can find a single block covering the union of newly_uncovered.
                            # If newly_uncovered is empty, we can just remove both.
                            # We will mark this as a success and handle removal carefully.
                            
                            pass # We will implement the replacement logic below.
                        
                        # Find a single block covering all newly_uncovered
                        if newly_uncovered:
                            candidates = set(triple_to_blocks[newly_uncovered[0]])
                            for t_idx in newly_uncovered[1:]:
                                candidates.intersection_update(triple_to_blocks[t_idx])
                            
                            # Exclude b1 and b2
                            candidates.discard(b1)
                            candidates.discard(b2)
                            
                            if candidates:
                                # Pick best candidate (min overlap)
                                best_candidate = -1
                                best_cost = float('inf')
                                for c_idx in candidates:
                                    cost = 0
                                    for t_idx in block_tsubs[c_idx]:
                                        cost += coverage_counts[t_idx]
                                    if cost < best_cost:
                                        best_cost = cost
                                        best_candidate = c_idx
                                
                                if best_candidate != -1:
                                    # Replace b1 and b2 with best_candidate
                                    # Remove b1 and b2 from solution
                                    # Add best_candidate
                                    # Update counts
                                    
                                    # We need to restore counts for b1, b2 first? 
                                    # No, they are currently removed (counts decremented).
                                    # We just add best_candidate.
                                    
                                    # Update solution list
                                    # Remove indices i and j
                                    # Note: if i < j, removing i shifts j to j-1.
                                    # We are in a loop over indices, so modifying list is dangerous.
                                    # We will collect changes and apply them after the loop?
                                    # Or just break and restart iteration?
                                    
                                    # Let's apply immediately but break the loops.
                                    solution[i] = best_candidate
                                    solution.pop(j)
                                    # Now solution has best_candidate at i, and old j is gone.
                                    # We need to update coverage_counts: 
                                    # b1, b2 are already decremented.
                                    # We increment for best_candidate.
                                    for t_idx in block_tsubs[best_candidate]:
                                        coverage_counts[t_idx] += 1
                                    
                                    improved = True
                                    break # Break inner loop
                        else:
                            # newly_uncovered is empty
                            # We can remove both b1 and b2
                            # Remove j then i (since i < j)
                            solution.pop(j)
                            solution.pop(i)
                            improved = True
                            break # Break inner loop
                        
                        # Restore counts if no change made
                        for t_idx in block_tsubs[b1]:
                            coverage_counts[t_idx] += 1
                        for t_idx in block_tsubs[b2]:
                            coverage_counts[t_idx] += 1
                    
                    if improved:
                        break # Break outer loop if inner broke due to improvement
                
            if not improved:
                break
        
        return solution

    best_solution = None
    best_len = float('inf')
    
    # Run multiple greedy starts
    num_starts = 20
    for s in range(num_starts):
        sol = run_greedy(s)
        sol = local_search(sol)
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol
            # Optional: early stop if we hit a known lower bound or good number
            # Lower bound is 69. If we hit 70, we can stop? 
            # But we want to minimize.
    
    return [list(all_blocks[i]) for i in best_solution]
