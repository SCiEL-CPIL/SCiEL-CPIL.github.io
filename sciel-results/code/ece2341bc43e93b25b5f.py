import itertools
import time
import numpy as np
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # 1. Precomputation
    # All t-subsets
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
    
    # All k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # block_to_tsets: list of lists of tset indices
    # tset_to_blocks: list of lists of block indices
    block_to_tsets = [[] for _ in range(num_blocks)]
    tset_to_blocks = [[] for _ in range(num_tsets)]
    
    for b_idx, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            ts_idx = tset_to_idx[ts]
            block_to_tsets[b_idx].append(ts_idx)
            tset_to_blocks[ts_idx].append(b_idx)
    
    # Convert to numpy arrays for faster operations if needed, 
    # but lists are often faster for sparse access in Python loops.
    # Let's stick to lists for the core logic but use sets for fast intersection.
    
    # Helper to get a valid solution from block indices
    def get_solution(indices):
        return [list(all_blocks[i]) for i in indices]

    def check_validity(indices):
        # Exhaustive check is O(num_tsets * avg_blocks_per_tset)
        # We maintain counts, so we just check if all counts > 0
        counts = np.zeros(num_tsets, dtype=np.int32)
        for b_idx in indices:
            for ts_idx in block_to_tsets[b_idx]:
                counts[ts_idx] += 1
        return np.all(counts > 0)

    def run_local_search(initial_indices, max_time_per_run=10.0):
        """
        Takes a list of block indices, applies local search improvements.
        Returns the best list of indices found.
        """
        current = list(initial_indices)
        if not current:
            return []
        
        # Calculate initial cover counts
        cover_counts = np.zeros(num_tsets, dtype=np.int32)
        for b_idx in current:
            for ts_idx in block_to_tsets[b_idx]:
                cover_counts[ts_idx] += 1
        
        best_size = len(current)
        best_solution = list(current)
        
        # To allow 1-for-1 swaps, we need to know which blocks cover specific tsets.
        # We have tset_to_blocks.
        
        improved = True
        iterations = 0
        local_start = time.time()
        
        while improved and (time.time() - local_start) < max_time_per_run:
            improved = False
            iterations += 1
            
            # Phase 1: Remove redundant blocks
            # A block is redundant if all tsets it covers have count > 1
            # We iterate in reverse to safely pop
            for i in range(len(current) - 1, -1, -1):
                b_idx = current[i]
                is_redundant = True
                for ts_idx in block_to_tsets[b_idx]:
                    if cover_counts[ts_idx] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove it
                    for ts_idx in block_to_tsets[b_idx]:
                        cover_counts[ts_idx] -= 1
                    current.pop(i)
                    improved = True
                    if len(current) < best_size:
                        best_size = len(current)
                        best_solution = list(current)
            
            if not current:
                break
                
            # Phase 2: 1-for-1 Swaps
            # We try to swap a block B_out with B_in if B_in covers all tsets 
            # that are currently uniquely covered by B_out.
            # To make the swap beneficial for *future* reductions, we prefer B_in 
            # that covers many tsets with count == 1 (other than those in B_out) 
            # or generally covers a lot.
            
            # Shuffle to avoid bias
            random.shuffle(current)
            
            for i in range(len(current)):
                if (time.time() - local_start) > max_time_per_run:
                    break
                    
                b_out = current[i]
                
                # Find tsets uniquely covered by b_out
                unique_tsets = []
                for ts_idx in block_to_tsets[b_out]:
                    if cover_counts[ts_idx] == 1:
                        unique_tsets.append(ts_idx)
                        
                if not unique_tsets:
                    # This block is redundant, should have been caught in Phase 1
                    # But if Phase 1 was interrupted or logic changed, handle it
                    continue

                # Find candidate blocks B_in that cover all unique_tsets
                # Start with the list of blocks covering the first unique tset
                # Then intersect with others.
                candidates = set(tset_to_blocks[unique_tsets[0]])
                candidates.discard(b_out)
                
                for ts_idx in unique_tsets[1:]:
                    if not candidates:
                        break
                    candidates.intersection_update(tset_to_blocks[ts_idx])
                
                if not candidates:
                    continue
                
                # Evaluate candidates
                # Score: Number of tsets in B_in that have count == 1 
                # (excluding those already in unique_tsets, which are forced)
                # Actually, simpler: Score = Total tsets covered by B_in that are NOT in B_out.
                # This increases the "density" of coverage, potentially making other blocks redundant.
                
                best_candidate = -1
                best_score = -1
                b_out_tsets_set = set(block_to_tsets[b_out])
                
                # Limit candidates to check to save time if list is huge
                candidate_list = list(candidates)
                if len(candidate_list) > 50:
                    random.shuffle(candidate_list)
                    candidate_list = candidate_list[:50]
                    
                for b_in in candidate_list:
                    # Calculate score
                    # Count tsets in b_in that are NOT in b_out
                    score = 0
                    for ts_idx in block_to_tsets[b_in]:
                        if ts_idx not in b_out_tsets_set:
                            score += 1
                            # Optional: prioritize tsets with low count
                            # if cover_counts[ts_idx] == 1: score += 10
                            
                    if score > best_score:
                        best_score = score
                        best_candidate = b_in
                        
                if best_candidate != -1:
                    # Perform swap
                    # Update cover_counts
                    for ts_idx in block_to_tsets[b_out]:
                        cover_counts[ts_idx] -= 1
                    for ts_idx in block_to_tsets[best_candidate]:
                        cover_counts[ts_idx] += 1
                    
                    current[i] = best_candidate
                    improved = True
                    # Note: We don't update best_solution here because size didn't change.
                    # But it might enable future removals.
                    
            # If we made swaps, loop back to Phase 1 to remove any newly redundant blocks.
            
        return best_solution

    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = np.ones(num_tsets, dtype=np.bool_)
        uncovered_count = num_tsets
        selected = []
        
        # Weights: 1 / frequency of tset
        tset_counts = np.zeros(num_tsets, dtype=np.int32)
        for b_idx in range(num_blocks):
            for ts_idx in block_to_tsets[b_idx]:
                tset_counts[ts_idx] += 1
        safe_counts = np.maximum(tset_counts, 1).astype(np.float64)
        tset_weights = 1.0 / safe_counts
        
        # Initial gains
        gains = np.zeros(num_blocks, dtype=np.float64)
        for b_idx in range(num_blocks):
            if block_to_tsets[b_idx]:
                gains[b_idx] = np.sum(tset_weights[block_to_tsets[b_idx]])
                
        selected_mask = np.zeros(num_blocks, dtype=np.bool_)
        
        while uncovered_count > 0 and time.time() < deadline - 15:
            # Find best block
            # We can use numpy argmax on a masked view, but updating gains is the bottleneck.
            # Let's stick to the loop but optimize it.
            best_idx = -1
            best_gain = -1.0
            
            # Optimization: If we have a priority queue, this is O(log N).
            # But updating N elements is O(N).
            # Given N=75k, a simple scan is ~0.01s. 
            # We expect ~300-500 selections. 500 * 0.01 = 5s. Acceptable.
            
            for i in range(num_blocks):
                if not selected_mask[i] and gains[i] > best_gain:
                    best_gain = gains[i]
                    best_idx = i
                    
            if best_idx == -1 or best_gain <= 0:
                break
                
            selected.append(best_idx)
            selected_mask[best_idx] = True
            
            # Update
            for ts_idx in block_to_tsets[best_idx]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    uncovered_count -= 1
                    w = tset_weights[ts_idx]
                    for b in tset_to_blocks[ts_idx]:
                        if not selected_mask[b]:
                            gains[b] -= w
                            
        return selected

    best_global_solution = None
    best_global_size = float('inf')
    
    # Strategy: Run multiple greedy+local search instances with different seeds
    # and time limits.
    
    # Run 1: Standard greedy
    t_start = time.time()
    init1 = greedy_cover(42)
    if init1:
        sol1 = run_local_search(init1, max_time_per_run=min(20.0, deadline - t_start - 5))
        if sol1 and len(sol1) < best_global_size:
            best_global_size = len(sol1)
            best_global_solution = sol1
            
    # Run 2: Greedy with different seed
    t_start = time.time()
    if time.time() < deadline - 30:
        init2 = greedy_cover(123)
        if init2:
            sol2 = run_local_search(init2, max_time_per_run=min(20.0, deadline - t_start - 5))
            if sol2 and len(sol2) < best_global_size:
                best_global_size = len(sol2)
                best_global_solution = sol2
                
    # Run 3: Try a random initialization? No, too large.
    # Instead, try local search on the best solution found with more time if available.
    
    if best_global_solution and time.time() < deadline - 10:
        remaining_time = deadline - time.time() - 2
        # Re-run local search on the best solution with remaining time
        improved_sol = run_local_search(best_global_solution, max_time_per_run=remaining_time)
        if improved_sol and len(improved_sol) < best_global_size:
            best_global_size = len(improved_sol)
            best_global_solution = improved_sol
            
    if not best_global_solution:
        # Fallback: Return a valid but potentially large solution
        # Just return the first greedy result if it exists
        if init1:
            return get_solution(init1)
        return []
        
    return get_solution(best_global_solution)

if __name__ == "__main__":
    # Test with the specific parameters
    v, k, t = 19, 8, 5
    res = construct(v, k, t)
    print(f"Number of blocks: {len(res)}")
    # Verify
    covered = set()
    for block in res:
        for ts in itertools.combinations(block, t):
            covered.add(ts)
    total_tsets = len(list(itertools.combinations(range(v), t)))
    print(f"Covered: {len(covered)}/{total_tsets}")
