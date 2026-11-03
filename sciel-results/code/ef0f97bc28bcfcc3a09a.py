import itertools
import time
import numpy as np
import random
import sys
from collections import defaultdict

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
        
        improved = True
        local_start = time.time()
        
        while improved and (time.time() - local_start) < max_time_per_run:
            improved = False
            
            # Phase 1: Remove redundant blocks (multiple passes until stable)
            while True:
                found_redundant = False
                for i in range(len(current) - 1, -1, -1):
                    if (time.time() - local_start) > max_time_per_run:
                        break
                    b_idx = current[i]
                    is_redundant = True
                    for ts_idx in block_to_tsets[b_idx]:
                        if cover_counts[ts_idx] == 1:
                            is_redundant = False
                            break
                    
                    if is_redundant:
                        for ts_idx in block_to_tsets[b_idx]:
                            cover_counts[ts_idx] -= 1
                        current.pop(i)
                        found_redundant = True
                        improved = True
                        if len(current) < best_size:
                            best_size = len(current)
                            best_solution = list(current)
                
                if not found_redundant or (time.time() - local_start) > max_time_per_run:
                    break
            
            if not current:
                break
                
            # Phase 2: 1-for-1 Swaps with improved scoring
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
                    continue

                # Find candidate blocks B_in that cover all unique_tsets
                candidates = set(tset_to_blocks[unique_tsets[0]])
                candidates.discard(b_out)
                
                for ts_idx in unique_tsets[1:]:
                    if not candidates:
                        break
                    candidates.intersection_update(tset_to_blocks[ts_idx])
                
                if not candidates:
                    continue
                
                best_candidate = -1
                best_score = -1
                b_out_tsets_set = set(block_to_tsets[b_out])
                
                candidate_list = list(candidates)
                if len(candidate_list) > 100:
                    random.shuffle(candidate_list)
                    candidate_list = candidate_list[:100]
                
                for b_in in candidate_list:
                    # Score: prioritize blocks that cover many currently-sparse tsets
                    # and don't overlap too much with b_out
                    score = 0
                    overlap = 0
                    for ts_idx in block_to_tsets[b_in]:
                        if ts_idx in b_out_tsets_set:
                            overlap += 1
                        else:
                            # Weight by how "needed" this tset is
                            cc = cover_counts[ts_idx]
                            if cc == 1:
                                score += 10
                            elif cc == 2:
                                score += 5
                            elif cc == 3:
                                score += 2
                            else:
                                score += 1
                    # Penalize high overlap (less new coverage)
                    score -= overlap * 0.5
                    
                    if score > best_score:
                        best_score = score
                        best_candidate = b_in
                    
                if best_candidate != -1:
                    for ts_idx in block_to_tsets[b_out]:
                        cover_counts[ts_idx] -= 1
                    for ts_idx in block_to_tsets[best_candidate]:
                        cover_counts[ts_idx] += 1
                    
                    current[i] = best_candidate
                    improved = True
            
            # Phase 3: 1-for-2 moves (remove one, add two) - can reduce size by 1
            if (time.time() - local_start) < max_time_per_run and len(current) > 2:
                # Try removing a block and seeing if we can cover its unique tsets with 2 new blocks
                # that together are more efficient
                random.shuffle(current)
                for i in range(min(len(current), 30)):  # Limit to 30 attempts
                    if (time.time() - local_start) > max_time_per_run:
                        break
                    
                    b_out = current[i]
                    unique_tsets = []
                    for ts_idx in block_to_tsets[b_out]:
                        if cover_counts[ts_idx] == 1:
                            unique_tsets.append(ts_idx)
                    
                    if len(unique_tsets) <= 1:
                        continue
                    
                    # Try to find 2 blocks that cover all unique_tsets
                    # and provide good additional coverage
                    # Use the first unique tset to get candidates
                    cand1 = set(tset_to_blocks[unique_tsets[0]])
                    cand1.discard(b_out)
                    
                    if len(cand1) > 50:
                        cand1 = set(random.sample(list(cand1), 50))
                    
                    found_pair = False
                    for b_a in cand1:
                        if (time.time() - local_start) > max_time_per_run:
                            break
                        # What tsets does b_a NOT cover from unique_tsets?
                        uncovered_by_a = [ts for ts in unique_tsets if ts not in set(block_to_tsets[b_a])]
                        if not uncovered_by_a:
                            # b_a alone covers all unique tsets - this is just a swap
                            continue
                        
                        # Find b_b that covers remaining
                        cand2 = set(tset_to_blocks[uncovered_by_a[0]])
                        cand2.discard(b_out)
                        cand2.discard(b_a)
                        for ts in uncovered_by_a[1:]:
                            if not cand2:
                                break
                            cand2.intersection_update(tset_to_blocks[ts])
                        
                        if not cand2:
                            continue
                        
                        if len(cand2) > 30:
                            cand2 = set(random.sample(list(cand2), 30))
                        
                        for b_b in cand2:
                            # Check if b_a and b_b together provide good coverage
                            # and if removing b_out + adding b_a,b_b reduces total or maintains
                            new_unique = 0
                            for ts in block_to_tsets[b_a]:
                                if cover_counts[ts] == 1 and ts not in set(block_to_tsets[b_b]):
                                    new_unique += 1
                            for ts in block_to_tsets[b_b]:
                                if cover_counts[ts] == 1 and ts not in set(block_to_tsets[b_a]):
                                    new_unique += 1
                            
                            # Accept if we don't create new unique tsets (or few)
                            if new_unique <= 2:
                                # Apply: remove b_out, add b_a and b_b
                                for ts_idx in block_to_tsets[b_out]:
                                    cover_counts[ts_idx] -= 1
                                for ts_idx in block_to_tsets[b_a]:
                                    cover_counts[ts_idx] += 1
                                for ts_idx in block_to_tsets[b_b]:
                                    cover_counts[ts_idx] += 1
                                
                                current[i] = b_a
                                current.append(b_b)
                                improved = True
                                found_pair = True
                                break
                        
                        if found_pair:
                            break
                    
                    if found_pair:
                        break
            
            # Phase 4: 2-for-1 moves (remove two, add one) - can reduce size by 1
            if (time.time() - local_start) < max_time_per_run and len(current) > 2:
                random.shuffle(current)
                for i in range(min(len(current) - 1, 20)):
                    if (time.time() - local_start) > max_time_per_run:
                        break
                    
                    b_out1 = current[i]
                    b_out2 = current[i + 1]
                    
                    # Combined unique tsets
                    tsets1 = set(block_to_tsets[b_out1])
                    tsets2 = set(block_to_tsets[b_out2])
                    combined_unique = []
                    for ts_idx in tsets1 | tsets2:
                        if cover_counts[ts_idx] == 2 and ts_idx in tsets1 and ts_idx in tsets2:
                            combined_unique.append(ts_idx)
                        elif cover_counts[ts_idx] == 1:
                            combined_unique.append(ts_idx)
                    
                    if not combined_unique:
                        continue
                    
                    # Find a single block that covers all combined_unique
                    candidates = set(tset_to_blocks[combined_unique[0]])
                    candidates.discard(b_out1)
                    candidates.discard(b_out2)
                    for ts in combined_unique[1:]:
                        if not candidates:
                            break
                        candidates.intersection_update(tset_to_blocks[ts])
                    
                    if not candidates:
                        continue
                    
                    # Pick best candidate by additional coverage
                    best_b = -1
                    best_add = -1
                    for b_in in list(candidates)[:50]:
                        add = 0
                        for ts in block_to_tsets[b_in]:
                            if cover_counts[ts] >= 1:
                                add += 1
                        if add > best_add:
                            best_add = add
                            best_b = b_in
                    
                    if best_b != -1:
                        for ts_idx in block_to_tsets[b_out1]:
                            cover_counts[ts_idx] -= 1
                        for ts_idx in block_to_tsets[b_out2]:
                            cover_counts[ts_idx] -= 1
                        for ts_idx in block_to_tsets[best_b]:
                            cover_counts[ts_idx] += 1
                        
                        current[i] = best_b
                        current.pop(i + 1)
                        improved = True
                        if len(current) < best_size:
                            best_size = len(current)
                            best_solution = list(current)
                        break
        
        # Final polish: aggressive redundancy removal
        if (time.time() - local_start) < max_time_per_run:
            while True:
                found_redundant = False
                for i in range(len(current) - 1, -1, -1):
                    if (time.time() - local_start) > max_time_per_run:
                        break
                    b_idx = current[i]
                    is_redundant = True
                    for ts_idx in block_to_tsets[b_idx]:
                        if cover_counts[ts_idx] == 1:
                            is_redundant = False
                            break
                    if is_redundant:
                        for ts_idx in block_to_tsets[b_idx]:
                            cover_counts[ts_idx] -= 1
                        current.pop(i)
                        found_redundant = True
                        if len(current) < best_size:
                            best_size = len(current)
                            best_solution = list(current)
                if not found_redundant:
                    break
        
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
    
    # Run 1: Standard greedy with extended local search
    t_start = time.time()
    init1 = greedy_cover(42)
    if init1:
        # Use more time for local search since it's the primary run
        ls_time = min(50.0, deadline - t_start - 10)
        sol1 = run_local_search(init1, max_time_per_run=ls_time)
        if sol1 and len(sol1) < best_global_size:
            best_global_size = len(sol1)
            best_global_solution = sol1
            
    # Run 2: Greedy with different seed if time permits
    t_start = time.time()
    if time.time() < deadline - 30:
        init2 = greedy_cover(123)
        if init2:
            ls_time = min(20.0, deadline - t_start - 10)
            sol2 = run_local_search(init2, max_time_per_run=ls_time)
            if sol2 and len(sol2) < best_global_size:
                best_global_size = len(sol2)
                best_global_solution = sol2
    
    # Run 3: Another greedy with different seed if time permits
    t_start = time.time()
    if time.time() < deadline - 15:
        init3 = greedy_cover(777)
        if init3:
            ls_time = min(10.0, deadline - t_start - 5)
            sol3 = run_local_search(init3, max_time_per_run=ls_time)
            if sol3 and len(sol3) < best_global_size:
                best_global_size = len(sol3)
                best_global_solution = sol3
    
    # Final polish: Re-run local search on the best solution with remaining time
    if best_global_solution and time.time() < deadline - 5:
        remaining_time = deadline - time.time() - 2
        if remaining_time > 5:
            improved_sol = run_local_search(best_global_solution, max_time_per_run=remaining_time)
            if improved_sol and len(improved_sol) < best_global_size:
                best_global_size = len(improved_sol)
                best_global_solution = improved_sol
    
    # Second polish pass with even more time if available
    if best_global_solution and time.time() < deadline - 3:
        remaining_time = deadline - time.time() - 1
        if remaining_time > 3:
            improved_sol2 = run_local_search(best_global_solution, max_time_per_run=remaining_time)
            if improved_sol2 and len(improved_sol2) < best_global_size:
                best_global_size = len(improved_sol2)
                best_global_solution = improved_sol2
            
    if not best_global_solution:
        # Fallback: Return a valid but potentially large solution
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
