import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for safety
    
    # 1. Generate all k-blocks and t-subsets
    points = range(v)
    k_blocks = list(itertools.combinations(points, k))
    t_subs = list(itertools.combinations(points, t))
    
    num_blocks = len(k_blocks)
    num_subs = len(t_subs)
    
    # Map subset to index
    sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    
    # For each block, list the indices of the t-subsets it covers
    block_subs = []
    for block in k_blocks:
        subs_in_block = [sub_to_idx[sub] for sub in itertools.combinations(block, t)]
        block_subs.append(subs_in_block)
        
    # For each subset, list the indices of the blocks that cover it
    # This is useful for local search
    sub_blocks = [[] for _ in range(num_subs)]
    for i, subs in enumerate(block_subs):
        for j in subs:
            sub_blocks[j].append(i)

    # 2. Greedy Construction
    # uncovered[j] = 1 if subset j is not covered yet
    uncovered = [1] * num_subs
    solution = []
    used_blocks = [False] * num_blocks
    
    # To speed up greedy, we can't easily use a heap because gains change dynamically.
    # However, for v=19, k=7, t=4, the number of blocks is ~50k.
    # A simple loop over all blocks to find max gain is O(num_blocks * num_subs_in_block).
    # num_subs_in_block is C(7,4)=35.
    # So one pass is 50k * 35 ~ 1.7M operations.
    # We need to do this until all covered.
    # Number of blocks in solution is ~150.
    # Total ops ~ 150 * 1.7M = 255M. This might be too slow in Python.
    
    # Optimization:
    # We can maintain a list of "active" blocks.
    # Or, we can just do it and hope it's fast enough with PyPy/CPython optimizations?
    # Actually, 255M simple integer ops in Python might take > 10s.
    # Let's try to optimize the greedy step.
    
    # Alternative Greedy:
    # Keep a count of how many times each subset is covered.
    # Initially 0.
    # In each step, find block b that maximizes sum(1 for s in block_subs[b] if covered_count[s] == 0).
    
    covered_count = [0] * num_subs
    remaining_uncovered = num_subs
    
    # Precompute block_subs as lists of ints for fast access
    # We already have block_subs as list of lists of ints.
    
    while remaining_uncovered > 0:
        if time.time() > deadline - 20:
            break
            
        best_block = -1
        best_gain = 0
        
        # Iterate through all blocks not yet used? Or all blocks?
        # Blocks already used might still have gain if they cover new subsets?
        # No, if a block is used, all its subsets are covered. Gain is 0.
        # So we only need to check unused blocks.
        
        for i in range(num_blocks):
            if used_blocks[i]:
                continue
                
            gain = 0
            # Check how many subsets in this block are currently uncovered
            for s_idx in block_subs[i]:
                if covered_count[s_idx] == 0:
                    gain += 1
                    if gain > best_gain:
                        # Early termination if we can't beat best? 
                        # Max possible gain is len(block_subs[i]) which is 35.
                        # If best_gain is already 35, we can't beat it.
                        # But we still need to find the specific block.
                        pass
            
            if gain > best_gain:
                best_gain = gain
                best_block = i
                if best_gain == len(block_subs[i]): # Can't get better for this block
                    # But another block might also have 35?
                    # We just want one best.
                    pass
        
        if best_block == -1 or best_gain == 0:
            break # Should not happen if valid covering exists
            
        # Add best_block to solution
        solution.append(best_block)
        used_blocks[best_block] = True
        
        # Update covered_count
        for s_idx in block_subs[best_block]:
            if covered_count[s_idx] == 0:
                remaining_uncovered -= 1
            covered_count[s_idx] += 1

    # 3. Local Search
    # We want to minimize len(solution).
    # Strategy:
    # 1. Remove redundant blocks.
    # 2. Swap blocks to enable further removals.
    
    # covered_count is currently the count from the greedy phase.
    # It represents how many blocks in 'solution' cover each subset.
    
    improved = True
    max_iters = 50
    iter_count = 0
    
    while improved and iter_count < max_iters and time.time() < deadline - 5:
        improved = False
        iter_count += 1
        
        # Pass 1: Remove redundant blocks
        for i in list(solution):
            # Check if any subset in block i has count == 1
            # If min count > 1, remove it
            min_count = min(covered_count[s_idx] for s_idx in block_subs[i])
            if min_count > 1:
                # Remove i
                for s_idx in block_subs[i]:
                    covered_count[s_idx] -= 1
                solution.remove(i)
                improved = True
                break # Restart scan to find new redundancies
        
        if improved:
            continue
        
        # Pass 2: Swap blocks
        # For each block i in solution:
        #   Let D = {s in block_subs[i] | covered_count[s] == 1}
        #   If D is empty, it would have been removed in Pass 1.
        #   Find a block j not in solution such that D is subset of block_subs[j].
        #   If found, swap i with j.
        
        sol_set = set(solution)
        
        for i in solution:
            if time.time() > deadline - 5:
                break
                
            D = []
            for s_idx in block_subs[i]:
                if covered_count[s_idx] == 1:
                    D.append(s_idx)
            
            if not D:
                continue # Should be removed, but maybe race condition?
                
            # Find candidate blocks j that cover all of D
            # Intersection of sub_blocks[s] for s in D
            # Start with the smallest list to minimize work
            D_sorted = sorted(D, key=lambda x: len(sub_blocks[x]))
            
            if not D_sorted:
                continue
                
            candidates = set(sub_blocks[D_sorted[0]])
            
            for s_idx in D_sorted[1:]:
                candidates.intersection_update(sub_blocks[s_idx])
                if not candidates:
                    break
            
            if not candidates:
                continue
            
            # Filter out blocks already in solution
            valid_candidates = [j for j in candidates if j not in sol_set]
            
            if not valid_candidates:
                continue
            
            # Pick a candidate. 
            # Heuristic: Pick the one that covers the most subsets with count == 1 (other than D)?
            # Or just pick the first one?
            # To improve chances of future removals, we want to increase counts of other subsets.
            # Let's pick the candidate that maximizes the number of subsets it covers that are currently count == 1.
            # This "consolidates" the coverage.
            
            best_j = -1
            best_score = -1
            
            for j in valid_candidates:
                score = 0
                for s_idx in block_subs[j]:
                    if covered_count[s_idx] == 1:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_j = j
                    
            if best_j != -1:
                # Perform swap
                # Remove i
                for s_idx in block_subs[i]:
                    covered_count[s_idx] -= 1
                sol_set.remove(i)
                solution.remove(i)
                
                # Add best_j
                for s_idx in block_subs[best_j]:
                    covered_count[s_idx] += 1
                sol_set.add(best_j)
                solution.append(best_j)
                
                improved = True
                break # Restart scan

    # Return the list of blocks
    return [list(k_blocks[i]) for i in solution]
