import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Pre-compute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # Pre-compute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map block index to list of t-subset indices it covers
    block_to_tsubs = []
    # Also create bitmask for each block
    block_masks = [0] * n_blocks
    
    for b_idx, b in enumerate(all_blocks):
        subs = []
        mask = 0
        for sub in itertools.combinations(b, t):
            idx = t_sub_index[sub]
            subs.append(idx)
            mask |= (1 << idx)
        block_to_tsubs.append(subs)
        block_masks[b_idx] = mask
        
    # Map t-subset index to list of block indices that cover it
    tsub_to_blocks = [[] for _ in range(n_t_subs)]
    for b_idx, subs in enumerate(block_to_tsubs):
        for s in subs:
            tsub_to_blocks[s].append(b_idx)
            
    best_solution = []
    best_size = float('inf')
    
    # To diversify, we can try different random seeds for tie-breaking in greedy
    # Or just run the greedy+local search multiple times if time permits.
    
    while time.time() < deadline:
        # --- Phase 1: Greedy Construction using Bitmasks ---
        uncovered_mask = (1 << n_t_subs) - 1
        solution_blocks = []
        solution_set = set()
        
        while uncovered_mask:
            if time.time() > deadline:
                break
                
            max_count = -1
            best_b_idx = -1
            # Find block with max uncovered coverage
            # Optimization: iterate through all blocks
            # For speed, we might want to filter candidates, but 50k blocks is small.
            for i in range(n_blocks):
                # Count bits in (block_mask[i] & uncovered_mask)
                c = (block_masks[i] & uncovered_mask).bit_count()
                if c > max_count:
                    max_count = c
                    best_b_idx = i
                    if c == 0:
                        break # No block covers anything new? Should not happen.
                    # If we found a block covering all remaining? 
                    # We can't easily check if c == remaining_count without counting bits in uncovered_mask
                    # But if c is high, it's good.
                    
            if best_b_idx == -1 or max_count == 0:
                break
                
            solution_blocks.append(best_b_idx)
            solution_set.add(best_b_idx)
            uncovered_mask &= ~block_masks[best_b_idx]
            
        if not solution_blocks:
            continue
            
        # --- Phase 2: Local Search ---
        # Initialize cover_count
        cover_count = [0] * n_t_subs
        for b_idx in solution_blocks:
            for s in block_to_tsubs[b_idx]:
                cover_count[s] += 1
                
        improved = True
        while improved and time.time() < deadline - 2: # Safety margin
            improved = False
            
            # Pass 1: Remove redundant blocks
            for i in range(len(solution_blocks)):
                b_idx = solution_blocks[i]
                redundant = True
                for s in block_to_tsubs[b_idx]:
                    if cover_count[s] == 1:
                        redundant = False
                        break
                if redundant:
                    for s in block_to_tsubs[b_idx]:
                        cover_count[s] -= 1
                    solution_blocks.pop(i)
                    solution_set.remove(b_idx)
                    improved = True
                    break # Restart loop
                    
            # Pass 2: Swap
            if not improved and time.time() < deadline - 5:
                for i in range(len(solution_blocks)):
                    b_idx = solution_blocks[i]
                    critical_subsets = []
                    for s in block_to_tsubs[b_idx]:
                        if cover_count[s] == 1:
                            critical_subsets.append(s)
                            
                    if not critical_subsets:
                        # Should have been removed
                        for s in block_to_tsubs[b_idx]:
                            cover_count[s] -= 1
                        solution_blocks.pop(i)
                        solution_set.remove(b_idx)
                        improved = True
                        break
                        
                    # Find candidate b' not in solution such that critical_subsets subset of b'
                    # Intersection of tsub_to_blocks for critical subsets
                    candidate_set = set(tsub_to_blocks[critical_subsets[0]])
                    for s in critical_subsets[1:]:
                        candidate_set.intersection_update(tsub_to_blocks[s])
                        if not candidate_set:
                            break
                            
                    if candidate_set:
                        # Filter out blocks already in solution
                        candidates = [c for c in candidate_set if c not in solution_set]
                        if candidates:
                            # Heuristic: pick candidate that covers most "critical" subsets of OTHER blocks
                            # i.e., maximize sum of 1 if cover_count[s] == 1 for s in b'
                            best_candidate = -1
                            max_score = -1
                            for c_idx in candidates:
                                score = 0
                                for s in block_to_tsubs[c_idx]:
                                    if cover_count[s] == 1:
                                        score += 1
                                if score > max_score:
                                    max_score = score
                                    best_candidate = c_idx
                                    
                            if best_candidate != -1:
                                # Perform swap
                                # Remove b_idx
                                for s in block_to_tsubs[b_idx]:
                                    cover_count[s] -= 1
                                # Add best_candidate
                                for s in block_to_tsubs[best_candidate]:
                                    cover_count[s] += 1
                                    
                                solution_blocks[i] = best_candidate
                                solution_set.remove(b_idx)
                                solution_set.add(best_candidate)
                                improved = True
                                break
                                
        # Check if this solution is better
        if len(solution_blocks) < best_size:
            best_size = len(solution_blocks)
            best_solution = [list(all_blocks[b_idx]) for b_idx in solution_blocks]
            
        # If we found a very good solution, we might stop early?
        # But we want to try to beat it.
        
    return best_solution
