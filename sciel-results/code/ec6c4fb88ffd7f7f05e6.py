import itertools
import random
import math
import time
import numpy as np

def construct(v, k, t):
    points = list(range(v))
    all_subsets = list(itertools.combinations(points, t))
    n_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    # Precompute subset indices for each block
    block_subsets = []
    for b in all_blocks:
        subs = [subset_to_idx[s] for s in itertools.combinations(b, t)]
        block_subsets.append(subs)
        
    # Precompute for each subset, which blocks contain it
    # This is useful for the swap local search
    # subset_to_blocks: list of lists of block indices
    subset_to_blocks = [[] for _ in range(n_subsets)]
    for b_idx, subs in enumerate(block_subsets):
        for s_idx in subs:
            subset_to_blocks[s_idx].append(b_idx)

    def greedy_cover(randomize=False):
        uncovered = set(range(n_subsets))
        selected = []
        
        while uncovered:
            best_gain = -1
            candidates = []
            
            for b_idx, subs in enumerate(block_subsets):
                if len(subs) <= best_gain:
                    continue
                gain = 0
                for s_idx in subs:
                    if s_idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    candidates = [b_idx]
                elif gain == best_gain and gain > 0:
                    candidates.append(b_idx)
            
            if not candidates:
                break
                
            if randomize and len(candidates) > 1:
                chosen = random.choice(candidates)
            else:
                chosen = candidates[0]
                
            selected.append(chosen)
            for s_idx in block_subsets[chosen]:
                if s_idx in uncovered:
                    uncovered.discard(s_idx)
                    
        return selected

    def local_search(selected):
        selected_set = set(selected)
        coverage_count = [0] * n_subsets
        
        for b_idx in selected:
            for s_idx in block_subsets[b_idx]:
                coverage_count[s_idx] += 1
                
        improved = True
        while improved:
            improved = False
            
            # Phase 1: Remove redundant blocks
            new_selected = []
            for b_idx in selected:
                is_redundant = True
                for s_idx in block_subsets[b_idx]:
                    if coverage_count[s_idx] == 1:
                        is_redundant = False
                        break
                if not is_redundant:
                    new_selected.append(b_idx)
                else:
                    # Remove from coverage
                    for s_idx in block_subsets[b_idx]:
                        coverage_count[s_idx] -= 1
                    selected_set.discard(b_idx)
                    improved = True
            selected = new_selected
            
            # Phase 2: Try swaps to enable further reduction
            # We iterate through a copy of selected to avoid modification during iteration issues
            # Note: Swapping might make other blocks redundant, so we loop back to Phase 1.
            current_selected = list(selected)
            for b_idx in current_selected:
                if b_idx not in selected_set:
                    continue
                    
                # Find unique subsets for this block
                unique_subs = []
                for s_idx in block_subsets[b_idx]:
                    if coverage_count[s_idx] == 1:
                        unique_subs.append(s_idx)
                        
                if not unique_subs:
                    continue # Should have been removed in Phase 1, but just in case
                
                # Find a block that covers all unique_subs
                # Intersect block lists
                candidate_blocks = set(subset_to_blocks[unique_subs[0]])
                for s_idx in unique_subs[1:]:
                    candidate_blocks &= set(subset_to_blocks[s_idx])
                    if not candidate_blocks:
                        break
                
                if not candidate_blocks:
                    continue
                    
                # Find a candidate not in selected_set
                swap_target = None
                for c_idx in candidate_blocks:
                    if c_idx not in selected_set:
                        swap_target = c_idx
                        break
                        
                if swap_target is not None:
                    # Perform swap
                    # Remove old
                    for s_idx in block_subsets[b_idx]:
                        coverage_count[s_idx] -= 1
                    selected_set.discard(b_idx)
                    
                    # Add new
                    for s_idx in block_subsets[swap_target]:
                        coverage_count[s_idx] += 1
                    selected_set.add(swap_target)
                    
                    # Update selected list
                    selected = [b for b in selected if b != b_idx]
                    selected.append(swap_target)
                    improved = True
                    break # Restart the while loop to re-evaluate redundancies
                    
        return selected

    # Strategy: Run multiple randomized greedy passes, keep the best one after local search
    best_cover = None
    best_size = float('inf')
    
    # Time limit check
    start_time = time.time()
    time_limit = 110 # Leave 10s buffer
    
    # First, run a deterministic greedy to get a baseline
    base = greedy_cover(randomize=False)
    base = local_search(base)
    if len(base) < best_size:
        best_size = len(base)
        best_cover = base
        
    # Run randomized iterations
    num_iters = 0
    while time.time() - start_time < time_limit:
        num_iters += 1
        cand = greedy_cover(randomize=True)
        cand = local_search(cand)
        
        if len(cand) < best_size:
            best_size = len(cand)
            best_cover = cand
            
    # Convert indices to blocks
    if best_cover is None:
        # Fallback
        best_cover = greedy_cover(randomize=False)
        
    return [list(all_blocks[i]) for i in best_cover]
