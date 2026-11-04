import itertools
import time
import random
import math
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    """
    Find a covering of t-subsets of v points using k-subsets (blocks).
    Returns a list of blocks (each is a sorted list of k distinct ints in 0..v-1).
    """
    start_time = time.time()
    deadline = start_time + 145.0
    
    def check_time(budget=5.0):
        return time.time() - start_time > deadline - budget
    
    # Precompute all t-subsets and index them
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # All possible k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Build incidence: for each block, which t-subsets it covers
    block_tsubs = []
    for b in all_blocks:
        tsubs_in_b = []
        for ts in itertools.combinations(b, t):
            tsubs_in_b.append(tsub_index[ts])
        block_tsubs.append(tsubs_in_b)
    
    # Precompute tsub_to_blocks for quick lookup
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for j, tsubs in enumerate(block_tsubs):
        for ts_idx in tsubs:
            tsub_to_blocks[ts_idx].append(j)
    
    # --- Greedy with priority queue approach ---
    def greedy_construction(weights, time_budget=10.0):
        """Greedy set cover using the given weights."""
        uncovered = set(range(num_tsubs))
        selected = []
        uncovered_count = num_tsubs
        
        # Score for each block: sum of weights of uncovered t-subsets it covers
        # Initially: score[j] = sum(weights[i] for i in block_tsubs[j])
        score = np.zeros(num_blocks, dtype=np.float64)
        for j in range(num_blocks):
            score[j] = np.sum(weights[block_tsubs[j]])
        
        selected_mask = np.zeros(num_blocks, dtype=bool)
        
        while uncovered_count > 0:
            if time.time() - start_time > deadline - time_budget:
                break
            
            # Find best block
            # Mask out already selected blocks
            scores = score.copy()
            scores[selected_mask] = -1.0
            b_idx = int(np.argmax(scores))
            
            if scores[b_idx] <= 0:
                break
            
            selected.append(b_idx)
            selected_mask[b_idx] = True
            
            # Update: remove newly covered t-subsets from uncovered
            # and update scores
            for t_idx in block_tsubs[b_idx]:
                if t_idx in uncovered:
                    uncovered.remove(t_idx)
                    uncovered_count -= 1
                    # Decrease score of all blocks that cover this t-subset
                    for b in tsub_to_blocks[t_idx]:
                        score[b] -= weights[t_idx]
        
        return selected
    
    # --- Reduction phase ---
    def reduce_blocks(block_indices, time_budget=10.0):
        """Remove redundant blocks."""
        if not block_indices:
            return []
        
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b in block_indices:
            for t in block_tsubs[b]:
                cover_count[t] += 1
        
        changed = True
        while changed:
            if time.time() - start_time > deadline - time_budget:
                break
            changed = False
            new_blocks = []
            for b in block_indices:
                tsubs = block_tsubs[b]
                if len(tsubs) > 0 and np.min(cover_count[tsubs]) > 1:
                    changed = True
                    cover_count[tsubs] -= 1
                else:
                    new_blocks.append(b)
            block_indices = new_blocks
        return block_indices
    
    # --- Swap-based local search ---
    def local_search(block_indices, time_budget=15.0):
        """Try swaps to improve the covering."""
        if not block_indices:
            return []
        
        current_blocks = list(block_indices)
        current_set = set(current_blocks)
        
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b in current_blocks:
            for t in block_tsubs[b]:
                cover_count[t] += 1
        
        improved = True
        while improved:
            if time.time() - start_time > deadline - time_budget:
                break
            improved = False
            
            # Try removing each block and see if covering is still valid
            # If not, try swapping with a better block
            blocks_to_try = list(current_blocks)
            random.shuffle(blocks_to_try)
            
            for b in blocks_to_try:
                if time.time() - start_time > deadline - time_budget:
                    break
                
                # Find t-subsets that are exclusively covered by b
                exclusive = [t for t in block_tsubs[b] if cover_count[t] == 1]
                if not exclusive:
                    # Can remove b
                    for t in block_tsubs[b]:
                        cover_count[t] -= 1
                    current_blocks = [x for x in current_blocks if x != b]
                    current_set.discard(b)
                    improved = True
                    break
                
                # Need to find a replacement block that covers all exclusive t-subsets
                # Find blocks that cover all exclusive t-subsets
                candidate_replacements = None
                for t in exclusive:
                    candidates = set(tsub_to_blocks[t])
                    if candidate_replacements is None:
                        candidate_replacements = candidates
                    else:
                        candidate_replacements &= candidates
                
                if candidate_replacements is None or len(candidate_replacements) == 0:
                    continue
                
                # Remove current block from candidates
                candidate_replacements.discard(b)
                if len(candidate_replacements) == 0:
                    continue
                
                # Pick the best replacement (one that covers the most other uncovered t-subsets)
                # For simplicity, pick randomly from candidates
                replacement = random.choice(list(candidate_replacements))
                
                # Perform swap
                for t in block_tsubs[b]:
                    cover_count[t] -= 1
                for t in block_tsubs[replacement]:
                    cover_count[t] += 1
                
                current_blocks = [x for x in current_blocks if x != b]
                current_blocks.append(replacement)
                current_set.discard(b)
                current_set.add(replacement)
                improved = True
                break
        
        return current_blocks
    
    # --- Main algorithm ---
    # Initial greedy with inverse degree weights
    degs = np.array([len(tsub_to_blocks[i]) for i in range(num_tsubs)], dtype=np.float64)
    degs[degs == 0] = 1.0
    base_weights = 1.0 / degs
    
    best_blocks = greedy_construction(base_weights, time_budget=15.0)
    best_blocks = reduce_blocks(best_blocks, time_budget=10.0)
    best_len = len(best_blocks)
    
    # Multiple restarts with perturbed weights
    num_restarts = 8
    for run in range(num_restarts):
        if time.time() - start_time > deadline - 20.0:
            break
        
        # Perturb weights
        if run == 0:
            weights_run = base_weights.copy()
        else:
            noise = 0.1 * np.random.randn(num_tsubs)
            weights_run = base_weights * (1.0 + noise)
            weights_run[weights_run < 0] = 1e-6
        
        cand_blocks = greedy_construction(weights_run, time_budget=12.0)
        cand_blocks = reduce_blocks(cand_blocks, time_budget=8.0)
        
        if len(cand_blocks) < best_len:
            best_len = len(cand_blocks)
            best_blocks = cand_blocks
            # Apply local search to the best
            if time.time() - start_time < deadline - 15.0:
                best_blocks = local_search(best_blocks, time_budget=12.0)
                best_len = len(best_blocks)
                best_blocks = reduce_blocks(best_blocks, time_budget=5.0)
                best_len = len(best_blocks)
    
    # Final reduction
    best_blocks = reduce_blocks(best_blocks, time_budget=5.0)
    best_len = len(best_blocks)
    
    return [list(all_blocks[b]) for b in best_blocks]
