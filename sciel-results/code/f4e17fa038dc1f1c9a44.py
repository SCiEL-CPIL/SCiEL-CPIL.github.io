import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute all blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map each block to the t-subsets it covers
    block_t_indices = []
    for block in all_blocks:
        idxs = [t_subset_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_t_indices.append(idxs)
    
    # Build reverse mapping: t-subset index -> list of block indices covering it
    t_to_blocks = defaultdict(list)
    for b_idx, t_idxs in enumerate(block_t_indices):
        for t_idx in t_idxs:
            t_to_blocks[t_idx].append(b_idx)
    
    # Greedy construction with diversity bias
    cover_counts = np.zeros(n_t, dtype=np.int32)
    selected_indices = []
    selected_set = set()
    
    # For diversity, track which points have been overused
    point_usage = np.zeros(v, dtype=np.int32)
    
    for _ in range(100):
        if time.time() > deadline - 60.0:
            break
        
        best_idx = -1
        best_score = -1
        
        # Sample a subset of blocks to speed up greedy
        if n_blocks > 50000:
            sample_indices = random.sample(range(n_blocks), 10000)
        else:
            sample_indices = range(n_blocks)
        
        for b_idx in sample_indices:
            if b_idx in selected_set:
                continue
            
            # Count how many NEW t-subsets this block covers
            new_cover = 0
            for t_idx in block_t_indices[b_idx]:
                if cover_counts[t_idx] == 0:
                    new_cover += 1
            
            # Diversity score: prefer blocks that use underutilized points
            diversity_bonus = 0
            block = all_blocks[b_idx]
            for p in block:
                diversity_bonus -= 0.1 * point_usage[p]
            
            score = new_cover + diversity_bonus
            
            if score > best_score:
                best_score = score
                best_idx = b_idx
        
        if best_idx == -1 or best_score <= 0:
            break
        
        # Add the block
        selected_indices.append(best_idx)
        selected_set.add(best_idx)
        
        # Update cover counts
        for t_idx in block_t_indices[best_idx]:
            cover_counts[t_idx] += 1
        
        # Update point usage
        for p in all_blocks[best_idx]:
            point_usage[p] += 1
        
        # Check if all covered
        if np.all(cover_counts > 0):
            break
    
    # Verify greedy produced a valid solution
    if np.any(cover_counts == 0):
        # Fallback: add blocks until valid
        while np.any(cover_counts == 0):
            uncovered = np.where(cover_counts == 0)[0]
            # Find a block that covers the most uncovered
            best_idx = -1
            best_new = 0
            for b_idx in random.sample(range(n_blocks), min(5000, n_blocks)):
                if b_idx in selected_set:
                    continue
                new = sum(1 for t_idx in block_t_indices[b_idx] if cover_counts[t_idx] == 0)
                if new > best_new:
                    best_new = new
                    best_idx = b_idx
            if best_idx == -1 or best_new == 0:
                break
            selected_indices.append(best_idx)
            selected_set.add(best_idx)
            for t_idx in block_t_indices[best_idx]:
                cover_counts[t_idx] += 1
            for p in all_blocks[best_idx]:
                point_usage[p] += 1
            if time.time() > deadline - 30.0:
                break
    
    current_solution = selected_indices.copy()
    current_size = len(current_solution)
    best_solution = current_solution.copy()
    best_size = current_size
    
    # Rebuild cover_counts for current solution
    cover_counts = np.zeros(n_t, dtype=np.int32)
    for b_idx in current_solution:
        cover_counts[block_t_indices[b_idx]] += 1
    
    current_uncovered = int(np.sum(cover_counts == 0))
    
    # Simulated Annealing with smarter moves
    T = 30.0
    T_min = 0.5
    cooling_rate = 0.998
    
    # Precompute block intersection sizes for faster evaluation? No, too memory intensive.
    # Instead, use smarter candidate generation.
    
    iteration = 0
    while time.time() < deadline - 10.0:
        iteration += 1
        
        # Periodic restart of temperature
        if iteration % 3000 == 0:
            T = max(T, 10.0)
        
        # Move 1: Try deletion (always beneficial)
        if current_size > 1:
            idx_to_remove = random.randint(0, current_size - 1)
            b_idx = current_solution[idx_to_remove]
            t_idxs = block_t_indices[b_idx]
            if np.min(cover_counts[t_idxs]) > 1:
                # Safe to remove
                cover_counts[t_idxs] -= 1
                current_solution.pop(idx_to_remove)
                current_size -= 1
                if current_uncovered == 0 and current_size < best_size:
                    best_size = current_size
                    best_solution = current_solution.copy()
                continue
        
        # Move 2: Smart swap
        if current_size == 0:
            b_idx = random.randint(0, n_blocks - 1)
            current_solution.append(b_idx)
            current_size += 1
            t_idxs = block_t_indices[b_idx]
            new_uncovered = 0
            for t in t_idxs:
                if cover_counts[t] == 0:
                    new_uncovered -= 1
                cover_counts[t] += 1
            current_uncovered += new_uncovered
            if current_uncovered == 0 and current_size < best_size:
                best_size = current_size
                best_solution = current_solution.copy()
            continue
        
        # Pick block A to remove
        idx_a = random.randint(0, current_size - 1)
        b_idx_a = current_solution[idx_a]
        t_idxs_a = block_t_indices[b_idx_a]
        
        # Find critical t-subsets in A (those with count == 1)
        criticals = [t for t in t_idxs_a if cover_counts[t] == 1]
        
        if not criticals:
            # A is redundant, handle in deletion phase
            continue
        
        # Find candidate blocks B that cover all criticals
        # For efficiency, only consider blocks that cover at least half the criticals
        # Use the reverse mapping
        candidate_blocks = set()
        for c in criticals:
            candidate_blocks.update(t_to_blocks[c])
        
        # Remove A itself and already-selected blocks
        candidate_blocks.discard(b_idx_a)
        candidate_blocks -= selected_set if 'selected_set' in dir() else set()
        
        # Filter: must cover all criticals
        valid_candidates = []
        for b_idx_b in candidate_blocks:
            if b_idx_b == b_idx_a:
                continue
            t_idxs_b = block_t_indices[b_idx_b]
            # Check if B covers all criticals
            b_set = set(t_idxs_b)
            if all(c in b_set for c in criticals):
                valid_candidates.append(b_idx_b)
        
        if not valid_candidates:
            T *= cooling_rate
            if T < T_min:
                T = T_min
            continue
        
        # Pick a random candidate
        b_idx_b = random.choice(valid_candidates)
        t_idxs_b = block_t_indices[b_idx_b]
        
        # Evaluate the swap
        # Remove A
        new_uncovered = current_uncovered
        for t in t_idxs_a:
            cover_counts[t] -= 1
            if cover_counts[t] == 0:
                new_uncovered += 1
        
        # Add B
        for t in t_idxs_b:
            if cover_counts[t] == 0:
                new_uncovered -= 1
            cover_counts[t] += 1
        
        delta_uncovered = new_uncovered - current_uncovered
        
        # Acceptance
        if delta_uncovered <= 0:
            accept = True
        else:
            lambda_cost = 5.0
            delta_E = lambda_cost * delta_uncovered
            p = np.exp(-delta_E / T)
            accept = random.random() < p
        
        if accept:
            current_uncovered = new_uncovered
            current_solution[idx_a] = b_idx_b
            if current_uncovered == 0 and current_size < best_size:
                best_size = current_size
                best_solution = current_solution.copy()
        else:
            # Revert
            for t in t_idxs_b:
                cover_counts[t] -= 1
            for t in t_idxs_a:
                cover_counts[t] += 1
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    # Final cleanup: delete redundant blocks
    final_solution = best_solution.copy()
    cover_counts = np.zeros(n_t, dtype=np.int32)
    for b_idx in final_solution:
        cover_counts[block_t_indices[b_idx]] += 1
    
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for i in range(len(final_solution) - 1, -1, -1):
            b_idx = final_solution[i]
            t_idxs = block_t_indices[b_idx]
            if np.min(cover_counts[t_idxs]) > 1:
                cover_counts[t_idxs] -= 1
                final_solution.pop(i)
                improved = True
                break
    
    # Final verification
    if np.any(cover_counts == 0):
        # Fallback to greedy solution
        final_solution = selected_indices.copy()
        cover_counts = np.zeros(n_t, dtype=np.int32)
        for b_idx in final_solution:
            cover_counts[block_t_indices[b_idx]] += 1
    
    return [list(all_blocks[i]) for i in final_solution]
