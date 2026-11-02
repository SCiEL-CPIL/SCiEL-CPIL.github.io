import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s
    
    # Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(k_subsets)
    
    # coverage_masks[i] is the bitmask of t-subsets covered by k_subset i
    coverage_masks = [0] * num_k_subsets
    # coverage_lists[i] is the list of t-subset indices for efficient count updates
    coverage_lists = [[] for _ in range(num_k_subsets)]
    
    for i, ks in enumerate(k_subsets):
        mask = 0
        indices = []
        for ts in itertools.combinations(ks, t):
            idx = t_subset_index[ts]
            mask |= (1 << idx)
            indices.append(idx)
        coverage_masks[i] = mask
        coverage_lists[i] = indices

    best_blocks = None
    best_count = float('inf')
    
    num_restarts = 50000
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # 1. Greedy Phase
        uncovered_mask = (1 << num_t_subsets) - 1
        current_blocks = []
        used_blocks = set()
        
        greedy_success = True
        while uncovered_mask:
            if time.time() > deadline:
                greedy_success = False
                break
                
            max_gain = 0
            candidates = []
            
            for i in range(num_k_subsets):
                if i in used_blocks:
                    continue
                gain = (coverage_masks[i] & uncovered_mask).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    candidates = [i]
                elif gain == max_gain and gain > 0:
                    candidates.append(i)
            
            if max_gain == 0:
                break
                
            chosen_idx = random.choice(candidates)
            current_blocks.append(chosen_idx)
            used_blocks.add(chosen_idx)
            uncovered_mask &= ~coverage_masks[chosen_idx]
            
        if not greedy_success or uncovered_mask:
            continue
            
        current_count = len(current_blocks)
        
        # 2. Local Search Phase
        counts = [0] * num_t_subsets
        for b_idx in current_blocks:
            for t_idx in coverage_lists[b_idx]:
                counts[t_idx] += 1
        
        improved = True
        max_local_iters = 50
        iter_count = 0
        
        while improved and time.time() < deadline and iter_count < max_local_iters:
            improved = False
            iter_count += 1
            
            # Pass 1: Remove Redundant
            indices_to_check = list(range(len(current_blocks)))
            random.shuffle(indices_to_check)
            
            for pos in indices_to_check:
                if time.time() > deadline:
                    break
                b_idx = current_blocks[pos]
                redundant = True
                for t_idx in coverage_lists[b_idx]:
                    if counts[t_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    current_blocks.pop(pos)
                    used_blocks.remove(b_idx)
                    for t_idx in coverage_lists[b_idx]:
                        counts[t_idx] -= 1
                    improved = True
                    break
            
            if improved:
                continue
            
            # Pass 2: Swap
            # Try to replace a block with another to enable future removals
            if len(current_blocks) > 0:
                # Pick a random block to try to swap
                pos = random.randint(0, len(current_blocks) - 1)
                b_idx = current_blocks[pos]
                
                # Find unique coverage of b_idx
                unique_mask = 0
                for t_idx in coverage_lists[b_idx]:
                    if counts[t_idx] == 1:
                        unique_mask |= (1 << t_idx)
                
                if unique_mask:
                    # Find a replacement B' that covers unique_mask
                    # and is not in used_blocks
                    # We sample random candidates to keep it fast
                    sample_size = min(100, num_k_subsets)
                    candidates = random.sample(range(num_k_subsets), sample_size)
                    
                    for cand_idx in candidates:
                        if cand_idx in used_blocks:
                            continue
                        
                        # Check if cand_idx covers all unique sets
                        if (coverage_masks[cand_idx] & unique_mask) == unique_mask:
                            # Perform swap
                            # Remove b_idx
                            current_blocks.pop(pos)
                            used_blocks.remove(b_idx)
                            for t_idx in coverage_lists[b_idx]:
                                counts[t_idx] -= 1
                            
                            # Add cand_idx
                            current_blocks.append(cand_idx)
                            used_blocks.add(cand_idx)
                            for t_idx in coverage_lists[cand_idx]:
                                counts[t_idx] += 1
                            
                            improved = True
                            break
                    if improved:
                        continue # Restart local search loop

        if len(current_blocks) < best_count:
            best_count = len(current_blocks)
            best_blocks = [list(k_subsets[b]) for b in current_blocks]

    if best_blocks is None:
        if time.time() < deadline:
             uncovered_mask = (1 << num_t_subsets) - 1
             current_blocks = []
             used_blocks = set()
             while uncovered_mask:
                 max_gain = 0
                 best_i = -1
                 for i in range(num_k_subsets):
                     if i in used_blocks: continue
                     gain = (coverage_masks[i] & uncovered_mask).bit_count()
                     if gain > max_gain:
                         max_gain = gain
                         best_i = i
                 if max_gain == 0: break
                 current_blocks.append(best_i)
                 used_blocks.add(best_i)
                 uncovered_mask &= ~coverage_masks[best_i]
             if not uncovered_mask:
                 best_blocks = [list(k_subsets[b]) for b in current_blocks]
                 best_count = len(current_blocks)

    return best_blocks
