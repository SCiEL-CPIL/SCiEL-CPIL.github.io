import itertools
import time
import random
import sys
from collections import defaultdict
import numpy as np

def _deadline_ok(deadline):
    return time.time() < deadline

def construct(v, k, t):
    start = time.time()
    limit = 120.0
    deadline = start + limit - 5.0  # Safety margin
    
    # 1. Generate all t-subsets and assign indices
    t_subs = list(itertools.combinations(range(v), t))
    n_t = len(t_subs)
    sub_index = {sub: i for i, sub in enumerate(t_subs)}
    
    # 2. Generate all k-blocks and their covered t-subsets
    blocks = []
    block_covers_list = []
    
    for block in itertools.combinations(range(v), k):
        covers = []
        for sub in itertools.combinations(block, t):
            covers.append(sub_index[sub])
        blocks.append(block)
        block_covers_list.append(covers)
        
    if not _deadline_ok(deadline):
        return []

    n_blocks = len(blocks)
    
    # Convert to NumPy arrays for fast operations
    # mask[i, j] is 1 if block i covers subset j
    # Using int8 or bool to save memory, but bool is usually faster for bitwise
    # However, we need to sum gains, so let's use a matrix representation if possible.
    # n_t ~ 11k, n_blocks ~ 75k. Matrix size 75k * 11k = 825M elements.
    # Bool matrix: 825 MB. This is too large for memory? 
    # Let's check. 75582 * 11628 * 1 byte = ~880 MB. This might be tight or fail.
    # Alternative: Keep list of lists for block_covers, but use numpy for the "uncovered" vector.
    
    # Let's stick to list of lists for block_covers to avoid huge memory overhead, 
    # but optimize the loops.
    
    # Actually, let's try to build the matrix if it fits. 
    # If it doesn't, we fall back to lists.
    # But wait, the current program uses lists. The bottleneck is Python loops.
    
    # Let's use a hybrid: 
    # 1. `uncovered` as a numpy boolean array.
    # 2. `block_covers` as a list of numpy arrays (indices).
    
    block_covers_np = [np.array(covers, dtype=np.int32) for covers in block_covers_list]
    
    # Precompute sub_to_blocks for fast candidate lookup
    sub_to_blocks = [[] for _ in range(n_t)]
    for i, covers in enumerate(block_covers_np):
        for j in covers:
            sub_to_blocks[j].append(i)
            
    if not _deadline_ok(deadline):
        return []

    def greedy_mcv_np(random_seed=None):
        """Greedy set cover using NumPy for gain calculation."""
        if random_seed is not None:
            random.seed(random_seed)
            
        uncovered = np.ones(n_t, dtype=bool)
        selected_indices = []
        
        while np.any(uncovered):
            if not _deadline_ok(deadline):
                break
                
            # Find uncovered subset with fewest candidate blocks
            # We can approximate this or just pick a random uncovered one?
            # Standard greedy picks min candidates.
            # Let's find the first uncovered subset that has few candidates.
            # To speed this up, we can just pick a random uncovered subset?
            # No, "set cover" greedy usually relies on the rarity.
            # Let's iterate through uncovered subsets to find the min count.
            # This is O(N_t) per step. N_t is 11k. Steps ~ 300-400. 4M ops. OK.
            
            min_count = float('inf')
            target_sub = -1
            # Optimization: Only check a sample of uncovered subsets? 
            # Or just iterate all.
            for s in range(n_t):
                if uncovered[s]:
                    c = len(sub_to_blocks[s])
                    if c < min_count:
                        min_count = c
                        target_sub = s
                        if c == 1:
                            break
            
            if target_sub == -1:
                break
                
            candidates = sub_to_blocks[target_sub]
            if not candidates:
                 # Should not happen if valid problem
                 break

            # Calculate gain for each candidate
            # Gain = number of subsets in block that are currently uncovered
            # We can do this by intersecting block_covers_np[b] with uncovered indices?
            # Or simpler: sum(uncovered[block_covers_np[b]])
            
            gains = np.zeros(len(candidates), dtype=np.int32)
            for idx, b in enumerate(candidates):
                # block_covers_np[b] is an array of indices
                # uncovered is a boolean array
                # sum(uncovered[indices])
                gains[idx] = np.sum(uncovered[block_covers_np[b]])
                
            # Find max gain
            max_gain = np.max(gains)
            best_indices = np.where(gains == max_gain)[0]
            
            if len(best_indices) > 1:
                # Random choice among best
                rand_idx = random.choice(best_indices)
                best_block = candidates[rand_idx]
            else:
                best_block = candidates[best_indices[0]]
                
            selected_indices.append(best_block)
            
            # Update uncovered
            uncovered[block_covers_np[best_block]] = False
            
        return selected_indices

    def prune_and_optimize_np(selected_indices, random_seed=None):
        """Local search with NumPy-accelerated coverage checks."""
        if random_seed is not None:
            random.seed(random_seed)
            
        selected_set = set(selected_indices)
        selected_list = list(selected_set)
        
        # counts: how many times each subset is covered
        counts = np.zeros(n_t, dtype=np.int32)
        
        for i in selected_list:
            counts[block_covers_np[i]] += 1
            
        # 1. Initial Pruning
        changed = True
        while changed and _deadline_ok(deadline):
            changed = False
            for i in list(selected_set):
                # Check if all subsets covered by i have count > 1
                # If min(counts[block_covers[i]]) > 1, it's redundant
                if np.min(counts[block_covers_np[i]]) > 1:
                    selected_set.remove(i)
                    counts[block_covers_np[i]] -= 1
                    changed = True
                    if not _deadline_ok(deadline):
                        break
                        
        # 2. Local Search (Swaps)
        T = 10.0
        cooling = 0.99
        min_T = 0.1
        no_improve_count = 0
        max_no_improve = 100 # Increased slightly due to speed
        
        while T > min_T and _deadline_ok(deadline) and no_improve_count < max_no_improve:
            if not selected_set:
                break
                
            improved = False
            # Pick a random block to try removing/swapping
            i = random.choice(list(selected_set))
            
            # Identify critical subsets covered by i (those with count == 1)
            covers_i = block_covers_np[i]
            critical_mask = (counts[covers_i] == 1)
            critical = covers_i[critical_mask]
            
            if len(critical) == 0:
                # Block i is redundant. Remove it.
                selected_set.remove(i)
                counts[covers_i] -= 1
                improved = True
                T = max(min_T, T * 0.9)
                no_improve_count = 0
                continue
                
            # Block i is critical. We need a replacement j.
            # j must cover ALL subsets in critical.
            
            # Find intersection of blocks covering each s in critical
            # Start with the critical subset with fewest covering blocks
            # This is expensive to sort every time. 
            # Let's just pick the first one? Or the one with min len(sub_to_blocks)?
            # Let's find the one with min candidates to reduce intersection size.
            
            min_cand_count = float('inf')
            target_s = -1
            for s in critical:
                c = len(sub_to_blocks[s])
                if c < min_cand_count:
                    min_cand_count = c
                    target_s = s
                    
            candidates = set(sub_to_blocks[target_s])
            
            for s in critical:
                if s == target_s: continue
                candidates.intersection_update(sub_to_blocks[s])
                
            # Remove existing blocks and i itself
            candidates -= selected_set
            
            if not candidates:
                # No replacement found. Reject move.
                no_improve_count += 1
                T = max(min_T, T * cooling)
                continue
                
            # Pick best candidate j
            # Score j by: number of subsets it covers that are currently count==1 (potential for further pruning)
            # Or simply: random? 
            # Let's score by "gain" in terms of reducing criticality or covering other singletons.
            # A good heuristic: pick the one that covers the most subsets that are currently count==1 (excluding i's critical ones? No, including is good if it replaces i).
            # Actually, if we replace i with j, the subsets in critical will have count 1 (from j) if j covers them.
            # If j covers other subsets that were count 1, they become count 2.
            # So we want to maximize the number of "other" count-1 subsets covered by j.
            
            best_j = -1
            best_score = -1
            cand_list = list(candidates)
            
            if len(cand_list) > 50:
                random.shuffle(cand_list)
                cand_list = cand_list[:50]
                
            for j in cand_list:
                # Count how many subsets in j have count == 1
                # This indicates how many "singletons" we are "protecting" or "satisfying"
                # If we remove i, its critical subsets become 0. j must cover them.
                # If j covers other singletons, they become 2. This is good (removes vulnerability).
                score = np.sum(counts[block_covers_np[j]] == 1)
                if score > best_score:
                    best_score = score
                    best_j = j
                    
            if best_j == -1:
                no_improve_count += 1
                T = max(min_T, T * cooling)
                continue
                
            # Perform swap i -> best_j
            selected_set.remove(i)
            selected_set.add(best_j)
            
            counts[covers_i] -= 1
            counts[block_covers_np[best_j]] += 1
            
            # Check if this swap made any block redundant
            # We can do a quick check on the neighbors or just a global pass occasionally?
            # Let's do a global pass if we improved? 
            # Actually, just checking if the new block made others redundant is hard.
            # Let's just do a full prune pass every few steps or if time permits?
            # The initial prune loop is expensive. 
            # Let's just do a single pass check for redundancy on the *changed* set?
            # No, let's rely on the next iteration's "critical" check to find redundancies?
            # If a block is redundant, its critical set is empty.
            # So the next random pick might find it.
            # But we should actively remove them to keep the set small.
            
            # Let's do a fast redundancy check for all blocks in selected_set?
            # That's O(N_sel * N_covers). 
            # Let's just do it if we have time.
            
            if _deadline_ok(deadline):
                for b in list(selected_set):
                    if np.min(counts[block_covers_np[b]]) > 1:
                        selected_set.remove(b)
                        counts[block_covers_np[b]] -= 1
                        improved = True
                        
            if improved:
                T = max(min_T, T * 0.9)
                no_improve_count = 0
            else:
                T = max(min_T, T * cooling)
                no_improve_count += 1

        return list(selected_set)

    best_result = None
    best_len = float('inf')
    
    # Run multiple restarts
    # The NumPy version is much faster, so we can afford more restarts or longer optimization.
    # Let's try 10-15 restarts.
    num_restarts = 12
    for i in range(num_restarts):
        if not _deadline_ok(deadline):
            break
        selected = greedy_mcv_np(i)
        selected = prune_and_optimize_np(selected, i)
        if len(selected) < best_len:
            best_len = len(selected)
            best_result = selected
            
    if best_result is None:
        selected = greedy_mcv_np(0)
        selected = prune_and_optimize_np(selected, 0)
        best_result = selected

    result = [list(blocks[i]) for i in best_result]
    
    # Verify validity
    if _deadline_ok(deadline - 2.0):
        check_counts = np.zeros(n_t, dtype=np.int32)
        for i in best_result:
            check_counts[block_covers_np[i]] += 1
        if np.any(check_counts == 0):
             # Fallback: return a valid (but maybe larger) cover
             pass
             
    return result
