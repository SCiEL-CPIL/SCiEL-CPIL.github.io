import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin under 120s
    
    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # For each k-subset, compute the set of t-subset indices it covers
    block_t_indices = []
    for block in all_k_subsets:
        indices = tuple(sorted(t_subset_to_idx[s] for s in itertools.combinations(block, t)))
        block_t_indices.append(indices)
    
    # For each t-subset, which blocks cover it
    t_subset_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, indices in enumerate(block_t_indices):
        for idx in indices:
            t_subset_blocks[idx].append(b_idx)
    
    # Precompute block masks as bitsets for fast coverage gain computation
    # Using Python integers as bitsets (since num_t_subsets can be large)
    block_masks = [0] * num_k_subsets
    for b_idx, indices in enumerate(block_t_indices):
        mask = 0
        for idx in indices:
            mask |= (1 << idx)
        block_masks[b_idx] = mask
    
    # Helper: minimize a given set of blocks (remove redundant blocks)
    def minimize(active_blocks_set, cover_counts):
        changed = True
        while changed:
            changed = False
            items = list(active_blocks_set)
            random.shuffle(items)
            for b_idx in items:
                if b_idx not in active_blocks_set:
                    continue
                can_remove = True
                for t_idx in block_t_indices[b_idx]:
                    if cover_counts[t_idx] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    active_blocks_set.remove(b_idx)
                    for t_idx in block_t_indices[b_idx]:
                        cover_counts[t_idx] -= 1
                    changed = True
        return active_blocks_set, cover_counts
    
    # Helper: Greedy Set Cover with randomization among ties
    def greedy_cover():
        uncovered_mask = (1 << num_t_subsets) - 1  # all bits set
        gain = [len(block_t_indices[i]) for i in range(num_k_subsets)]
        selected_blocks = []
        
        while uncovered_mask != 0:
            # Find max gain
            max_gain = max(gain)
            if max_gain == 0:
                break
            
            # Find all blocks with max gain, pick randomly
            candidates = [i for i in range(num_k_subsets) if gain[i] == max_gain]
            best_block = random.choice(candidates)
            
            selected_blocks.append(best_block)
            
            # Update gains using bitset operations
            new_covered = block_masks[best_block] & uncovered_mask
            uncovered_mask ^= new_covered  # clear newly covered bits
            
            # Update gains: for each block, subtract the number of newly covered bits it has
            # This is expensive to do naively, so we use a different approach:
            # For each t_idx in new_covered bits, decrement gain for blocks covering it
            # But this is also expensive. Let's use a hybrid:
            # Actually, let's just recompute gain for affected blocks
            # Find which t_indices are in new_covered
            newly_covered_indices = []
            temp = new_covered
            while temp:
                # Find lowest set bit
                lsb = temp & (-temp)
                idx = lsb.bit_length() - 1
                newly_covered_indices.append(idx)
                temp ^= lsb
            
            for t_idx in newly_covered_indices:
                for b_idx in t_subset_blocks[t_idx]:
                    gain[b_idx] -= 1
                        
        return selected_blocks
    
    # Helper: Local Search with swap + minimize
    def local_search(active_blocks, cover_count, time_limit):
        start_time_ls = time.time()
        
        while time.time() - start_time_ls < time_limit:
            improved = False
            current_blocks = list(active_blocks)
            random.shuffle(current_blocks)
            
            for B in current_blocks:
                if time.time() - start_time_ls >= time_limit:
                    break
                if B not in active_blocks:
                    continue
                
                # Find t-subsets uniquely covered by B
                uncovered_by_B = []
                for t_idx in block_t_indices[B]:
                    if cover_count[t_idx] == 1:
                        uncovered_by_B.append(t_idx)
                
                if not uncovered_by_B:
                    continue

                # Find candidate blocks that cover all of uncovered_by_B
                # Start with the smallest t_subset_blocks list
                uncovered_by_B.sort(key=lambda x: len(t_subset_blocks[x]))
                
                candidates = None
                for t_idx in uncovered_by_B:
                    blocks_list = t_subset_blocks[t_idx]
                    if candidates is None:
                        candidates = set(blocks_list)
                    else:
                        candidates.intersection_update(blocks_list)
                    
                    if not candidates:
                        break
                
                if not candidates:
                    continue
                
                # Filter: must not be B, must not already be in active_blocks
                valid_candidates = [c for c in candidates if c != B and c not in active_blocks]
                
                if not valid_candidates:
                    continue
                
                # Perform swap
                B_new = random.choice(valid_candidates)
                
                active_blocks.remove(B)
                active_blocks.add(B_new)
                
                for t_idx in block_t_indices[B]:
                    cover_count[t_idx] -= 1
                for t_idx in block_t_indices[B_new]:
                    cover_count[t_idx] += 1
                
                # Minimize after swap
                active_blocks, cover_count = minimize(active_blocks, cover_count)
                improved = True
                
            if not improved:
                break
        
        return active_blocks, cover_count
    
    # Helper: Simulated Annealing for block replacement
    def simulated_annealing(active_blocks, cover_count, time_limit):
        start_time_sa = time.time()
        current_temp = 50.0  # Initial temperature
        cooling_rate = 0.95
        min_temp = 0.5
        
        while time.time() - start_time_sa < time_limit and current_temp > min_temp:
            # Pick a random block to potentially replace
            current_blocks = list(active_blocks)
            random.shuffle(current_blocks)
            
            improved_this_round = False
            
            for B in current_blocks:
                if time.time() - start_time_sa >= time_limit:
                    break
                if B not in active_blocks:
                    continue
                
                # Find t-subsets uniquely covered by B
                uncovered_by_B = []
                for t_idx in block_t_indices[B]:
                    if cover_count[t_idx] == 1:
                        uncovered_by_B.append(t_idx)
                
                if not uncovered_by_B:
                    # B is redundant, remove it
                    active_blocks.remove(B)
                    for t_idx in block_t_indices[B]:
                        cover_count[t_idx] -= 1
                    improved_this_round = True
                    continue

                # Find candidate blocks that cover all of uncovered_by_B
                uncovered_by_B.sort(key=lambda x: len(t_subset_blocks[x]))
                
                candidates = None
                for t_idx in uncovered_by_B:
                    blocks_list = t_subset_blocks[t_idx]
                    if candidates is None:
                        candidates = set(blocks_list)
                    else:
                        candidates.intersection_update(blocks_list)
                    
                    if not candidates:
                        break
                
                if not candidates:
                    continue
                
                # Filter: must not be B, must not already be in active_blocks
                valid_candidates = [c for c in candidates if c != B and c not in active_blocks]
                
                if not valid_candidates:
                    continue
                
                # Evaluate candidates based on how many additional t-subsets they cover
                best_candidate = None
                best_score = -float('inf')
                
                # Compute current coverage of B
                B_coverage_mask = block_masks[B]
                
                for C in valid_candidates:
                    # Score: number of new t-subsets covered by C that aren't covered by B
                    C_mask = block_masks[C]
                    new_coverage = C_mask & ~B_coverage_mask
                    score = new_coverage.bit_count()
                    
                    # Add some randomness to break ties
                    score += random.uniform(-1, 1)
                    
                    if score > best_score:
                        best_score = score
                        best_candidate = C
                
                if best_candidate is None:
                    continue
                
                # Compute the change in uncovered count
                # Removing B: uncovered_by_B becomes uncovered
                # Adding C: may cover some of those
                C_mask = block_masks[best_candidate]
                
                # How many of uncovered_by_B does C cover?
                covered_by_C = 0
                for t_idx in uncovered_by_B:
                    if (C_mask >> t_idx) & 1:
                        covered_by_C += 1
                
                # Net change in uncovered: len(uncovered_by_B) - covered_by_C
                # If negative, we're improving
                delta_uncovered = len(uncovered_by_B) - covered_by_C
                
                # Accept if improving or with probability exp(-delta/temp)
                if delta_uncovered <= 0 or random.random() < np.exp(-delta_uncovered / current_temp):
                    active_blocks.remove(B)
                    active_blocks.add(best_candidate)
                    
                    for t_idx in block_t_indices[B]:
                        cover_count[t_idx] -= 1
                    for t_idx in block_t_indices[best_candidate]:
                        cover_count[t_idx] += 1
                    
                    improved_this_round = True
            
            # Cool down
            current_temp *= cooling_rate
        
        # Final minimize
        active_blocks, cover_count = minimize(active_blocks, cover_count)
        
        return active_blocks, cover_count
    
    # Main Loop: Multi-Start
    best_blocks = None
    best_size = float('inf')
    
    # First run: quick greedy to get a baseline
    selected = greedy_cover()
    active_blocks = set(selected)
    cover_count = [0] * num_t_subsets
    for b_idx in active_blocks:
        for t_idx in block_t_indices[b_idx]:
            cover_count[t_idx] += 1
    active_blocks, cover_count = minimize(active_blocks, cover_count)
    best_size = len(active_blocks)
    best_blocks = list(active_blocks)
    
    # Continue with more iterations if time allows
    while time.time() < deadline - 2.0:
        # Greedy with randomization
        selected = greedy_cover()
        
        # Minimize
        active_blocks = set(selected)
        cover_count = [0] * num_t_subsets
        for b_idx in active_blocks:
            for t_idx in block_t_indices[b_idx]:
                cover_count[t_idx] += 1
        active_blocks, cover_count = minimize(active_blocks, cover_count)
        
        # Local Search with remaining time budget
        remaining_time = deadline - 2.0 - time.time()
        if remaining_time > 0.5:
            # Give local search and simulated annealing time
            ls_time = max(0.2, remaining_time * 0.25)
            sa_time = max(0.2, remaining_time * 0.25)
            
            active_blocks, cover_count = local_search(active_blocks, cover_count, ls_time)
            active_blocks, cover_count = simulated_annealing(active_blocks, cover_count, sa_time)
        
        # Update Best
        current_size = len(active_blocks)
        if current_size < best_size:
            best_size = current_size
            best_blocks = list(active_blocks)
            
            # Lower bound is 131; if we hit it, stop
            if best_size <= 131:
                break
    
    if best_blocks is None:
        selected = greedy_cover()
        best_blocks = selected

    result = [list(all_k_subsets[b_idx]) for b_idx in best_blocks]
    return result
