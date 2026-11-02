import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    
    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Build incidence structure
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(ts_list)

    # Convert to numpy arrays for fast operations
    block_to_ts_np = [np.array(lst, dtype=np.int32) for lst in block_to_ts]
    ts_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in ts_to_blocks]
    
    # Bitmask representation for fast coverage tracking
    masks = [0] * num_k_blocks
    for i, ts_list in enumerate(block_to_ts):
        mask = 0
        for idx in ts_list:
            mask |= (1 << idx)
        masks[i] = mask

    def run_greedy(seed, explore_prob=0.0):
        random.seed(seed)
        gain = np.full(num_k_blocks, len(block_to_ts[0]) if num_k_blocks > 0 else 0, dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []
        
        while True:
            if time.time() > deadline - 5.0:
                return None
            
            max_gain = int(np.max(gain))
            if max_gain <= 0:
                break
                
            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break
                
            if explore_prob > 0 and random.random() < explore_prob:
                # Explore: pick from blocks with gain >= max_gain - 3
                explore_cands = np.where((gain >= max_gain - 3) & (~is_selected))[0]
                if len(explore_cands) > 0:
                    b_idx = int(explore_cands[random.randint(0, len(explore_cands) - 1)])
                else:
                    b_idx = int(candidates[random.randint(0, len(candidates) - 1)])
            else:
                b_idx = int(candidates[random.randint(0, len(candidates) - 1)])
            
            sol.append(b_idx)
            is_selected[b_idx] = True
            gain[b_idx] = -1
            
            # Identify newly covered t-subsets
            bts = block_to_ts_np[b_idx]
            newly_covered_mask = is_uncovered[bts]
            newly_covered = bts[newly_covered_mask]
            
            if len(newly_covered) == 0:
                continue
                
            is_uncovered[newly_covered] = False
            
            # Vectorized gain update using bincount
            blocks_to_decrement = []
            for ts_idx in newly_covered:
                blocks_to_decrement.extend(ts_to_blocks[ts_idx])
                
            if blocks_to_decrement:
                counts = np.bincount(blocks_to_decrement, minlength=num_k_blocks)
                gain -= counts
                gain[is_selected] = -1000

        return sol

    def optimize_redundancy(block_indices):
        """Remove redundant blocks and perform local search for swaps"""
        if not block_indices:
            return []
        
        current_blocks = list(block_indices)
        
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for idx in current_blocks:
            for ts_idx in block_to_ts[idx]:
                cover_count[ts_idx] += 1
        
        # Remove redundant blocks
        changed = True
        while changed:
            changed = False
            for i in range(len(current_blocks)):
                if time.time() > deadline - 3.0:
                    break
                
                idx = current_blocks[i]
                redundant = True
                for ts_idx in block_to_ts[idx]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                        
                if redundant:
                    for ts_idx in block_to_ts[idx]:
                        cover_count[ts_idx] -= 1
                    current_blocks.pop(i)
                    changed = True
                    break
        
        # Local search: try to swap blocks to improve coverage efficiency
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            random.shuffle(current_blocks)
            
            for i in range(len(current_blocks)):
                if time.time() > deadline - 3.0:
                    break
                    
                idx = current_blocks[i]
                # Find t-subsets uniquely covered by this block
                unique_ts = [ts_idx for ts_idx in block_to_ts[idx] if cover_count[ts_idx] == 1]
                
                if not unique_ts:
                    # This block is redundant, remove it
                    for ts_idx in block_to_ts[idx]:
                        cover_count[ts_idx] -= 1
                    current_blocks.pop(i)
                    improved = True
                    break
                
                # Find candidate blocks that cover all unique_ts
                candidates = set(ts_to_blocks[unique_ts[0]])
                for ts_idx in unique_ts[1:]:
                    candidates.intersection_update(ts_to_blocks[ts_idx])
                    if not candidates:
                        break
                
                if not candidates:
                    continue
                
                # Filter out already selected blocks
                selected_set = set(current_blocks)
                valid_candidates = [c for c in candidates if c not in selected_set]
                
                if not valid_candidates:
                    continue
                
                # Choose a candidate that maximizes new coverage
                best_candidate = None
                best_gain = -1
                for c in valid_candidates[:20]:  # Limit candidates for speed
                    new_gain = 0
                    for ts_idx in block_to_ts[c]:
                        if cover_count[ts_idx] == 0:
                            new_gain += 1
                    if new_gain > best_gain:
                        best_gain = new_gain
                        best_candidate = c
                
                if best_candidate is not None and best_gain > 0:
                    # Perform swap
                    for ts_idx in block_to_ts[idx]:
                        cover_count[ts_idx] -= 1
                    for ts_idx in block_to_ts[best_candidate]:
                        cover_count[ts_idx] += 1
                    current_blocks[i] = best_candidate
                    improved = True
                    break
        
        return current_blocks

    def spectral_prune(block_indices):
        """Spectral sparsification: prune blocks with low spectral contribution"""
        if not block_indices or len(block_indices) <= 10:
            return block_indices
        
        current_blocks = list(block_indices)
        n_blocks = len(current_blocks)
        
        # Compute coverage uniformity metric
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for idx in current_blocks:
            for ts_idx in block_to_ts[idx]:
                cover_count[ts_idx] += 1
        
        # Identify blocks with low "spectral contribution"
        # A block has low contribution if it covers many already-well-covered t-subsets
        block_scores = np.zeros(n_blocks, dtype=np.float64)
        for i, idx in enumerate(current_blocks):
            score = 0.0
            for ts_idx in block_to_ts[idx]:
                # Weight by inverse of coverage count (well-covered t-subsets contribute less)
                score += 1.0 / (cover_count[ts_idx] + 1)
            block_scores[i] = score
        
        # Sort blocks by score (ascending) and try removing low-score blocks
        order = np.argsort(block_scores)
        
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            for pos in order:
                if time.time() > deadline - 3.0:
                    break
                if pos >= len(current_blocks):
                    continue
                    
                idx = current_blocks[pos]
                # Check if this block is redundant
                redundant = True
                for ts_idx in block_to_ts[idx]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                        
                if redundant:
                    for ts_idx in block_to_ts[idx]:
                        cover_count[ts_idx] -= 1
                    current_blocks.pop(pos)
                    # Update scores for remaining blocks
                    for i in range(len(current_blocks)):
                        if i >= pos:
                            order[i] -= 1
                    improved = True
                    break
        
        return current_blocks

    # Main search loop
    best_solution = None
    best_len = float('inf')
    
    # Phase 1: Deterministic greedy
    sol = run_greedy(0)
    if sol is not None:
        optimized = optimize_redundancy(sol)
        spectral = spectral_prune(optimized)
        best_solution = spectral
        best_len = len(spectral)
    
    # Phase 2: Randomized greedy with varying exploration
    num_runs = 150
    for i in range(1, num_runs):
        if time.time() > deadline - 15.0:
            break
        
        # Vary exploration probability
        explore = 0.0 if i < 50 else (0.1 if i < 100 else 0.2)
        sol = run_greedy(i, explore_prob=explore)
        if sol is not None:
            optimized = optimize_redundancy(sol)
            spectral = spectral_prune(optimized)
            if len(spectral) < best_len:
                best_len = len(spectral)
                best_solution = spectral
    
    # Convert block indices to actual blocks
    if best_solution is None:
        return []
    
    return [list(all_k_blocks[idx]) for idx in best_solution]
