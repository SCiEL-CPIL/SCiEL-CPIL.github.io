import itertools
import random
import time
import sys

def construct(v, k, t):
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # Map each block to the indices of the t-subsets it covers
    block_t_indices = []
    for block in all_k_subsets:
        indices = [t_subset_to_idx[s] for s in itertools.combinations(block, t)]
        block_t_indices.append(indices)
        
    # Map each t-subset to the blocks that cover it
    t_subset_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, indices in enumerate(block_t_indices):
        for idx in indices:
            t_subset_blocks[idx].append(b_idx)
            
    # Precompute degrees of t-subsets (number of blocks covering each)
    t_subset_degrees = [len(lst) for lst in t_subset_blocks]

    # 2. Phase 1: Weighted Greedy Independent Set (Packing)
    pool = set(range(num_k_subsets))
    core_blocks = []
    uncovered_mask = [1] * num_t_subsets
    
    # Calculate initial "scarcity" weight for each block
    block_weights = [0.0] * num_k_subsets
    for b_idx, indices in enumerate(block_t_indices):
        w = 0.0
        for t_idx in indices:
            deg = t_subset_degrees[t_idx]
            if deg > 0:
                w += 1.0 / deg
        block_weights[b_idx] = w

    while pool:
        best_block = -1
        max_score = -1.0
        
        for b_idx in pool:
            cov_uncovered = 0
            scarcity_sum = 0.0
            for t_idx in block_t_indices[b_idx]:
                if uncovered_mask[t_idx]:
                    cov_uncovered += 1
                    scarcity_sum += 1.0 / t_subset_degrees[t_idx]
            
            if cov_uncovered == 0:
                continue
                
            score = cov_uncovered * 100.0 + scarcity_sum
            
            if score > max_score:
                max_score = score
                best_block = b_idx
                
        if best_block == -1:
            break
            
        core_blocks.append(best_block)
        
        for t_idx in block_t_indices[best_block]:
            uncovered_mask[t_idx] = 0
            
        neighbors_to_remove = set()
        for t_idx in block_t_indices[best_block]:
            neighbors_to_remove.update(t_subset_blocks[t_idx])
        
        pool.difference_update(neighbors_to_remove)

    # 3. Phase 2: Solve Remaining Set Cover (optimized)
    active_blocks = set(core_blocks)
    cover_count = [0] * num_t_subsets
    
    for b_idx in active_blocks:
        for t_idx in block_t_indices[b_idx]:
            cover_count[t_idx] += 1
            
    still_uncovered = [i for i in range(num_t_subsets) if cover_count[i] == 0]
    
    if still_uncovered:
        residual_uncovered = set(still_uncovered)
        
        while residual_uncovered:
            best_block = -1
            max_gain = 0
            
            for b_idx in range(num_k_subsets):
                if b_idx in active_blocks:
                    continue
                gain = 0
                for t_idx in block_t_indices[b_idx]:
                    if t_idx in residual_uncovered:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_block = b_idx
                    
            if best_block == -1 or max_gain == 0:
                break
                
            active_blocks.add(best_block)
            for t_idx in block_t_indices[best_block]:
                if t_idx in residual_uncovered:
                    residual_uncovered.remove(t_idx)
                cover_count[t_idx] += 1

    # 4. Minimize
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

    active_blocks, cover_count = minimize(active_blocks, cover_count)

    # 5. Local Search (Enhanced with 2-for-1 swaps)
    start_time = time.time()
    time_limit = 110.0
    
    def try_remove(b_idx):
        nonlocal active_blocks, cover_count
        if b_idx not in active_blocks:
            return False
        for t_idx in block_t_indices[b_idx]:
            if cover_count[t_idx] <= 1:
                return False
        active_blocks.remove(b_idx)
        for t_idx in block_t_indices[b_idx]:
            cover_count[t_idx] -= 1
        return True

    def try_add(b_idx):
        nonlocal active_blocks, cover_count
        if b_idx in active_blocks:
            return False
        active_blocks.add(b_idx)
        for t_idx in block_t_indices[b_idx]:
            cover_count[t_idx] += 1
        return True

    def try_swap(old_idx, new_idx):
        nonlocal active_blocks, cover_count
        if old_idx not in active_blocks or new_idx in active_blocks:
            return False
        
        unique_subsets = []
        for t_idx in block_t_indices[old_idx]:
            if cover_count[t_idx] == 1:
                unique_subsets.append(t_idx)
                
        if not unique_subsets:
            pass
        else:
            for t_idx in unique_subsets:
                if t_idx not in block_t_indices[new_idx]:
                    return False

        active_blocks.remove(old_idx)
        active_blocks.add(new_idx)
        
        for t_idx in block_t_indices[old_idx]:
            cover_count[t_idx] -= 1
        for t_idx in block_t_indices[new_idx]:
            cover_count[t_idx] += 1
        return True

    def try_2for1_swap(old1, old2, new_idx):
        """Replace two blocks with one if the new block covers all unique subsets of both old blocks."""
        nonlocal active_blocks, cover_count
        if old1 not in active_blocks or old2 not in active_blocks or new_idx in active_blocks:
            return False
        if old1 == old2:
            return False
        
        # Check if new_idx covers all unique subsets of old1 and old2
        for t_idx in block_t_indices[old1]:
            if cover_count[t_idx] == 1 and t_idx not in block_t_indices[new_idx]:
                return False
        for t_idx in block_t_indices[old2]:
            if cover_count[t_idx] == 1 and t_idx not in block_t_indices[new_idx]:
                return False
        
        # Perform the swap
        active_blocks.remove(old1)
        active_blocks.remove(old2)
        active_blocks.add(new_idx)
        
        for t_idx in block_t_indices[old1]:
            cover_count[t_idx] -= 1
        for t_idx in block_t_indices[old2]:
            cover_count[t_idx] -= 1
        for t_idx in block_t_indices[new_idx]:
            cover_count[t_idx] += 1
        return True

    while time.time() - start_time < time_limit:
        improved = False
        current_blocks = list(active_blocks)
        random.shuffle(current_blocks)
        
        for B in current_blocks:
            if B not in active_blocks:
                continue
            
            # Try Remove
            if try_remove(B):
                active_blocks, cover_count = minimize(active_blocks, cover_count)
                improved = True
                continue
            
            # Try Swap
            unique_subsets = []
            for t_idx in block_t_indices[B]:
                if cover_count[t_idx] == 1:
                    unique_subsets.append(t_idx)
            
            if unique_subsets:
                candidates = None
                for t_idx in unique_subsets:
                    blocks_list = t_subset_blocks[t_idx]
                    if candidates is None:
                        candidates = set(blocks_list)
                    else:
                        candidates.intersection_update(blocks_list)
                    if not candidates:
                        break
                
                if candidates:
                    valid_candidates = [c for c in candidates if c != B and c not in active_blocks]
                    if valid_candidates:
                        B_new = random.choice(valid_candidates)
                        if try_swap(B, B_new):
                            active_blocks, cover_count = minimize(active_blocks, cover_count)
                            improved = True
            else:
                pass
        
        # Try 2-for-1 swaps
        if time.time() - start_time < time_limit - 5:
            current_blocks = list(active_blocks)
            random.shuffle(current_blocks)
            n_blocks = len(current_blocks)
            # Only try a limited number of pairs to save time
            max_pairs = min(200, n_blocks * (n_blocks - 1) // 2)
            pairs_tried = 0
            for i in range(n_blocks):
                if pairs_tried >= max_pairs:
                    break
                B1 = current_blocks[i]
                if B1 not in active_blocks:
                    continue
                for j in range(i + 1, n_blocks):
                    if pairs_tried >= max_pairs:
                        break
                    B2 = current_blocks[j]
                    if B2 not in active_blocks:
                        continue
                    pairs_tried += 1
                    
                    # Find candidates that cover unique subsets of both B1 and B2
                    unique1 = [t_idx for t_idx in block_t_indices[B1] if cover_count[t_idx] == 1]
                    unique2 = [t_idx for t_idx in block_t_indices[B2] if cover_count[t_idx] == 1]
                    
                    if unique1 or unique2:
                        all_unique = set(unique1) | set(unique2)
                        candidates = None
                        for t_idx in all_unique:
                            blocks_list = t_subset_blocks[t_idx]
                            if candidates is None:
                                candidates = set(blocks_list)
                            else:
                                candidates.intersection_update(blocks_list)
                            if not candidates:
                                break
                        
                        if candidates:
                            valid_candidates = [c for c in candidates if c not in active_blocks]
                            if valid_candidates:
                                B_new = random.choice(valid_candidates)
                                if try_2for1_swap(B1, B2, B_new):
                                    active_blocks, cover_count = minimize(active_blocks, cover_count)
                                    improved = True
        
        if not improved:
            # Try random perturbation: add a random block, then minimize
            inactive_blocks = [b for b in range(num_k_subsets) if b not in active_blocks]
            if inactive_blocks:
                B_rand = random.choice(inactive_blocks)
                try_add(B_rand)
                active_blocks, cover_count = minimize(active_blocks, cover_count)
                # Check if we improved (size decreased or same)
                # Since we added one and minimized, if size is now <= original, keep it
                # Otherwise, revert (but we don't have the old state easily)
                # Just keep it for now; next iterations will try to improve further
            else:
                break

    # 6. Final reduction pass
    active_blocks, cover_count = minimize(active_blocks, cover_count)
    
    result = [list(all_k_subsets[b_idx]) for b_idx in active_blocks]
    return result
