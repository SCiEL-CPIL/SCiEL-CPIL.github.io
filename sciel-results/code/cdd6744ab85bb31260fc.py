import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = 150.0
    safety_margin = 10.0

    # Precompute t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {subset: i for i, subset in enumerate(t_subsets)}
    n_t = len(t_subsets)

    # Precompute k-subsets and their t-subset coverage
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    n_t_in_k = len(list(itertools.combinations(range(k), t)))

    # Build mapping: for each block, which t-subsets does it cover?
    block_t_indices = np.zeros((n_k, n_t_in_k), dtype=np.int32)
    for i, block in enumerate(k_subsets):
        for j, t_combo in enumerate(itertools.combinations(block, t)):
            block_t_indices[i, j] = t_sub_to_idx[t_combo]

    def greedy_cover(num_kicks=0, kick_prob=0.3, kick_window=2):
        """
        Greedy set cover with optional random kicks.
        num_kicks: total number of kicks to apply
        kick_prob: probability of kicking at each step when kicks remain
        kick_window: how many gain levels below max to consider when kicking
        """
        uncovered = np.ones(n_t, dtype=bool)
        selected_blocks = []
        selected_set = set()
        kicks_remaining = num_kicks

        while np.any(uncovered):
            if time.time() - start_time > deadline - safety_margin:
                break

            # Calculate gain for each block (vectorized)
            gains = uncovered[block_t_indices].sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break

            # Find all blocks with max gain
            max_mask = gains == max_gain
            candidates = np.where(max_mask)[0]

            # Apply randomized kick if available
            if kicks_remaining > 0 and len(candidates) > 1 and random.random() < kick_prob:
                # Pick from blocks with gain >= max_gain - kick_window
                threshold = max_gain - kick_window
                perturbed_candidates = np.where(gains >= threshold)[0]
                if len(perturbed_candidates) > 1:
                    candidates = perturbed_candidates
                    kicks_remaining -= 1

            best_idx = candidates[random.randint(0, len(candidates) - 1)]

            if best_idx not in selected_set:
                selected_blocks.append(k_subsets[best_idx])
                selected_set.add(best_idx)
                uncovered[block_t_indices[best_idx]] = False

        return selected_blocks

    def remove_redundant(selected_blocks):
        """Remove blocks that don't cover any uniquely covered t-subset."""
        if not selected_blocks:
            return selected_blocks

        # Build coverage count
        coverage_count = np.zeros(n_t, dtype=np.int32)
        for block in selected_blocks:
            for t_combo in itertools.combinations(block, t):
                coverage_count[t_sub_to_idx[t_combo]] += 1

        changed = True
        while changed:
            if time.time() - start_time > deadline - safety_margin:
                break
            changed = False
            for i in range(len(selected_blocks)):
                block = selected_blocks[i]
                is_redundant = True
                for t_combo in itertools.combinations(block, t):
                    if coverage_count[t_sub_to_idx[t_combo]] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for t_combo in itertools.combinations(block, t):
                        coverage_count[t_sub_to_idx[t_combo]] -= 1
                    selected_blocks.pop(i)
                    changed = True
                    break

        return selected_blocks

    def local_search(selected_blocks, max_iters=50):
        """
        Try to improve by swapping one block for another that covers
        the same unique t-subsets but also covers more already-covered ones
        (or equivalently, try to find a block that can replace another
        to reduce total block count via a different mechanism).
        
        Actually, a simpler local search: try to find if removing a block
        and re-greedy-ing the uncovered parts gives a smaller set.
        This is expensive, so we limit iterations.
        """
        if not selected_blocks:
            return selected_blocks
        
        best = list(selected_blocks)
        best_size = len(best)
        
        for _ in range(max_iters):
            if time.time() - start_time > deadline - safety_margin:
                break
            
            # Try removing each block and see if we can cover the gap with fewer new blocks
            for i in range(len(best)):
                if time.time() - start_time > deadline - safety_margin:
                    break
                
                # Temporarily remove block i
                removed_block = best[i]
                remaining = [b for j, b in enumerate(best) if j != i]
                
                # Find t-subsets that were only covered by removed_block
                coverage_count = np.zeros(n_t, dtype=np.int32)
                for block in remaining:
                    for t_combo in itertools.combinations(block, t):
                        coverage_count[t_sub_to_idx[t_combo]] += 1
                
                uncovered = np.zeros(n_t, dtype=bool)
                for t_combo in itertools.combinations(removed_block, t):
                    idx = t_sub_to_idx[t_combo]
                    if coverage_count[idx] == 0:
                        uncovered[idx] = True
                
                if not np.any(uncovered):
                    # Block was redundant, remove it
                    best = remaining
                    best_size = len(best)
                    break
                
                # Try to cover the uncovered t-subsets with 1 new block
                # Find blocks that cover all uncovered t-subsets
                gains = uncovered[block_t_indices].sum(axis=1)
                total_uncovered = np.sum(uncovered)
                
                # A block that covers all uncovered t-subsets
                perfect_mask = gains == total_uncovered
                perfect_candidates = np.where(perfect_mask)[0]
                
                if len(perfect_candidates) > 0:
                    # Check if any of these blocks is already in remaining
                    new_block_idx = perfect_candidates[0]
                    if new_block_idx not in [k_subsets.index(b) for b in remaining]:
                        # Adding one block to replace one block: no improvement
                        pass
                    # Actually we need to add exactly 1 block, so size stays same
                    # No improvement from this swap
                # If we can't cover with 1 block, we'd need more, which is worse
                # So no improvement possible via single-block swap
                
            if time.time() - start_time > deadline - safety_margin:
                break
        
        return best

    # Multiple restarts with different configurations
    best_result = None
    best_score = float('inf')
    
    configs = [
        (0, 0.0, 0),      # deterministic
        (20, 0.3, 1),     # light kicks
        (30, 0.3, 2),     # moderate kicks
        (50, 0.4, 2),     # more kicks
        (40, 0.35, 3),    # wider window
        (60, 0.4, 1),     # many kicks, narrow window
    ]
    
    seeds = [42, 123, 777, 2024, 31415, 271828]
    
    for i, (num_kicks, kick_prob, kick_window) in enumerate(configs):
        if time.time() - start_time > deadline - safety_margin - 5:
            break
        
        seed = seeds[i % len(seeds)]
        random.seed(seed)
        
        result = greedy_cover(num_kicks, kick_prob, kick_window)
        result = remove_redundant(result)
        
        if len(result) < best_score:
            best_score = len(result)
            best_result = result
    
    # Try additional restarts if time permits
    additional_configs = [
        (25, 0.3, 2),
        (35, 0.35, 2),
        (45, 0.4, 2),
        (30, 0.3, 3),
    ]
    
    for i, (num_kicks, kick_prob, kick_window) in enumerate(additional_configs):
        if time.time() - start_time > deadline - safety_margin - 5:
            break
        
        seed = seeds[(i + 6) % len(seeds)]
        random.seed(seed)
        
        result = greedy_cover(num_kicks, kick_prob, kick_window)
        result = remove_redundant(result)
        
        if len(result) < best_score:
            best_score = len(result)
            best_result = result
    
    # Final verification
    if best_result is None:
        random.seed(42)
        best_result = greedy_cover(0, 0.0, 0)
        best_result = remove_redundant(best_result)
    
    # Verify coverage
    covered = np.zeros(n_t, dtype=bool)
    for block in best_result:
        for t_combo in itertools.combinations(block, t):
            covered[t_sub_to_idx[t_combo]] = True
    
    if not np.all(covered):
        # Fallback
        random.seed(0)
        best_result = greedy_cover(0, 0.0, 0)
        best_result = remove_redundant(best_result)
    
    return [list(b) for b in best_result]
