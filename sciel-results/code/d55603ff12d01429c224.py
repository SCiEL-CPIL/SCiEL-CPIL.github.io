import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    points = range(v)
    all_subsets = list(itertools.combinations(points, t))
    num_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute block_to_subsets and subset_to_blocks
    block_to_subsets = [[] for _ in range(num_blocks)]
    subset_to_blocks = [[] for _ in range(num_subsets)]
    
    for b_idx, block in enumerate(all_blocks):
        for s in itertools.combinations(block, t):
            s_idx = subset_to_idx[s]
            block_to_subsets[b_idx].append(s_idx)
            subset_to_blocks[s_idx].append(b_idx)
    
    # Precompute numpy arrays for fast operations
    block_subset_arrays = [np.array(subs, dtype=np.int32) for subs in block_to_subsets]
    subset_block_arrays = [np.array(blocks, dtype=np.int32) for blocks in subset_to_blocks]
    
    # Precompute subset_to_blocks as sets for fast intersection in swap
    subset_to_blocks_sets = [set(blocks) for blocks in subset_to_blocks]
    
    # Precompute unique coverage for each block (how many subsets it uniquely covers given a coverage array)
    # This is computed on-the-fly in local search
    
    def run_greedy_full(random_seed=None):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        uncovered = np.ones(num_subsets, dtype=bool)
        gains = np.array([len(subs) for subs in block_to_subsets], dtype=np.int32)
        
        selected_blocks = []
        
        while True:
            if not np.any(uncovered):
                break
                
            # Use a smooth surrogate: add small random perturbation to break ties
            # and encourage exploration. Scale noise with current max gain for adaptivity.
            max_gain = np.max(gains)
            noise_scale = 0.1 + 0.2 * (1.0 / (1.0 + max_gain))
            noise = rng.uniform(0.0, noise_scale, size=num_blocks)
            scores = gains.astype(np.float64) + noise
            best_idx = int(np.argmax(scores))
            
            if gains[best_idx] == 0:
                break
                
            selected_blocks.append(best_idx)
            
            b_subs = block_subset_arrays[best_idx]
            newly_covered = b_subs[uncovered[b_subs]]
            
            if len(newly_covered) == 0:
                continue
                
            uncovered[newly_covered] = False
            # Vectorized gain update using np.add.at for scatter
            for s in newly_covered:
                blocks_covering = subset_block_arrays[s]
                gains[blocks_covering] -= 1
                    
        return selected_blocks

    def run_greedy_partial(uncovered_mask, random_seed=None):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            return []
            
        gains = np.zeros(num_blocks, dtype=np.int32)
        for s_idx in uncovered_indices:
            blocks_covering = subset_block_arrays[s_idx]
            gains[blocks_covering] += 1
                
        selected_blocks = []
        current_uncovered = uncovered_mask.copy()
        
        while True:
            if not np.any(current_uncovered):
                break
                
            # Adaptive noise scaling based on current max gain
            max_gain = np.max(gains)
            noise_scale = 0.1 + 0.2 * (1.0 / (1.0 + max_gain))
            noise = rng.uniform(0.0, noise_scale, size=num_blocks)
            scores = gains.astype(np.float64) + noise
            best_idx = int(np.argmax(scores))
            
            if gains[best_idx] == 0:
                break
                
            selected_blocks.append(best_idx)
            
            b_subs = block_subset_arrays[best_idx]
            newly_covered = b_subs[current_uncovered[b_subs]]
            
            if len(newly_covered) == 0:
                continue
                
            current_uncovered[newly_covered] = False
            for s in newly_covered:
                blocks_covering = subset_block_arrays[s]
                gains[blocks_covering] -= 1
                    
        return selected_blocks

    def run_greedy_hardest(random_seed=None):
        """Greedy that prioritizes the 'hardest' uncovered subset (lowest degree among uncovered).
        
        At each step, find the uncovered 5-subset with the fewest available blocks
        (blocks not yet selected that cover it), then pick a block covering it
        that maximizes total new coverage. This ensures structurally isolated
        subsets are addressed early.
        """
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        uncovered = np.ones(num_subsets, dtype=bool)
        selected_set = set()
        selected_blocks = []
        
        # Precompute: for each subset, list of blocks covering it (already have subset_block_arrays)
        
        while True:
            if not np.any(uncovered):
                break
            
            # Find the hardest uncovered subset: one with fewest non-selected covering blocks
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            
            # Sample a subset of hardest candidates to avoid O(num_subsets * num_blocks) per step
            # Use the degree (number of blocks covering it minus already selected)
            # Approximate: just pick randomly from uncovered, but weight toward low-degree ones
            
            # For efficiency, check a random sample of uncovered subsets
            sample_size = min(50, len(uncovered_indices))
            if len(uncovered_indices) <= sample_size:
                candidates = uncovered_indices
            else:
                candidates = uncovered_indices[rng.choice(len(uncovered_indices), size=sample_size, replace=False)]
            
            best_subset = -1
            best_degree = float('inf')
            best_blocks_for_subset = None
            
            for s_idx in candidates:
                # Count how many blocks cover this subset that are NOT yet selected
                covering = subset_block_arrays[s_idx]
                non_selected = 0
                for b in covering:
                    if b not in selected_set:
                        non_selected += 1
                if non_selected == 0:
                    # This subset can't be covered - shouldn't happen in valid state
                    break
                if non_selected < best_degree:
                    best_degree = non_selected
                    best_subset = s_idx
                    best_blocks_for_subset = covering
            
            if best_subset == -1 or best_degree == 0:
                # Fallback: pick any uncovered subset
                s_idx = uncovered_indices[0]
                covering = subset_block_arrays[s_idx]
                non_sel = [b for b in covering if b not in selected_set]
                if not non_sel:
                    break
                # Pick the one with max new coverage
                best_b = -1
                best_gain = -1
                for b in non_sel:
                    b_subs = block_subset_arrays[b]
                    gain = np.sum(uncovered[b_subs])
                    if gain > best_gain:
                        best_gain = gain
                        best_b = b
                if best_b == -1:
                    break
                selected_blocks.append(best_b)
                selected_set.add(best_b)
                b_subs = block_subset_arrays[best_b]
                newly = b_subs[uncovered[b_subs]]
                uncovered[newly] = False
                continue
            
            # Among blocks covering best_subset (and not yet selected), pick one with max new coverage
            covering = best_blocks_for_subset
            non_sel = [b for b in covering if b not in selected_set]
            if not non_sel:
                break
            
            best_b = -1
            best_gain = -1
            for b in non_sel:
                b_subs = block_subset_arrays[b]
                gain = np.sum(uncovered[b_subs])
                if gain > best_gain:
                    best_gain = gain
                    best_b = b
            
            if best_b == -1 or best_gain == 0:
                break
            
            selected_blocks.append(best_b)
            selected_set.add(best_b)
            b_subs = block_subset_arrays[best_b]
            newly = b_subs[uncovered[b_subs]]
            uncovered[newly] = False
            
        return selected_blocks

    def minimize_with_swaps(blocks_indices, coverage_counts, time_limit, random_seed=None):
        """Local search with deletion and swap operations, vectorized."""
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        current_blocks = list(blocks_indices)
        
        if not current_blocks:
            return current_blocks, coverage_counts.copy()
            
        cover_count = coverage_counts.copy()
        current_blocks_set = set(current_blocks)
        
        start_time = time.time()
        
        # Phase 1: Aggressive deletion (vectorized check)
        changed = True
        while changed and (time.time() - start_time) < time_limit:
            changed = False
            idx_list = list(range(len(current_blocks)))
            rng.shuffle(idx_list)
            for i in idx_list:
                if i >= len(current_blocks):
                    continue
                if (time.time() - start_time) > time_limit * 0.3:
                    break
                b_idx = current_blocks[i]
                b_subs = block_subset_arrays[b_idx]
                # Vectorized: check if any subset is uniquely covered
                if not np.any(cover_count[b_subs] == 1):
                    cover_count[b_subs] -= 1
                    current_blocks_set.discard(b_idx)
                    current_blocks.pop(i)
                    changed = True
        
        # Phase 2: Swap-based local search (more aggressive)
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            indices = list(range(len(current_blocks)))
            rng.shuffle(indices)
            
            for i in indices:
                if i >= len(current_blocks):
                    continue
                if (time.time() - start_time) > time_limit:
                    break
                    
                old_block = current_blocks[i]
                old_subs = block_subset_arrays[old_block]
                
                # Find unique subsets covered by old_block (vectorized)
                unique_mask = cover_count[old_subs] == 1
                unique_subs = old_subs[unique_mask]
                
                if len(unique_subs) == 0:
                    # Block is not critical - swap with random non-selected block
                    attempts = 0
                    while attempts < 5:
                        new_block = rng.randint(0, num_blocks - 1)
                        if new_block != old_block and new_block not in current_blocks_set:
                            # Check swap doesn't break coverage
                            # Remove old, add new, verify no subset goes to 0
                            cover_count[old_subs] -= 1
                            new_subs = block_subset_arrays[new_block]
                            if np.any(cover_count[old_subs] < 1):
                                # This would break coverage - revert
                                cover_count[old_subs] += 1
                                attempts += 1
                                continue
                            cover_count[new_subs] += 1
                            current_blocks_set.discard(old_block)
                            current_blocks_set.add(new_block)
                            current_blocks[i] = new_block
                            improved = True
                            break
                        attempts += 1
                else:
                    # Block is critical - find candidates covering all unique_subs
                    candidates = None
                    for s_idx in unique_subs:
                        if candidates is None:
                            candidates = subset_to_blocks_sets[s_idx].copy()
                        else:
                            candidates.intersection_update(subset_to_blocks_sets[s_idx])
                            if not candidates:
                                break
                    
                    if candidates:
                        candidates.discard(old_block)
                        if candidates:
                            # Pick a random candidate that isn't already selected
                            cand_list = [c for c in candidates if c not in current_blocks_set]
                            if cand_list:
                                new_block = rng.choice(cand_list)
                                # Verify swap maintains coverage
                                cover_count[old_subs] -= 1
                                new_subs = block_subset_arrays[new_block]
                                if not np.any(cover_count[old_subs] < 1):
                                    cover_count[new_subs] += 1
                                    current_blocks_set.discard(old_block)
                                    current_blocks_set.add(new_block)
                                    current_blocks[i] = new_block
                                    improved = True
                                else:
                                    cover_count[old_subs] += 1
        
        # Phase 3: Final aggressive deletion polish
        if time.time() < start_time + time_limit:
            for _ in range(5):
                changed = True
                while changed and (time.time() - start_time) < time_limit:
                    changed = False
                    idx_list = list(range(len(current_blocks)))
                    rng.shuffle(idx_list)
                    for i in idx_list:
                        if i >= len(current_blocks):
                            continue
                        b_idx = current_blocks[i]
                        b_subs = block_subset_arrays[b_idx]
                        if not np.any(cover_count[b_subs] == 1):
                            cover_count[b_subs] -= 1
                            current_blocks_set.discard(b_idx)
                            current_blocks.pop(i)
                            changed = True
        
        return current_blocks, cover_count

    def compute_coverage(blocks_indices):
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in blocks_indices:
            b_subs = block_subset_arrays[b_idx]
            cover_count[b_subs] += 1
        return cover_count

    def lll_random_construction(random_seed=None, p=None):
        """LLL-inspired probabilistic construction.
        
        Sample blocks with probability p, then iteratively remove locally redundant blocks.
        A block is locally redundant if none of its covered subsets are uniquely covered by it
        within the current set of selected blocks.
        """
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
        
        if p is None:
            # Estimate good p: we want ~expected number of blocks that gives good coverage
            # Each block covers C(k,t) subsets out of C(v,t) total
            # Probability a random subset is NOT in a given block = 1 - C(k,t)/C(v,t)
            # For n blocks, prob not covered = (1 - C(k,t)/C(v,t))^n
            # We want this small, say < 0.01
            from math import comb, log
            p_block = comb(k, t) / comb(v, t)
            # n * p_block should be large enough
            # Let's try a few values
            p = 0.15  # Start with a reasonable default
        
        # Sample blocks
        mask = rng.random(num_blocks) < p
        selected = np.where(mask)[0].tolist()
        
        if not selected:
            return []
        
        # Compute coverage
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in selected:
            b_subs = block_subset_arrays[b_idx]
            cover_count[b_subs] += 1
        
        # Check if all subsets are covered
        if np.any(cover_count == 0):
            # Need to repair - add blocks to cover uncovered subsets
            uncovered_mask = (cover_count == 0).astype(bool)
            repair_res = run_greedy_partial(uncovered_mask, random_seed=random_seed)
            selected.extend(repair_res)
            # Recompute coverage
            cover_count = np.zeros(num_subsets, dtype=np.int32)
            for b_idx in selected:
                b_subs = block_subset_arrays[b_idx]
                cover_count[b_subs] += 1
        
        # Iteratively remove locally redundant blocks
        # A block is locally redundant if all its subsets are covered by other blocks
        changed = True
        start_time = time.time()
        while changed and (time.time() - start_time) < 2.0:
            changed = False
            # Shuffle for randomization
            rng.shuffle(selected)
            for b_idx in selected[:]:
                if b_idx not in selected:
                    continue
                b_subs = block_subset_arrays[b_idx]
                is_redundant = True
                for s_idx in b_subs:
                    if cover_count[s_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for s_idx in b_subs:
                        cover_count[s_idx] -= 1
                    selected.remove(b_idx)
                    changed = True
        
        return selected

    best_blocks = None
    best_len = float('inf')
    best_coverage = None
    
    start_time = time.time()
    deadline = start_time + 115.0
    
    trial = 0
    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining < 10.0:
            break
            
        # Strategy 1: Full Greedy + Local Search (deletion + swap)
        greedy_res = run_greedy_full(trial)
        coverage = compute_coverage(greedy_res)
        
        # Adaptive time limit for local search - give more time to local search
        ls_time = min(5.0, max(2.0, remaining * 0.2))
        min_res, min_cov = minimize_with_swaps(greedy_res, coverage, time_limit=ls_time, random_seed=trial)
        
        if len(min_res) < best_len:
            best_len = len(min_res)
            best_blocks = min_res
            best_coverage = min_cov
            
        # Strategy 2: Hardest-subset Greedy + Local Search
        if time.time() < deadline - 12:
            hardest_res = run_greedy_hardest(trial + 5000)
            if hardest_res:
                hardest_cov = compute_coverage(hardest_res)
                ls_time_h = min(4.0, max(1.5, (deadline - time.time()) * 0.15))
                min_res_h, min_cov_h = minimize_with_swaps(hardest_res, hardest_cov, time_limit=ls_time_h, random_seed=trial + 60000)
                
                if len(min_res_h) < best_len:
                    best_len = len(min_res_h)
                    best_blocks = min_res_h
                    best_coverage = min_cov_h

        # Strategy 3: LLL-inspired probabilistic construction + Local Search
        if time.time() < deadline - 12:
            p_vals = [0.12, 0.15, 0.18, 0.22, 0.28]
            p = p_vals[trial % len(p_vals)]
            lll_res = lll_random_construction(random_seed=trial + 10000, p=p)
            
            if lll_res:
                lll_coverage = compute_coverage(lll_res)
                
                ls_time_lla = min(4.0, max(1.5, (deadline - time.time()) * 0.15))
                min_res_lla, min_cov_lla = minimize_with_swaps(lll_res, lll_coverage, time_limit=ls_time_lla, random_seed=trial + 20000)
                
                if len(min_res_lla) < best_len:
                    best_len = len(min_res_lla)
                    best_blocks = min_res_lla
                    best_coverage = min_cov_lla

        # Strategy 4: Perturb Best + Repair + Local Search
        if best_blocks is not None and len(best_blocks) > 20 and time.time() < deadline - 12:
            perturbed = list(best_blocks)
            num_remove = min(max(3, len(perturbed) // (8 + trial % 12)), len(perturbed) - 1)
            if num_remove > 0:
                cover_count_temp = best_coverage.copy()
                unique_counts = []
                for b_idx in perturbed:
                    b_subs = block_subset_arrays[b_idx]
                    unique = int(np.sum(cover_count_temp[b_subs] == 1))
                    unique_counts.append(unique)
                
                sorted_indices = sorted(range(len(perturbed)), key=lambda i: unique_counts[i])
                candidates = sorted_indices[:max(num_remove * 2, 6)]
                remove_indices = random.sample(candidates, min(num_remove, len(candidates)))
                blocks_to_remove = set(perturbed[i] for i in remove_indices)
                
                cover_count_temp2 = np.zeros(num_subsets, dtype=np.int32)
                for b_idx in perturbed:
                    if b_idx in blocks_to_remove:
                        continue
                    b_subs = block_subset_arrays[b_idx]
                    cover_count_temp2[b_subs] += 1
                        
                uncovered_mask = (cover_count_temp2 == 0).astype(bool)
                
                repair_res = run_greedy_partial(uncovered_mask, trial + 30000)
                
                new_set = [b for b in perturbed if b not in blocks_to_remove]
                new_set.extend(repair_res)
                
                new_coverage = compute_coverage(new_set)
                
                ls_time2 = min(4.0, max(1.5, (deadline - time.time()) * 0.15))
                min_res2, min_cov2 = minimize_with_swaps(new_set, new_coverage, time_limit=ls_time2, random_seed=trial + 40000)
                
                if len(min_res2) < best_len:
                    best_len = len(min_res2)
                    best_blocks = min_res2
                    best_coverage = min_cov2

        # Strategy 5: Pure swap-based diversification on best solution
        if best_blocks is not None and len(best_blocks) > 20 and time.time() < deadline - 8:
            ls_time3 = min(5.0, max(2.0, (deadline - time.time()) * 0.25))
            swap_res, swap_cov = minimize_with_swaps(best_blocks, best_coverage, time_limit=ls_time3, random_seed=trial + 50000)
            
            if len(swap_res) < best_len:
                best_len = len(swap_res)
                best_blocks = swap_res
                best_coverage = swap_cov

        trial += 1
        
    # Final polish: aggressive deletion pass on best solution
    if best_blocks is not None and time.time() < deadline - 5:
        final_cov = best_coverage.copy()
        final_blocks = list(best_blocks)
        final_set = set(final_blocks)
        for _ in range(8):
            changed = True
            while changed and time.time() < deadline - 5:
                changed = False
                idx_list = list(range(len(final_blocks)))
                random.shuffle(idx_list)
                for i in idx_list:
                    if i >= len(final_blocks):
                        continue
                    b_idx = final_blocks[i]
                    b_subs = block_subset_arrays[b_idx]
                    if not np.any(final_cov[b_subs] == 1):
                        final_cov[b_subs] -= 1
                        final_set.discard(b_idx)
                        final_blocks.pop(i)
                        changed = True
        if len(final_blocks) < best_len:
            best_len = len(final_blocks)
            best_blocks = final_blocks
            best_coverage = final_cov
        
        # Additional swap-based diversification in final phase
        if best_blocks is not None and time.time() < deadline - 3:
            remaining_time = deadline - time.time() - 3
            if remaining_time > 0.5:
                final_swap_res, final_swap_cov = minimize_with_swaps(
                    best_blocks, best_coverage, time_limit=remaining_time, random_seed=99999
                )
                if len(final_swap_res) < best_len:
                    best_len = len(final_swap_res)
                    best_blocks = final_swap_res
                    best_coverage = final_swap_cov
        
    if best_blocks is None:
        greedy_res = run_greedy_full(0)
        coverage = compute_coverage(greedy_res)
        best_blocks, best_coverage = minimize_with_swaps(greedy_res, coverage, time_limit=5.0, random_seed=0)
        
    return [list(all_blocks[i]) for i in best_blocks]
