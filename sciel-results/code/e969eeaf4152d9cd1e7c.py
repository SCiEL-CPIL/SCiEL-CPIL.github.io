import numpy as np
import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill
    
    points = range(v)
    all_t_subsets = list(itertools.combinations(points, t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_k_subsets = list(itertools.combinations(points, k))
    num_k_subsets = len(all_k_subsets)
    
    # Precompute block -> t-subset indices
    block_to_t_indices = []
    for ks in all_k_subsets:
        block_to_t_indices.append([t_subset_to_idx[ts] for ts in itertools.combinations(ks, t)])
    
    # Precompute t-subset -> block indices
    t_subset_to_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, t_indices in enumerate(block_to_t_indices):
        for idx in t_indices:
            t_subset_to_blocks[idx].append(b_idx)
    
    # Convert to numpy arrays for speed
    block_to_t_arrays = [np.array(t_indices, dtype=np.int32) for t_indices in block_to_t_indices]
    t_subset_to_block_arrays = [np.array(b_indices, dtype=np.int32) for b_indices in t_subset_to_blocks]
    
    # Precompute block sizes
    block_sizes = np.array([len(t_indices) for t_indices in block_to_t_indices], dtype=np.int32)
    
    def compute_coverage(block_indices):
        """Compute coverage count array for given block indices."""
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in block_indices:
            cover_count[block_to_t_arrays[b_idx]] += 1
        return cover_count
    
    def greedy_construct(rng, use_criticality=True):
        """Greedy construction with optional criticality weighting."""
        uncovered = np.ones(num_t_subsets, dtype=bool)
        remaining_count = np.array([len(b_indices) for b_indices in t_subset_to_blocks], dtype=np.int32)
        gains = block_sizes.copy()
        used_blocks = np.zeros(num_k_subsets, dtype=bool)
        selected = []
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
            
            # Find top candidates
            if use_criticality:
                noise = rng.uniform(0, 0.5, size=num_k_subsets)
                scores = gains.astype(np.float64) + noise
                scores[used_blocks] = -1
                top_k = min(500, num_k_subsets)
                top_indices = np.argpartition(scores, -top_k)[-top_k:]
                
                best_idx = -1
                best_score = -1
                for idx in top_indices:
                    if gains[idx] == 0:
                        continue
                    b_t = block_to_t_arrays[idx]
                    unc_mask = uncovered[b_t]
                    if not np.any(unc_mask):
                        continue
                    unc_t = b_t[unc_mask]
                    crit_bonus = np.sum(1.0 / remaining_count[unc_t])
                    final_score = gains[idx] + 0.5 * crit_bonus + rng.uniform(0, 0.1)
                    if final_score > best_score:
                        best_score = final_score
                        best_idx = idx
            else:
                # Simple greedy: pick random uncovered t-subset, then best block
                unc_indices = np.where(uncovered)[0]
                s_idx = int(unc_indices[rng.integers(len(unc_indices))])
                candidates = t_subset_to_block_arrays[s_idx]
                if len(candidates) == 0:
                    break
                best_idx = -1
                best_gain = -1
                for idx in candidates:
                    if used_blocks[idx]:
                        continue
                    gain = gains[idx]
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx
            
            if best_idx == -1 or gains[best_idx] == 0:
                break
            
            selected.append(best_idx)
            used_blocks[best_idx] = True
            
            b_t = block_to_t_arrays[best_idx]
            newly_covered = b_t[uncovered[b_t]]
            
            if len(newly_covered) == 0:
                continue
            
            uncovered[newly_covered] = False
            remaining_count[b_t] -= 1
            
            # Update gains
            for s in newly_covered:
                for b in t_subset_to_block_arrays[s]:
                    gains[b] -= 1
        
        return selected
    
    def repair_covering(block_indices, max_time=5.0, rng=None):
        """Repair a covering by adding blocks to cover missing t-subsets."""
        if rng is None:
            rng = np.random.default_rng()
        
        cover_count = compute_coverage(block_indices)
        uncovered_mask = (cover_count == 0)
        
        if not np.any(uncovered_mask):
            return list(block_indices)
        
        result = list(block_indices)
        result_set = set(result)
        repair_deadline = time.time() + max_time
        
        while np.any(uncovered_mask) and time.time() < repair_deadline:
            unc_indices = np.where(uncovered_mask)[0]
            if len(unc_indices) == 0:
                break
            
            # Find blocks covering most uncovered
            gains = np.zeros(num_k_subsets, dtype=np.int32)
            for s_idx in unc_indices:
                for b_idx in t_subset_to_block_arrays[s_idx]:
                    if b_idx not in result_set:
                        gains[b_idx] += 1
            
            if np.max(gains) == 0:
                break
            
            noise = rng.uniform(0, 0.1, size=num_k_subsets)
            scores = gains.astype(np.float64) + noise
            best_idx = int(np.argmax(scores))
            
            if gains[best_idx] == 0:
                break
            
            result.append(best_idx)
            result_set.add(best_idx)
            
            b_t = block_to_t_arrays[best_idx]
            newly_covered = b_t[uncovered_mask[b_t]]
            uncovered_mask[newly_covered] = False
        
        return result
    
    def deletion_pass(blocks, ref_count):
        """Remove redundant blocks. Returns new blocks list and updated ref_count."""
        changed = False
        i = 0
        while i < len(blocks):
            b = blocks[i]
            t_indices = block_to_t_arrays[b]
            
            # Tentatively remove
            ref_count[t_indices] -= 1
            
            if np.min(ref_count) > 0:
                # Redundant
                blocks.pop(i)
                changed = True
            else:
                # Restore
                ref_count[t_indices] += 1
                i += 1
        return blocks, ref_count, changed
    
    def try_swap(blocks, ref_count, rng):
        """Try to replace a block with another that enables deletion."""
        if len(blocks) == 0:
            return False, blocks, ref_count
        
        i = int(rng.integers(len(blocks)))
        b_old = blocks[i]
        old_t = block_to_t_arrays[b_old]
        
        # Find critical t-subsets (count == 1 after removing b_old)
        criticals = old_t[ref_count[old_t] == 1]
        
        if len(criticals) == 0:
            return False, blocks, ref_count
        
        # Union of points in critical t-subsets
        u_set = set()
        for idx in criticals:
            u_set.update(all_t_subsets[idx])
        
        if len(u_set) > k:
            return False, blocks, ref_count
        
        # Generate candidate blocks containing u_set
        remaining_points = [p for p in points if p not in u_set]
        if len(remaining_points) < k - len(u_set):
            return False, blocks, ref_count
        
        block_set = set(blocks)
        
        # Try multiple random completions
        for _ in range(30):
            if len(remaining_points) >= k - len(u_set):
                extra = list(rng.choice(remaining_points, size=k - len(u_set), replace=False))
                b_new_tuple = tuple(sorted(list(u_set) + extra))
            else:
                break
            
            if b_new_tuple == b_old:
                continue
            if b_new_tuple in block_set:
                continue
            
            # Find block index
            b_new = None
            # Use a lookup - but we don't have one for k-subsets to index
            # Instead, just work with the tuple directly
            new_t_indices = [t_subset_to_idx[ts] for ts in itertools.combinations(b_new_tuple, t)]
            new_t = np.array(new_t_indices, dtype=np.int32)
            
            # Check validity
            ref_count[old_t] -= 1
            ref_count[new_t] += 1
            
            if np.min(ref_count) > 0:
                # Valid swap
                blocks[i] = b_new_tuple
                return True, blocks, ref_count
            else:
                # Revert
                ref_count[new_t] -= 1
                ref_count[old_t] += 1
        
        return False, blocks, ref_count
    
    def optimize_covering(blocks, max_time=30.0, rng=None):
        """Optimize a covering via deletion and swaps."""
        if rng is None:
            rng = np.random.default_rng()
        
        blocks = list(blocks)
        ref_count = compute_coverage(blocks)
        opt_deadline = time.time() + max_time
        
        while time.time() < opt_deadline:
            # Deletion pass
            blocks, ref_count, changed = deletion_pass(blocks, ref_count)
            
            # Try swaps
            num_swaps = 20
            for _ in range(num_swaps):
                if time.time() > opt_deadline:
                    break
                _, blocks, ref_count = try_swap(blocks, ref_count, rng)
            
            # Another deletion pass
            blocks, ref_count, _ = deletion_pass(blocks, ref_count)
        
        return blocks
    
    # Main optimization loop
    best_blocks = None
    best_size = float('inf')
    
    # Strategy 1: Criticality-weighted greedy with multiple seeds
    for seed in range(5):
        if time.time() > deadline - 10:
            break
        
        rng = np.random.default_rng(seed * 12345 + 67890)
        blocks = greedy_construct(rng, use_criticality=True)
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = list(blocks)
        
        # Optimize
        blocks = optimize_covering(blocks, max_time=15.0, rng=rng)
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = list(blocks)
    
    # Strategy 2: LLL-inspired random initialization
    # Pick random subset of blocks with probability p, repair, optimize
    for trial in range(3):
        if time.time() > deadline - 15:
            break
        
        rng = np.random.default_rng(99999 + trial * 7777)
        
        # Choose p based on expected coverage
        # Each block covers C(8,5)=56 t-subsets out of C(19,5)=11628
        # Need enough blocks: n * 56 >= 11628 => n >= 208
        # With p, expected blocks = p * 75582
        # Try p values that give ~250-400 blocks
        p = 0.003 + trial * 0.002  # 0.003, 0.005, 0.007
        
        mask = rng.random(num_k_subsets) < p
        initial_blocks = np.where(mask)[0].tolist()
        
        if len(initial_blocks) == 0:
            continue
        
        # Repair to valid covering
        initial_blocks = repair_covering(initial_blocks, max_time=5.0, rng=rng)
        
        if len(initial_blocks) < best_size:
            best_size = len(initial_blocks)
            best_blocks = list(initial_blocks)
        
        # Optimize
        initial_blocks = optimize_covering(initial_blocks, max_time=15.0, rng=rng)
        if len(initial_blocks) < best_size:
            best_size = len(initial_blocks)
            best_blocks = list(initial_blocks)
    
    # Strategy 3: Simple greedy (random t-subset) as fallback
    if time.time() < deadline - 10:
        rng = np.random.default_rng(424242)
        blocks = greedy_construct(rng, use_criticality=False)
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = list(blocks)
        
        blocks = optimize_covering(blocks, max_time=15.0, rng=rng)
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = list(blocks)
    
    # Final: ensure we have a valid covering
    if best_blocks is None:
        # Fallback: just use all blocks (shouldn't happen)
        best_blocks = list(range(num_k_subsets))
    
    # Convert block indices back to actual blocks (tuples of points)
    result = [all_k_subsets[b] for b in best_blocks]
    
    # Verify validity (exhaustive check)
    # For safety, check all t-subsets are covered
    cover_count = np.zeros(num_t_subsets, dtype=np.int32)
    for b in result:
        for ts in itertools.combinations(b, t):
            cover_count[t_subset_to_idx[ts]] += 1
    
    if np.min(cover_count) < 1:
        # Invalid - fall back to greedy
        rng = np.random.default_rng(0)
        result = greedy_construct(rng, use_criticality=True)
        result = [all_k_subsets[b] for b in result]
    
    return result
