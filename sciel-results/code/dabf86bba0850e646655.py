"""Optimized covering design search for C(24,7,3).

Strategy:
1. Fast greedy construction (multiple seeds) with vectorized gain.
2. Fast reduction (remove redundant blocks).
3. Simulated annealing local search: swap/replace blocks to minimize count.
"""
import time
import random
import itertools
import numpy as np

# EVOLVE-BLOCK-START

def construct(v, k, t):
    deadline = time.time() + 110.0  # Safety margin
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {subset: i for i, subset in enumerate(all_t_subsets)}
    
    # Precompute: for each (t-1)-subset, which points complete it to a t-subset
    # And for each point x and (t-1)-subset, the t-subset index
    point_to_tsubset_map = [{} for _ in range(v)]
    for idx, subset in enumerate(all_t_subsets):
        for i in range(t):
            rest = list(subset[:i] + subset[i+1:])
            rest.sort()
            p = subset[i]
            point_to_tsubset_map[p][tuple(rest)] = idx
    
    # Precompute all k-subsets as frozensets for fast lookup
    all_k_subsets = list(itertools.combinations(range(v), k))
    all_k_frozensets = [frozenset(b) for b in all_k_subsets]
    
    # Map from frozenset to its list of t-subset indices
    ksubset_to_tsubsets = {}
    for b in all_k_subsets:
        ts = []
        for subset in itertools.combinations(b, t):
            ts.append(t_subset_to_idx[subset])
        ksubset_to_tsubsets[frozenset(b)] = ts
    
    def greedy_cover(seed):
        """Fast greedy with randomization."""
        rng = random.Random(seed)
        uncovered = np.ones(n_t_subsets, dtype=bool)
        uncovered_count = n_t_subsets
        blocks = []
        
        while uncovered_count > 0:
            if time.time() > deadline - 25:
                return None
            
            # Pick a random uncovered t-subset
            # Use rejection sampling for speed
            while True:
                idx = rng.randint(0, n_t_subsets - 1)
                if uncovered[idx]:
                    break
            
            uncovered[idx] = False
            uncovered_count -= 1
            
            block = list(all_t_subsets[idx])
            rng.shuffle(block)
            block_set = set(block)
            
            # Add points to the block greedily
            while len(block) < k:
                if time.time() > deadline - 25:
                    return None
                
                best_x = -1
                best_gain = -1
                tie_x = []
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        r_tuple = tuple(sorted(rest))
                        sidx = point_to_tsubset_map[x].get(r_tuple, -1)
                        if sidx != -1 and uncovered[sidx]:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        tie_x = [x]
                    elif gain == best_gain and gain >= 0:
                        tie_x.append(x)
                
                if best_x < 0:
                    return None
                
                if len(tie_x) > 1:
                    best_x = tie_x[rng.randint(0, len(tie_x) - 1)]
                
                block.append(best_x)
                block_set.add(best_x)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for subset in itertools.combinations(block, t):
                sidx = t_subset_to_idx[subset]
                if uncovered[sidx]:
                    uncovered[sidx] = False
                    uncovered_count -= 1
        
        return blocks

    def reduce_blocks(blocks):
        """Remove redundant blocks using vectorized operations."""
        if not blocks:
            return []
        
        blocks = list(blocks)
        n = len(blocks)
        
        # Precompute t-subset indices for each block
        block_t_indices = []
        for block in blocks:
            ts = []
            for subset in itertools.combinations(block, t):
                ts.append(t_subset_to_idx[subset])
            block_t_indices.append(ts)
        
        # Build coverage matrix: n_blocks x n_t_subsets (sparse-like via counting)
        # For efficiency, use a 2D array only if feasible
        if n * n_t_subsets < 10_000_000:
            cov_matrix = np.zeros((n, n_t_subsets), dtype=np.int8)
            for bi, t_idx_list in enumerate(block_t_indices):
                for idx in t_idx_list:
                    cov_matrix[bi, idx] += 1
            coverage_count = cov_matrix.sum(axis=0)
            
            # Iteratively remove redundant blocks
            active = np.ones(n, dtype=bool)
            improved = True
            while improved:
                if time.time() > deadline - 5:
                    break
                improved = False
                for bi in range(n):
                    if not active[bi]:
                        continue
                    # Check if block bi is redundant
                    # A block is redundant if for every t-subset it covers,
                    # some other active block also covers it
                    # i.e., coverage_count[idx] > 1 for all idx in block_t_indices[bi]
                    t_idx = block_t_indices[bi]
                    if len(t_idx) == 0:
                        active[bi] = False
                        improved = True
                        continue
                    # Check: is there any t-subset in this block with coverage == 1?
                    # If all have coverage > 1, block is redundant
                    covers = coverage_count[t_idx]
                    if covers.min() > 1:
                        active[bi] = False
                        # Update coverage
                        coverage_count[t_idx] -= 1
                        improved = True
                        if time.time() > deadline - 5:
                            break
            result = [blocks[i] for i in range(n) if active[i]]
            return result
        else:
            # Fallback: sequential approach with numpy for coverage
            coverage_count = np.zeros(n_t_subsets, dtype=np.int32)
            for t_idx_list in block_t_indices:
                for idx in t_idx_list:
                    coverage_count[idx] += 1
            
            active = list(range(n))
            improved = True
            while improved:
                if time.time() > deadline - 5:
                    break
                improved = False
                for bi in list(active):
                    t_idx = block_t_indices[bi]
                    if len(t_idx) == 0:
                        active.remove(bi)
                        improved = True
                        continue
                    covers = coverage_count[t_idx]
                    if covers.min() > 1:
                        active.remove(bi)
                        coverage_count[t_idx] -= 1
                        improved = True
                        if time.time() > deadline - 5:
                            break
            return [blocks[i] for i in active]

    def simulated_annealing(blocks, seed):
        """SA to minimize block count by removals and single-block swaps."""
        rng = random.Random(seed)
        blocks = list(blocks)
        current_count = len(blocks)
        best_count = current_count
        best_blocks = list(blocks)
        
        # Precompute t-subset indices for each block
        def get_block_t_indices(block):
            return [t_subset_to_idx[subset] for subset in itertools.combinations(block, t)]
        
        # Precompute coverage for current solution
        coverage = np.zeros(n_t_subsets, dtype=np.int32)
        block_t_idx = [get_block_t_indices(b) for b in blocks]
        for t_idx_list in block_t_idx:
            for idx in t_idx_list:
                coverage[idx] += 1
        
        n_blocks = len(blocks)
        temp = 3.0
        cooling = 0.995
        min_temp = 0.05
        max_iter = 5000
        
        for it in range(max_iter):
            if time.time() > deadline - 8:
                break
            if temp < min_temp:
                break
            
            if n_blocks <= 1:
                break
            
            # Randomly choose move type: removal attempt (40%) or swap (60%)
            move_type = rng.random()
            bi = rng.randint(0, n_blocks - 1)
            
            if move_type < 0.4:
                # Try removal
                removed_t_idx = block_t_idx[bi]
                for idx in removed_t_idx:
                    coverage[idx] -= 1
                
                uncovered_mask = (coverage == 0)
                n_uncovered = np.count_nonzero(uncovered_mask)
                
                if n_uncovered == 0:
                    # Can remove this block!
                    blocks.pop(bi)
                    block_t_idx.pop(bi)
                    n_blocks -= 1
                    current_count -= 1
                    if current_count < best_count:
                        best_count = current_count
                        best_blocks = list(blocks)
                    temp = max(temp, 1.5)
                else:
                    # Revert
                    for idx in removed_t_idx:
                        coverage[idx] += 1
            else:
                # Try swap: replace block bi with a different block
                # Generate a candidate new block
                # Strategy: pick a random uncovered-or-low-coverage t-subset, build around it
                # Or: pick a random point set and build greedily targeting low-coverage areas
                
                # Build a candidate block targeting under-covered t-subsets
                # Weight by inverse coverage to find useful replacements
                candidate = None
                
                # Try a few random starting points
                for attempt in range(3):
                    # Pick a random t-subset with low coverage
                    low_cov_mask = (coverage <= 2)
                    if not low_cov_mask.any():
                        low_cov_mask = np.ones(n_t_subsets, dtype=bool)
                    low_cov_indices = np.where(low_cov_mask)[0]
                    if len(low_cov_indices) == 0:
                        continue
                    ui = low_cov_indices[rng.randint(0, len(low_cov_indices) - 1)]
                    
                    start_block = list(all_t_subsets[ui])
                    rng.shuffle(start_block)
                    nb_set = set(start_block)
                    
                    while len(start_block) < k:
                        best_x = -1
                        best_gain = -1
                        for x in range(v):
                            if x in nb_set:
                                continue
                            gain = 0
                            for rest in itertools.combinations(start_block, t - 1):
                                r_tuple = tuple(sorted(rest))
                                sidx = point_to_tsubset_map[x].get(r_tuple, -1)
                                if sidx != -1 and coverage[sidx] <= 2:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_x = x
                        
                        if best_x < 0:
                            break
                        start_block.append(best_x)
                        nb_set.add(best_x)
                    
                    if len(start_block) < k:
                        continue
                    start_block.sort()
                    candidate = start_block
                    break
                
                if candidate is None:
                    continue
                
                cand_t_idx = get_block_t_indices(candidate)
                
                # Check: does the new block cover all t-subsets that the old block uniquely covers?
                # i.e., for each t-subset in old block with coverage == 1, new block must also cover it
                old_t_idx = block_t_idx[bi]
                valid_swap = True
                for idx in old_t_idx:
                    if coverage[idx] == 1 and idx not in cand_t_idx:
                        valid_swap = False
                        break
                
                if not valid_swap:
                    continue
                
                # Apply swap
                # Remove old block's coverage
                for idx in old_t_idx:
                    coverage[idx] -= 1
                # Add new block's coverage
                for idx in cand_t_idx:
                    coverage[idx] += 1
                
                blocks[bi] = candidate
                block_t_idx[bi] = cand_t_idx
                
                # Count didn't change, but we may enable future removals
                # Slight temperature boost on swap to encourage exploration
                temp = max(temp, 0.5)
            
            temp *= cooling
        
        # Final reduction
        result = reduce_blocks(best_blocks)
        return result

    best_blocks = None
    best_count = float('inf')
    
    # Phase 1: Greedy constructions
    seed = 0
    while time.time() < deadline - 40:
        result = greedy_cover(seed)
        if result is not None:
            reduced = reduce_blocks(result)
            if len(reduced) < best_count:
                best_count = len(reduced)
                best_blocks = reduced
        seed += 1
    
    if best_blocks is None:
        # Fallback: deterministic greedy
        uncovered = np.ones(n_t_subsets, dtype=bool)
        uncovered_count = n_t_subsets
        blocks = []
        
        while uncovered_count > 0:
            # Find first uncovered
            idx = 0
            while idx < n_t_subsets and not uncovered[idx]:
                idx += 1
            if idx >= n_t_subsets:
                break
            
            uncovered[idx] = False
            uncovered_count -= 1
            
            block = list(all_t_subsets[idx])
            block_set = set(block)
            
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        r_tuple = tuple(sorted(rest))
                        sidx = point_to_tsubset_map[x].get(r_tuple, -1)
                        if sidx != -1 and uncovered[sidx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                
                if best_x < 0:
                    break
                block.append(best_x)
                block_set.add(best_x)
            
            if len(block) < k:
                break
            
            block.sort()
            blocks.append(block)
            
            for subset in itertools.combinations(block, t):
                sidx = t_subset_to_idx[subset]
                if uncovered[sidx]:
                    uncovered[sidx] = False
                    uncovered_count -= 1
        
        best_blocks = blocks
        best_count = len(best_blocks)
    
    # Phase 2: Simulated annealing refinement (multiple restarts)
    sa_seed = seed
    while time.time() < deadline - 15:
        sa_result = simulated_annealing(best_blocks, sa_seed)
        if sa_result is not None and len(sa_result) < best_count:
            best_count = len(sa_result)
            best_blocks = sa_result
        sa_seed += 1
    
    # Phase 3: Final reduction (multiple passes)
    if time.time() < deadline - 3:
        final = reduce_blocks(best_blocks)
        if len(final) < best_count:
            best_count = len(final)
            best_blocks = final
        # One more pass after SA changes
        if time.time() < deadline - 1:
            final2 = reduce_blocks(best_blocks)
            if len(final2) < best_count:
                best_count = len(final2)
                best_blocks = final2
    
    return best_blocks

# EVOLVE-BLOCK-END
