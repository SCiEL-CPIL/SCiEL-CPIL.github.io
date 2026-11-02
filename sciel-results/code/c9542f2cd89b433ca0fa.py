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
        """Remove redundant blocks."""
        if not blocks:
            return []
        
        blocks = list(blocks)
        improved = True
        
        while improved:
            if time.time() > deadline - 10:
                break
            improved = False
            
            # Compute coverage counts
            coverage_count = np.zeros(n_t_subsets, dtype=np.int32)
            for block in blocks:
                for subset in itertools.combinations(block, t):
                    idx = t_subset_to_idx[subset]
                    coverage_count[idx] += 1
            
            # Try removing each block (random order for variety)
            indices = list(range(len(blocks)))
            random.shuffle(indices)
            
            for bi in indices:
                if time.time() > deadline - 10:
                    break
                
                block = blocks[bi]
                can_remove = True
                affected_indices = []
                for subset in itertools.combinations(block, t):
                    idx = t_subset_to_idx[subset]
                    coverage_count[idx] -= 1
                    affected_indices.append(idx)
                    if coverage_count[idx] == 0:
                        can_remove = False
                        break
                
                if can_remove:
                    blocks.pop(bi)
                    improved = True
                    break
                else:
                    for a_idx in affected_indices:
                        coverage_count[a_idx] += 1
        
        return blocks

    def simulated_annealing(blocks, seed):
        """SA to minimize block count by swapping blocks."""
        rng = random.Random(seed)
        blocks = list(blocks)
        current_count = len(blocks)
        best_count = current_count
        best_blocks = list(blocks)
        
        # Precompute coverage for current solution
        coverage = np.zeros(n_t_subsets, dtype=np.int32)
        for block in blocks:
            for subset in itertools.combinations(block, t):
                idx = t_subset_to_idx[subset]
                coverage[idx] += 1
        
        n_blocks = len(blocks)
        temp = 5.0
        cooling = 0.98
        min_temp = 0.1
        max_iter = 3000
        
        for it in range(max_iter):
            if time.time() > deadline - 8:
                break
            if temp < min_temp:
                break
            
            # Try to remove a random block and re-cover uncovered t-subsets
            if n_blocks <= 1:
                break
            
            bi = rng.randint(0, n_blocks - 1)
            removed_block = blocks[bi]
            
            # Temporarily remove
            for subset in itertools.combinations(removed_block, t):
                idx = t_subset_to_idx[subset]
                coverage[idx] -= 1
            
            uncovered_indices = [i for i in range(n_t_subsets) if coverage[i] == 0]
            
            if not uncovered_indices:
                # Can remove this block!
                blocks.pop(bi)
                n_blocks -= 1
                current_count -= 1
                if current_count < best_count:
                    best_count = current_count
                    best_blocks = list(blocks)
                # Reset temp to explore more
                temp = max(temp, 2.0)
                continue
            
            # Need to re-cover: try to find a small set of new blocks
            # Greedily cover uncovered t-subsets
            new_blocks = []
            local_uncovered = set(uncovered_indices)
            
            while local_uncovered:
                if time.time() > deadline - 8:
                    break
                
                # Pick random uncovered
                ui = list(local_uncovered)[rng.randint(0, len(local_uncovered) - 1)]
                local_uncovered.discard(ui)
                
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
                            if sidx != -1 and sidx in local_uncovered:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                    
                    if best_x < 0:
                        break
                    start_block.append(best_x)
                    nb_set.add(best_x)
                
                if len(start_block) < k:
                    break
                
                start_block.sort()
                new_blocks.append(start_block)
                
                for subset in itertools.combinations(start_block, t):
                    sidx = t_subset_to_idx[subset]
                    if sidx in local_uncovered:
                        local_uncovered.discard(sidx)
            
            if local_uncovered:
                # Couldn't fully re-cover; revert
                for subset in itertools.combinations(removed_block, t):
                    idx = t_subset_to_idx[subset]
                    coverage[idx] += 1
                continue
            
            # Check if new blocks are better (fewer or equal)
            if len(new_blocks) < 1:
                # Shouldn't happen but safety
                for subset in itertools.combinations(removed_block, t):
                    idx = t_subset_to_idx[subset]
                    coverage[idx] += 1
                continue
            
            # Replace: remove old block, add new blocks
            # Update coverage for new blocks
            for nb in new_blocks:
                for subset in itertools.combinations(nb, t):
                    idx = t_subset_to_idx[subset]
                    coverage[idx] += 1
            
            # Accept if it doesn't increase count, or with SA probability if it does
            delta = len(new_blocks) - 1
            if delta <= 0:
                # Accept
                blocks[bi] = new_blocks[0] if len(new_blocks) == 1 else new_blocks[0]
                # Actually we need to handle multiple new blocks properly
                # For simplicity, only accept single-block replacements
                if len(new_blocks) == 1:
                    blocks[bi] = new_blocks[0]
                    # n_blocks stays same
                else:
                    # Revert; multi-block replacement is complex
                    for nb in new_blocks:
                        for subset in itertools.combinations(nb, t):
                            idx = t_subset_to_idx[subset]
                            coverage[idx] -= 1
                    for subset in itertools.combinations(removed_block, t):
                        idx = t_subset_to_idx[subset]
                        coverage[idx] += 1
            else:
                # Delta > 0: accept with probability exp(-delta/temp)
                if rng.random() < np.exp(-delta / temp):
                    # Accept: replace one block with multiple
                    # This changes n_blocks
                    # For simplicity, only do single-block swaps
                    for nb in new_blocks:
                        for subset in itertools.combinations(nb, t):
                            idx = t_subset_to_idx[subset]
                            coverage[idx] -= 1
                    for subset in itertools.combinations(removed_block, t):
                        idx = t_subset_to_idx[subset]
                        coverage[idx] += 1
                else:
                    for nb in new_blocks:
                        for subset in itertools.combinations(nb, t):
                            idx = t_subset_to_idx[subset]
                            coverage[idx] -= 1
                    for subset in itertools.combinations(removed_block, t):
                        idx = t_subset_to_idx[subset]
                        coverage[idx] += 1
            
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
    
    # Phase 2: Simulated annealing refinement
    if time.time() < deadline - 15:
        sa_result = simulated_annealing(best_blocks, seed)
        if sa_result is not None and len(sa_result) < best_count:
            best_count = len(sa_result)
            best_blocks = sa_result
    
    # Phase 3: Final reduction
    if time.time() < deadline - 3:
        final = reduce_blocks(best_blocks)
        if len(final) < best_count:
            best_count = len(final)
            best_blocks = final
    
    return best_blocks

# EVOLVE-BLOCK-END
