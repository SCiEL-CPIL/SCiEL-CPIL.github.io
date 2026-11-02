"""
Strategy: Simulated Annealing over valid coverings with bitmask representation.
- Build an initial valid covering using a fast greedy (bitmask-accelerated).
- Generate a pool of candidate blocks (including greedy + random perturbations).
- Use simulated annealing to minimize block count:
  * Move types: remove a redundant block, swap one block for another, add a block (rare).
  * Track coverage counts per triple (C(24,3)=2024 triples) to quickly test validity.
  * Accept invalid states with a small temperature-dependent probability to escape local minima.
- Return the best valid covering found within the time budget.
"""

import itertools
import random
import time
import math

def _build_initial_greedy(v, k, t, rng):
    """
    Fast greedy construction of a valid covering using bitmask for uncovered triples.
    Returns list of blocks (each a sorted tuple of k points).
    """
    # Precompute triple indices
    triple_list = list(itertools.combinations(range(v), t))
    triple_index = {triple: i for i, triple in enumerate(triple_list)}
    n_triples = len(triple_list)  # 2024
    
    # Bitmask of uncovered triples (1 = uncovered)
    uncovered = (1 << n_triples) - 1
    
    # Precompute for each block, the bitmask of triples it covers
    # But we build blocks incrementally, so we compute on the fly.
    
    blocks = []
    while uncovered:
        # Find first uncovered triple
        first_idx = 0
        for i in range(n_triples):
            if (uncovered >> i) & 1:
                first_idx = i
                break
        first_triple = triple_list[first_idx]
        
        block = list(first_triple)
        
        # Greedily add points to cover the most uncovered triples
        while len(block) < k:
            best_point = None
            best_gain = -1
            block_set = set(block)
            for x in range(v):
                if x in block_set:
                    continue
                # Compute gain: number of newly covered triples
                gain = 0
                for subset in itertools.combinations(block, t - 1):
                    new_triple = tuple(sorted(subset + (x,)))
                    idx = triple_index[new_triple]
                    if (uncovered >> idx) & 1:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            if best_point is None:
                break
            block.append(best_point)
        
        block.sort()
        blocks.append(tuple(block))
        
        # Update uncovered
        for triple in itertools.combinations(block, t):
            idx = triple_index[triple]
            uncovered &= ~(1 << idx)
    
    return blocks

def _generate_pool(v, k, t, rng, greedy_blocks, num_extra=500):
    """
    Generate a pool of candidate blocks: greedy blocks + random perturbations.
    """
    pool = list(set(greedy_blocks))
    
    seen = set(pool)
    for _ in range(num_extra):
        # Random block
        block = tuple(sorted(rng.sample(range(v), k)))
        if block not in seen:
            pool.append(block)
            seen.add(block)
        
        # Perturb a greedy block: swap 1-2 points
        if pool:
            base = rng.choice(pool)
            new_block = list(base)
            for _ in range(rng.randint(1, 2)):
                idx_remove = rng.randint(0, k - 1)
                removed = new_block.pop(idx_remove)
                candidates = [x for x in range(v) if x not in new_block]
                if candidates:
                    new_block.append(rng.choice(candidates))
            new_block.sort()
            new_tuple = tuple(new_block)
            if new_tuple not in seen:
                pool.append(new_tuple)
                seen.add(new_tuple)
    
    return pool

def _precompute_block_triples(pool, v, t, triple_index):
    """
    For each block in pool, precompute list of triple indices it covers.
    """
    block_triples = []
    for block in pool:
        triples = []
        for triple in itertools.combinations(block, t):
            triples.append(triple_index[triple])
        block_triples.append(triples)
    return block_triples

def _count_triple_coverage(selected, block_triples, n_triples):
    """
    Count how many selected blocks cover each triple.
    Returns list of counts, length n_triples.
    """
    counts = [0] * n_triples
    for idx in selected:
        for t_idx in block_triples[idx]:
            counts[t_idx] += 1
    return counts

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    rng = random.Random(123)
    
    # Precompute triple indexing
    triple_list = list(itertools.combinations(range(v), t))
    triple_index = {triple: i for i, triple in enumerate(triple_list)}
    n_triples = len(triple_list)
    
    # Step 1: Build initial greedy covering
    greedy_blocks = _build_initial_greedy(v, k, t, rng)
    initial_count = len(greedy_blocks)
    
    # Step 2: Generate candidate pool
    pool = _generate_pool(v, k, t, rng, greedy_blocks, num_extra=600)
    
    # Step 3: Precompute triple indices for each block in pool
    block_triples = _precompute_block_triples(pool, v, t, triple_index)
    
    # Map greedy blocks to pool indices
    greedy_pool_indices = []
    pool_set = {block: i for i, block in enumerate(pool)}
    for gb in greedy_blocks:
        if gb in pool_set:
            greedy_pool_indices.append(pool_set[gb])
        else:
            # Should not happen, but just in case
            pool.append(gb)
            pool_set[gb] = len(pool) - 1
            block_triples.append([triple_index[tr] for tr in itertools.combinations(gb, t)])
            greedy_pool_indices.append(len(pool) - 1)
    
    # Step 4: Simulated Annealing
    # State: selected = list of pool indices
    selected = greedy_pool_indices[:]
    counts = _count_triple_coverage(selected, block_triples, n_triples)
    valid = all(c >= 1 for c in counts)
    
    best_selected = selected[:] if valid else None
    best_count = len(selected) if valid else float('inf')
    
    T = 1.0
    cooling = 0.995
    min_T = 0.01
    
    while T > min_T and time.time() < deadline - 5.0:
        # Choose a move
        move_type = rng.random()
        
        if move_type < 0.4 and len(selected) > 1:
            # Try removing a block
            i_remove = rng.randint(0, len(selected) - 1)
            idx = selected[i_remove]
            
            # Check if any triple becomes uncovered
            removable = True
            for t_idx in block_triples[idx]:
                if counts[t_idx] <= 1:
                    removable = False
                    break
            
            if removable:
                # Apply removal
                for t_idx in block_triples[idx]:
                    counts[t_idx] -= 1
                selected.pop(i_remove)
                new_count = len(selected)
                new_valid = all(c >= 1 for c in counts)
                
                if new_valid:
                    if new_count < best_count:
                        best_count = new_count
                        best_selected = selected[:]
                else:
                    # Rollback
                    selected.insert(i_remove, idx)
                    for t_idx in block_triples[idx]:
                        counts[t_idx] += 1
            
            elif rng.random() < 0.1:
                # Occasionally try removing even if not immediately valid
                # (this is rare; mostly for exploration)
                pass
        
        elif move_type < 0.8:
            # Try swapping one block for another
            i_remove = rng.randint(0, len(selected) - 1)
            idx_remove = selected[i_remove]
            
            # Pick a random different block to add
            i_add = rng.randint(0, len(pool) - 1)
            idx_add = pool[i_add] if False else i_add  # pool index
            
            # Check if idx_add is already in selected
            if idx_add in selected:
                continue
            
            # Compute delta
            # Removing idx_remove, adding idx_add
            # Check if all triples remain covered after swap
            # Temporarily decrease counts for idx_remove
            can_swap = True
            for t_idx in block_triples[idx_remove]:
                if counts[t_idx] <= 1:
                    # This triple is only covered by idx_remove
                    # Check if idx_add covers it
                    if t_idx not in block_triples[idx_add]:
                        can_swap = False
                        break
            
            if can_swap:
                # Apply swap
                for t_idx in block_triples[idx_remove]:
                    counts[t_idx] -= 1
                for t_idx in block_triples[idx_add]:
                    counts[t_idx] += 1
                
                selected[i_remove] = idx_add
                new_count = len(selected)
                new_valid = all(c >= 1 for c in counts)
                
                if new_valid:
                    if new_count < best_count:
                        best_count = new_count
                        best_selected = selected[:]
                else:
                    # Rollback
                    for t_idx in block_triples[idx_add]:
                        counts[t_idx] -= 1
                    for t_idx in block_triples[idx_remove]:
                        counts[t_idx] += 1
                    selected[i_remove] = idx_remove
            
            elif rng.random() < 0.05:
                # Small chance to accept invalid swap for exploration
                for t_idx in block_triples[idx_remove]:
                    counts[t_idx] -= 1
                for t_idx in block_triples[idx_add]:
                    counts[t_idx] += 1
                
                selected[i_remove] = idx_add
                
                # Check validity
                new_valid = all(c >= 1 for c in counts)
                if not new_valid and rng.random() < 0.01:
                    # Rollback if too invalid
                    for t_idx in block_triples[idx_add]:
                        counts[t_idx] -= 1
                    for t_idx in block_triples[idx_remove]:
                        counts[t_idx] += 1
                    selected[i_remove] = idx_remove
                elif new_valid:
                    if len(selected) < best_count:
                        best_count = len(selected)
                        best_selected = selected[:]
        
        else:
            # Rarely add a block (to escape local minima)
            if len(selected) < 100:
                idx_add = rng.randint(0, len(pool) - 1)
                if idx_add not in selected:
                    selected.append(idx_add)
                    for t_idx in block_triples[idx_add]:
                        counts[t_idx] += 1
                    # Check if this addition allows a later removal
                    # For now, just accept
                    new_valid = all(c >= 1 for c in counts)
                    if new_valid and len(selected) < best_count:
                        best_count = len(selected)
                        best_selected = selected[:]
        
        T *= cooling
    
    # Step 5: Final cleanup - try to remove redundant blocks from best_selected
    if best_selected:
        # Iteratively remove redundant blocks
        changed = True
        while changed:
            changed = False
            counts_best = _count_triple_coverage(best_selected, block_triples, n_triples)
            for i in range(len(best_selected)):
                idx = best_selected[i]
                removable = True
                for t_idx in block_triples[idx]:
                    if counts_best[t_idx] <= 1:
                        removable = False
                        break
                if removable:
                    for t_idx in block_triples[idx]:
                        counts_best[t_idx] -= 1
                    best_selected.pop(i)
                    changed = True
                    break
    
    # Step 6: Convert to final blocks
    if best_selected:
        final_blocks = [list(pool[i]) for i in best_selected]
    else:
        # Fallback
        final_blocks = [list(b) for b in greedy_blocks]
    
    # Final validation
    covered = set()
    for block in final_blocks:
        for triple in itertools.combinations(block, t):
            covered.add(tuple(triple))
    
    if covered != set(triple_list):
        # Fallback to greedy
        final_blocks = [list(b) for b in greedy_blocks]
    
    return final_blocks
