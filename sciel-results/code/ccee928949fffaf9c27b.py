"""Covering design C(19,8,5) - Fast Bitmask Greedy + Local Search.

Uses Python integer bitmasks for efficient coverage tracking.
Precomputes bitmask for each of the C(19,8) blocks.
Runs randomized greedy with tie-breaking and local search (removal).
"""
import itertools
import random
import time


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Fallback for other parameters (simple greedy, not optimized)
    if not (v == 19 and k == 8 and t == 5):
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            target = min(uncovered)
            tset = set(target)
            remaining = [x for x in range(v) if x not in tset]
            need = k - t
            best_block, best_gain = None, -1
            for comb in itertools.combinations(remaining, need):
                block = tuple(sorted(target + comb))
                gain = sum(1 for sub in itertools.combinations(block, t) if sub in uncovered)
                if gain > best_gain:
                    best_gain, best_block = gain, block
            if best_block is None:
                break
            blocks.append(list(best_block))
            uncovered.difference_update(itertools.combinations(best_block, t))
            if time.time() > deadline:
                break
        return blocks

    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    n_t = len(all_t_subsets)
    full_mask = (1 << n_t) - 1
    
    # Map t-subsets to indices
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute bitmask for each block
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    # block_mask[j] is an integer where bit i is 1 if block j covers t-subset i
    block_masks = [0] * n_blocks
    for j, block in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            mask |= (1 << idx)
        block_masks[j] = mask

    def run_greedy(seed):
        """Randomized greedy with bitmask coverage."""
        rng = random.Random(seed)
        coverage = 0
        selected_indices = []
        
        # We need to track which blocks are selected to avoid re-selection
        # Since we don't remove blocks in greedy, we can just keep a set of selected indices
        selected_set = set()
        
        while coverage != full_mask:
            if time.time() > deadline - 5.0:
                return None
            
            best_gain = -1
            best_cands = []
            
            # Find block with max marginal gain
            # Optimization: If we have many blocks, this loop is the bottleneck.
            # 75k blocks * ~300 steps = 22.5M iterations.
            # Each iteration: bit_count(mask & ~coverage).
            # This should be fast enough in Python if we minimize overhead.
            
            for j in range(n_blocks):
                if j in selected_set:
                    continue
                new_cov = block_masks[j] & ~coverage
                gain = new_cov.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_cands = [j]
                elif gain == best_gain:
                    best_cands.append(j)
            
            if not best_cands:
                break
                
            b = rng.choice(best_cands)
            selected_indices.append(b)
            selected_set.add(b)
            coverage |= block_masks[b]
            
        return selected_indices

    def local_search_remove(selected_indices):
        """Remove redundant blocks."""
        if not selected_indices:
            return selected_indices
            
        # Compute current coverage
        coverage = 0
        for b in selected_indices:
            coverage |= block_masks[b]
            
        if coverage != full_mask:
            return selected_indices # Should not happen if greedy was complete
            
        changed = True
        while changed:
            changed = False
            if time.time() > deadline - 2.0:
                break
                
            for i in range(len(selected_indices)):
                b = selected_indices[i]
                # Check if removing b still covers everything
                # new_coverage = coverage & ~block_masks[b]
                # If new_coverage == full_mask, it's redundant
                if (coverage & ~block_masks[b]) == full_mask:
                    selected_indices.pop(i)
                    coverage &= ~block_masks[b]
                    changed = True
                    break # Restart loop as indices changed
                    
        return selected_indices

    best_sol = None
    best_len = float('inf')
    
    # First run
    sol = run_greedy(0)
    if sol is not None:
        sol = local_search_remove(sol)
        best_sol = sol
        best_len = len(sol)
        
    # Randomized restarts
    seed = 1
    while time.time() < deadline - 5.0:
        sol = run_greedy(seed)
        seed += 1
        if sol is None:
            continue
            
        sol = local_search_remove(sol)
        
        if len(sol) < best_len:
            best_len = len(sol)
            best_sol = sol
            
    if best_sol is None:
        # Fallback: return empty or a trivial valid set if greedy failed (unlikely)
        # But greedy should always find a covering.
        return []
        
    return [list(all_blocks[b]) for b in best_sol]
