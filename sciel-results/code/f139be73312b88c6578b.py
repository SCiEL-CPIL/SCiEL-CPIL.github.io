import itertools
import time
import random
import sys
from collections import Counter

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using:
    1. Fast greedy construction for a valid initial covering.
    2. Simulated Annealing with block-swap moves and bitmask coverage tracking.
    
    Returns a list of blocks (each a sorted list of k distinct integers in 0..v-1).
    """
    start_time = time.time()
    deadline = start_time + 143.0  # Safety margin

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}

    # Precompute bitmask for each t-subset
    t_masks = [0] * num_t_subsets
    for i, ts in enumerate(all_t_subsets):
        mask = 0
        for pt in ts:
            mask |= (1 << pt)
        t_masks[i] = mask

    # Precompute: for each k-subset (block), the list of t-subset indices it covers
    # This is expensive to precompute fully, so we compute on-the-fly but cache
    block_t_indices_cache = {}
    
    def get_block_t_indices(block):
        """Return list of t-subset indices covered by this block."""
        key = tuple(block)
        if key in block_t_indices_cache:
            return block_t_indices_cache[key]
        indices = []
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            indices.append(idx)
        block_t_indices_cache[key] = indices
        return indices

    def greedy_construct(seed, time_limit):
        """Fast greedy: pick uncovered t-subset, build block around it."""
        rng = random.Random(seed)
        covered = [0] * num_t_subsets
        blocks = []
        
        while True:
            # Find an uncovered t-subset
            uncovered = [i for i in range(num_t_subsets) if covered[i] == 0]
            if not uncovered:
                break
            
            target_idx = rng.choice(uncovered)
            target_subset = all_t_subsets[target_idx]
            
            # Build a block containing target_subset
            block_points = set(target_subset)
            remaining = [x for x in range(v) if x not in block_points]
            rng.shuffle(remaining)
            
            # Greedily add points that cover the most new t-subsets
            for _ in range(k - t):
                if not remaining:
                    break
                best_point = None
                best_gain = -1
                for x in remaining:
                    gain = 0
                    # Count how many currently-uncovered t-subsets would be covered
                    # by adding x to the block
                    for rest in itertools.combinations(block_points, t - 1):
                        new_sub = tuple(sorted(rest + (x,)))
                        idx = t_subset_to_idx.get(new_sub)
                        if idx is not None and covered[idx] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point is not None:
                    block_points.add(best_point)
                    remaining.remove(best_point)
                else:
                    if remaining:
                        p = remaining.pop(0)
                        block_points.add(p)
            
            block = tuple(sorted(block_points))
            blocks.append(block)
            
            # Update coverage
            for idx in get_block_t_indices(block):
                covered[idx] += 1
            
            if time.time() > time_limit:
                break
        
        return blocks, covered

    # Phase 1: Greedy construction
    initial_blocks, covered = greedy_construct(42, start_time + 5.0)
    n_blocks = len(initial_blocks)
    
    # Ensure validity
    if any(c == 0 for c in covered):
        # Patch: add blocks for uncovered t-subsets
        for i in range(num_t_subsets):
            if covered[i] == 0:
                ts = all_t_subsets[i]
                block_points = set(ts)
                remaining = [x for x in range(v) if x not in block_points]
                random.shuffle(remaining)
                for x in remaining:
                    if len(block_points) >= k:
                        break
                    block_points.add(x)
                block = tuple(sorted(block_points))
                initial_blocks.append(block)
                for idx in get_block_t_indices(block):
                    covered[idx] += 1
        n_blocks = len(initial_blocks)

    # Phase 2: Simulated Annealing with block swaps
    # Coverage count array
    cov = [0] * num_t_subsets
    for b in initial_blocks:
        for idx in get_block_t_indices(b):
            cov[idx] += 1

    best_n = n_blocks
    best_blocks = list(initial_blocks)
    best_cov = list(cov)

    rng = random.Random(12345)
    
    # Precompute all k-subsets for random swap targets (for v=19,k=8: C(19,8)=75582)
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_k_subsets = len(all_k_subsets)
    
    # Precompute t-indices for all k-subsets (memory: 75582 * C(8,5)=56 = ~4.2M ints, ~34MB)
    # Actually C(8,5) = 56, so each block covers 56 t-subsets
    # 75582 * 56 = 4,232,592 entries. Store as list of arrays.
    all_block_t_indices = []
    for ks in all_k_subsets:
        indices = []
        for ts in itertools.combinations(ks, t):
            idx = t_subset_to_idx[ts]
            indices.append(idx)
        all_block_t_indices.append(indices)
    
    # Map from k-subset to its index in all_k_subsets for quick lookup
    k_subset_to_ks_idx = {ks: i for i, ks in enumerate(all_k_subsets)}

    # SA parameters
    initial_temp = 2.0
    cooling_rate = 0.9995
    min_temp = 0.01
    
    current_temp = initial_temp
    sa_start = time.time()
    
    # Track current state
    cur_blocks = list(initial_blocks)
    cur_cov = list(cov)
    cur_n = n_blocks

    while time.time() < deadline - 5.0 and current_temp > min_temp:
        # Pick a random block to replace
        i = rng.randrange(cur_n)
        old_block = cur_blocks[i]
        old_t_indices = all_block_t_indices[k_subset_to_ks_idx[old_block]]
        
        # Pick a random new block (different from old)
        new_ks_idx = rng.randrange(n_k_subsets)
        if all_k_subsets[new_ks_idx] == old_block:
            continue
        new_block = all_k_subsets[new_ks_idx]
        new_t_indices = all_block_t_indices[new_ks_idx]
        
        # Compute delta: how coverage changes
        # For each t-subset in old_block: cov decreases by 1
        # For each t-subset in new_block: cov increases by 1
        # Delta in "badness" = number of t-subsets that go from 1->0 (bad) minus 0->1 (good)
        delta_bad = 0
        # Temporarily compute
        for idx in old_t_indices:
            if cur_cov[idx] == 1:
                delta_bad += 1  # This t-subset would become uncovered
        for idx in new_t_indices:
            if cur_cov[idx] == 0:
                delta_bad -= 1  # This t-subset would become covered
        
        # Accept if delta_bad <= 0 (improvement or neutral) or with probability exp(-delta_bad/temp)
        if delta_bad <= 0:
            accept = True
        else:
            # delta_bad > 0 means we'd create delta_bad newly uncovered t-subsets
            # Accept with probability exp(-delta_bad / current_temp)
            import math
            accept = (rng.random() < math.exp(-delta_bad / current_temp))
        
        if accept:
            # Apply move
            for idx in old_t_indices:
                cur_cov[idx] -= 1
            for idx in new_t_indices:
                cur_cov[idx] += 1
            cur_blocks[i] = new_block
            
            # Check if we improved (no newly uncovered)
            if delta_bad <= 0:
                # Check if this is a new best
                # Actually we need to check if any cov went to 0
                # If delta_bad == 0, it's neutral; if < 0, improvement
                pass
        
        # Cool down
        current_temp *= cooling_rate

        # Periodic: check if we can reduce (remove blocks with all cov > 1)
        # This is a different move type: try removing a block
        if rng.random() < 0.1 and cur_n > 1:
            # Try to find a removable block
            removable_idx = -1
            for j in range(cur_n):
                b = cur_blocks[j]
                t_inds = all_block_t_indices[k_subset_to_ks_idx[b]]
                can_remove = True
                for idx in t_inds:
                    if cur_cov[idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    removable_idx = j
                    break
            
            if removable_idx >= 0:
                b = cur_blocks[removable_idx]
                t_inds = all_block_t_indices[k_subset_to_ks_idx[b]]
                for idx in t_inds:
                    cur_cov[idx] -= 1
                cur_blocks.pop(removable_idx)
                cur_n -= 1
                
                if cur_n < best_n:
                    best_n = cur_n
                    best_blocks = list(cur_blocks)
                    best_cov = list(cur_cov)

        # Also periodically try: replace a block with one that covers more uncovered
        # (greedy repair if we ever create uncovered, but we shouldn't in SA)
    
    # Final reduction pass on best
    final_blocks = list(best_blocks)
    final_cov = [0] * num_t_subsets
    for b in final_blocks:
        for idx in all_block_t_indices[k_subset_to_ks_idx[b]]:
            final_cov[idx] += 1
    
    # Remove redundant blocks
    changed = True
    while changed and time.time() < deadline - 2.0:
        changed = False
        for i in range(len(final_blocks)):
            b = final_blocks[i]
            t_inds = all_block_t_indices[k_subset_to_ks_idx[b]]
            can_remove = True
            for idx in t_inds:
                if final_cov[idx] == 1:
                    can_remove = False
                    break
            if can_remove:
                for idx in t_inds:
                    final_cov[idx] -= 1
                final_blocks.pop(i)
                changed = True
                break
    
    # Validate
    if any(c == 0 for c in final_cov):
        # Fallback to initial
        final_blocks = list(initial_blocks)
    
    return [list(b) for b in final_blocks]
