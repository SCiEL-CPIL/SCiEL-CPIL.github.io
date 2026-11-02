"""Combined covering design solver: greedy construction + Max-SAT-inspired local search."""
import itertools
import random
import time
import numpy as np


def _precompute(v, k, t):
    """Precompute all t-subsets and mappings for fast operations."""
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # For each point, the list of t-subset indices that contain it
    point_to_tsubs = [[] for _ in range(v)]
    for ts in all_tsubs:
        idx = tsub_to_idx[ts]
        for p in ts:
            point_to_tsubs[p].append(idx)
    
    # For each k-block, precompute its t-subset indices
    # We'll compute on the fly, but provide a helper
    
    return all_tsubs, tsub_to_idx, point_to_tsubs


def _greedy_construct(v, k, t, all_tsubs, tsub_to_idx, point_to_tsubs, rng, deadline):
    """Fast greedy construction. Returns list of blocks."""
    n_tsubs = len(all_tsubs)
    uncovered = np.ones(n_tsubs, dtype=np.uint8)
    blocks = []
    
    # Precompute for each k-subset its t-subset indices? Too many (C(19,8)=75582).
    # Instead, compute on the fly but cache.
    block_tsubs_cache = {}
    
    def get_block_tsubs(block):
        key = tuple(sorted(block))
        if key not in block_tsubs_cache:
            subs = []
            for ts in itertools.combinations(block, t):
                subs.append(tsub_to_idx[ts])
            block_tsubs_cache[key] = subs
        return block_tsubs_cache[key]
    
    while np.any(uncovered) and time.time() < deadline:
        # Find best block to add: maximizes number of newly covered t-subsets
        # Use random sampling of candidate blocks for speed
        best_gain = -1
        best_block = None
        
        # Sample a set of candidate blocks
        # Start with blocks containing random uncovered t-subsets
        uncovered_indices = np.where(uncovered)[0]
        if len(uncovered_indices) == 0:
            break
        
        # Pick a few random uncovered t-subsets as seeds
        n_seeds = min(20, len(uncovered_indices))
        seed_indices = rng.sample(list(uncovered_indices), n_seeds)
        
        candidates = set()
        for si in seed_indices:
            ts = all_tsubs[si]
            # Generate random k-blocks containing this t-subset
            remaining = [p for p in range(v) if p not in ts]
            for _ in range(5):
                if len(remaining) < k - t:
                    break
                extra = rng.sample(remaining, k - t)
                block = tuple(sorted(list(ts) + extra))
                candidates.add(block)
        
        # Also add some fully random blocks
        for _ in range(30):
            block = tuple(sorted(rng.sample(range(v), k)))
            candidates.add(block)
        
        # Evaluate candidates
        for block in candidates:
            if time.time() > deadline:
                break
            subs = get_block_tsubs(block)
            gain = 0
            for idx in subs:
                if uncovered[idx]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = block
        
        if best_block is None:
            # Fallback: random block
            best_block = tuple(sorted(rng.sample(range(v), k)))
            best_gain = 0
        
        # Add the block
        blocks.append(list(best_block))
        subs = get_block_tsubs(best_block)
        for idx in subs:
            uncovered[idx] = 0
    
    return blocks


def _local_search(blocks, v, k, t, all_tsubs, tsub_to_idx, point_to_tsubs, rng, deadline):
    """
    Max-SAT-inspired local search.
    State: set of selected blocks.
    Objective: minimize number of blocks while covering all t-subsets.
    Moves:
      1. Remove a redundant block.
      2. Swap a point in a block to make another block redundant.
      3. Add a block to enable removal of two others.
    """
    n_tsubs = len(all_tsubs)
    if not blocks:
        return blocks
    
    # Build data structures
    block_tsubs = []
    cover_count = np.zeros(n_tsubs, dtype=np.int32)
    
    for block in blocks:
        subs = []
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[ts]
            subs.append(idx)
            cover_count[idx] += 1
        block_tsubs.append(subs)
    
    n_blocks = len(blocks)
    best_count = n_blocks
    no_improve = 0
    max_no_improve = 50
    
    while time.time() < deadline - 5 and no_improve < max_no_improve:
        no_improve += 1
        improved = False
        
        # Phase 1: Remove redundant blocks
        for i in range(len(blocks)):
            if time.time() > deadline - 5:
                break
            redundant = True
            for idx in block_tsubs[i]:
                if cover_count[idx] == 1:
                    redundant = False
                    break
            if redundant:
                for idx in block_tsubs[i]:
                    cover_count[idx] -= 1
                blocks.pop(i)
                block_tsubs.pop(i)
                best_count = len(blocks)
                improved = True
                no_improve = 0
                break
        
        if improved:
            continue
        
        # Phase 2: Try swaps to enable removal
        # For each block i, find its critical t-subsets (covered only by i)
        # Try to modify another block j to cover one of these critical t-subsets
        for i in range(len(blocks)):
            if time.time() > deadline - 5:
                break
            
            criticals = [idx for idx in block_tsubs[i] if cover_count[idx] == 1]
            if not criticals:
                # Should have been caught in Phase 1
                continue
            
            # Try to cover criticals[0] by modifying another block
            crit_idx = criticals[0]
            crit_ts = all_tsubs[crit_idx]
            crit_set = set(crit_ts)
            
            for j in range(len(blocks)):
                if j == i or time.time() > deadline - 5:
                    continue
                
                block_j = blocks[j]
                block_j_set = set(block_j)
                
                # Check if block_j already covers crit_ts
                if crit_idx in block_tsubs[j]:
                    continue
                
                # Find how many points of crit_ts are in block_j
                common = block_j_set & crit_set
                missing = crit_set - block_j_set
                
                # We need to add |missing| points from crit_ts and remove |missing| other points
                # Only try if |missing| <= 2 for speed
                if len(missing) > 2:
                    continue
                
                missing_list = list(missing)
                # Points in block_j not in crit_ts
                extra = list(block_j_set - crit_set)
                
                if len(extra) < len(missing_list):
                    continue
                
                # Try all combinations of points to remove
                for rem_combo in itertools.combinations(extra, len(missing_list)):
                    if time.time() > deadline - 5:
                        break
                    
                    new_block_j = [p for p in block_j if p not in set(rem_combo)] + missing_list
                    new_block_j.sort()
                    new_block_j_set = set(new_block_j)
                    
                    # Compute new t-subsets for block_j
                    new_subs = set()
                    for ts in itertools.combinations(new_block_j, t):
                        new_subs.add(tsub_to_idx[ts])
                    
                    old_subs_set = set(block_tsubs[j])
                    
                    # Check if any t-subset lost unique coverage
                    can_swap = True
                    for idx in old_subs_set:
                        if idx not in new_subs and cover_count[idx] == 1:
                            can_swap = False
                            break
                    
                    if not can_swap:
                        continue
                    
                    # Check if new block_j covers crit_idx
                    if crit_idx not in new_subs:
                        continue
                    
                    # Perform the swap
                    # Remove old t-subsets
                    for idx in old_subs_set:
                        cover_count[idx] -= 1
                    # Add new t-subsets
                    for idx in new_subs:
                        cover_count[idx] += 1
                    
                    blocks[j] = new_block_j
                    block_tsubs[j] = list(new_subs)
                    
                    # Now try to remove block i
                    still_critical = False
                    for idx in block_tsubs[i]:
                        if cover_count[idx] == 1:
                            still_critical = True
                            break
                    
                    if not still_critical:
                        # Remove block i
                        for idx in block_tsubs[i]:
                            cover_count[idx] -= 1
                        blocks.pop(i)
                        block_tsubs.pop(i)
                        best_count = len(blocks)
                        improved = True
                        no_improve = 0
                        break
                    else:
                        # Revert
                        for idx in new_subs:
                            cover_count[idx] -= 1
                        for idx in old_subs_set:
                            cover_count[idx] += 1
                        blocks[j] = sorted(block_j_set)  # restore
                        block_tsubs[j] = list(old_subs_set)
                    
                    if improved:
                        break
                
                if improved:
                    break
            
            if improved:
                break
        
        # Phase 3: Random perturbation if stuck
        if no_improve > 20 and len(blocks) > 10:
            # Randomly modify a random block
            j = rng.randint(0, len(blocks) - 1)
            block_j = blocks[j]
            old_subs_set = set(block_tsubs[j])
            
            # Replace one random point
            p_remove = rng.choice(block_j)
            remaining = [p for p in range(v) if p != p_remove]
            p_add = rng.choice(remaining)
            
            new_block_j = [p for p in block_j if p != p_remove] + [p_add]
            new_block_j.sort()
            
            new_subs = set()
            for ts in itertools.combinations(new_block_j, t):
                new_subs.add(tsub_to_idx[ts])
            
            # Check validity
            can_swap = True
            for idx in old_subs_set:
                if idx not in new_subs and cover_count[idx] == 1:
                    can_swap = False
                    break
            
            if can_swap:
                for idx in old_subs_set:
                    cover_count[idx] -= 1
                for idx in new_subs:
                    cover_count[idx] += 1
                blocks[j] = new_block_j
                block_tsubs[j] = list(new_subs)
                no_improve = 0
    
    return blocks


def construct(v, k, t):
    """Construct a covering design C(v,k,t)."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    rng = random.Random(42)
    
    # Precompute
    all_tsubs, tsub_to_idx, point_to_tsubs = _precompute(v, k, t)
    
    best_blocks = None
    best_count = float('inf')
    
    # Multiple restarts
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        # Greedy construction
        blocks = _greedy_construct(v, k, t, all_tsubs, tsub_to_idx, point_to_tsubs, rng, deadline - 20)
        
        if not blocks:
            continue
        
        # Verify coverage
        cover_count = np.zeros(len(all_tsubs), dtype=np.uint8)
        for block in blocks:
            for ts in itertools.combinations(block, t):
                cover_count[tsub_to_idx[ts]] += 1
        
        if np.any(cover_count == 0):
            # Invalid, skip
            continue
        
        # Local search
        blocks = _local_search(blocks, v, k, t, all_tsubs, tsub_to_idx, point_to_tsubs, rng, deadline - 5)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
    
    if best_blocks is None:
        # Fallback
        blocks = _greedy_construct(v, k, t, all_tsubs, tsub_to_idx, point_to_tsubs, rng, deadline)
        best_blocks = blocks
    
    return best_blocks
