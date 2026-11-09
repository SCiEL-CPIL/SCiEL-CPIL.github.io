import itertools
import random
import time
import sys

# Global cache
_CACHE = {}

def _precompute(v, k, t):
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]
    
    start_time = time.time()
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    triples = list(itertools.combinations(points, t))
    triple_to_idx = {tri: i for i, tri in enumerate(triples)}
    n_triples = len(triples)
    
    # block_masks[i] is a bitmask of triples covered by block i
    block_masks = [0] * n_blocks
    # blocks_containing[j] is a list of block indices containing triple j
    blocks_containing = [[] for _ in range(n_triples)]
    # block_triples[i] is a list of triple indices in block i
    block_triples = [[] for _ in range(n_blocks)]
    
    for i, block in enumerate(all_blocks):
        mask = 0
        for tri in itertools.combinations(block, t):
            idx = triple_to_idx[tri]
            mask |= (1 << idx)
            blocks_containing[idx].append(i)
            block_triples[i].append(idx)
        block_masks[i] = mask
        
    full_mask = (1 << n_triples) - 1
    
    _CACHE[key] = (all_blocks, block_masks, full_mask, blocks_containing, n_blocks, n_triples, block_triples)
    return _CACHE[key]

def _greedy_construct(v, k, t, all_blocks, block_masks, full_mask, blocks_containing, n_blocks, seed, block_triples):
    rng = random.Random(seed)
    uncovered = full_mask
    blocks_used = []
    used_mask = 0
    active_blocks = list(range(n_blocks))
    
    while uncovered:
        uncovered_count = uncovered.bit_count()
        
        # Compute gains for all active blocks - track top candidates efficiently
        max_gain = 0
        best_indices = []
        near_best_indices = []
        
        for i, idx in enumerate(active_blocks):
            gain = (block_masks[idx] & uncovered).bit_count()
            if gain > max_gain:
                max_gain = gain
                best_indices = [i]
                near_best_indices = [i]
                if gain == uncovered_count:
                    break
            elif gain == max_gain:
                best_indices.append(i)
                near_best_indices.append(i)
            elif gain >= max_gain - 1:
                near_best_indices.append(i)
        
        if max_gain == 0:
            break
        
        # Pick from best or near-best with some randomness
        if rng.random() < 0.7:
            selected_pos = rng.choice(best_indices)
        else:
            selected_pos = rng.choice(near_best_indices)
        
        selected_idx = active_blocks[selected_pos]
        
        blocks_used.append(selected_idx)
        used_mask |= block_masks[selected_idx]
        uncovered &= ~used_mask
        active_blocks.pop(selected_pos)
        
    return blocks_used

def _local_search(v, k, t, initial_blocks, block_masks, full_mask, blocks_containing, n_blocks, n_triples, block_triples, seed, deadline):
    rng = random.Random(seed)
    
    selected = set(initial_blocks)
    coverage_count = [0] * n_triples
    
    for b in selected:
        for tri_idx in block_triples[b]:
            coverage_count[tri_idx] += 1
    
    best_size = len(selected)
    best_selected = set(selected)
    
    no_improve_time = time.time()
    max_no_improve = 1.5  # Shorter patience to allow more iterations
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try to remove a redundant block
        if selected:
            items = list(selected)
            rng.shuffle(items)
            for b in items:
                redundant = True
                for tri_idx in block_triples[b]:
                    if coverage_count[tri_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    for tri_idx in block_triples[b]:
                        coverage_count[tri_idx] -= 1
                    selected.remove(b)
                    improved = True
                    if len(selected) < best_size:
                        best_size = len(selected)
                        best_selected = set(selected)
                    no_improve_time = time.time()
                    break
        
        if not improved:
            # 2. Try swaps: relaxed - allow replacement if it covers at least half the unique triples
            swap_done = False
            sel_list = list(selected)
            rng.shuffle(sel_list)
            
            for b in sel_list:
                if time.time() >= deadline:
                    break
                    
                # Find unique triples for b
                unique_tris = []
                for tri_idx in block_triples[b]:
                    if coverage_count[tri_idx] == 1:
                        unique_tris.append(tri_idx)
                
                if not unique_tris:
                    continue
                
                # Try multiple unique triples as starting points
                for tri_idx in rng.sample(unique_tris, min(3, len(unique_tris))):
                    candidates = blocks_containing[tri_idx]
                    
                    if len(candidates) <= 1:
                        continue
                    
                    cand_list = list(candidates)
                    rng.shuffle(cand_list)
                    
                    for b_prime in cand_list[:50]:
                        if b_prime in selected:
                            continue
                        
                        # Check how many unique triples b_prime covers
                        covered_unique = 0
                        for ut in unique_tris:
                            if block_masks[b_prime] & (1 << ut):
                                covered_unique += 1
                        
                        # Require covering at least 50% of unique triples (more relaxed)
                        if covered_unique < len(unique_tris) * 0.5:
                            continue
                        
                        # Check that removing b doesn't leave any triple uncovered
                        safe = True
                        for tri_idx2 in block_triples[b]:
                            if coverage_count[tri_idx2] == 1 and not (block_masks[b_prime] & (1 << tri_idx2)):
                                safe = False
                                break
                        
                        if not safe:
                            continue
                        
                        # Perform swap
                        for tri_idx2 in block_triples[b]:
                            coverage_count[tri_idx2] -= 1
                        selected.remove(b)
                        
                        for tri_idx2 in block_triples[b_prime]:
                            coverage_count[tri_idx2] += 1
                        selected.add(b_prime)
                        
                        swap_done = True
                        improved = True
                        if len(selected) < best_size:
                            best_size = len(selected)
                            best_selected = set(selected)
                        no_improve_time = time.time()
                        break
                    
                    if swap_done:
                        break
                
                if swap_done:
                    break
        
        if not improved:
            # 3. Try 2-opt: remove 2 blocks, add 1 block
            if len(selected) >= 3:
                sel_list2 = list(selected)
                rng.shuffle(sel_list2)
                two_opt_done = False
                
                for i in range(min(20, len(sel_list2))):
                    if time.time() >= deadline:
                        break
                    b1 = sel_list2[i]
                    
                    unique_tris_1 = [tri_idx for tri_idx in block_triples[b1] if coverage_count[tri_idx] == 1]
                    if not unique_tris_1:
                        continue
                    
                    for j in range(i + 1, min(i + 10, len(sel_list2))):
                        if time.time() >= deadline:
                            break
                        b2 = sel_list2[j]
                        
                        unique_tris_2 = [tri_idx for tri_idx in block_triples[b2] if coverage_count[tri_idx] == 1]
                        
                        all_unique = set(unique_tris_1) | set(unique_tris_2)
                        if not all_unique:
                            continue
                        
                        first_tri = next(iter(all_unique))
                        candidates = blocks_containing[first_tri]
                        
                        if len(candidates) <= 1:
                            continue
                        
                        cand_list2 = list(candidates)
                        rng.shuffle(cand_list2)
                        
                        for b_prime in cand_list2[:30]:
                            if b_prime in selected:
                                continue
                            
                            # Check coverage of unique triples (relaxed: at least 40%)
                            covered = sum(1 for ut in all_unique if block_masks[b_prime] & (1 << ut))
                            if covered < len(all_unique) * 0.4:
                                continue
                            
                            # Safety check: no triple becomes uncovered
                            safe = True
                            for tri_idx2 in block_triples[b1]:
                                if coverage_count[tri_idx2] == 1 and not (block_masks[b_prime] & (1 << tri_idx2)):
                                    safe = False
                                    break
                            if safe:
                                for tri_idx2 in block_triples[b2]:
                                    if coverage_count[tri_idx2] == 1 and not (block_masks[b_prime] & (1 << tri_idx2)):
                                        safe = False
                                        break
                            
                            if not safe:
                                continue
                            
                            # Perform 2-opt
                            for tri_idx2 in block_triples[b1]:
                                coverage_count[tri_idx2] -= 1
                            selected.remove(b1)
                            
                            for tri_idx2 in block_triples[b2]:
                                coverage_count[tri_idx2] -= 1
                            selected.remove(b2)
                            
                            for tri_idx2 in block_triples[b_prime]:
                                coverage_count[tri_idx2] += 1
                            selected.add(b_prime)
                            
                            two_opt_done = True
                            improved = True
                            if len(selected) < best_size:
                                best_size = len(selected)
                                best_selected = set(selected)
                            no_improve_time = time.time()
                            break
                        
                        if two_opt_done:
                            break
                    
                    if two_opt_done:
                        break
        
        if not improved:
            if time.time() - no_improve_time > max_no_improve:
                break
    
    # Final fill: add blocks to cover any remaining uncovered triples
    uncovered_mask = 0
    for tri_idx in range(n_triples):
        if coverage_count[tri_idx] == 0:
            uncovered_mask |= (1 << tri_idx)
    
    if uncovered_mask:
        # Greedily add blocks to cover uncovered triples
        active = [idx for idx in range(n_blocks) if idx not in selected]
        while uncovered_mask and time.time() < deadline:
            best_idx = -1
            max_gain = 0
            for idx in active:
                gain = (block_masks[idx] & uncovered_mask).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_idx = idx
                if gain == uncovered_mask.bit_count():
                    break
            if best_idx == -1 or max_gain == 0:
                break
            for tri_idx2 in block_triples[best_idx]:
                coverage_count[tri_idx2] += 1
            selected.add(best_idx)
            uncovered_mask &= ~block_masks[best_idx]
            if len(selected) < best_size:
                best_size = len(selected)
                best_selected = set(selected)
    
    return list(best_selected)

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Precompute
    all_blocks, block_masks, full_mask, blocks_containing, n_blocks, n_triples, block_triples = _precompute(v, k, t)
    
    best_blocks = None
    best_len = float('inf')
    
    seed = 0
    
    # Run multiple restarts with adaptive local search time
    while time.time() < deadline - 3:
        # 1. Greedy (fast)
        initial_blocks = _greedy_construct(v, k, t, all_blocks, block_masks, full_mask, blocks_containing, n_blocks, seed, block_triples)
        
        if not initial_blocks:
            seed += 1
            continue
        
        # 2. Local Search with adaptive time allocation
        remaining = deadline - time.time()
        
        # Allocate more time to early restarts, less to later ones
        # First few restarts get more time, later ones get less
        if seed < 3:
            ls_time = min(15.0, max(4.0, remaining - 2.0))
        elif seed < 8:
            ls_time = min(10.0, max(3.0, remaining - 2.0))
        else:
            ls_time = min(6.0, max(2.0, remaining - 2.0))
        
        ls_deadline = time.time() + ls_time
        
        if ls_deadline > deadline - 2.0:
            ls_deadline = deadline - 2.0
        
        if ls_deadline <= time.time() + 0.5:
            break
            
        blocks = _local_search(v, k, t, initial_blocks, block_masks, full_mask, blocks_containing, n_blocks, n_triples, block_triples, seed, ls_deadline)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            print(f"New best: {best_len} blocks at t={time.time()-start_time:.1f}s", flush=True)
        
        seed += 1
    
    if best_blocks is None:
        blocks = _greedy_construct(v, k, t, all_blocks, block_masks, full_mask, blocks_containing, n_blocks, 999, block_triples)
        best_blocks = blocks
    
    result = [list(all_blocks[idx]) for idx in best_blocks]
    
    return result
