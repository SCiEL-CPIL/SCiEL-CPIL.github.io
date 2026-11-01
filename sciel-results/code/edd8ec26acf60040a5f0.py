import itertools
import time
import random
import sys

_PRECOMPUTED = {}

def _precompute(v, k, t):
    if (v, k, t) in _PRECOMPUTED:
        return _PRECOMPUTED[(v, k, t)]
        
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    block_masks = [0] * num_blocks
    blocks_by_triple = [[] for _ in range(num_triples)]
    block_triples = [[] for _ in range(num_blocks)]
    
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        tris = []
        for sub in itertools.combinations(block, t):
            t_idx = triple_to_idx[sub]
            mask |= (1 << t_idx)
            blocks_by_triple[t_idx].append(b_idx)
            tris.append(t_idx)
        block_masks[b_idx] = mask
        block_triples[b_idx] = tris
        
    _PRECOMPUTED[(v, k, t)] = {
        'all_blocks': all_blocks,
        'block_masks': block_masks,
        'blocks_by_triple': blocks_by_triple,
        'block_triples': block_triples,
        'num_triples': num_triples,
        'triple_to_idx': triple_to_idx
    }
    return _PRECOMPUTED[(v, k, t)]

def _greedy(v, k, t, data, seed):
    rng = random.Random(seed)
    all_blocks = data['all_blocks']
    block_masks = data['block_masks']
    blocks_by_triple = data['blocks_by_triple']
    num_triples = data['num_triples']
    
    uncovered_mask = (1 << num_triples) - 1
    selected_indices = set()
    
    while uncovered_mask:
        # Pick random uncovered triple
        while True:
            t_idx = rng.randint(0, num_triples - 1)
            if (uncovered_mask >> t_idx) & 1:
                break
        
        candidates = blocks_by_triple[t_idx]
        best_gain = -1
        best_block_idx = -1
        
        for b_idx in candidates:
            if b_idx in selected_indices:
                continue
            gain = (block_masks[b_idx] & uncovered_mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block_idx = b_idx
            elif gain == best_gain and gain > 0 and rng.random() < 0.5:
                best_block_idx = b_idx
                
        if best_block_idx == -1:
            break
            
        selected_indices.add(best_block_idx)
        uncovered_mask &= ~block_masks[best_block_idx]
        
    return [all_blocks[i] for i in selected_indices]

def _reduce(blocks, data):
    if not blocks:
        return []
        
    block_triples = data['block_triples']
    all_blocks = data['all_blocks']
    num_triples = data['num_triples']
    
    # Map block tuple to index in all_blocks
    block_idx_map = {b: i for i, b in enumerate(all_blocks)}
    
    # Get indices of selected blocks
    selected_indices = [block_idx_map[b] for b in blocks]
    
    cov_count = [0] * num_triples
    for idx in selected_indices:
        for ti in block_triples[idx]:
            cov_count[ti] += 1
            
    active = [True] * len(selected_indices)
    changed = True
    while changed:
        changed = False
        for i, idx in enumerate(selected_indices):
            if not active[i]:
                continue
            redundant = True
            for ti in block_triples[idx]:
                if cov_count[ti] == 1:
                    redundant = False
                    break
            if redundant:
                active[i] = False
                changed = True
                for ti in block_triples[idx]:
                    cov_count[ti] -= 1
                    
    return [all_blocks[selected_indices[i]] for i in range(len(selected_indices)) if active[i]]

def _local_search_swap_remove(blocks, data, deadline):
    all_blocks = data['all_blocks']
    block_triples = data['block_triples']
    num_triples = data['num_triples']
    block_idx_map = {b: i for i, b in enumerate(all_blocks)}
    
    blocks = [list(b) for b in blocks]
    selected_indices = [block_idx_map[b] for b in blocks]
    
    cov_count = [0] * num_triples
    for idx in selected_indices:
        for ti in block_triples[idx]:
            cov_count[ti] += 1
            
    rng = random.Random(42)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        # Shuffle to avoid bias
        order = list(range(len(blocks)))
        rng.shuffle(order)
        
        for i in order:
            if time.time() >= deadline:
                break
            
            idx = selected_indices[i]
            # Check if removable
            critical_ti = -1
            for ti in block_triples[idx]:
                if cov_count[ti] == 1:
                    critical_ti = ti
                    break
            
            if critical_ti == -1:
                # Removable
                for ti in block_triples[idx]:
                    cov_count[ti] -= 1
                blocks.pop(i)
                selected_indices.pop(i)
                improved = True
                break # Restart loop as indices changed
                
            # Try to enable removal by swapping in another block
            # Get the triple
            # triple_to_idx is not directly indexed by ti in data? 
            # We need the triple itself to find points.
            # Let's store triples in data too.
            triple = data['all_triples'][critical_ti]
            
            # Find a block j != i that contains 2 points of triple
            # And try to swap
            found_swap = False
            for j in range(len(blocks)):
                if j == i:
                    continue
                idx_j = selected_indices[j]
                block_j = blocks[j]
                
                # Check if block_j contains exactly 2 points of triple
                # triple is (a, b, c)
                count_in_block = sum(1 for p in triple if p in block_j)
                if count_in_block != 2:
                    continue
                
                # Identify the 2 points in block_j and the 1 missing
                points_in = [p for p in triple if p in block_j]
                missing_point = [p for p in triple if p not in block_j][0]
                
                # Choose a point y in block_j to swap out
                # y must not be in points_in
                candidates_y = [p for p in block_j if p not in points_in]
                
                for y in candidates_y:
                    # Swap y with missing_point
                    new_block_j = [p if p != y else missing_point for p in block_j]
                    
                    # Check validity
                    # Lost triples: contain y, not missing_point
                    # Gained triples: contain missing_point, not y
                    # We only care about lost triples becoming uncovered
                    valid = True
                    for ti_lost in block_triples[idx_j]:
                        # Check if this triple contains y
                        # We need to check if y is in the triple
                        # triple = data['all_triples'][ti_lost]
                        # if y in triple and missing_point not in triple:
                        #    if cov_count[ti_lost] == 1: valid = False
                        pass # This is slow. 
                    
                    # Faster check:
                    # The lost triples are those in block_triples[idx_j] that contain y.
                    # We can precompute for each block index and each point, the list of triple indices containing that point?
                    # Too much memory.
                    
                    # Let's just check the 15 triples formed by y and 2 other points in new_block_j
                    # Wait, the lost triples are those that had y and did NOT have missing_point.
                    # Since missing_point was not in block_j, any triple with y was not having missing_point.
                    # So we need to check all triples in block_j that contain y.
                    # There are C(6, 2) = 15 such triples.
                    
                    # Generate them
                    other_points = [p for p in block_j if p != y]
                    for u_idx in range(len(other_points)):
                        for v_idx in range(u_idx + 1, len(other_points)):
                            u = other_points[u_idx]
                            v = other_points[v_idx]
                            tri = tuple(sorted((y, u, v)))
                            ti = data['triple_to_idx'].get(tri)
                            if ti is not None and cov_count[ti] == 1:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                    
                    # Swap is valid
                    # Update cov_count
                    # Decrement for lost (triples with y, not missing_point)
                    # Increment for gained (triples with missing_point, not y)
                    
                    # Lost: y in tri, missing_point not in tri
                    for u_idx in range(len(other_points)):
                        for v_idx in range(u_idx + 1, len(other_points)):
                            u = other_points[u_idx]
                            v = other_points[v_idx]
                            tri = tuple(sorted((y, u, v)))
                            ti = data['triple_to_idx'].get(tri)
                            if ti is not None:
                                cov_count[ti] -= 1
                                
                    # Gained: missing_point in tri, y not in tri
                    # The triples in new_block_j containing missing_point
                    # missing_point, u, v where u,v in other_points
                    for u_idx in range(len(other_points)):
                        for v_idx in range(u_idx + 1, len(other_points)):
                            u = other_points[u_idx]
                            v = other_points[v_idx]
                            tri = tuple(sorted((missing_point, u, v)))
                            ti = data['triple_to_idx'].get(tri)
                            if ti is not None:
                                cov_count[ti] += 1
                                
                    # Apply swap
                    blocks[j] = new_block_j
                    # Update selected_indices? No, index j is same, but block content changed.
                    # We need to update the index in all_blocks?
                    # The block might not be in all_blocks if it's a new combination?
                    # No, all_blocks contains ALL combinations. So new_block_j is in all_blocks.
                    new_idx_j = block_idx_map[tuple(sorted(new_block_j))]
                    selected_indices[j] = new_idx_j
                    # Note: block_triples[new_idx_j] is different from block_triples[idx_j]
                    
                    found_swap = True
                    break
                if found_swap:
                    break
                    
            if found_swap:
                improved = True
                break # Restart loop
                
    return blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    data = _precompute(v, k, t)
    data['all_triples'] = list(itertools.combinations(range(v), t))
    
    best_blocks = None
    best_len = float('inf')
    
    seed = 0
    while time.time() < deadline - 10.0:
        blocks = _greedy(v, k, t, data, seed)
        blocks = _reduce(blocks, data)
        
        if time.time() < deadline - 5.0:
            blocks = _local_search_swap_remove(blocks, data, deadline - 2.0)
            blocks = _reduce(blocks, data)
            
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
        if best_len == 69:
            break
            
        seed += 1
        
    if best_blocks is None:
        blocks = _greedy(v, k, t, data, 0)
        best_blocks = _reduce(blocks, data)
        
    return best_blocks
