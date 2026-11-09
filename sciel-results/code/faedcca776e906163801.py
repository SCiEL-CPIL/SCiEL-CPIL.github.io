import itertools
import random
import time
import sys
import numpy as np

# Global cache
_CACHE = {}

def _precompute(v, k, t):
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]
    
    points = range(v)
    # Generate all blocks
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    # Generate all triples
    triples = list(itertools.combinations(points, t))
    triple_to_idx = {tri: i for i, tri in enumerate(triples)}
    n_triples = len(triples)
    
    # Precompute block_triple_indices: list of triple indices for each block
    block_triple_indices = [None] * n_blocks
    
    # Precompute triple_to_blocks: list of block indices for each triple
    triple_to_blocks = [[] for _ in range(n_triples)]
    
    for i, block in enumerate(all_blocks):
        indices = []
        for tri in itertools.combinations(block, t):
            idx = triple_to_idx[tri]
            indices.append(idx)
            triple_to_blocks[idx].append(i)
        block_triple_indices[i] = indices
        
    full_mask = (1 << n_triples) - 1
    
    _CACHE[key] = (all_blocks, block_triple_indices, triple_to_blocks, full_mask, n_triples)
    return _CACHE[key]

def _greedy_construct(v, k, t, all_blocks, block_triple_indices, triple_to_blocks, full_mask, n_triples, seed):
    rng = random.Random(seed)
    
    # Use a set for uncovered triples for O(1) membership checks
    uncovered_set = set(range(n_triples))
    
    blocks_used = []
    used_blocks_set = set()
    
    # Precompute constraint (number of blocks covering each triple) for "Most Constrained" heuristic
    triple_constraints = [len(triple_to_blocks[i]) for i in range(n_triples)]
    
    while uncovered_set:
        # Pick the uncovered triple with the minimum number of covering blocks
        min_count = float('inf')
        candidates_for_tri = []
        
        for tri in uncovered_set:
            count = triple_constraints[tri]
            if count < min_count:
                min_count = count
                candidates_for_tri = [tri]
            elif count == min_count:
                candidates_for_tri.append(tri)
                
        if not candidates_for_tri:
            break
            
        target_tri = candidates_for_tri[rng.randint(0, len(candidates_for_tri) - 1)]
        
        # Get all blocks covering this triple
        candidate_blocks = triple_to_blocks[target_tri]
        
        if not candidate_blocks:
            break
            
        # Evaluate gain for each candidate block
        best_block = None
        max_gain = -1
        
        for b_idx in candidate_blocks:
            if b_idx in used_blocks_set:
                continue
                
            gain = 0
            for t_idx in block_triple_indices[b_idx]:
                if t_idx in uncovered_set:
                    gain += 1
                    
            if gain > max_gain:
                max_gain = gain
                best_block = b_idx
            elif gain == max_gain and best_block is not None:
                if rng.random() < 0.5:
                    best_block = b_idx
                    
        if best_block is None:
            break
            
        blocks_used.append(best_block)
        used_blocks_set.add(best_block)
        
        # Update uncovered_set
        for t_idx in block_triple_indices[best_block]:
            if t_idx in uncovered_set:
                uncovered_set.remove(t_idx)
                
    return blocks_used

def _local_search(v, k, t, initial_blocks, block_triple_indices, triple_to_blocks, full_mask, n_triples, deadline):
    current_blocks = list(initial_blocks)
    
    # Precompute coverage counts
    triple_counts = [0] * n_triples
    for b in current_blocks:
        for idx in block_triple_indices[b]:
            triple_counts[idx] += 1
            
    num_uncovered = sum(1 for c in triple_counts if c == 0)
    if num_uncovered != 0:
        return current_blocks
        
    best_blocks = list(current_blocks)
    best_len = len(current_blocks)
    
    rng = random.Random(42)
    
    # Simulated annealing parameters
    temperature = 3.0
    cooling_rate = 0.995
    min_temperature = 0.05
    
    n_blocks_total = len(triple_to_blocks[0]) if triple_to_blocks else 0
    
    while time.time() < deadline:
        if len(current_blocks) < best_len:
            best_len = len(current_blocks)
            best_blocks = list(current_blocks)
            
        # Phase 1: Delete all redundant blocks
        changed = True
        while changed:
            changed = False
            for i in range(len(current_blocks) - 1, -1, -1):
                b_out = current_blocks[i]
                redundant = True
                for idx in block_triple_indices[b_out]:
                    if triple_counts[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    del current_blocks[i]
                    for idx in block_triple_indices[b_out]:
                        triple_counts[idx] -= 1
                    changed = True
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = list(current_blocks)
                    break
        
        if not current_blocks:
            break
            
        # Phase 2: Try moves with simulated annealing
        move_type = rng.random()
        
        if move_type < 0.7:
            # 1-exchange: remove one block, add one block
            pos = rng.randint(0, len(current_blocks) - 1)
            b_out = current_blocks[pos]
            
            uncovered_triples = []
            for idx in block_triple_indices[b_out]:
                if triple_counts[idx] == 1:
                    uncovered_triples.append(idx)
                    
            if not uncovered_triples:
                del current_blocks[pos]
                for idx in block_triple_indices[b_out]:
                    triple_counts[idx] -= 1
                continue

            min_triple = min(uncovered_triples, key=lambda x: len(triple_to_blocks[x]))
            candidates = set(triple_to_blocks[min_triple])
            
            for tri in uncovered_triples:
                if tri == min_triple:
                    continue
                candidates.intersection_update(triple_to_blocks[tri])
                if not candidates:
                    break
            
            current_set = set(current_blocks)
            candidates.difference_update(current_set)
            
            if candidates:
                candidate_list = list(candidates)
                rng.shuffle(candidate_list)
                
                # Try up to 3 candidates
                max_tries = min(3, len(candidate_list))
                best_candidate = None
                best_gain = -1
                
                for b_in in candidate_list[:max_tries]:
                    gain = 0
                    for idx in block_triple_indices[b_in]:
                        if triple_counts[idx] == 0:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_candidate = b_in
                
                if best_candidate is not None:
                    # Compute quality: penalize unique coverage, reward redundancy
                    # Quality = sum over triples of (1 if count >= 2 else 0) - lambda * (count == 1)
                    # Simpler: quality = sum of min(count, 2) for all triples
                    # But we need a faster metric. Use: number of triples with count == 1 (bad) 
                    # and number with count >= 2 (good).
                    
                    # Old quality: count of triples with count >= 2
                    old_redundant = 0
                    for idx in block_triple_indices[b_out]:
                        if triple_counts[idx] >= 2:
                            old_redundant += 1
                    # Also need to account for triples NOT in b_out that might be affected? No, b_out only affects its own triples.
                    
                    # New quality: simulate
                    new_redundant = 0
                    for idx in block_triple_indices[b_out]:
                        new_c = triple_counts[idx] - 1
                        if new_c >= 2:
                            new_redundant += 1
                    for idx in block_triple_indices[best_candidate]:
                        if idx in block_triple_indices[b_out]:
                            # Already counted above
                            pass
                        else:
                            new_c = triple_counts[idx] + 1
                            if new_c >= 2:
                                new_redundant += 1
                    
                    # Also count triples in b_out that are NOT in best_candidate
                    # And triples in best_candidate that are NOT in b_out
                    # The above logic is flawed. Let's just compute the delta properly.
                    
                    # delta_quality = (new_redundant - old_redundant)
                    # But we also care about not creating new unique coverages.
                    # Let's use a penalty for triples that become unique (count 1) from being redundant (count >= 2)
                    # or from being unique to 0 (which is invalid, so we avoid those).
                    
                    # Better metric: sum of (1 if count >= 2 else 0) - 0.5 * (1 if count == 1 else 0)
                    # This rewards redundancy and penalizes unique coverage.
                    
                    old_metric = 0
                    for idx in block_triple_indices[b_out]:
                        c = triple_counts[idx]
                        if c >= 2:
                            old_metric += 1
                        elif c == 1:
                            old_metric -= 0.5
                    for idx in block_triple_indices[best_candidate]:
                        if idx not in block_triple_indices[b_out]:
                            c = triple_counts[idx]
                            if c >= 2:
                                old_metric += 1
                            elif c == 1:
                                old_metric -= 0.5
                            
                    new_metric = 0
                    # We need to compute the new counts for affected triples
                    # Affected triples are those in b_out or best_candidate
                    affected = set(block_triple_indices[b_out]) | set(block_triple_indices[best_candidate])
                    for idx in affected:
                        c = triple_counts[idx]
                        if idx in block_triple_indices[b_out]:
                            c -= 1
                        if idx in block_triple_indices[best_candidate]:
                            c += 1
                        if c >= 2:
                            new_metric += 1
                        elif c == 1:
                            new_metric -= 0.8 # Heavier penalty for unique coverage
                    
                    delta_quality = new_metric - old_metric
                    
                    if delta_quality >= 0 or rng.random() < np.exp(delta_quality / temperature):
                        del current_blocks[pos]
                        for idx in block_triple_indices[b_out]:
                            triple_counts[idx] -= 1
                        
                        current_blocks.append(best_candidate)
                        for idx in block_triple_indices[best_candidate]:
                            triple_counts[idx] += 1
            
            temperature *= cooling_rate
            if temperature < min_temperature:
                temperature = min_temperature
                
        elif move_type < 0.9:
            # 2-exchange: remove two blocks, add one block
            if len(current_blocks) < 2:
                continue
                
            pos1 = rng.randint(0, len(current_blocks) - 1)
            pos2 = rng.randint(0, len(current_blocks) - 1)
            if pos1 == pos2:
                continue
                
            b_out1 = current_blocks[pos1]
            b_out2 = current_blocks[pos2]
            
            # Temporarily remove both
            temp_counts = list(triple_counts)
            for idx in block_triple_indices[b_out1]:
                temp_counts[idx] -= 1
            for idx in block_triple_indices[b_out2]:
                temp_counts[idx] -= 1
            
            # Find uncovered triples
            uncovered = [i for i in range(n_triples) if temp_counts[i] == 0]
            if not uncovered:
                # Both are redundant
                # Remove the one with lower index first to avoid issues
                if pos1 > pos2:
                    pos1, pos2 = pos2, pos1
                    b_out1, b_out2 = b_out2, b_out1
                del current_blocks[pos1]
                for idx in block_triple_indices[b_out1]:
                    triple_counts[idx] -= 1
                del current_blocks[pos2 - 1]
                for idx in block_triple_indices[b_out2]:
                    triple_counts[idx] -= 1
                continue
            
            # Find a block that covers all uncovered triples
            min_triple = min(uncovered, key=lambda x: len(triple_to_blocks[x]))
            candidates = set(triple_to_blocks[min_triple])
            
            for tri in uncovered:
                if tri == min_triple:
                    continue
                candidates.intersection_update(triple_to_blocks[tri])
                if not candidates:
                    break
            
            current_set = set(current_blocks)
            candidates.difference_update(current_set)
            
            if candidates:
                candidate_list = list(candidates)
                rng.shuffle(candidate_list)
                
                max_tries = min(3, len(candidate_list))
                best_candidate = None
                best_gain = -1
                
                for b_in in candidate_list[:max_tries]:
                    gain = 0
                    for idx in block_triple_indices[b_in]:
                        if temp_counts[idx] == 0:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_candidate = b_in
                
                if best_candidate is not None:
                    # This reduces block count by 1, so always accept
                    # Remove both blocks
                    if pos1 > pos2:
                        pos1, pos2 = pos2, pos1
                        b_out1, b_out2 = b_out2, b_out1
                    del current_blocks[pos1]
                    for idx in block_triple_indices[b_out1]:
                        triple_counts[idx] -= 1
                    del current_blocks[pos2 - 1]
                    for idx in block_triple_indices[b_out2]:
                        triple_counts[idx] -= 1
                    
                    current_blocks.append(best_candidate)
                    for idx in block_triple_indices[best_candidate]:
                        triple_counts[idx] += 1
                    
                    temperature *= cooling_rate
                    if temperature < min_temperature:
                        temperature = min_temperature
    
    return best_blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110
    
    all_blocks, block_triple_indices, triple_to_blocks, full_mask, n_triples = _precompute(v, k, t)
    
    best_blocks_global = None
    best_len_global = float('inf')
    
    seed = 0
    while time.time() < deadline - 5:
        remaining = deadline - time.time()
        if remaining < 2:
            break
            
        ls_deadline = time.time() + min(12, remaining - 2)
        
        greedy_blocks = _greedy_construct(v, k, t, all_blocks, block_triple_indices, triple_to_blocks, full_mask, n_triples, seed)
        seed += 1
        
        if not greedy_blocks:
            continue
            
        ls_blocks = _local_search(
            v, k, t, 
            greedy_blocks, 
            block_triple_indices, 
            triple_to_blocks, 
            full_mask, 
            n_triples, 
            ls_deadline
        )
        
        if len(ls_blocks) < best_len_global:
            best_len_global = len(ls_blocks)
            best_blocks_global = ls_blocks

    if best_blocks_global is None:
        if time.time() < deadline - 2:
            greedy_blocks = _greedy_construct(v, k, t, all_blocks, block_triple_indices, triple_to_blocks, full_mask, n_triples, 999)
            best_blocks_global = greedy_blocks
        else:
            best_blocks_global = _greedy_construct(v, k, t, all_blocks, block_triple_indices, triple_to_blocks, full_mask, n_triples, 42)

    result = [list(all_blocks[idx]) for idx in best_blocks_global]
    return result
