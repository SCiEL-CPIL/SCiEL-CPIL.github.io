import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    # Precompute all t-subsets and map them to IDs
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_id = {sub: i for i, sub in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)
    
    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute for each block, the list of t-subset IDs it covers
    block_ids = []
    for b in all_blocks:
        ids = []
        for sub in itertools.combinations(b, t):
            ids.append(tsub_to_id[sub])
        block_ids.append(ids)
        
    # Precompute for each t-subset ID, the list of block indices that contain it
    sub_blocks = [[] for _ in range(n_tsubs)]
    for b_idx, ids in enumerate(block_ids):
        for id in ids:
            sub_blocks[id].append(b_idx)
            
    best_blocks = None
    best_len = float('inf')
    
    # Multiple restarts
    num_restarts = 30
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        rng = random.Random(r * 7919 + 13)
        
        # 1. Greedy Construction with ratio-based scoring
        blocks = _greedy_cover_ratio(v, k, t, n_tsubs, block_ids, sub_blocks, all_blocks, rng, deadline)
        
        if blocks is None:
            continue
            
        # 2. Local Search / Refinement
        blocks = _local_search_v3(blocks, v, k, t, n_tsubs, block_ids, sub_blocks, all_blocks, rng, deadline)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        # Early exit if we hit the known best or better
        if best_len <= 76:
            break

    if best_blocks is None:
        return []
        
    return [list(b) for b in best_blocks]

def _greedy_cover_ratio(v, k, t, n_tsubs, block_ids, sub_blocks, all_blocks, rng, deadline):
    """
    Greedy covering using ratio-based scoring:
    score = newly_covered / (1 + already_covered_overlap)
    This penalizes blocks that overlap heavily with existing blocks.
    """
    covered = [False] * n_tsubs
    uncovered_count = n_tsubs
    blocks = []
    
    while uncovered_count > 0:
        if time.time() > deadline:
            return None
            
        # Pick a random uncovered t-subset as seed
        seed_id = rng.randint(0, n_tsubs - 1)
        if covered[seed_id]:
            found = False
            for i in range(n_tsubs):
                if not covered[i]:
                    seed_id = i
                    found = True
                    break
            if not found:
                break
                
        candidates = sub_blocks[seed_id]
        if not candidates:
            return None
            
        best_block = -1
        best_score = -1.0
        best_tiebreak = []
        
        # Calculate ratio-based score for each candidate
        for b_idx in candidates:
            newly_covered = 0
            already_covered = 0
            for id in block_ids[b_idx]:
                if covered[id]:
                    already_covered += 1
                else:
                    newly_covered += 1
                    
            if newly_covered == 0:
                continue
                
            # Ratio: new coverage penalized by overlap
            # Use a smooth penalty: score = newly / (1 + alpha * already)
            alpha = 0.5
            score = newly_covered / (1.0 + alpha * already_covered)
            
            if score > best_score:
                best_score = score
                best_block = b_idx
                best_tiebreak = [b_idx]
            elif abs(score - best_score) < 1e-9:
                best_tiebreak.append(b_idx)
                
        if best_block == -1:
            return None
            
        # Pick randomly from tiebreaks to add diversity
        chosen_block = rng.choice(best_tiebreak)
        blocks.append(all_blocks[chosen_block])
        
        # Update covered
        for id in block_ids[chosen_block]:
            if not covered[id]:
                covered[id] = True
                uncovered_count -= 1
                
    return blocks

def _local_search_v3(blocks, v, k, t, n_tsubs, block_ids, sub_blocks, all_blocks, rng, deadline):
    """
    Enhanced local search with:
    1. Redundant block removal
    2. Single-block replacement to reduce unique coverage
    3. Two-block swap to enable removal of a third block
    """
    block_val_to_idx = {b: i for i, b in enumerate(all_blocks)}
    current_block_indices = []
    for b in blocks:
        current_block_indices.append(block_val_to_idx[b])
        
    cover_count = [0] * n_tsubs
    for b_idx in current_block_indices:
        for id in block_ids[b_idx]:
            cover_count[id] += 1
            
    improved_indices = list(current_block_indices)
    
    # Track how many times we've tried each improvement type
    max_iterations = 500
    iteration = 0
    
    while iteration < max_iterations:
        if time.time() > deadline:
            break
        iteration += 1
        
        # 1. Remove redundant blocks
        removed = False
        for i in range(len(improved_indices) - 1, -1, -1):
            b_idx = improved_indices[i]
            is_redundant = True
            for id in block_ids[b_idx]:
                if cover_count[id] == 1:
                    is_redundant = False
                    break
                    
            if is_redundant:
                for id in block_ids[b_idx]:
                    cover_count[id] -= 1
                improved_indices.pop(i)
                removed = True
                break
                
        if removed:
            continue
            
        n = len(improved_indices)
        if n == 0:
            break
            
        # 2. Try single-block replacement
        b_idx_pos = rng.randint(0, n - 1)
        b_idx = improved_indices[b_idx_pos]
        
        unique_subs = [id for id in block_ids[b_idx] if cover_count[id] == 1]
        
        if not unique_subs:
            # Should have been caught by redundant removal
            for id in block_ids[b_idx]:
                cover_count[id] -= 1
            improved_indices.pop(b_idx_pos)
            continue
            
        # Find blocks that cover all unique subsets
        candidate_indices = set(sub_blocks[unique_subs[0]])
        for id in unique_subs[1:]:
            candidate_indices.intersection_update(sub_blocks[id])
            if not candidate_indices:
                break
                
        if candidate_indices:
            current_set = set(improved_indices)
            valid_candidates = [c for c in candidate_indices if c not in current_set]
            
            if valid_candidates:
                # Score candidates by how many additional unique subsets they cover
                # (i.e., subsets currently covered only by b_idx that the new block also covers)
                best_new = None
                best_score = -1
                for c in valid_candidates:
                    # Count how many of b_idx's unique subs are also in c
                    overlap = 0
                    for id in unique_subs:
                        if id in block_ids[c]:
                            overlap += 1
                    # Prefer blocks that cover more unique subsets
                    if overlap > best_score:
                        best_score = overlap
                        best_new = c
                        
                if best_new is not None:
                    for id in block_ids[b_idx]:
                        cover_count[id] -= 1
                    for id in block_ids[best_new]:
                        cover_count[id] += 1
                    improved_indices[b_idx_pos] = best_new
                    continue
        
        # 3. Try two-block swap: replace block i with block j, then see if we can remove block k
        if n >= 3:
            i = rng.randint(0, n - 1)
            b_i = improved_indices[i]
            
            # Find unique subsets of b_i
            unique_i = [id for id in block_ids[b_i] if cover_count[id] == 1]
            
            if unique_i:
                # Look for a block that covers some (but not all) of unique_i
                # and whose addition allows removal of another block
                candidates_for_i = set(sub_blocks[unique_i[0]])
                for uid in unique_i[1:]:
                    candidates_for_i.intersection_update(sub_blocks[uid])
                    if not candidates_for_i:
                        break
                
                if candidates_for_i:
                    current_set = set(improved_indices)
                    # Try a few random candidates
                    valid_cands = [c for c in candidates_for_i if c not in current_set]
                    if valid_cands:
                        # Shuffle and try up to 3
                        rng.shuffle(valid_cands)
                        for new_b in valid_cands[:3]:
                            # Simulate: remove b_i, add new_b
                            temp_cover = list(cover_count)
                            for id in block_ids[b_i]:
                                temp_cover[id] -= 1
                            for id in block_ids[new_b]:
                                temp_cover[id] += 1
                                
                            # Check if any block becomes removable
                            for j in range(n):
                                if j == i:
                                    continue
                                b_j = improved_indices[j]
                                can_remove = True
                                for id in block_ids[b_j]:
                                    if temp_cover[id] == 1:
                                        can_remove = False
                                        break
                                if can_remove:
                                    # Apply the swap and removal
                                    for id in block_ids[b_i]:
                                        cover_count[id] -= 1
                                    for id in block_ids[new_b]:
                                        cover_count[id] += 1
                                    for id in block_ids[b_j]:
                                        cover_count[id] -= 1
                                    
                                    improved_indices[i] = new_b
                                    improved_indices.pop(j)
                                    break
                            else:
                                continue
                            break
        
    return [all_blocks[i] for i in improved_indices]
