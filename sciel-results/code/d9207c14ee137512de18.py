import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 150s limit

    points = list(range(v))
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_index = {sub: i for i, sub in enumerate(all_t_subsets)}

    # Precompute t-subsets for each k-block efficiently
    # For each k-subset, precompute its t-subset indices
    # C(19,8) = 75582 blocks, each with C(8,5)=56 t-subsets
    # Total: 4.2M entries - manageable

    # Build a mapping from k-block (as frozenset) to list of t-subset indices
    # But we can't precompute all C(19,8) blocks upfront due to memory
    # Instead, compute on the fly with caching

    # Precompute for each point, which t-subsets contain it
    point_to_tsubs = [set() for _ in range(v)]
    for sub in all_t_subsets:
        for p in sub:
            point_to_tsubs[p].add(t_index[sub])

    # Precompute for each pair of points, which t-subsets contain both
    # This helps in incremental gain calculation
    # C(19,2) = 170 pairs, each with C(19-2, t-2) = C(17,3) = 680 t-subsets
    pair_to_tsubs = {}
    for i in range(v):
        for j in range(i+1, v):
            subs = []
            for sub in all_t_subsets:
                if i in sub and j in sub:
                    subs.append(t_index[sub])
            pair_to_tsubs[(i,j)] = subs

    def get_block_tsubs(block):
        """Get all t-subset indices for a given block"""
        return [t_index[sub] for sub in itertools.combinations(block, t)]

    def greedy_once(random_seed=None):
        rng = random.Random(random_seed)
        
        # Coverage count for each t-subset
        cover_count = np.zeros(n_t, dtype=np.int32)
        blocks = []
        
        while True:
            if time.time() - start_time > time_limit - 5.0:
                return None
            
            # Find uncovered t-subsets
            uncovered = np.where(cover_count == 0)[0]
            if len(uncovered) == 0:
                break
            
            # Bottleneck heuristic: pick the t-subset with minimum "degree"
            # i.e., the one that is hardest to cover
            # Use a smooth surrogate: weight by how many blocks could cover it
            # For simplicity, pick randomly among uncovered, but bias toward
            # t-subsets that appear in fewer potential blocks
            
            # Pick a random uncovered t-subset
            idx = rng.choice(uncovered)
            s = all_t_subsets[idx]
            s_set = set(s)
            remaining_points = [p for p in points if p not in s_set]
            
            best_block = None
            best_gain = -1
            best_min_coverage = -1  # For bottleneck tie-breaking
            
            # Generate candidate blocks and evaluate
            # To speed up, sample a subset of candidates if too many
            all_candidates = list(itertools.combinations(remaining_points, k - t))
            
            # If too many candidates, sample
            max_candidates = 2000
            if len(all_candidates) > max_candidates:
                candidates = rng.sample(all_candidates, max_candidates)
            else:
                candidates = all_candidates
            
            for combo in candidates:
                block = tuple(sorted(s + combo))
                block_tsubs = get_block_tsubs(block)
                
                # Calculate gain: how many uncovered t-subsets does this block cover?
                gain = 0
                min_cov = float('inf')
                for sub_idx in block_tsubs:
                    if cover_count[sub_idx] == 0:
                        gain += 1
                    min_cov = min(min_cov, cover_count[sub_idx])
                
                # Bottleneck heuristic: prefer blocks that cover t-subsets with low coverage
                # Use a smooth combination: gain * (1 + 0.1 * min_cov_penalty)
                # where min_cov_penalty rewards covering hard-to-cover t-subsets
                score = gain
                if gain > 0 and min_cov == 0:
                    score += 0.5  # Bonus for covering uncovered t-subsets
                
                if score > best_gain:
                    best_gain = score
                    best_block = block
                    best_min_coverage = min_cov
                elif score == best_gain and gain > 0:
                    # Random tie break to diversify
                    if rng.random() < 0.3:
                        best_block = block
                        best_min_coverage = min_cov
            
            if best_block is None:
                break
                
            blocks.append(best_block)
            
            # Update coverage
            for sub_idx in get_block_tsubs(best_block):
                cover_count[sub_idx] += 1
        
        return blocks

    def local_search_remove_replace(blocks, random_seed=None):
        """Try to remove redundant blocks or replace them with better ones."""
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
        
        n_blocks = len(blocks)
        if n_blocks == 0:
            return blocks
        
        # Build coverage count
        cover_count = np.zeros(n_t, dtype=np.int32)
        block_tsubs = []
        for block in blocks:
            tsubs = get_block_tsubs(block)
            block_tsubs.append(tsubs)
            for idx in tsubs:
                cover_count[idx] += 1
        
        kept = list(range(n_blocks))
        
        # Pass 1: Remove redundant blocks
        changed = True
        max_passes = 3
        for _pass in range(max_passes):
            if time.time() - start_time > time_limit - 3.0:
                break
            changed = False
            order = list(kept)
            rng.shuffle(order)
            for bi in order:
                if bi not in kept:
                    continue
                # Check if all t-subsets of this block are covered by others
                removable = True
                for idx in block_tsubs[bi]:
                    if cover_count[idx] <= 1:
                        removable = False
                        break
                if removable:
                    kept.remove(bi)
                    for idx in block_tsubs[bi]:
                        cover_count[idx] -= 1
                    changed = True
            
            if not changed:
                break
        
        # Pass 2: Try to replace blocks with better ones
        # For each block, try to find a replacement that covers the same
        # uniquely-covered t-subsets with fewer total t-subsets
        for _replace_pass in range(2):
            if time.time() - start_time > time_limit - 3.0:
                break
            
            for bi in list(kept):
                if time.time() - start_time > time_limit - 3.0:
                    break
                
                # Find t-subsets uniquely covered by this block
                unique_tsubs = [idx for idx in block_tsubs[bi] if cover_count[idx] == 1]
                if len(unique_tsubs) == 0:
                    continue
                
                # Try to find a replacement block that covers all unique_tsubs
                # and has fewer total t-subsets or same
                old_block = blocks[bi]
                old_size = len(block_tsubs[bi])
                
                # Generate candidate replacements
                # Must contain all points that are in unique t-subsets
                must_include = set()
                for idx in unique_tsubs:
                    must_include.update(all_t_subsets[idx])
                
                if len(must_include) > k:
                    continue
                
                remaining = [p for p in points if p not in must_include]
                need = k - len(must_include)
                
                if need < 0:
                    continue
                
                # Sample some candidates
                if need == 0:
                    candidates = [tuple(sorted(must_include))]
                else:
                    all_combos = list(itertools.combinations(remaining, need))
                    if len(all_combos) > 500:
                        candidates = [tuple(sorted(must_include | set(c))) for c in rng.sample(all_combos, 500)]
                    else:
                        candidates = [tuple(sorted(must_include | set(c))) for c in all_combos]
                
                best_candidate = None
                best_candidate_size = old_size
                
                for cand in candidates:
                    cand_tsubs = get_block_tsubs(cand)
                    cand_size = len(cand_tsubs)
                    
                    # Check if it covers all unique_tsubs
                    cand_set = set(cand_tsubs)
                    if all(idx in cand_set for idx in unique_tsubs):
                        if cand_size < best_candidate_size:
                            best_candidate = cand
                            best_candidate_size = cand_size
                
                if best_candidate is not None and best_candidate != old_block:
                    # Replace
                    # Remove old block's coverage
                    for idx in block_tsubs[bi]:
                        cover_count[idx] -= 1
                    
                    # Add new block's coverage
                    new_tsubs = get_block_tsubs(best_candidate)
                    for idx in new_tsubs:
                        cover_count[idx] += 1
                    
                    blocks[bi] = best_candidate
                    block_tsubs[bi] = new_tubs if False else new_tsubs
        
        return [blocks[i] for i in kept]

    best_blocks = None
    best_len = float('inf')
    
    num_restarts = 0
    while time.time() - start_time < time_limit - 5.0:
        seed = num_restarts
        res = greedy_once(seed)
        num_restarts += 1
        
        if res is not None:
            # Apply local search
            if time.time() - start_time < time_limit - 3.0:
                res = local_search_remove_replace(res, seed)
            
            if len(res) < best_len:
                best_len = len(res)
                best_blocks = res
        
        if best_len <= 264:
            break

    if best_blocks is None:
        if time.time() - start_time < time_limit - 5.0:
            best_blocks = greedy_once(0)
            if best_blocks is not None:
                best_blocks = local_search_remove_replace(best_blocks, 0)
    
    return best_blocks
