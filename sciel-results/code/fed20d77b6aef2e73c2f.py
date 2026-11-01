import numpy as np
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    time_limit = 118.0  # Safety margin for 120s limit
    
    # Precomputation
    points = range(v)
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}
    
    # Generate all k-subsets
    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)
    
    # Precompute the t-subsets contained in each k-subset
    def nCr(n, r):
        if r > n: return 0
        if r == 0: return 1
        return int(np.prod([n-i for i in range(r)]) / np.prod([i+1 for i in range(r)]))
        
    t_count = nCr(k, t)
    
    # block_t_indices[i] is an array of indices into all_ts for the i-th block
    block_t_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    for i, block in enumerate(all_ks):
        subs = list(itertools.combinations(block, t))
        for j, sub in enumerate(subs):
            block_t_indices[i, j] = idx_map[sub]
    
    # Build reverse index: for each t-subset, which blocks contain it
    triple_to_blocks = [[] for _ in range(n_ts)]
    for b_idx in range(n_ks):
        for sub_idx in block_t_indices[b_idx]:
            triple_to_blocks[sub_idx].append(b_idx)
    triple_to_blocks = [np.array(lst, dtype=np.int32) for lst in triple_to_blocks]
    
    # Greedy Construction with Randomization
    covered = np.zeros(n_ts, dtype=bool)
    blocks_indices = []
    point_load = np.zeros(v, dtype=int)
    
    greedy_deadline = start_time + time_limit * 0.50
    
    while time.time() < greedy_deadline:
        uncovered_flags = ~covered[block_t_indices]
        gains = uncovered_flags.sum(axis=1)
        
        max_gain = np.max(gains)
        if max_gain == 0:
            break
            
        candidates = np.where(gains == max_gain)[0]
        
        # Tie-breaking with smooth surrogate: use negative point-load sum as soft penalty
        if len(candidates) > 1:
            load_sums = np.zeros(len(candidates), dtype=np.int32)
            for ci, c_idx in enumerate(candidates):
                block = all_ks[c_idx]
                load_sums[ci] = sum(point_load[p] for p in block)
            
            min_load = np.min(load_sums)
            low_load_mask = load_sums <= min_load + 1
            low_load_indices = np.where(low_load_mask)[0]
            best_local = low_load_indices[np.random.randint(len(low_load_indices))]
            best_idx = candidates[best_local]
        else:
            best_idx = candidates[0]
        
        blocks_indices.append(best_idx)
        block = all_ks[best_idx]
        covered[block_t_indices[best_idx]] = True
        for p in block:
            point_load[p] += 1
            
        if np.all(covered):
            break
            
    # Local Search: Try to remove redundant blocks
    coverage_count = np.zeros(n_ts, dtype=np.int32)
    for b_idx in blocks_indices:
        coverage_count[block_t_indices[b_idx]] += 1
        
    current_blocks = list(blocks_indices)
    
    ls_start = time.time()
    ls_time_limit = time_limit - (ls_start - start_time) - 5.0
    
    improved = True
    while improved and (time.time() - ls_start) < ls_time_limit:
        improved = False
        new_current_blocks = []
        
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx]] += 1
            
        for b_idx in current_blocks:
            subs_in_block = block_t_indices[b_idx]
            min_cov = np.min(coverage_count[subs_in_block])
            
            if min_cov > 1:
                improved = True
                coverage_count[subs_in_block] -= 1
            else:
                new_current_blocks.append(b_idx)
                
        current_blocks = new_current_blocks
    
    # Block replacement local search: try to replace pairs of blocks with a single block
    if (time.time() - ls_start) < ls_time_limit * 0.3:
        replacement_improved = True
        while replacement_improved and (time.time() - ls_start) < ls_time_limit * 0.6:
            replacement_improved = False
            
            coverage_count = np.zeros(n_ts, dtype=np.int32)
            for b_idx in current_blocks:
                coverage_count[block_t_indices[b_idx]] += 1
            
            n_blocks = len(current_blocks)
            
            # Try all pairs (i, j) with i < j
            for i in range(n_blocks):
                if (time.time() - ls_start) > ls_time_limit * 0.6:
                    break
                for j in range(i + 1, n_blocks):
                    if (time.time() - ls_start) > ls_time_limit * 0.6:
                        break
                    
                    b1 = current_blocks[i]
                    b2 = current_blocks[j]
                    
                    subs1 = block_t_indices[b1]
                    subs2 = block_t_indices[b2]
                    
                    # Find critical triples: those that would be uncovered if we remove both b1 and b2
                    # A triple is critical if:
                    # - It's in b1 with cov==1 (only b1 covers it)
                    # - It's in b2 with cov==1 (only b2 covers it)
                    # - It's in both b1 and b2 with cov==2 (only b1 and b2 cover it)
                    
                    critical_set = set()
                    
                    for sub_idx in subs1:
                        if coverage_count[sub_idx] == 1:
                            critical_set.add(sub_idx)
                    
                    for sub_idx in subs2:
                        if coverage_count[sub_idx] == 1:
                            critical_set.add(sub_idx)
                    
                    # Also check for triples in both with cov==2
                    for sub_idx in subs1:
                        if coverage_count[sub_idx] == 2 and sub_idx in set(subs2):
                            critical_set.add(sub_idx)
                    
                    if len(critical_set) == 0:
                        continue
                    
                    # Now check if any single block covers all critical triples
                    # For efficiency, check blocks that contain at least one critical triple
                    candidate_blocks = set()
                    for c_idx in critical_set:
                        for blk in triple_to_blocks[c_idx]:
                            candidate_blocks.add(blk)
                    
                    # Filter out b1 and b2
                    candidate_blocks.discard(b1)
                    candidate_blocks.discard(b2)
                    
                    # Check each candidate
                    found_replacement = False
                    for cand in candidate_blocks:
                        if cand in current_blocks:
                            continue
                        
                        cand_subs = set(block_t_indices[cand].tolist())
                        all_covered = True
                        for c in critical_set:
                            if c not in cand_subs:
                                all_covered = False
                                break
                        
                        if all_covered:
                            # Found a replacement!
                            new_blocks = [b for b in current_blocks if b != b1 and b != b2]
                            if cand not in new_blocks:
                                new_blocks.append(cand)
                            current_blocks = new_blocks
                            replacement_improved = True
                            found_replacement = True
                            break
                    
                    if found_replacement:
                        break
                
                if replacement_improved:
                    break
            
            # After a replacement, also do a quick removal pass
            if replacement_improved:
                coverage_count = np.zeros(n_ts, dtype=np.int32)
                for b_idx in current_blocks:
                    coverage_count[block_t_indices[b_idx]] += 1
                
                new_blocks = []
                for b_idx in current_blocks:
                    subs_in_block = block_t_indices[b_idx]
                    if np.min(coverage_count[subs_in_block]) > 1:
                        coverage_count[subs_in_block] -= 1
                    else:
                        new_blocks.append(b_idx)
                current_blocks = new_blocks
    
    # Convert indices back to blocks
    final_blocks = [list(all_ks[i]) for i in current_blocks]
    
    return final_blocks
