import numpy as np
import itertools
import time
import random
from collections import Counter

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0
    
    points = range(v)
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}
    
    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)
    
    def nCr(n, r):
        if r > n: return 0
        if r == 0: return 1
        res = 1
        for i in range(r):
            res *= (n - i)
            res //= (i + 1)
        return res
            
    t_count = nCr(k, t)
    
    # Precompute: for each block, which t-subsets it covers
    block_t_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    for i, block in enumerate(all_ks):
        subs = itertools.combinations(block, t)
        for j, sub in enumerate(subs):
            block_t_indices[i, j] = idx_map[sub]
    
    # Build reverse index: for each t-subset, which blocks cover it
    t_to_blocks = [set() for _ in range(n_ts)]
    for i in range(n_ks):
        for j in range(t_count):
            t_to_blocks[block_t_indices[i, j]].add(i)
    
    # Build point-to-tsubsets index for peeling
    point_to_ts = [set() for _ in range(v)]
    for ts_idx, ts in enumerate(all_ts):
        for p in ts:
            point_to_ts[p].add(ts_idx)
    
    best_solution = None
    best_score = float('inf')
    
    def fast_deletion(current_blocks, coverage_count):
        """Remove all blocks whose t-subsets are all covered by other blocks."""
        if time.time() - start_time > time_limit - 3.0:
            return current_blocks
        improved = True
        while improved:
            if time.time() - start_time > time_limit - 3.0:
                break
            improved = False
            new_blocks = []
            for b_idx in current_blocks:
                subs_in_block = block_t_indices[b_idx]
                if np.all(coverage_count[subs_in_block] > 1):
                    improved = True
                    coverage_count[subs_in_block] -= 1
                else:
                    new_blocks.append(b_idx)
            current_blocks = new_blocks
        return current_blocks
    
    def hypergraph_peeling(covered, blocks_indices):
        """
        Use hypergraph peeling: iteratively find the point with minimum degree 
        in the residual hypergraph of uncovered triples, then add the block that 
        maximally covers the triples incident to that point.
        """
        if time.time() - start_time > time_limit - 5.0:
            return blocks_indices, covered
        
        # Work with a copy of covered
        covered = covered.copy()
        remaining = np.sum(~covered)
        
        if remaining == 0:
            return blocks_indices, covered
        
        # Build current uncovered set per point
        # For each point, track how many uncovered t-subsets contain it
        point_uncovered_count = np.zeros(v, dtype=np.int32)
        for p in range(v):
            ts_list = list(point_to_ts[p])
            if ts_list:
                point_uncovered_count[p] = np.sum(~covered[ts_list])
        
        # Iteratively peel
        iterations = 0
        while remaining > 0 and iterations < 50:
            if time.time() - start_time > time_limit - 5.0:
                break
            iterations += 1
            
            # Find point with minimum positive degree
            min_degree = float('inf')
            best_point = -1
            for p in range(v):
                if point_uncovered_count[p] > 0 and point_uncovered_count[p] < min_degree:
                    min_degree = point_uncovered_count[p]
                    best_point = p
            
            if best_point == -1:
                break
            
            # Find all uncovered t-subsets containing best_point
            uncovered_ts_with_point = []
            for ts_idx in point_to_ts[best_point]:
                if not covered[ts_idx]:
                    uncovered_ts_with_point.append(ts_idx)
            
            if not uncovered_ts_with_point:
                break
            
            # Find the block that covers the most of these uncovered t-subsets
            # Also consider blocks that cover many other uncovered t-subsets
            best_block = -1
            best_gain = -1
            best_total_gain = -1
            
            # Candidates: blocks that contain best_point
            candidates = []
            for b_idx in range(n_ks):
                if best_point in all_ks[b_idx]:
                    candidates.append(b_idx)
            
            # If too many candidates, sample
            if len(candidates) > 100:
                candidates = random.sample(candidates, 100)
            
            for cand in candidates:
                subs = block_t_indices[cand]
                # Count how many of the uncovered t-subsets with best_point this block covers
                cand_set = set(subs.tolist())
                local_gain = sum(1 for ts in uncovered_ts_with_point if ts in cand_set)
                
                if local_gain > best_gain or (local_gain == best_gain and local_gain > 0):
                    # Compute total gain (all newly covered t-subsets)
                    total_gain = np.sum(~covered[subs])
                    if local_gain > best_gain or (local_gain == best_gain and total_gain > best_total_gain):
                        best_gain = local_gain
                        best_total_gain = total_gain
                        best_block = cand
            
            if best_block == -1 or best_gain <= 0:
                # Fallback: standard greedy pick
                covered_counts = covered[block_t_indices].sum(axis=1)
                gains = t_count - covered_counts
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                max_gain_blocks = np.where(gains == max_gain)[0]
                if len(max_gain_blocks) > 50:
                    best_block = random.choice(max_gain_blocks)
                else:
                    best_block = max_gain_blocks[0]
            else:
                best_block = int(best_block)
            
            # Add this block
            blocks_indices.append(best_block)
            subs_to_cover = block_t_indices[best_block]
            newly_covered = ~covered[subs_to_cover]
            new_covers = np.sum(newly_covered)
            remaining -= new_covers
            covered[subs_to_cover] = True
            
            # Update point uncovered counts
            for p in all_ks[best_block]:
                ts_list = list(point_to_ts[p])
                if ts_list:
                    point_uncovered_count[p] = np.sum(~covered[ts_list])
        
        return blocks_indices, covered
    
    def swap_local_search(current_blocks, coverage_count):
        """Try to replace pairs of blocks with a single block that covers all their unique t-subsets."""
        if time.time() - start_time > time_limit - 3.0:
            return current_blocks, coverage_count
        
        blocks_set = set(current_blocks)
        improved = True
        while improved:
            if time.time() - start_time > time_limit - 3.0:
                break
            improved = False
            
            n = len(current_blocks)
            tried = set()
            for i in range(n):
                if time.time() - start_time > time_limit - 3.0:
                    return current_blocks, coverage_count
                b1 = current_blocks[i]
                for j in range(i + 1, n):
                    b2 = current_blocks[j]
                    pair_key = (min(b1, b2), max(b1, b2))
                    if pair_key in tried:
                        continue
                    tried.add(pair_key)
                    
                    subs1 = block_t_indices[b1]
                    subs2 = block_t_indices[b2]
                    
                    unique_to_1 = []
                    unique_to_2 = []
                    for sub in subs1:
                        if coverage_count[sub] == 1:
                            unique_to_1.append(sub)
                    for sub in subs2:
                        if coverage_count[sub] == 1:
                            unique_to_2.append(sub)
                    
                    if not unique_to_1 and not unique_to_2:
                        improved = True
                        coverage_count[subs1] -= 1
                        coverage_count[subs2] -= 1
                        current_blocks = [b for b in current_blocks if b not in (b1, b2)]
                        blocks_set.discard(b1)
                        blocks_set.discard(b2)
                        break
                    elif unique_to_1 and not unique_to_2:
                        candidates = t_to_blocks[unique_to_1[0]]
                        for cand in candidates:
                            if cand in blocks_set and cand != b1 and cand != b2:
                                cand_subs = set(block_t_indices[cand].tolist())
                                if all(s in cand_subs for s in unique_to_1):
                                    improved = True
                                    coverage_count[subs1] -= 1
                                    coverage_count[subs2] -= 1
                                    coverage_count[block_t_indices[cand]] += 1
                                    current_blocks = [b for b in current_blocks if b not in (b1, b2)]
                                    blocks_set.discard(b1)
                                    blocks_set.discard(b2)
                                    blocks_set.add(cand)
                                    break
                        if improved:
                            break
                    elif unique_to_2 and not unique_to_1:
                        candidates = t_to_blocks[unique_to_2[0]]
                        for cand in candidates:
                            if cand in blocks_set and cand != b1 and cand != b2:
                                cand_subs = set(block_t_indices[cand].tolist())
                                if all(s in cand_subs for s in unique_to_2):
                                    improved = True
                                    coverage_count[subs1] -= 1
                                    coverage_count[subs2] -= 1
                                    coverage_count[block_t_indices[cand]] += 1
                                    current_blocks = [b for b in current_blocks if b not in (b1, b2)]
                                    blocks_set.discard(b1)
                                    blocks_set.discard(b2)
                                    blocks_set.add(cand)
                                    break
                        if improved:
                            break
                    else:
                        pass
            if improved:
                coverage_count = np.zeros(n_ts, dtype=np.int32)
                for b_idx in current_blocks:
                    coverage_count[block_t_indices[b_idx]] += 1
                blocks_set = set(current_blocks)
        return current_blocks, coverage_count
    
    def run_greedy(seed):
        random.seed(seed)
        np.random.seed(seed)
        
        covered = np.zeros(n_ts, dtype=bool)
        blocks_indices = []
        remaining = n_ts
        
        cover_freq = np.zeros(n_ts, dtype=np.int32)
        
        # Phase 1: Standard greedy for most of the coverage
        while remaining > 0:
            if time.time() - start_time > time_limit - 8.0:
                break
            
            covered_counts = covered[block_t_indices].sum(axis=1)
            gains = t_count - covered_counts
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            max_gain_blocks = np.where(gains == max_gain)[0]
            
            if len(max_gain_blocks) > 50:
                candidates = max_gain_blocks[random.sample(range(len(max_gain_blocks)), min(50, len(max_gain_blocks)))]
            else:
                candidates = max_gain_blocks
            
            best_score_candidate = -1
            best_candidate = candidates[0]
            for cand in candidates:
                subs = block_t_indices[cand]
                newly_covered = ~covered[subs]
                if np.any(newly_covered):
                    bonus = np.sum(1.0 / (1.0 + cover_freq[subs[newly_covered]]))
                else:
                    bonus = 0.0
                score_cand = max_gain + bonus * 0.1
                if score_cand > best_score_candidate:
                    best_score_candidate = score_cand
                    best_candidate = cand
            
            blocks_indices.append(best_candidate)
            
            subs_to_cover = block_t_indices[best_candidate]
            newly_covered = ~covered[subs_to_cover]
            new_covers = np.sum(newly_covered)
            remaining -= new_covers
            cover_freq[subs_to_cover] += 1
            covered[subs_to_cover] = True
            
            # Stop early when we're in the "hard tail" region
            if remaining < n_ts * 0.15 and len(blocks_indices) > 50:
                break
        
        # Phase 2: Hypergraph peeling for the residual
        if remaining > 0 and time.time() - start_time < time_limit - 5.0:
            blocks_indices, covered = hypergraph_peeling(covered, blocks_indices)
        
        # Phase 3: Finish with standard greedy if needed
        remaining = np.sum(~covered)
        while remaining > 0:
            if time.time() - start_time > time_limit - 5.0:
                break
            
            covered_counts = covered[block_t_indices].sum(axis=1)
            gains = t_count - covered_counts
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            max_gain_blocks = np.where(gains == max_gain)[0]
            best_candidate = max_gain_blocks[0]
            blocks_indices.append(best_candidate)
            
            subs_to_cover = block_t_indices[best_candidate]
            newly_covered = ~covered[subs_to_cover]
            new_covers = np.sum(newly_covered)
            remaining -= new_covers
            covered[subs_to_cover] = True
        
        current_blocks = list(blocks_indices)
        
        # Build coverage count
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx]] += 1
        
        # Fast deletion
        current_blocks = fast_deletion(current_blocks, coverage_count)
        
        # Rebuild coverage count after deletion
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx]] += 1
        
        # Swap local search
        current_blocks, coverage_count = swap_local_search(current_blocks, coverage_count)
        
        # Final deletion pass
        current_blocks = fast_deletion(current_blocks, coverage_count)
        
        return current_blocks, len(current_blocks)

    restart_count = 0
    while time.time() - start_time < time_limit - 10.0:
        seed = restart_count
        blocks, score = run_greedy(seed)
        
        if blocks is None:
            break
            
        if score < best_score:
            best_score = score
            best_solution = blocks
            
        restart_count += 1
    
    if best_solution is None:
        blocks, score = run_greedy(0)
        if blocks is not None:
            best_solution = blocks
            best_score = score
        else:
            return []

    final_blocks = [list(all_ks[i]) for i in best_solution]
    
    return final_blocks
