import numpy as np
import itertools
import time
import random
from math import comb, log, sqrt

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    # Total number of t-subsets
    total_tsubsets = comb(v, t)
    
    # Generate all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Track which t-subsets are covered: 1 = covered, 0 = not
    covered = np.zeros(total_tsubsets, dtype=np.int8)
    
    # For each point, precompute which t-subsets contain it
    point_tsubsets = [[] for _ in range(v)]
    for i, ts in enumerate(all_tsubsets):
        for p in ts:
            point_tsubsets[p].append(i)
    
    # Convert to numpy arrays for speed
    point_tsubset_arrays = [np.array(pts, dtype=np.int32) for pts in point_tsubsets]
    
    # Precompute: for each pair of points, which t-subsets contain both
    pair_tsubsets = {}
    for i, ts in enumerate(all_tsubsets):
        for a in range(t):
            for b in range(a+1, t):
                pair = (ts[a], ts[b])
                if pair not in pair_tsubsets:
                    pair_tsubsets[pair] = []
                pair_tsubsets[pair].append(i)
    
    # Precompute: for each triple of points (if t>=3), which t-subsets contain all three
    triple_tsubsets = {}
    if t >= 3:
        for i, ts in enumerate(all_tsubsets):
            for a in range(t):
                for b in range(a+1, t):
                    for c in range(b+1, t):
                        triple = (ts[a], ts[b], ts[c])
                        if triple not in triple_tsubsets:
                            triple_tsubsets[triple] = []
                        triple_tsubsets[triple].append(i)
    
    blocks = []
    block_set_cache = set()
    
    # UCB parameters
    exploration_factor = 1.5  # sqrt(2 * log(n_total))
    
    # Candidate block pool for UCB selection
    # Each arm: (block_tuple, total_gain, pull_count)
    candidate_pool = []
    total_pulls = 0
    
    def compute_block_gain(block_list, covered_arr):
        """Compute how many uncovered t-subsets this block covers."""
        gain = 0
        for ts in itertools.combinations(block_list, t):
            idx = tsubset_to_idx[ts]
            if covered_arr[idx] == 0:
                gain += 1
        return gain
    
    def generate_candidate_blocks(covered_arr, n_candidates=20):
        """Generate candidate blocks using various heuristics."""
        candidates = []
        
        # Find uncovered t-subsets
        uncovered_mask = covered_arr == 0
        if not np.any(uncovered_mask):
            return candidates
        
        uncovered_indices = np.where(uncovered_mask)[0]
        
        # Strategy 1: For each of several random uncovered t-subsets, build a greedy block
        n_samples = min(5, len(uncovered_indices))
        sample_indices = np.random.choice(uncovered_indices, size=n_samples, replace=False)
        
        for si in sample_indices:
            target_ts = all_tsubsets[si]
            block_set = set(target_ts)
            block_list = list(target_ts)
            
            # Greedily add points
            while len(block_list) < k:
                if time.time() > deadline - 5:
                    break
                
                best_point = -1
                best_gain = -1
                
                for p in range(v):
                    if p in block_set:
                        continue
                    
                    # Calculate gain: how many uncovered t-subsets would be covered
                    gain = 0
                    if t - 1 <= len(block_list):
                        for rest in itertools.combinations(block_list, t - 1):
                            candidate = tuple(sorted(rest + (p,)))
                            if candidate in tsubset_to_idx:
                                idx = tsubset_to_idx[candidate]
                                if covered_arr[idx] == 0:
                                    gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = p
                
                if best_point == -1 or best_gain == 0:
                    available = [p for p in range(v) if p not in block_set]
                    if available:
                        best_point = available[random.randint(0, len(available)-1)]
                    else:
                        break
                
                block_set.add(best_point)
                block_list.append(best_point)
            
            block_list.sort()
            bt = tuple(block_list)
            if bt not in block_set_cache:
                gain = compute_block_gain(block_list, covered_arr)
                candidates.append((bt, gain, 0))
        
        # Strategy 2: Random blocks biased toward uncovered points
        for _ in range(3):
            block_list = set()
            # Start with a random uncovered t-subset
            if len(uncovered_indices) > 0:
                idx = uncovered_indices[random.randint(0, len(uncovered_indices)-1)]
                block_list.update(all_tsubsets[idx])
            
            # Fill remaining slots
            while len(block_list) < k:
                if time.time() > deadline - 5:
                    break
                p = random.randint(0, v-1)
                if p not in block_list:
                    block_list.add(p)
            
            block_list = sorted(block_list)
            bt = tuple(block_list)
            if bt not in block_set_cache:
                gain = compute_block_gain(block_list, covered_arr)
                candidates.append((bt, gain, 0))
        
        return candidates
    
    def ucb_select(candidates, covered_arr):
        """Select a candidate using UCB1 formula."""
        if not candidates:
            return None
        
        # Update gains based on current coverage
        updated = []
        for bt, old_gain, pulls in candidates:
            gain = compute_block_gain(list(bt), covered_arr)
            updated.append((bt, gain, pulls))
        
        if not updated:
            return None
        
        # UCB1: mean_reward + exploration_factor * sqrt(log(total_pulls) / pulls)
        if total_pulls == 0:
            # All unexplored, pick max gain
            best_idx = max(range(len(updated)), key=lambda i: updated[i][1])
        else:
            log_n = log(total_pulls + 1)
            ucb_scores = []
            for i, (bt, gain, pulls) in enumerate(updated):
                if pulls == 0:
                    # Infinite UCB for unexplored arms
                    ucb_scores.append(float('inf'))
                else:
                    mean_reward = gain / pulls
                    exploration_bonus = exploration_factor * sqrt(log_n / pulls)
                    ucb_scores.append(mean_reward + exploration_bonus)
            
            best_idx = max(range(len(updated)), key=lambda i: ucb_scores[i])
        
        return updated[best_idx]
    
    while True:
        if time.time() > deadline - 10:
            break
        
        # Find uncovered t-subsets
        uncovered_indices = np.where(covered == 0)[0]
        if len(uncovered_indices) == 0:
            break
        
        # Generate candidate blocks
        new_candidates = generate_candidate_blocks(covered, n_candidates=15)
        
        # Add new candidates to pool (avoiding duplicates)
        for bt, gain, pulls in new_candidates:
            if bt not in block_set_cache:
                candidate_pool.append((bt, gain, pulls))
        
        # Remove candidates with zero gain from pool
        candidate_pool = [(bt, g, p) for bt, g, p in candidate_pool if g > 0]
        
        if not candidate_pool:
            # Fallback: greedy construction
            target_ts = all_tsubsets[uncovered_indices[0]]
            block_set = set(target_ts)
            block_list = list(target_ts)
            
            while len(block_list) < k:
                if time.time() > deadline - 5:
                    break
                
                best_point = -1
                best_gain = -1
                
                for p in range(v):
                    if p in block_set:
                        continue
                    
                    gain = 0
                    if t - 1 <= len(block_list):
                        for rest in itertools.combinations(block_list, t - 1):
                            candidate = tuple(sorted(rest + (p,)))
                            if candidate in tsubset_to_idx:
                                idx = tsubset_to_idx[candidate]
                                if covered[idx] == 0:
                                    gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = p
                
                if best_point == -1 or best_gain == 0:
                    available = [p for p in range(v) if p not in block_set]
                    if available:
                        best_point = available[random.randint(0, len(available)-1)]
                    else:
                        break
                
                block_set.add(best_point)
                block_list.append(best_point)
            
            block_list.sort()
            bt = tuple(block_list)
            if bt not in block_set_cache:
                blocks.append(block_list)
                block_set_cache.add(bt)
                
                # Update covered array
                for ts in itertools.combinations(block_list, t):
                    idx = tsubset_to_idx[ts]
                    covered[idx] = 1
            continue
        
        # Select best candidate using UCB
        selected = ucb_select(candidate_pool, covered)
        
        if selected is None:
            # Fallback
            target_ts = all_tsubsets[uncovered_indices[0]]
            block_set = set(target_ts)
            block_list = list(target_ts)
            
            while len(block_list) < k:
                if time.time() > deadline - 5:
                    break
                
                best_point = -1
                best_gain = -1
                
                for p in range(v):
                    if p in block_set:
                        continue
                    
                    gain = 0
                    if t - 1 <= len(block_list):
                        for rest in itertools.combinations(block_list, t - 1):
                            candidate = tuple(sorted(rest + (p,)))
                            if candidate in tsubset_to_idx:
                                idx = tsubset_to_idx[candidate]
                                if covered[idx] == 0:
                                    gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = p
                
                if best_point == -1 or best_gain == 0:
                    available = [p for p in range(v) if p not in block_set]
                    if available:
                        best_point = available[random.randint(0, len(available)-1)]
                    else:
                        break
                
                block_set.add(best_point)
                block_list.append(best_point)
            
            block_list.sort()
            bt = tuple(block_list)
            if bt not in block_set_cache:
                blocks.append(block_list)
                block_set_cache.add(bt)
                
                # Update covered array
                for ts in itertools.combinations(block_list, t):
                    idx = tsubset_to_idx[ts]
                    covered[idx] = 1
            continue
        
        bt, gain, pulls = selected
        
        # Check if this block is already in our list
        if bt in block_set_cache:
            # Update pull count
            candidate_pool = [(b, g, p if b != bt else p + 1) for b, g, p in candidate_pool]
            total_pulls += 1
            continue
        
        block_list = list(bt)
        blocks.append(block_list)
        block_set_cache.add(bt)
        
        # Update pull count for selected arm
        candidate_pool = [(b, g, p if b != bt else p + 1) for b, g, p in candidate_pool]
        total_pulls += 1
        
        # Update covered array
        for ts in itertools.combinations(block_list, t):
            idx = tsubset_to_idx[ts]
            covered[idx] = 1
        
        # Check if we've covered everything
        if np.all(covered == 1):
            break
    
    # Post-processing: try to remove redundant blocks
    if time.time() < deadline - 2 and len(blocks) > 0:
        i = 0
        while i < len(blocks):
            if time.time() > deadline - 2:
                break
            
            # Temporarily remove block i
            temp_blocks = blocks[:i] + blocks[i+1:]
            
            # Check if all t-subsets are still covered
            still_covered = np.zeros(total_tsubsets, dtype=np.int8)
            for b in temp_blocks:
                for ts in itertools.combinations(b, t):
                    still_covered[tsubset_to_idx[ts]] = 1
            
            if np.all(still_covered == 1):
                # Block i is redundant, remove it
                blocks.pop(i)
            else:
                i += 1
    
    return blocks
