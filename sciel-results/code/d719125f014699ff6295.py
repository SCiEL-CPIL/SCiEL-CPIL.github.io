import time
import random
import itertools
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Helper function for combinations count
    def C(n, r):
        if r > n or r < 0:
            return 0
        if r == 0 or r == n:
            return 1
        r = min(r, n - r)
        result = 1
        for i in range(1, r + 1):
            result = result * (n - i + 1) // i
        return result

    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    
    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    block_to_ts_np = []
    
    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(ts_list)
        block_to_ts_np.append(np.array(ts_list, dtype=np.int32))

    # State
    gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
    is_uncovered = np.full(num_t_subsets, True, dtype=bool)
    is_selected = np.full(num_k_blocks, False, dtype=bool)
    
    solution_blocks_idx = []
    
    # Greedy Phase with LP-inspired weighting
    # Use a price-based approach: each uncovered t-subset has a "price"
    # that reflects how many blocks can cover it (rarity)
    # Blocks covering rare t-subsets get higher priority
    
    # Precompute rarity: number of blocks covering each t-subset
    ts_rarity = np.array([len(ts_to_blocks[i]) for i in range(num_t_subsets)], dtype=np.float64)
    
    # Initial price = 1/rarity (higher price for rarer subsets)
    ts_price = np.maximum(1.0 / ts_rarity, 1e-6)
    
    while True:
        if time.time() > deadline - 15.0:
            break
            
        max_gain = gain.max()
        if max_gain <= 0:
            break
        
        # Score blocks by weighted gain: sum of prices of uncovered t-subsets they cover
        # This is a smooth surrogate that considers global coverage pressure
        candidates = np.where(gain == max_gain)[0]
        
        if len(candidates) <= 50:
            # Few candidates: pick the one with highest price-weighted gain
            best_idx = candidates[0]
            best_score = -1
            for idx in candidates:
                score = 0.0
                for ts_idx in block_to_ts[idx]:
                    if is_uncovered[ts_idx]:
                        score += ts_price[ts_idx]
                if score > best_score:
                    best_score = score
                    best_idx = idx
        else:
            # Many candidates: sample to find good ones
            sample_size = min(200, len(candidates))
            sample = random.sample(candidates.tolist(), sample_size)
            best_idx = sample[0]
            best_score = -1
            for idx in sample:
                score = 0.0
                for ts_idx in block_to_ts[idx]:
                    if is_uncovered[ts_idx]:
                        score += ts_price[ts_idx]
                if score > best_score:
                    best_score = score
                    best_idx = idx
        
        solution_blocks_idx.append(best_idx)
        is_selected[best_idx] = True
        gain[best_idx] = -1
        
        for ts_idx in block_to_ts[best_idx]:
            if is_uncovered[ts_idx]:
                is_uncovered[ts_idx] = False
                for b_idx in ts_to_blocks[ts_idx]:
                    if not is_selected[b_idx]:
                        gain[b_idx] -= 1
    
    best_solution_idx = solution_blocks_idx[:]
    best_len = len(solution_blocks_idx)
    
    # Fast cover count calculator
    def get_cover_counts(x):
        counts = np.zeros(num_t_subsets, dtype=np.int32)
        for j in range(num_k_blocks):
            if x[j]:
                counts[block_to_ts_np[j]] += 1
        return counts
    
    # Initial state from Greedy
    x = np.zeros(num_k_blocks, dtype=bool)
    for j in best_solution_idx:
        x[j] = True
        
    cover_count = get_cover_counts(x)
    
    # Optimized Local Search
    def local_search_hill_climb(x, cover_count, max_iters=5000):
        x = x.copy()
        cover_count = cover_count.copy()
        current_cost = int(np.sum(x))
        best_cost = current_cost
        best_x = x.copy()
        
        no_improvement = 0
        lb = (C(v, t) + C(k, t) - 1) // C(k, t)
        
        active_blocks = np.where(x)[0].tolist()
        
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            if not active_blocks:
                break
                
            improved = False
            random.shuffle(active_blocks)
            
            for j in active_blocks:
                if np.all(cover_count[block_to_ts_np[j]] > 1):
                    x[j] = False
                    cover_count[block_to_ts_np[j]] -= 1
                    current_cost -= 1
                    active_blocks.remove(j)
                    improved = True
                    break
                    
            if improved:
                no_improvement = 0
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_x = x.copy()
                    if best_cost == lb:
                        return best_x, best_cost
            else:
                no_improvement += 1
                if no_improvement > 30:
                    break
                    
        return best_x, best_cost

    def local_search_with_swap(x, cover_count, max_iters=2000):
        x = x.copy()
        cover_count = cover_count.copy()
        current_cost = int(np.sum(x))
        best_cost = current_cost
        best_x = x.copy()
        lb = (C(v, t) + C(k, t) - 1) // C(k, t)
        
        active_blocks = np.where(x)[0].tolist()
        inactive_blocks = np.where(~x)[0].tolist()
        
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            if not active_blocks:
                break
                
            removed_any = False
            random.shuffle(active_blocks)
            for j in active_blocks:
                if np.all(cover_count[block_to_ts_np[j]] > 1):
                    x[j] = False
                    cover_count[block_to_ts_np[j]] -= 1
                    current_cost -= 1
                    active_blocks.remove(j)
                    inactive_blocks.append(j)
                    removed_any = True
                    break
                    
            if removed_any:
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_x = x.copy()
                    if best_cost == lb:
                        return best_x, best_cost
                continue
                
            if not inactive_blocks:
                break
                
            j = random.choice(active_blocks)
            unique_ts_j = set(ts_idx for ts_idx in block_to_ts_np[j] if cover_count[ts_idx] == 1)
            
            if not unique_ts_j:
                if np.all(cover_count[block_to_ts_np[j]] > 1):
                    x[j] = False
                    cover_count[block_to_ts_np[j]] -= 1
                    current_cost -= 1
                    active_blocks.remove(j)
                    inactive_blocks.append(j)
                    continue
                else:
                    continue
            
            found = False
            sample_size = min(80, len(inactive_blocks))
            sample = random.sample(inactive_blocks, sample_size)
            
            for j_prime in sample:
                if unique_ts_j.issubset(set(block_to_ts_np[j_prime])):
                    x[j] = False
                    cover_count[block_to_ts_np[j]] -= 1
                    x[j_prime] = True
                    cover_count[block_to_ts_np[j_prime]] += 1
                    
                    if np.min(cover_count) >= 1:
                        active_blocks.remove(j)
                        active_blocks.append(j_prime)
                        inactive_blocks.remove(j_prime)
                        inactive_blocks.append(j)
                        found = True
                        break
                    else:
                        x[j_prime] = False
                        cover_count[block_to_ts_np[j_prime]] -= 1
                        x[j] = True
                        cover_count[block_to_ts_np[j]] += 1
                        
            if found:
                continue
                
            if len(active_blocks) < 2:
                break
                
            critical_blocks = []
            for j in active_blocks:
                if not np.all(cover_count[block_to_ts_np[j]] > 1):
                    critical_blocks.append(j)
                    
            if len(critical_blocks) < 2:
                continue
                
            j1 = random.choice(critical_blocks)
            j2_candidates = [j for j in critical_blocks if j != j1]
            if not j2_candidates:
                continue
            j2 = random.choice(j2_candidates)
            
            unique_ts_12 = set()
            for ts_idx in block_to_ts_np[j1]:
                if cover_count[ts_idx] == 1:
                    unique_ts_12.add(ts_idx)
            for ts_idx in block_to_ts_np[j2]:
                if cover_count[ts_idx] == 1:
                    unique_ts_12.add(ts_idx)
            
            if not unique_ts_12:
                continue
                
            sample_size = min(120, len(inactive_blocks))
            sample = random.sample(inactive_blocks, sample_size)
            
            found_swap = False
            for j3 in sample:
                if unique_ts_12.issubset(set(block_to_ts_np[j3])):
                    x[j1] = False
                    cover_count[block_to_ts_np[j1]] -= 1
                    x[j2] = False
                    cover_count[block_to_ts_np[j2]] -= 1
                    x[j3] = True
                    cover_count[block_to_ts_np[j3]] += 1
                    
                    if np.min(cover_count) >= 1:
                        active_blocks.remove(j1)
                        active_blocks.remove(j2)
                        active_blocks.append(j3)
                        inactive_blocks.remove(j3)
                        inactive_blocks.append(j1)
                        inactive_blocks.append(j2)
                        current_cost -= 1
                        found_swap = True
                        
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_x = x.copy()
                            if best_cost == lb:
                                return best_x, best_cost
                        break
                    else:
                        x[j3] = False
                        cover_count[block_to_ts_np[j3]] -= 1
                        x[j2] = True
                        cover_count[block_to_ts_np[j2]] += 1
                        x[j1] = True
                        cover_count[block_to_ts_np[j1]] += 1
                        current_cost += 1

        return best_x, best_cost

    # Run the search
    current_x = x
    current_cc = cover_count
    
    # First pass: Pure hill climbing (removals)
    current_x, current_cost = local_search_hill_climb(current_x, current_cc, max_iters=3000)
    if current_cost < best_len:
        best_len = current_cost
        best_solution_idx = np.where(current_x)[0].tolist()
        current_cc = get_cover_counts(current_x)
        
    # Second pass: Hill climbing with swaps
    for restart in range(15):
        if time.time() > deadline - 5.0:
            break
        random.seed(restart)
        new_x, new_cost = local_search_with_swap(current_x, current_cc, max_iters=1500)
        
        if new_cost < best_len:
            best_len = new_cost
            best_solution_idx = np.where(new_x)[0].tolist()
            current_x = new_x
            current_cc = get_cover_counts(new_x)
            
            current_x, current_cost = local_search_hill_climb(current_x, current_cc, max_iters=1500)
            if current_cost < best_len:
                best_len = current_cost
                best_solution_idx = np.where(current_x)[0].tolist()
                current_cc = get_cover_counts(current_x)
            
    # Final cleanup
    final_x = np.zeros(num_k_blocks, dtype=bool)
    for j in best_solution_idx:
        final_x[j] = True
    final_cc = get_cover_counts(final_x)
    
    final_x, final_cost = local_search_hill_climb(final_x, final_cc, max_iters=2000)
    if final_cost < best_len:
        best_len = final_cost
        best_solution_idx = np.where(final_x)[0].tolist()

    final_blocks = [list(all_k_blocks[j]) for j in best_solution_idx]
    
    return final_blocks
