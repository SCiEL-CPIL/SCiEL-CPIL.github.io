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
    # Generate all t-subsets and map them to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Generate all possible k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    
    # Map t-subset to index
    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Build incidence structures:
    # block_to_ts[j]: list of indices of t-subsets contained in block j
    # ts_to_blocks[i]: list of indices of blocks containing t-subset i
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    
    # Precompute numpy arrays for block_to_ts for vectorized operations
    block_to_ts_np = []
    
    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(ts_list)
        block_to_ts_np.append(np.array(ts_list, dtype=np.int32))

    # State for Greedy initialization
    gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
    is_uncovered = np.full(num_t_subsets, True, dtype=bool)
    is_selected = np.full(num_k_blocks, False, dtype=bool)
    
    solution_blocks_idx = []
    
    # Greedy Phase (Deterministic) to get a valid initial solution
    while True:
        if time.time() > deadline - 10.0:
            break
            
        max_gain = gain.max()
        if max_gain <= 0:
            break
        
        # Find all blocks with max gain, break ties by block index for determinism
        candidates = np.where(gain == max_gain)[0]
        max_idx = candidates[0]
        
        solution_blocks_idx.append(max_idx)
        is_selected[max_idx] = True
        gain[max_idx] = -1
        
        # Vectorized update of gains
        # We need to decrement gain for all blocks that cover the newly covered t-subsets
        # But only for those t-subsets that were previously uncovered.
        # This is tricky to vectorize fully without knowing which ts were uncovered.
        # We'll stick to loop for greedy as it's fast enough for initialization.
        for ts_idx in block_to_ts[max_idx]:
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
        
        # Maintain a list of active blocks to avoid np.where every time
        active_blocks = np.where(x)[0].tolist()
        
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            if not active_blocks:
                break
                
            improved = False
            random.shuffle(active_blocks)
            
            for j in active_blocks:
                # Vectorized check: is cover_count[ts] > 1 for all ts in block j?
                if np.all(cover_count[block_to_ts_np[j]] > 1):
                    # Remove block j
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
                if no_improvement > 50:
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
                
            # 1. Try removals (fast path)
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
                
            # 2. Try 1-swap (remove j, add j')
            # We need j' to cover all unique ts of j
            # To speed up, we can sample a few j's
            
            if not inactive_blocks:
                break
                
            # Pick a random active block to try to swap
            j = random.choice(active_blocks)
            unique_ts_j = set(ts_idx for ts_idx in block_to_ts_np[j] if cover_count[ts_idx] == 1)
            
            if not unique_ts_j:
                # Should have been removed in step 1, but maybe due to race condition or logic
                # Just try to remove it if possible
                if np.all(cover_count[block_to_ts_np[j]] > 1):
                     x[j] = False
                     cover_count[block_to_ts_np[j]] -= 1
                     current_cost -= 1
                     active_blocks.remove(j)
                     inactive_blocks.append(j)
                     continue
                else:
                    continue
            
            # Find a block in inactive that covers unique_ts_j
            # Sample a few
            found = False
            sample_size = min(50, len(inactive_blocks))
            sample = random.sample(inactive_blocks, sample_size)
            
            for j_prime in sample:
                # Check if j_prime covers all unique_ts_j
                # We can check if unique_ts_j is a subset of block_to_ts_np[j_prime]
                # Convert unique_ts_j to numpy array for fast comparison? Or just set intersection.
                # Set intersection is probably faster for small sets.
                if unique_ts_j.issubset(set(block_to_ts_np[j_prime])):
                    # Perform swap
                    x[j] = False
                    cover_count[block_to_ts_np[j]] -= 1
                    x[j_prime] = True
                    cover_count[block_to_ts_np[j_prime]] += 1
                    
                    # Check validity: min cover count >= 1
                    if np.min(cover_count) >= 1:
                        active_blocks.remove(j)
                        active_blocks.append(j_prime)
                        inactive_blocks.remove(j_prime)
                        inactive_blocks.append(j)
                        found = True
                        break
                    else:
                        # Revert
                        x[j_prime] = False
                        cover_count[block_to_ts_np[j_prime]] -= 1
                        x[j] = True
                        cover_count[block_to_ts_np[j]] += 1
                        
            if found:
                # Cost is same (1 out, 1 in), so no improvement in cost, but might enable removals
                # We don't update best_cost here as cost didn't change
                continue
                
            # 3. Try 2-exchange (remove j1, j2, add j3)
            # This is expensive, so we do it less frequently
            if len(active_blocks) < 2:
                break
                
            # Find critical blocks
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
                
            # Find j3 in inactive that covers unique_ts_12
            sample_size = min(100, len(inactive_blocks))
            sample = random.sample(inactive_blocks, sample_size)
            
            found_swap = False
            for j3 in sample:
                if unique_ts_12.issubset(set(block_to_ts_np[j3])):
                    # Perform 2-exchange
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
                        # Revert
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
    current_x, current_cost = local_search_hill_climb(current_x, current_cc, max_iters=2000)
    if current_cost < best_len:
        best_len = current_cost
        best_solution_idx = np.where(current_x)[0].tolist()
        current_cc = get_cover_counts(current_x)
        
    # Second pass: Hill climbing with swaps (perturbation)
    # Run multiple times
    for restart in range(10):
        if time.time() > deadline - 5.0:
            break
        random.seed(restart)
        new_x, new_cost = local_search_with_swap(current_x, current_cc, max_iters=1000)
        
        if new_cost < best_len:
            best_len = new_cost
            best_solution_idx = np.where(new_x)[0].tolist()
            current_x = new_x
            current_cc = get_cover_counts(current_x)
            
            # After improvement, do another round of pure removals
            current_x, current_cost = local_search_hill_climb(current_x, current_cc, max_iters=1000)
            if current_cost < best_len:
                best_len = current_cost
                best_solution_idx = np.where(current_x)[0].tolist()
                current_cc = get_cover_counts(current_x)
            
    # Final cleanup
    final_x = np.zeros(num_k_blocks, dtype=bool)
    for j in best_solution_idx:
        final_x[j] = True
    final_cc = get_cover_counts(final_x)
    
    final_x, final_cost = local_search_hill_climb(final_x, final_cc, max_iters=1000)
    if final_cost < best_len:
        best_len = final_cost
        best_solution_idx = np.where(final_x)[0].tolist()

    # Convert indices back to blocks
    final_blocks = [list(all_k_blocks[j]) for j in best_solution_idx]
    
    return final_blocks
