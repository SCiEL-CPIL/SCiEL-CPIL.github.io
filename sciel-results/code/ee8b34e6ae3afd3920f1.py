import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin for 150s limit

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

    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    
    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(ts_list)

    # Precompute as numpy arrays
    block_to_ts_np = [np.array(lst, dtype=np.int32) for lst in block_to_ts]
    ts_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in ts_to_blocks]
    
    block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}
    
    def get_block_idx(block):
        return block_tuple_to_idx.get(tuple(block), -1)

    # Precompute for faster local search: for each block, store the t-subset indices as a frozenset
    # and also precompute overlap counts between blocks to speed up swap evaluation
    # Actually, let's use a different approach: precompute a matrix of shared t-subsets
    # This is too large (75582 x 75582), so instead we'll use the existing structures

    def run_greedy(seed, explore_prob=0.0):
        random.seed(seed)
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []
        
        while True:
            if time.time() > deadline - 5.0:
                return None
            
            max_gain = int(np.max(gain))
            if max_gain <= 0:
                break
                
            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break
                
            if explore_prob > 0 and random.random() < explore_prob:
                explore_cands = np.where((gain >= max_gain - 3) & (~is_selected))[0]
                if len(explore_cands) > 0:
                    b_idx = int(explore_cands[random.randint(0, len(explore_cands) - 1)])
                else:
                    b_idx = int(candidates[random.randint(0, len(candidates) - 1)])
            else:
                b_idx = int(candidates[random.randint(0, len(candidates) - 1)])
            
            sol.append(list(all_k_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1
            
            bts = block_to_ts_np[b_idx]
            newly_covered_mask = is_uncovered[bts]
            newly_covered = bts[newly_covered_mask]
            
            if len(newly_covered) == 0:
                continue
                
            is_uncovered[newly_covered] = False
            
            # Use np.bincount for faster decrement
            blocks_to_decrement = []
            for ts_idx in newly_covered:
                blocks_to_decrement.extend(ts_to_blocks[ts_idx])
                
            if blocks_to_decrement:
                counts = np.bincount(blocks_to_decrement, minlength=num_k_blocks)
                gain -= counts
                gain[is_selected] = -1000

        return sol

    # Initial deterministic greedy
    best_solution = run_greedy(0)
    if best_solution is None:
        best_solution = [list(all_k_blocks[0])]
    best_len = len(best_solution)
    
    # Run multiple randomized greedy with varying exploration
    # Use more runs since greedy is fast
    num_runs = 150
    for i in range(1, num_runs):
        if time.time() > deadline - 40.0:
            break
        
        explore = 0.0 if i < 50 else (0.1 if i < 100 else 0.2)
        sol = run_greedy(i, explore_prob=explore)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol

    # Fast local search using bitset-like numpy operations
    def fast_local_search(blocks, max_iters=5000):
        n_blocks = len(blocks)
        if n_blocks == 0:
            return blocks, 0
        
        # Build cover count
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        block_ts_indices = []
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                block_ts_indices.append(block_to_ts_np[idx])
            else:
                ts_list = []
                for ts in itertools.combinations(b, t):
                    ts_list.append(ts_index_map[ts])
                arr = np.array(ts_list, dtype=np.int32)
                block_ts_indices.append(arr)
        
        for ts_arr in block_ts_indices:
            np.add.at(cover_count, ts_arr, 1)
        
        # Track which blocks are in solution
        sol_block_set = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_set.add(idx)
        
        improved_count = 0
        
        # Phase A: Remove redundant blocks
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 3.0:
                    return blocks, improved_count
                
                ts_arr = block_ts_indices[i]
                # Check if all t-subsets in this block are covered by >= 2 blocks
                if np.all(cover_count[ts_arr] >= 2):
                    # Remove block i
                    np.subtract.at(cover_count, ts_arr, 1)
                    blocks.pop(i)
                    block_ts_indices.pop(i)
                    idx = get_block_idx(blocks[i] if i < len(blocks) else [0]*k)
                    if idx >= 0:
                        sol_block_set.discard(idx)
                    improved_count += 1
                    changed = True
                    break
        
        # Phase B: Try to replace blocks with better ones
        iters = 0
        while iters < max_iters:
            iters += 1
            if time.time() > deadline - 3.0:
                break
            if len(blocks) == 0:
                break
            
            i = random.randint(0, len(blocks) - 1)
            ts_arr_i = block_ts_indices[i]
            
            # Find t-subsets uniquely covered by block i
            unique_mask = cover_count[ts_arr_i] == 1
            unique_ts = ts_arr_i[unique_mask]
            
            if len(unique_ts) == 0:
                # Block is redundant, remove it
                np.subtract.at(cover_count, ts_arr_i, 1)
                blocks.pop(i)
                block_ts_indices.pop(i)
                improved_count += 1
                continue
            
            # Find candidate blocks that cover all unique_ts
            candidates = set(ts_to_blocks[int(unique_ts[0])])
            for ts_idx in unique_ts[1:]:
                candidates.intersection_update(ts_to_blocks[int(ts_idx)])
                if not candidates:
                    break
            
            if not candidates:
                continue
            
            valid_candidates = [c for c in candidates if c not in sol_block_set]
            if not valid_candidates:
                continue
            
            new_block_idx = random.choice(valid_candidates)
            new_ts_arr = block_to_ts_np[new_block_idx]
            
            # Check if swap is valid: after removing block i and adding new_block,
            # all t-subsets still covered
            # Temporarily remove block i's coverage
            np.subtract.at(cover_count, ts_arr_i, 1)
            # Add new block's coverage
            np.add.at(cover_count, new_ts_arr, 1)
            
            if np.min(cover_count) >= 1:
                # Swap is valid
                blocks[i] = list(all_k_blocks[new_block_idx])
                block_ts_indices[i] = new_ts_arr
                sol_block_set.discard(get_block_idx(blocks[i]))
                sol_block_set.add(new_block_idx)
                improved_count += 1
            else:
                # Revert
                np.subtract.at(cover_count, new_ts_arr, 1)
                np.add.at(cover_count, ts_arr_i, 1)
        
        return blocks, improved_count

    # Phase 2: Iterative local search with restarts
    # Keep track of best solution and try multiple local search passes
    for _ in range(300):
        if time.time() > deadline - 5.0:
            break
        best_solution, improved = fast_local_search(best_solution, max_iters=3000)
        if len(best_solution) < best_len:
            best_len = len(best_solution)
        
        # Occasionally try a fresh local search from a slightly perturbed version
        if improved == 0 and random.random() < 0.3 and time.time() < deadline - 10.0:
            # Try swapping one block randomly and see if it helps
            if len(best_solution) > 2:
                idx_to_remove = random.randint(0, len(best_solution) - 1)
                ts_arr_removed = block_to_ts_np[get_block_idx(best_solution[idx_to_remove])]
                
                # Find a replacement that covers the unique subsets
                cover_count = np.zeros(num_t_subsets, dtype=np.int32)
                for b in best_solution:
                    idx = get_block_idx(b)
                    if idx >= 0:
                        np.add.at(cover_count, block_to_ts_np[idx], 1)
                
                unique_mask = cover_count[ts_arr_removed] == 1
                unique_ts = ts_arr_removed[unique_mask]
                
                if len(unique_ts) == 0:
                    # Just remove it
                    best_solution.pop(idx_to_remove)
                    best_len = len(best_solution)
                else:
                    candidates = set(ts_to_blocks[int(unique_ts[0])])
                    for ts_idx in unique_ts[1:]:
                        candidates.intersection_update(ts_to_blocks[int(ts_idx)])
                        if not candidates:
                            break
                    
                    sol_set = set(get_block_idx(b) for b in best_solution if get_block_idx(b) >= 0)
                    valid_cands = [c for c in candidates if c not in sol_set]
                    if valid_cands:
                        new_idx = random.choice(valid_cands)
                        best_solution[idx_to_remove] = list(all_k_blocks[new_idx])
                        # Check validity
                        cover_count2 = np.zeros(num_t_subsets, dtype=np.int32)
                        for b in best_solution:
                            idx = get_block_idx(b)
                            if idx >= 0:
                                np.add.at(cover_count2, block_to_ts_np[idx], 1)
                        if np.min(cover_count2) >= 1:
                            best_len = len(best_solution)
                        else:
                            # Revert
                            best_solution[idx_to_remove] = list(all_k_blocks[get_block_idx(best_solution[idx_to_remove])])

    return best_solution
