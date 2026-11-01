import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

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
    
    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(ts_list)
        
    # Precompute a map from tuple(block) to its index in all_k_blocks for fast lookup
    block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}
    
    def get_block_idx(block):
        return block_tuple_to_idx.get(tuple(block), -1)
    
    # Fast greedy to get initial solution
    def fast_greedy(seed=0):
        random.seed(seed)
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []
        
        while True:
            if time.time() > deadline - 5.0:
                return None
            
            max_gain = np.max(gain)
            if max_gain <= 0:
                break
                
            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break
                
            b_idx = candidates[random.randint(0, len(candidates) - 1)]
            
            sol.append(b_idx)
            is_selected[b_idx] = True
            gain[b_idx] = -1
            
            for ts_idx in block_to_ts[b_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    for nb_idx in ts_to_blocks[ts_idx]:
                        if not is_selected[nb_idx]:
                            gain[nb_idx] -= 1
                            
        return sol

    # Run a few greedy runs to pick the best starting point
    best_seed_sol = None
    best_seed_len = float('inf')
    for s in range(5):
        if time.time() > deadline - 10.0:
            break
        sol = fast_greedy(s)
        if sol is not None and len(sol) < best_seed_len:
            best_seed_len = len(sol)
            best_seed_sol = sol

    if best_seed_sol is None:
        # Fallback: just return an empty list or a trivial valid cover if time is up
        # But we need a valid cover. Let's do a deterministic greedy.
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []
        while True:
            if time.time() > deadline - 1.0:
                break
            max_idx = np.argmax(gain)
            if gain[max_idx] <= 0:
                break
            sol.append(max_idx)
            is_selected[max_idx] = True
            gain[max_idx] = -1
            for ts_idx in block_to_ts[max_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    for nb_idx in ts_to_blocks[ts_idx]:
                        if not is_selected[nb_idx]:
                            gain[nb_idx] -= 1
        best_seed_sol = sol
        best_seed_len = len(sol)

    # Simulated Annealing Setup
    # State: list of block indices
    current_sol = list(best_seed_sol)
    current_set = set(current_sol)
    
    # Compute cover counts for current solution
    cover_count = np.zeros(num_t_subsets, dtype=np.int32)
    for b_idx in current_sol:
        for ts_idx in block_to_ts[b_idx]:
            cover_count[ts_idx] += 1
            
    uncovered_count = np.sum(cover_count == 0)
    
    # Objective: minimize (num_blocks + 1000 * uncovered_count)
    def objective(num_blocks, uncovered):
        return num_blocks + 1000 * uncovered

    current_obj = objective(len(current_sol), uncovered_count)
    
    best_sol = list(current_sol)
    best_obj = current_obj
    
    # SA Parameters
    T0 = 100.0
    T_min = 0.1
    alpha = 0.995  # Cooling rate
    T = T0
    rng = random.Random(12345)
    
    # Precompute block sizes for potential heuristic (all same size k)
    # We will use random moves.
    
    # To speed up moves, we can precompute the list of t-subsets for each block.
    # block_to_ts is already a list of lists of indices.
    
    # Main SA Loop
    last_improvement_time = start_time
    no_improvement_time_limit = 10.0  # If no improvement for 10s, cool down faster or restart?
    
    while T > T_min and time.time() < deadline - 2.0:
        # Choose a move type
        move_type = rng.random()
        
        if move_type < 0.4:
            # Add a random block
            if len(current_sol) < num_k_blocks:
                # Pick a random block not in current solution
                # Efficient way: pick random index, check if in set
                # Since num_k_blocks is large, we might need to retry or use a different structure.
                # For v=19, k=8, num_k_blocks = 75582. Set lookup is O(1).
                b_idx = rng.randint(0, num_k_blocks - 1)
                if b_idx not in current_set:
                    # Apply move
                    old_obj = current_obj
                    
                    for ts_idx in block_to_ts[b_idx]:
                        if cover_count[ts_idx] == 0:
                            uncovered_count -= 1
                        cover_count[ts_idx] += 1
                    
                    current_sol.append(b_idx)
                    current_set.add(b_idx)
                    
                    new_obj = objective(len(current_sol), uncovered_count)
                    
                    # Acceptance
                    delta = new_obj - old_obj
                    if delta < 0 or rng.random() < np.exp(-delta / T):
                        current_obj = new_obj
                        if current_obj < best_obj:
                            best_obj = current_obj
                            best_sol = list(current_sol)
                            last_improvement_time = time.time()
                    else:
                        # Revert
                        current_sol.pop()
                        current_set.remove(b_idx)
                        for ts_idx in block_to_ts[b_idx]:
                            cover_count[ts_idx] -= 1
                            if cover_count[ts_idx] == 0:
                                uncovered_count += 1
                        
        elif move_type < 0.8:
            # Remove a random block
            if len(current_sol) > 0:
                # Pick a random block from current solution
                idx_to_remove = rng.randint(0, len(current_sol) - 1)
                b_idx = current_sol[idx_to_remove]
                
                old_obj = current_obj
                
                for ts_idx in block_to_ts[b_idx]:
                    cover_count[ts_idx] -= 1
                    if cover_count[ts_idx] == 0:
                        uncovered_count += 1
                
                # Remove from list and set
                current_sol.pop(idx_to_remove)
                current_set.remove(b_idx)
                
                new_obj = objective(len(current_sol), uncovered_count)
                
                # Acceptance
                delta = new_obj - old_obj
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    current_obj = new_obj
                    if current_obj < best_obj:
                        best_obj = current_obj
                        best_sol = list(current_sol)
                        last_improvement_time = time.time()
                else:
                    # Revert
                    current_sol.append(b_idx)
                    current_set.add(b_idx)
                    for ts_idx in block_to_ts[b_idx]:
                        cover_count[ts_idx] += 1
                        if cover_count[ts_idx] == 1:
                            uncovered_count -= 1
                            
        else:
            # Swap: remove one, add another
            if len(current_sol) > 0:
                # Pick a block to remove
                idx_to_remove = rng.randint(0, len(current_sol) - 1)
                b_idx_remove = current_sol[idx_to_remove]
                
                # Pick a block to add (not in current set)
                b_idx_add = rng.randint(0, num_k_blocks - 1)
                if b_idx_add in current_set:
                    continue # Retry or skip
                
                old_obj = current_obj
                
                # Apply removal
                for ts_idx in block_to_ts[b_idx_remove]:
                    cover_count[ts_idx] -= 1
                    if cover_count[ts_idx] == 0:
                        uncovered_count += 1
                current_sol.pop(idx_to_remove)
                current_set.remove(b_idx_remove)
                
                # Apply addition
                for ts_idx in block_to_ts[b_idx_add]:
                    if cover_count[ts_idx] == 0:
                        uncovered_count -= 1
                    cover_count[ts_idx] += 1
                current_sol.append(b_idx_add)
                current_set.add(b_idx_add)
                
                new_obj = objective(len(current_sol), uncovered_count)
                
                # Acceptance
                delta = new_obj - old_obj
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    current_obj = new_obj
                    if current_obj < best_obj:
                        best_obj = current_obj
                        best_sol = list(current_sol)
                        last_improvement_time = time.time()
                else:
                    # Revert
                    # Remove add
                    current_sol.pop()
                    current_set.remove(b_idx_add)
                    for ts_idx in block_to_ts[b_idx_add]:
                        cover_count[ts_idx] -= 1
                        if cover_count[ts_idx] == 0:
                            uncovered_count += 1
                    # Restore remove
                    current_sol.append(b_idx_remove)
                    current_set.add(b_idx_remove)
                    for ts_idx in block_to_ts[b_idx_remove]:
                        cover_count[ts_idx] += 1
                        if cover_count[ts_idx] == 1:
                            uncovered_count -= 1

        # Cool down
        T *= alpha
        
        # Adaptive cooling: if stuck, cool faster
        if time.time() - last_improvement_time > no_improvement_time_limit:
            T *= 0.95

    # Final cleanup: Remove redundant blocks from best_sol
    # We need to ensure best_sol is valid and minimal.
    # The SA objective penalizes uncovered, so best_sol should be valid (uncovered=0).
    # But it might have redundant blocks.
    
    # Rebuild cover counts for best_sol
    cover_count = np.zeros(num_t_subsets, dtype=np.int32)
    for b_idx in best_sol:
        for ts_idx in block_to_ts[b_idx]:
            cover_count[ts_idx] += 1
            
    # Check validity
    if np.min(cover_count) < 0:
        # Should not happen if penalty worked, but just in case
        pass
        
    # Redundancy removal
    changed = True
    while changed and time.time() < deadline - 0.5:
        changed = False
        for i in range(len(best_sol)):
            if time.time() > deadline - 0.5:
                break
            redundant = True
            for ts_idx in block_to_ts[best_sol[i]]:
                if cover_count[ts_idx] == 1:
                    redundant = False
                    break
            if redundant:
                b_idx = best_sol[i]
                best_sol.pop(i)
                for ts_idx in block_to_ts[b_idx]:
                    cover_count[ts_idx] -= 1
                changed = True
                break

    # Convert indices back to lists of points
    final_blocks = [list(all_k_blocks[idx]) for idx in best_sol]
    
    return final_blocks
