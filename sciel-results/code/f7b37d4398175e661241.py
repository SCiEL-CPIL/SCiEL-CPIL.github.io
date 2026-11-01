import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

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
    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}

    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)

    # block_to_ts: for each block, list of t-subset indices it covers
    block_to_ts = []
    for block in all_k_blocks:
        ts_list = []
        for ts in itertools.combinations(block, t):
            ts_list.append(ts_index_map[ts])
        block_to_ts.append(ts_list)

    # ts_to_blocks: for each t-subset, list of block indices covering it
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    for j, ts_list in enumerate(block_to_ts):
        for ts_idx in ts_list:
            ts_to_blocks[ts_idx].append(j)

    # Precompute as numpy arrays for speed
    block_to_ts_arr = [np.array(ts_list, dtype=np.int32) for ts_list in block_to_ts]

    # Build block tuple to index map for fast lookup
    block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}

    def get_block_idx(block):
        return block_tuple_to_idx.get(tuple(block), -1)

    # Randomized greedy
    def randomized_greedy(seed):
        random.seed(seed)
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []

        while True:
            if time.time() > deadline - 3.0:
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

            for ts_idx in block_to_ts_arr[b_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    for nb_idx in ts_to_blocks[ts_idx]:
                        if not is_selected[nb_idx]:
                            gain[nb_idx] -= 1

        return sol

    # Redundancy elimination: remove any block whose removal keeps all t-subsets covered
    def eliminate_redundancy(selected_blocks):
        """
        selected_blocks: list of block indices
        Returns: reduced list of block indices
        """
        selected = set(selected_blocks)
        n = len(selected)
        
        # Compute cover counts
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in selected:
            for ts_idx in block_to_ts_arr[b_idx]:
                cover_count[ts_idx] += 1
        
        # Iteratively remove redundant blocks
        changed = True
        while changed:
            if time.time() > deadline - 5.0:
                break
            changed = False
            for b_idx in list(selected):
                # Check if removing b_idx leaves all t-subsets covered
                # A t-subset is uncovered after removal iff cover_count[ts] == 1 and b_idx covers ts
                redundant = True
                for ts_idx in block_to_ts_arr[b_idx]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove b_idx
                    selected.discard(b_idx)
                    for ts_idx in block_to_ts_arr[b_idx]:
                        cover_count[ts_idx] -= 1
                    changed = True
        
        return list(selected)

    # 2-exchange local search: remove two blocks, add one block that covers their unique t-subsets
    def local_search_2exchange(selected_blocks):
        """
        selected_blocks: list of block indices
        Returns: improved list of block indices
        """
        selected = set(selected_blocks)
        
        # Compute cover counts
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in selected:
            for ts_idx in block_to_ts_arr[b_idx]:
                cover_count[ts_idx] += 1
        
        improved = True
        while improved:
            if time.time() > deadline - 5.0:
                break
            improved = False
            
            # Try all pairs of blocks to remove
            # For efficiency, only consider pairs where the union of their uniquely-covered t-subsets can be covered by a single block
            # This is expensive, so we sample pairs
            n_selected = len(selected)
            if n_selected < 2:
                break
            
            # Sample pairs
            num_samples = min(500, n_selected * (n_selected - 1) // 2)
            for _ in range(num_samples):
                if time.time() > deadline - 5.0:
                    break
                    
                b1, b2 = random.sample(list(selected), 2)
                
                # Compute the set of t-subsets uniquely covered by b1 or b2
                # i.e., t-subsets where cover_count[ts] == 1 and (b1 covers ts or b2 covers ts)
                unique_ts = set()
                for ts_idx in block_to_ts_arr[b1]:
                    if cover_count[ts_idx] == 1:
                        unique_ts.add(ts_idx)
                for ts_idx in block_to_ts_arr[b2]:
                    if cover_count[ts_idx] == 1:
                        unique_ts.add(ts_idx)
                
                if not unique_ts:
                    # Both blocks are redundant (should have been removed by eliminate_redundancy)
                    # Remove both
                    selected.discard(b1)
                    selected.discard(b2)
                    for ts_idx in block_to_ts_arr[b1]:
                        cover_count[ts_idx] -= 1
                    for ts_idx in block_to_ts_arr[b2]:
                        cover_count[ts_idx] -= 1
                    improved = True
                    break
                
                # Find a block that covers all t-subsets in unique_ts
                # This is a set intersection problem
                # For each candidate block, check if it covers all ts in unique_ts
                # This is expensive, so we limit the search
                found_block = None
                for candidate in range(num_k_blocks):
                    if candidate in selected:
                        continue
                    # Check if candidate covers all ts in unique_ts
                    covers_all = True
                    for ts_idx in unique_ts:
                        if ts_idx not in block_to_ts_arr[candidate]:
                            covers_all = False
                            break
                    if covers_all:
                        found_block = candidate
                        break
                
                if found_block is not None:
                    # Perform the swap
                    selected.discard(b1)
                    selected.discard(b2)
                    for ts_idx in block_to_ts_arr[b1]:
                        cover_count[ts_idx] -= 1
                    for ts_idx in block_to_ts_arr[b2]:
                        cover_count[ts_idx] -= 1
                    selected.add(found_block)
                    for ts_idx in block_to_ts_arr[found_block]:
                        cover_count[ts_idx] += 1
                    improved = True
                    break
        
        return list(selected)

    # Find best initial solution via randomized greedy
    best_initial = None
    best_initial_len = float('inf')

    # Run several randomized greedy to find good starting points
    num_greedy_runs = 20
    for i in range(num_greedy_runs):
        if time.time() > deadline - 30.0:
            break
        sol = randomized_greedy(i)
        if sol is not None and len(sol) < best_initial_len:
            best_initial_len = len(sol)
            best_initial = sol

    if best_initial is None:
        # Fallback: deterministic greedy
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []
        while True:
            if time.time() > deadline - 2.0:
                break
            max_idx = np.argmax(gain)
            if gain[max_idx] <= 0:
                break
            sol.append(max_idx)
            is_selected[max_idx] = True
            gain[max_idx] = -1
            for ts_idx in block_to_ts_arr[max_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    for b_idx in ts_to_blocks[ts_idx]:
                        if not is_selected[b_idx]:
                            gain[b_idx] -= 1
        best_initial = sol
        best_initial_len = len(sol)

    # Apply redundancy elimination
    current_solution = eliminate_redundancy(best_initial)
    
    # Apply 2-exchange local search
    current_solution = local_search_2exchange(current_solution)
    
    # Iterate: eliminate redundancy, then 2-exchange
    while time.time() < deadline - 10.0:
        new_solution = eliminate_redundancy(current_solution)
        if len(new_solution) == len(current_solution):
            new_solution = local_search_2exchange(new_solution)
        if len(new_solution) < len(current_solution):
            current_solution = new_solution
        else:
            # Try a few more 2-exchange rounds
            for _ in range(3):
                if time.time() > deadline - 5.0:
                    break
                new_solution = local_search_2exchange(current_solution)
                if len(new_solution) < len(current_solution):
                    current_solution = new_solution
                    break
            else:
                break

    # Convert to list of blocks
    result = [list(all_k_blocks[b_idx]) for b_idx in current_solution]

    # Ensure distinctness (should be, but just in case)
    seen = set()
    unique_result = []
    for block in result:
        tblock = tuple(block)
        if tblock not in seen:
            seen.add(tblock)
            unique_result.append(block)

    return unique_result
