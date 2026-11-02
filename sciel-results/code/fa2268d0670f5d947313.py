import time
import random
import itertools
import numpy as np
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin for 150s limit

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
    # block_to_ts[j]: numpy array of indices of t-subsets contained in block j
    # ts_to_blocks[i]: list of indices of blocks containing t-subset i
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(np.array(ts_list, dtype=np.int32))

    # Precompute set of block indices for each t-subset as a set for O(1) lookup
    ts_to_blocks_set = [set(lst) for lst in ts_to_blocks]
    
    # State for greedy
    gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
    is_uncovered = np.full(num_t_subsets, True, dtype=bool)
    is_selected = np.full(num_k_blocks, False, dtype=bool)
    
    solution_blocks = []
    
    # Greedy Phase with tie-breaking: prefer blocks with less overlap with already selected
    # Precompute block overlap matrix for tie-breaking (only for first ~500 blocks to save memory)
    # Actually, let's use a simpler approach: track which t-subsets are "critical" for each block
    
    # Better greedy: when multiple blocks have same max gain, prefer one that covers
    # t-subsets that are in fewer other blocks (more "unique" coverage)
    
    # Precompute: for each t-subset, how many blocks contain it (static)
    ts_block_count = np.array([len(ts_to_blocks[i]) for i in range(num_t_subsets)], dtype=np.int32)
    
    # Precompute for each block: sum of 1/ts_block_count for its t-subsets (higher = more unique)
    # This is a soft measure of how "valuable" the block is
    block_uniqueness = np.zeros(num_k_blocks, dtype=np.float64)
    for j in range(num_k_blocks):
        for ts_idx in block_to_ts[j]:
            block_uniqueness[j] += 1.0 / max(ts_block_count[ts_idx], 1)
    
    # Combine gain and uniqueness into a score for tie-breaking
    # Use gain as primary, uniqueness as secondary
    while True:
        if time.time() > deadline - 15.0:
            break
            
        max_gain = int(np.max(gain))
        if max_gain <= 0:
            break
            
        # Find all blocks with max gain
        candidates = np.where((gain == max_gain) & (~is_selected))[0]
        
        if len(candidates) == 1:
            max_idx = int(candidates[0])
        else:
            # Tie-break by uniqueness score
            max_idx = int(candidates[np.argmax(block_uniqueness[candidates])])
            
        solution_blocks.append(list(all_k_blocks[max_idx]))
        is_selected[max_idx] = True
        gain[max_idx] = -1
        
        # Vectorized gain update: for each newly covered t-subset, decrement gain of all blocks containing it
        newly_covered = block_to_ts[max_idx][is_uncovered[block_to_ts[max_idx]]]
        if len(newly_covered) > 0:
            is_uncovered[newly_covered] = False
            # For each newly covered t-subset, find all blocks containing it and decrement their gain
            for ts_idx in newly_covered:
                for b_idx in ts_to_blocks[ts_idx]:
                    if not is_selected[b_idx]:
                        gain[b_idx] -= 1
                    
    best_solution = solution_blocks
    best_len = len(solution_blocks)
    
    # Function to run a randomized greedy with temperature
    def randomized_greedy(seed, temperature=0.0):
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
                
            if temperature > 0:
                # Boltzmann-style selection: weight by gain^temperature
                # Among candidates, pick with probability proportional to 1 (uniform) 
                # but allow some exploration by occasionally picking lower-gain blocks
                if random.random() < 0.1:
                    # Explore: pick from blocks with gain >= max_gain - 2
                    explore_candidates = np.where((gain >= max_gain - 2) & (~is_selected))[0]
                    if len(explore_candidates) > 0:
                        b_idx = int(explore_candidates[random.randint(0, len(explore_candidates) - 1)])
                    else:
                        b_idx = int(candidates[random.randint(0, len(candidates) - 1)])
                else:
                    b_idx = int(candidates[random.randint(0, len(candidates) - 1)])
            else:
                b_idx = int(candidates[random.randint(0, len(candidates) - 1)])
            
            sol.append(list(all_k_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1
            
            # Vectorized gain update
            newly_covered = block_to_ts[b_idx][is_uncovered[block_to_ts[b_idx]]]
            if len(newly_covered) > 0:
                is_uncovered[newly_covered] = False
                for ts_idx in newly_covered:
                    for nb_idx in ts_to_blocks[ts_idx]:
                        if not is_selected[nb_idx]:
                            gain[nb_idx] -= 1
                            
        return sol

    # Run randomized greedy multiple times to find a better starting point
    num_runs = 80
    for i in range(num_runs):
        if time.time() > deadline - 5.0:
            break
            
        temp = 0.0 if i < 40 else 0.5  # First half deterministic-like, second half exploratory
        sol = randomized_greedy(i, temperature=temp)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol
            
    # Local Search: Remove redundant blocks (optimized)
    def optimize_redundancy(blocks):
        if not blocks:
            return blocks
        
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        
        for b in blocks:
            b_idx = get_block_idx(b)
            if b_idx >= 0:
                for ts_idx in block_to_ts[b_idx]:
                    cover_count[ts_idx] += 1
                
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 2.0:
                    return blocks
                
                b_idx = get_block_idx(blocks[i])
                if b_idx < 0:
                    continue
                    
                redundant = True
                for ts_idx in block_to_ts[b_idx]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                        
                if redundant:
                    for ts_idx in block_to_ts[b_idx]:
                        cover_count[ts_idx] -= 1
                    blocks.pop(i)
                    changed = True
                    break
                    
        return blocks

    # Precompute a map from tuple(block) to its index in all_k_blocks for fast lookup
    block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}
    
    def get_block_idx(block):
        return block_tuple_to_idx.get(tuple(block), -1)
    
    def local_search_swap(blocks):
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            b_idx = get_block_idx(b)
            if b_idx >= 0:
                for ts_idx in block_to_ts[b_idx]:
                    cover_count[ts_idx] += 1
                
        # Build set of block indices currently in solution for O(1) lookup
        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)
                
        improved = False
        for i in range(len(blocks)):
            if time.time() > deadline - 1.0:
                return blocks, improved
            
            old_b_idx = get_block_idx(blocks[i])
            if old_b_idx < 0:
                continue
                
            unique_ts = []
            for ts_idx in block_to_ts[old_b_idx]:
                if cover_count[ts_idx] == 1:
                    unique_ts.append(ts_idx)
                    
            if not unique_ts:
                continue
            
            # Find candidate blocks that cover ALL unique_ts
            candidates = ts_to_blocks_set[unique_ts[0]].copy()
            for ts_idx in unique_ts[1:]:
                candidates.intersection_update(ts_to_blocks_set[ts_idx])
                if not candidates:
                    break
            
            if not candidates:
                continue
            
            # Filter out blocks already in solution
            valid_candidates = [c for c in candidates if c not in sol_block_indices]
            if not valid_candidates:
                continue
                
            new_block_idx = random.choice(valid_candidates)
            new_block = list(all_k_blocks[new_block_idx])
            
            old_block = blocks[i]
            
            blocks[i] = new_block
            sol_block_indices.discard(old_b_idx)
            sol_block_indices.add(new_block_idx)
            
            for ts_idx in block_to_ts[old_b_idx]:
                cover_count[ts_idx] -= 1
            for ts_idx in block_to_ts[new_block_idx]:
                cover_count[ts_idx] += 1
            
            improved = True
            
        return blocks, improved

    # 2-exchange local search: try to replace two blocks with one or two different blocks
    def local_search_2exchange(blocks):
        if len(blocks) < 2:
            return blocks, False
            
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            b_idx = get_block_idx(b)
            if b_idx >= 0:
                for ts_idx in block_to_ts[b_idx]:
                    cover_count[ts_idx] += 1
                
        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)
                
        improved = False
        n = len(blocks)
        
        # Try pairs of blocks to remove; see if we can find replacement blocks
        for i in range(n):
            if time.time() > deadline - 1.0:
                return blocks, improved
            for j in range(i + 1, n):
                if time.time() > deadline - 1.0:
                    return blocks, improved
                    
                old_i_idx = get_block_idx(blocks[i])
                old_j_idx = get_block_idx(blocks[j])
                if old_i_idx < 0 or old_j_idx < 0:
                    continue
                    
                # Collect unique t-subsets for blocks[i] and blocks[j]
                unique_ts_i = []
                for ts_idx in block_to_ts[old_i_idx]:
                    if cover_count[ts_idx] == 1:
                        unique_ts_i.append(ts_idx)
                        
                unique_ts_j = []
                for ts_idx in block_to_ts[old_j_idx]:
                    if cover_count[ts_idx] == 1:
                        unique_ts_j.append(ts_idx)
                        
                if not unique_ts_i and not unique_ts_j:
                    continue
                    
                # Try to find a single block covering all unique t-subsets of both
                all_unique = unique_ts_i + unique_ts_j
                if not all_unique:
                    continue
                    
                candidates = ts_to_blocks_set[all_unique[0]].copy()
                for ts_idx in all_unique[1:]:
                    candidates.intersection_update(ts_to_blocks_set[ts_idx])
                    if not candidates:
                        break
                        
                if candidates:
                    valid = [c for c in candidates if c not in sol_block_indices]
                    if valid:
                        new_block_idx = random.choice(valid)
                        new_block = list(all_k_blocks[new_block_idx])
                        
                        # Remove both blocks, add one new block
                        for ts_idx in block_to_ts[old_i_idx]:
                            cover_count[ts_idx] -= 1
                        for ts_idx in block_to_ts[old_j_idx]:
                            cover_count[ts_idx] -= 1
                        for ts_idx in block_to_ts[new_block_idx]:
                            cover_count[ts_idx] += 1
                        
                        sol_block_indices.discard(old_i_idx)
                        sol_block_indices.discard(old_j_idx)
                        sol_block_indices.add(new_block_idx)
                        
                        blocks.pop(j)
                        blocks[i] = new_block
                        improved = True
                        break
                        
                # If single block doesn't work, try two separate blocks
                if unique_ts_i:
                    cand_i = ts_to_blocks_set[unique_ts_i[0]].copy()
                    for ts_idx in unique_ts_i[1:]:
                        cand_i.intersection_update(ts_to_blocks_set[ts_idx])
                        if not cand_i:
                            break
                else:
                    cand_i = set(range(num_k_blocks))
                    
                if unique_ts_j:
                    cand_j = ts_to_blocks_set[unique_ts_j[0]].copy()
                    for ts_idx in unique_ts_j[1:]:
                        cand_j.intersection_update(ts_to_blocks_set[ts_idx])
                        if not cand_j:
                            break
                else:
                    cand_j = set(range(num_k_blocks))
                    
                valid_i = [c for c in cand_i if c not in sol_block_indices]
                valid_j = [c for c in cand_j if c not in sol_block_indices and c not in cand_i]
                
                if valid_i and valid_j:
                    new_i_idx = random.choice(valid_i)
                    new_j_idx = random.choice(valid_j)
                    new_i = list(all_k_blocks[new_i_idx])
                    new_j = list(all_k_blocks[new_j_idx])
                    
                    for ts_idx in block_to_ts[old_i_idx]:
                        cover_count[ts_idx] -= 1
                    for ts_idx in block_to_ts[old_j_idx]:
                        cover_count[ts_idx] -= 1
                    for ts_idx in block_to_ts[new_i_idx]:
                        cover_count[ts_idx] += 1
                    for ts_idx in block_to_ts[new_j_idx]:
                        cover_count[ts_idx] += 1
                        
                    sol_block_indices.discard(old_i_idx)
                    sol_block_indices.discard(old_j_idx)
                    sol_block_indices.add(new_i_idx)
                    sol_block_indices.add(new_j_idx)
                    
                    blocks[i] = new_i
                    blocks[j] = new_j
                    improved = True
                    break
            if improved:
                break
                
        return blocks, improved

    # Simulated Annealing style polish: allow temporary worsening to escape local optima
    def sa_polish(blocks, max_iters=5000):
        if not blocks:
            return blocks
            
        # Build initial cover count
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)
                for ts_idx in block_to_ts[idx]:
                    cover_count[ts_idx] += 1
        
        best_len = len(blocks)
        best_blocks = list(blocks)
        current_len = len(blocks)
        
        T = 100.0  # Initial temperature
        T_min = 1.0
        cooling = 0.995
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
                
            # Find uncovered t-subsets
            uncovered = np.where(cover_count == 0)[0]
            
            # Objective: minimize (uncovered_count * 1000 + block_count)
            # We prefer covering everything, then minimize blocks
            current_obj = len(uncovered) * 1000 + current_len
            
            # Choose a move
            if len(uncovered) > 0 and random.random() < 0.6:
                # Cover an uncovered t-subset
                ts_idx = int(uncovered[random.randint(0, len(uncovered) - 1)])
                candidate_blocks = ts_to_blocks[ts_idx]
                if not candidate_blocks:
                    continue
                
                # Choose between swap or add
                if random.random() < 0.5 and sol_block_indices:
                    # Swap
                    existing_list = list(sol_block_indices)
                    old_idx = existing_list[random.randint(0, len(existing_list) - 1)]
                    
                    valid_candidates = [c for c in candidate_blocks if c not in sol_block_indices]
                    if not valid_candidates:
                        continue
                    
                    new_idx = valid_candidates[random.randint(0, len(valid_candidates) - 1)]
                    
                    # Compute delta
                    old_covered = block_to_ts[old_idx]
                    new_covered = block_to_ts[new_idx]
                    
                    # Count how many t-subsets would become uncovered or newly covered
                    delta_uncovered = 0
                    for ts_idx2 in old_covered:
                        if cover_count[ts_idx2] == 1:
                            delta_uncovered += 1
                    for ts_idx2 in new_covered:
                        if cover_count[ts_idx2] == 0:
                            delta_uncovered -= 1
                    
                    delta_obj = delta_uncovered * 1000  # Block count unchanged
                    
                    # Accept or reject
                    if delta_obj <= 0 or random.random() < np.exp(-delta_obj / T):
                        for ts_idx2 in old_covered:
                            cover_count[ts_idx2] -= 1
                        for ts_idx2 in new_covered:
                            cover_count[ts_idx2] += 1
                        sol_block_indices.discard(old_idx)
                        sol_block_indices.add(new_idx)
                else:
                    # Add
                    valid_candidates = [c for c in candidate_blocks if c not in sol_block_indices]
                    if not valid_candidates:
                        continue
                    
                    new_idx = valid_candidates[random.randint(0, len(valid_candidates) - 1)]
                    new_covered = block_to_ts[new_idx]
                    
                    delta_uncovered = 0
                    for ts_idx2 in new_covered:
                        if cover_count[ts_idx2] == 0:
                            delta_uncovered -= 1
                    
                    delta_obj = delta_uncovered * 1000 + 1  # Block count increases by 1
                    
                    if delta_obj <= 0 or random.random() < np.exp(-delta_obj / T):
                        for ts_idx2 in new_covered:
                            cover_count[ts_idx2] += 1
                        sol_block_indices.add(new_idx)
                        current_len += 1
            else:
                # Try to remove a redundant block or swap to reduce block count
                if not sol_block_indices:
                    continue
                    
                # Find blocks that are "less critical" (cover fewer uniquely-covered t-subsets)
                block_criticality = {}
                for b_idx in sol_block_indices:
                    crit = 0
                    for ts_idx2 in block_to_ts[b_idx]:
                        if cover_count[ts_idx2] == 1:
                            crit += 1
                    block_criticality[b_idx] = crit
                
                # Pick a block with low criticality
                sorted_blocks = sorted(block_criticality.items(), key=lambda x: x[1])
                if sorted_blocks[0][1] == 0:
                    # Can remove without uncovering anything
                    old_idx = sorted_blocks[0][0]
                    old_covered = block_to_ts[old_idx]
                    
                    for ts_idx2 in old_covered:
                        cover_count[ts_idx2] -= 1
                    sol_block_indices.discard(old_idx)
                    current_len -= 1
                elif len(sorted_blocks) > 1:
                    # Swap: remove a low-criticality block, add a block that covers its unique t-subsets
                    old_idx = sorted_blocks[0][0]
                    old_covered = block_to_ts[old_idx]
                    unique_ts = [ts_idx2 for ts_idx2 in old_covered if cover_count[ts_idx2] == 1]
                    
                    if unique_ts:
                        candidates = ts_to_blocks_set[unique_ts[0]].copy()
                        for ts_idx2 in unique_ts[1:]:
                            candidates.intersection_update(ts_to_blocks_set[ts_idx2])
                            if not candidates:
                                break
                        
                        valid = [c for c in candidates if c not in sol_block_indices]
                        if valid:
                            new_idx = random.choice(valid)
                            new_covered = block_to_ts[new_idx]
                            
                            for ts_idx2 in old_covered:
                                cover_count[ts_idx2] -= 1
                            for ts_idx2 in new_covered:
                                cover_count[ts_idx2] += 1
                            sol_block_indices.discard(old_idx)
                            sol_block_indices.add(new_idx)
                    
            # Rebuild blocks list from sorted indices for determinism
            sorted_indices = sorted(sol_block_indices)
            blocks = [list(all_k_blocks[idx]) for idx in sorted_indices]
            
            # Try to remove redundant blocks
            blocks = optimize_redundancy(blocks)
            current_len = len(blocks)
            
            # Update best
            if len(uncovered) == 0 and current_len < best_len:
                best_len = current_len
                best_blocks = list(blocks)
            
            # Cool down
            T = max(T_min, T * cooling)
        
        return best_blocks

    # Run local search for a few iterations
    for _ in range(30):
        if time.time() > deadline - 8.0:
            break
        best_solution, improved = local_search_swap(best_solution)
        if improved:
            best_solution = optimize_redundancy(best_solution)
            if len(best_solution) < best_len:
                best_len = len(best_solution)
        else:
            # Try 2-exchange
            best_solution, improved2 = local_search_2exchange(best_solution)
            if improved2:
                best_solution = optimize_redundancy(best_solution)
                if len(best_solution) < best_len:
                    best_len = len(best_solution)
            else:
                break

    # Final SA polish
    if time.time() < deadline - 10.0:
        polished = sa_polish(best_solution, max_iters=3000)
        if len(polished) < best_len:
            best_solution = polished
            best_len = len(best_solution)
    
    # Final high-precision polish: aggressive redundancy removal and swap search
    if time.time() < deadline - 3.0:
        # Multiple passes of redundancy removal
        for _ in range(5):
            if time.time() > deadline - 2.0:
                break
            new_sol = optimize_redundancy(best_solution)
            if len(new_sol) < len(best_solution):
                best_solution = new_sol
                best_len = len(best_solution)
        
        # Final swap pass
        if time.time() < deadline - 2.0:
            for _ in range(10):
                if time.time() > deadline - 1.0:
                    break
                best_solution, improved = local_search_swap(best_solution)
                if improved:
                    best_solution = optimize_redundancy(best_solution)
                    if len(best_solution) < best_len:
                        best_len = len(best_solution)
                else:
                    break

    return best_solution
