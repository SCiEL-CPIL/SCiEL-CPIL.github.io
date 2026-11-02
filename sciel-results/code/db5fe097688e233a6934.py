import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0

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

    if v != 19 or k != 8 or t != 5:
        # Fallback
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            target = min(uncovered)
            best_block = None
            best_gain = -1
            remaining = [x for x in range(v) if x not in target]
            need = k - t
            for comb in itertools.combinations(remaining, need):
                block = tuple(sorted(target + comb))
                gain = sum(1 for sub in itertools.combinations(block, t) if sub in uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            if best_block is None:
                break
            blocks.append(list(best_block))
            uncovered.difference_update(itertools.combinations(best_block, t))
            if time.time() > deadline:
                break
        return blocks

    points = range(v)
    all_t_subsets = list(itertools.combinations(points, t))
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    n_t = len(all_t_subsets)
    
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(n_t)]
    
    for j, block in enumerate(all_blocks):
        covered_indices = []
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            covered_indices.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(covered_indices)
        
    block_ts_arrays = [np.array(lst, dtype=np.int32) for lst in block_to_ts]
    # Precompute block ts sets for faster membership tests
    block_ts_sets = [set(lst) for lst in block_to_ts]

    def run_greedy(seed):
        random.seed(seed)
        gain = np.full(n_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(n_t, True, dtype=bool)
        is_selected = np.full(n_blocks, False, dtype=bool)
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
            
            sol.append(list(all_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1
            
            newly_covered_ts = []
            for ts_idx in block_to_ts[b_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    newly_covered_ts.append(ts_idx)
            
            if not newly_covered_ts:
                continue
                
            blocks_to_decrement = []
            for ts_idx in newly_covered_ts:
                blocks_to_decrement.extend(ts_to_blocks[ts_idx])
                
            if blocks_to_decrement:
                counts = np.bincount(blocks_to_decrement, minlength=n_blocks)
                gain -= counts
                gain[is_selected] = -1000

        return sol

    def get_block_indices(blocks):
        indices = []
        for b in blocks:
            indices.append(block_to_idx[tuple(b)])
        return indices

    def remove_single_redundancy(blocks):
        if not blocks:
            return blocks
        indices = get_block_indices(blocks)
        cover_count = np.zeros(n_t, dtype=np.int32)
        block_ts_lists = [block_to_ts[i] for i in indices]
        
        for ts_list in block_ts_lists:
            for ts_idx in ts_list:
                cover_count[ts_idx] += 1
                
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 2.0:
                    return blocks
                redundant = True
                for ts_idx in block_ts_lists[i]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    blocks.pop(i)
                    for ts_idx in block_ts_lists[i]:
                        cover_count[ts_idx] -= 1
                    block_ts_lists.pop(i)
                    changed = True
                    break
        return blocks

    def remove_pair_redundancy(blocks):
        if len(blocks) < 2:
            return blocks
            
        indices = get_block_indices(blocks)
        cover_count = np.zeros(n_t, dtype=np.int32)
        block_ts_lists = [block_to_ts[i] for i in indices]
        # Use precomputed sets for fast membership
        block_ts_sets = [block_ts_sets[i] for i in indices]
        
        for ts_list in block_ts_lists:
            for ts_idx in ts_list:
                cover_count[ts_idx] += 1
                
        n = len(blocks)
        changed = True
        while changed:
            changed = False
            n = len(blocks)
            for i in range(n):
                if time.time() > deadline - 2.0:
                    return blocks
                for j in range(i + 1, n):
                    safe = True
                    # Check ts in block i
                    for ts_idx in block_ts_lists[i]:
                        if ts_idx in block_ts_sets[j]:
                            if cover_count[ts_idx] == 2:
                                safe = False
                                break
                        else:
                            if cover_count[ts_idx] == 1:
                                safe = False
                                break
                    if not safe:
                        continue
                    
                    # Check ts in block j not in i
                    for ts_idx in block_ts_lists[j]:
                        if ts_idx not in block_ts_sets[i]:
                            if cover_count[ts_idx] == 1:
                                safe = False
                                break
                    if not safe:
                        continue
                    
                    # Remove i and j
                    for ts_idx in block_ts_lists[j]:
                        cover_count[ts_idx] -= 1
                    for ts_idx in block_ts_lists[i]:
                        cover_count[ts_idx] -= 1
                    
                    blocks.pop(j)
                    block_ts_lists.pop(j)
                    block_ts_sets.pop(j)
                    blocks.pop(i)
                    block_ts_lists.pop(i)
                    block_ts_sets.pop(i)
                    
                    changed = True
                    break
                if changed:
                    break
        return blocks

    def local_search_swap(blocks, seed):
        """
        Structural change: Add 1-swap local search to escape local minima of greedy.
        Try to replace a block with another that covers all its critical t-subsets.
        """
        if not blocks:
            return blocks
        
        rng = random.Random(seed)
        indices = get_block_indices(blocks)
        cover_count = np.zeros(n_t, dtype=np.int32)
        block_ts_lists = [block_to_ts[i] for i in indices]
        
        for ts_list in block_ts_lists:
            for ts_idx in ts_list:
                cover_count[ts_idx] += 1
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            n = len(blocks)
            # Shuffle indices to avoid bias
            order = list(range(n))
            rng.shuffle(order)
            
            for i in order:
                if time.time() > deadline - 2.0:
                    return blocks
                bi = indices[i]
                ts_idxs = block_ts_lists[i]
                
                # Check if block i is redundant
                redundant = True
                for ts_idx in ts_idxs:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove it
                    for ts_idx in ts_idxs:
                        cover_count[ts_idx] -= 1
                    blocks.pop(i)
                    indices.pop(i)
                    block_ts_lists.pop(i)
                    improved = True
                    break
                
                # Try to swap block i with another block
                # Find critical t-subsets (covered only by block i)
                critical_ts = [ts_idx for ts_idx in ts_idxs if cover_count[ts_idx] == 1]
                
                if not critical_ts:
                    continue
                
                # Pick a random critical t-subset to try to cover
                target_ts = critical_ts[rng.randrange(len(critical_ts))]
                
                # Find candidate blocks that cover target_ts
                candidates = ts_to_blocks[target_ts]
                # Filter out already selected blocks
                selected_set = set(indices)
                candidates = [c for c in candidates if c not in selected_set]
                
                if not candidates:
                    continue
                
                # Shuffle candidates
                rng.shuffle(candidates)
                
                for c in candidates:
                    # Check if candidate covers all critical_ts of block i
                    c_ts_set = block_ts_sets[c]
                    if all(ts in c_ts_set for ts in critical_ts):
                        # Swap
                        # Remove i
                        for ts_idx in ts_idxs:
                            cover_count[ts_idx] -= 1
                        # Add c
                        c_ts_list = block_to_ts[c]
                        for ts_idx in c_ts_list:
                            cover_count[ts_idx] += 1
                        
                        indices[i] = c
                        block_ts_lists[i] = c_ts_list
                        improved = True
                        break
                if improved:
                    break
        return blocks

    best_blocks = None
    best_len = float('inf')
    num_restarts = 200
    
    for i in range(num_restarts):
        if time.time() > deadline - 25.0:
            break
            
        sol = run_greedy(i)
        if sol is None:
            continue
        
        sol = remove_single_redundancy(sol)
        sol = remove_pair_redundancy(sol)
        sol = remove_single_redundancy(sol)
        
        # Apply local search swap
        sol = local_search_swap(sol, seed=i)
        sol = remove_single_redundancy(sol)
        sol = remove_pair_redundancy(sol)
        
        if len(sol) < best_len:
            best_len = len(sol)
            best_blocks = [list(b) for b in sol]
            
        if best_len <= 264:
            break

    if best_blocks is None:
        best_blocks = run_greedy(0)
        if best_blocks is None:
            return []
        best_blocks = remove_single_redundancy(best_blocks)

    return best_blocks
