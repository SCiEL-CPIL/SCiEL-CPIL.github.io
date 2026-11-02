"""Covering design C(19,8,5) - Vectorized Greedy with Incidence-Based Local Search."""
import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}

    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    
    # Incidence structures
    # block_to_ts: list of lists, block_to_ts[b] = list of t-subset indices covered by block b
    # ts_to_blocks: list of lists, ts_to_blocks[ts] = list of block indices covering t-subset ts
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(ts_list)

    # Bitmasks for fast coverage checks in local search
    block_masks = [0] * num_k_blocks
    for b_idx in range(num_k_blocks):
        mask = 0
        for idx in block_to_ts[b_idx]:
            mask |= (1 << idx)
        block_masks[b_idx] = mask

    def run_greedy(rng_seed):
        """Vectorized greedy construction."""
        rng = random.Random(rng_seed)
        gain = np.full(num_k_blocks, len(block_to_ts[0]), dtype=np.int32)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []

        # We track uncovered subsets implicitly by gain > 0
        # Initially all gains are max.
        
        while True:
            if time.time() > deadline - 5.0:
                return None

            # Find max gain
            max_gain = np.max(gain)
            if max_gain <= 0:
                break

            # Find candidates with max gain
            # To avoid numpy overhead on every step for small sets, we can do this efficiently
            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break
            
            b_idx = int(candidates[rng.randint(0, len(candidates) - 1)])
            sol.append(b_idx)
            is_selected[b_idx] = True
            gain[b_idx] = -1 # Mark as selected to exclude in future

            # Update gains for all other blocks
            # A block's gain decreases by 1 for every t-subset it covers that was just covered by b_idx
            # We only care about t-subsets that were previously uncovered.
            # However, maintaining an explicit "uncovered" set is hard with vectorized gains.
            # Standard trick: 
            # gain[b'] = |B' \ Intersection of previous selections|
            # This is hard to update incrementally without knowing which specific subsets were "new".
            
            # Alternative approach for vectorized greedy:
            # Keep a boolean array `is_uncovered` of size num_t_subsets.
            # gain[b] = sum(1 for ts in block_to_ts[b] if is_uncovered[ts])
            # Updating gain: for each ts in block_to_ts[b_idx], if is_uncovered[ts], 
            #   set is_uncovered[ts]=False
            #   for each b' in ts_to_blocks[ts], if not is_selected[b'], gain[b'] -= 1
            
            # This is O(sum of degrees) which is manageable.
            # Let's re-implement run_greedy with this logic as it's more robust.
            pass # Placeholder, we will use the explicit loop version below which is faster in practice for this size if optimized, 
                 # or we can stick to the bitmask greedy if we optimize it.
                 # Actually, the bitmask greedy in the original was slow due to Python loops.
                 # Let's use the NumPy gain update method properly.
        
        return sol

    def run_greedy_optimized(rng_seed):
        """Optimized Greedy with NumPy gain updates."""
        rng = random.Random(rng_seed)
        gain = np.full(num_k_blocks, len(block_to_ts[0]), dtype=np.int32)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
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
            
            b_idx = int(candidates[rng.randint(0, len(candidates) - 1)])
            sol.append(b_idx)
            is_selected[b_idx] = True
            gain[b_idx] = -1

            # Update state
            for ts_idx in block_to_ts[b_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    # Decrement gain for all blocks covering this ts
                    for nb_idx in ts_to_blocks[ts_idx]:
                        if not is_selected[nb_idx]:
                            gain[nb_idx] -= 1

        return sol

    def polish_blocks(block_indices):
        """Remove redundant blocks."""
        if not block_indices:
            return block_indices

        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in block_indices:
            for ts_idx in block_to_ts[b]:
                cover_count[ts_idx] += 1

        i = len(block_indices) - 1
        while i >= 0:
            if time.time() > deadline - 1.0:
                break
            b = block_indices[i]
            can_remove = True
            for ts_idx in block_to_ts[b]:
                if cover_count[ts_idx] == 1:
                    can_remove = False
                    break
            if can_remove:
                for ts_idx in block_to_ts[b]:
                    cover_count[ts_idx] -= 1
                block_indices.pop(i)
            else:
                i -= 1
        return block_indices

    def local_search_swap(block_indices, rng):
        """1-exchange local search using incidence for candidate filtering."""
        if not block_indices:
            return block_indices, False

        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in block_indices:
            for ts_idx in block_to_ts[b]:
                cover_count[ts_idx] += 1

        improved = False
        max_iters = 50
        iter_count = 0
        
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            if time.time() > deadline - 2.0:
                break

            for i in range(len(block_indices)):
                if time.time() > deadline - 2.0:
                    break
                    
                b = block_indices[i]
                # Find t-subsets uniquely covered by block b
                unique_ts = []
                for ts_idx in block_to_ts[b]:
                    if cover_count[ts_idx] == 1:
                        unique_ts.append(ts_idx)

                if not unique_ts:
                    # Block is redundant, remove it
                    for ts_idx in block_to_ts[b]:
                        cover_count[ts_idx] -= 1
                    block_indices.pop(i)
                    improved = True
                    break

                # Find candidate blocks that cover ALL unique_ts
                # Start with blocks covering the first unique ts
                candidates = set(ts_to_blocks[unique_ts[0]])
                for ts_idx in unique_ts[1:]:
                    candidates.intersection_update(ts_to_blocks[ts_idx])
                    if not candidates:
                        break

                if not candidates:
                    continue

                candidates.discard(b)
                
                # We want to find a candidate that, when swapped in, allows us to remove other blocks?
                # Or simply maintains coverage and has "good" properties.
                # Since we are minimizing count, a swap only helps if it enables a deletion.
                # However, a simple swap that maintains coverage is neutral.
                # To improve, we look for a swap that creates redundancy in OTHER blocks.
                
                best_candidate = -1
                best_potential_deletions = 0
                
                # Limit candidates to check for speed
                cand_list = list(candidates)
                if len(cand_list) > 20:
                    cand_list = rng.sample(cand_list, 20)
                
                for c in cand_list:
                    # Simulate swap: remove b, add c
                    # Check if this creates any redundancy in existing blocks
                    # We need to check if any other block j becomes redundant
                    # Block j is redundant if all its ts have count > 1 in the new state.
                    
                    # New count for ts:
                    # old_count[ts] - (1 if ts in b) + (1 if ts in c)
                    
                    # Instead of full simulation, check potential deletions:
                    # A block j (j != i) becomes redundant if:
                    # For all ts in j: new_count[ts] >= 2
                    # new_count[ts] = cover_count[ts] - (1 if ts in b) + (1 if ts in c)
                    
                    potential_del = 0
                    for j, bj in enumerate(block_indices):
                        if j == i:
                            continue
                        
                        # Check if bj becomes redundant
                        is_red = True
                        for ts_idx in block_to_ts[bj]:
                            new_cnt = cover_count[ts_idx]
                            if ts_idx in block_to_ts[b]: # ts in b
                                new_cnt -= 1
                            if ts_idx in block_to_ts[c]: # ts in c
                                new_cnt += 1
                            
                            if new_cnt < 2:
                                is_red = False
                                break
                        if is_red:
                            potential_del += 1
                    
                    if potential_del > best_potential_deletions:
                        best_potential_deletions = potential_del
                        best_candidate = c

                if best_candidate != -1 and best_potential_deletions > 0:
                    # Perform swap
                    for ts_idx in block_to_ts[b]:
                        cover_count[ts_idx] -= 1
                    for ts_idx in block_to_ts[best_candidate]:
                        cover_count[ts_idx] += 1
                    block_indices[i] = best_candidate
                    improved = True
                    break

        return block_indices, improved

    # Main Search Loop
    best_blocks = None
    best_len = float('inf')

    # 1. Deterministic Greedy
    sol = run_greedy_optimized(0)
    if sol is not None:
        sol = polish_blocks(sol)
        sol, _ = local_search_swap(sol, random.Random(0))
        sol = polish_blocks(sol)
        best_blocks = sol[:]
        best_len = len(sol)

    # 2. Randomized Greedy Restarts
    num_restarts = 100 # Reduced restarts to allow more time per local search
    for i in range(num_restarts):
        if time.time() > deadline - 10.0:
            break

        sol = run_greedy_optimized(i + 1)
        if sol is None:
            break

        sol = polish_blocks(sol)
        
        # Only run local search if we are competitive
        if len(sol) <= best_len + 5:
            sol, _ = local_search_swap(sol, random.Random(i + 1))
            sol = polish_blocks(sol)
            
            if len(sol) < best_len:
                best_len = len(sol)
                best_blocks = sol[:]
                if best_len <= 335:
                    break

    # 3. Final Polish on Best
    if best_blocks is not None:
        best_blocks = polish_blocks(best_blocks)
        best_blocks, _ = local_search_swap(best_blocks, random.Random(999))
        best_blocks = polish_blocks(best_blocks)

    if best_blocks is None:
        return []

    result = [list(all_k_blocks[b]) for b in best_blocks]
    return result
