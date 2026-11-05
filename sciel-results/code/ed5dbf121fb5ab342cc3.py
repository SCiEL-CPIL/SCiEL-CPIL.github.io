import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0

    # Precompute all blocks and t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    all_t_subs = list(itertools.combinations(range(v), t))
    num_t_subs = len(all_t_subs)

    # Map each t-subset to a unique index
    t_sub_index = {ts: i for i, ts in enumerate(all_t_subs)}

    # For each block, precompute the list of t-subset indices it covers
    block_t_indices = []
    for block in all_blocks:
        idxs = [t_sub_index[ts] for ts in itertools.combinations(block, t)]
        block_t_indices.append(np.array(idxs, dtype=np.int32))

    # Build a reverse index: for each t-subset, which blocks cover it
    # This is expensive to build fully, so we'll use a sampling approach in local search

    def time_left():
        return deadline - time.time()

    def greedy_construct(seed_val, use_weighted=True):
        """
        Greedy construction with optional weighted scoring.
        Weighted: score = gain / (frequency of elements in block + 1)
        """
        random.seed(seed_val)
        counts = np.zeros(num_t_subs, dtype=np.int32)
        uncovered_count = num_t_subs
        selected_blocks = []

        # Precompute element frequencies across all blocks for weighting
        if use_weighted:
            elem_freq = np.zeros(v, dtype=np.int32)
            for b in all_blocks:
                for e in b:
                    elem_freq[e] += 1
            # Normalize: lower freq = higher weight (rarity)
            # weight = 1 / (freq + 1)
            block_weights = np.array([1.0 / (sum(elem_freq[e] for e in b) + 1.0) for b in all_blocks], dtype=np.float64)
        else:
            block_weights = np.ones(num_blocks, dtype=np.float64)

        # Track which blocks are already selected
        selected_set = set()

        while uncovered_count > 0 and time_left() > 10.0:
            # Compute gain for each block: number of uncovered t-subsets it covers
            # We need to compute this efficiently
            # For each block, gain = sum of (counts[idx] == 0) for idx in block_t_indices[block]
            # This is expensive to do for all blocks every iteration.
            
            # Optimization: only recompute gains for blocks that might have changed
            # But since we're adding blocks, all gains can only decrease or stay same
            # Actually, gains decrease as we cover more t-subsets
            
            # Simple approach: compute all gains each iteration
            # But this is O(num_blocks * C(k,t)) which is expensive
            
            # Better: maintain gains incrementally
            # When we add a block, only t-subsets it covers have their count change
            # So only blocks that share t-subsets with the new block have changed gains
            
            # For simplicity and given time constraints, let's do a full recompute
            # but only every few iterations, and use a fast path
            
            # Actually, let's use a different approach:
            # Maintain a list of candidate blocks and their gains
            # Update gains only for affected blocks
            
            # For now, let's do a full scan but with numpy vectorization
            gains = np.zeros(num_blocks, dtype=np.int32)
            for i in range(num_blocks):
                if i in selected_set:
                    continue
                idxs = block_t_indices[i]
                gains[i] = np.sum(counts[idxs] == 0)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break

            # Find best blocks
            best_mask = (gains == max_gain)
            best_indices = np.where(best_mask)[0]
            
            if use_weighted and len(best_indices) > 1:
                # Weighted selection among ties
                weights = block_weights[best_indices]
                # Soft-max style: pick with probability proportional to weight
                total_w = np.sum(weights)
                if total_w > 0:
                    probs = weights / total_w
                    r = random.random()
                    cumsum = np.cumsum(probs)
                    idx = np.searchsorted(cumsum, r)
                    if idx >= len(best_indices):
                        idx = len(best_indices) - 1
                    best_idx = best_indices[idx]
                else:
                    best_idx = best_indices[random.randint(0, len(best_indices) - 1)]
            elif len(best_indices) > 1:
                best_idx = best_indices[random.randint(0, len(best_indices) - 1)]
            else:
                best_idx = best_indices[0]
            
            selected_blocks.append(best_idx)
            selected_set.add(best_idx)
            
            # Update counts and uncovered_count
            idxs = block_t_indices[best_idx]
            newly_covered = np.sum(counts[idxs] == 0)
            uncovered_count -= newly_covered
            counts[idxs] += 1

        return selected_blocks, counts

    def reduce_covering(selected_blocks, counts):
        """
        Remove redundant blocks iteratively.
        """
        if not selected_blocks:
            return selected_blocks, counts

        current_blocks = selected_blocks[:]
        cur_counts = counts.copy()
        changed = True

        while changed and time_left() > 10.0:
            changed = False
            new_blocks = []
            for b_idx in current_blocks:
                idxs = block_t_indices[b_idx]
                if np.any(cur_counts[idxs] == 1):
                    new_blocks.append(b_idx)
                else:
                    cur_counts[idxs] -= 1
                    changed = True
            current_blocks = new_blocks

        return current_blocks, cur_counts

    def deletion_based_reduce(selected_blocks, counts):
        """
        Try to delete blocks one at a time, checking if coverage is maintained.
        More aggressive than simple reduction.
        """
        if not selected_blocks:
            return selected_blocks, counts

        current_blocks = selected_blocks[:]
        cur_counts = counts.copy()
        
        # Try each block in random order
        random.shuffle(current_blocks)
        
        i = 0
        while i < len(current_blocks) and time_left() > 10.0:
            b_idx = current_blocks[i]
            idxs = block_t_indices[b_idx]
            
            # Check if any t-subset would become uncovered
            if np.any(cur_counts[idxs] == 1):
                i += 1
            else:
                # Safe to remove
                cur_counts[idxs] -= 1
                current_blocks.pop(i)
                # Don't increment i, check the same position again

        return current_blocks, cur_counts

    def swap_local_search(selected_blocks, counts, seed_val, max_iters=200):
        """
        Local search: try to swap blocks to enable further deletions.
        """
        random.seed(seed_val)
        if not selected_blocks:
            return selected_blocks, counts

        sel_blocks = selected_blocks[:]
        cur_counts = counts.copy()
        n = len(sel_blocks)
        
        # Precompute for faster lookup
        # For each t-subset, we could track which blocks cover it, but that's expensive
        # Instead, we'll sample candidates
        
        for _ in range(max_iters):
            if time_left() < 15.0:
                break
            
            # Pick a random block to try to remove
            rem_pos = random.randint(0, len(sel_blocks) - 1)
            rem_idx = sel_blocks[rem_pos]
            rem_ts = block_t_indices[rem_idx]
            
            # Check if removing it leaves any t-subset uncovered
            if np.any(cur_counts[rem_ts] == 1):
                # It would break coverage
                broken_ts = rem_ts[cur_counts[rem_ts] == 1]
                num_broken = len(broken_ts)
                
                # Try to find a replacement block that covers all broken t-subsets
                # Sample candidates
                best_add = -1
                best_score = -1
                
                # Sample 50 random blocks
                sample_size = min(50, num_blocks)
                candidates = random.sample(range(num_blocks), sample_size)
                
                for b_idx in candidates:
                    if b_idx in sel_blocks:
                        continue
                    add_ts = block_t_indices[b_idx]
                    
                    # Count how many broken_ts are covered
                    # Use set intersection for efficiency
                    broken_set = set(broken_ts.tolist())
                    add_set = set(add_ts.tolist())
                    covered = len(broken_set & add_set)
                    
                    if covered == num_broken:
                        # Perfect cover - also check if this enables deletion
                        # Score: prefer blocks that cover more already-covered t-subsets
                        # (so they're more likely to be deletable later)
                        extra_covered = len(add_set) - covered
                        score = -extra_covered  # Minimize extra covered
                        if score > best_score:
                            best_score = score
                            best_add = b_idx
                
                if best_add != -1:
                    # Perform swap
                    cur_counts[rem_ts] -= 1
                    add_ts = block_t_indices[best_add]
                    cur_counts[add_ts] += 1
                    sel_blocks[rem_pos] = best_add
                    
                    # Try to delete the new block if it's redundant
                    new_ts = block_t_indices[best_add]
                    if not np.any(cur_counts[new_ts] == 1):
                        cur_counts[new_ts] -= 1
                        sel_blocks.pop(rem_pos)
            else:
                # Safe to remove
                cur_counts[rem_ts] -= 1
                sel_blocks.pop(rem_pos)

        return sel_blocks, cur_counts

    # Main loop: try multiple strategies
    best_solution = None
    best_len = float('inf')

    # Strategy 1: Standard greedy + reduction
    for seed in range(10):
        if time_left() < 20.0:
            break
        sol, counts = greedy_construct(seed * 12345 + 1, use_weighted=False)
        sol, counts = reduce_covering(sol, counts)
        sol, counts = deletion_based_reduce(sol, counts)
        current_len = len(sol)
        if current_len < best_len:
            best_len = current_len
            best_solution = sol[:]

    # Strategy 2: Weighted greedy + reduction + local search
    for seed in range(10):
        if time_left() < 30.0:
            break
        sol, counts = greedy_construct(seed * 54321 + 2, use_weighted=True)
        sol, counts = reduce_covering(sol, counts)
        sol, counts = deletion_based_reduce(sol, counts)
        if time_left() > 25.0:
            sol, counts = swap_local_search(sol, counts, seed * 777 + 3, max_iters=100)
        current_len = len(sol)
        if current_len < best_len:
            best_len = current_len
            best_solution = sol[:]

    # Final polish: more aggressive local search on best solution
    if best_solution is not None and time_left() > 20.0:
        # Reconstruct counts for best solution
        final_counts = np.zeros(num_t_subs, dtype=np.int32)
        for b_idx in best_solution:
            final_counts[block_t_indices[b_idx]] += 1
        best_solution, final_counts = swap_local_search(best_solution, final_counts, 999, max_iters=200)
        best_solution, final_counts = deletion_based_reduce(best_solution, final_counts)

    if best_solution is None:
        # Fallback: trivial covering (all blocks)
        best_solution = list(range(num_blocks))

    final_blocks = [list(all_blocks[i]) for i in best_solution]
    return final_blocks
