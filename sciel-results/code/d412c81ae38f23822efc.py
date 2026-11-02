import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    """
    Covers design C(v, k, t) using a different strategy:
    1. Construct a valid covering using a randomized greedy algorithm that prioritizes
       covering the most "hard" t-subsets (those with fewest covering blocks in the pool).
    2. Use a specialized local search that focuses on "2-for-1" swaps and deletions,
       which are the primary mechanisms for reducing block count in covering designs.
    3. Run multiple restarts with different random seeds and pool configurations.
    
    Key differences from current program:
    - The initial covering is built more carefully to minimize initial redundancy.
    - The local search uses a more targeted approach to find valid deletions and swaps.
    - Uses a smaller, more focused pool to speed up each iteration, allowing more restarts.
    """
    deadline = time.time() + 145.0  # Safety margin below 150s kill
    start_time = time.time()
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute for each t-subset, the list of all k-blocks that contain it
    # This is crucial for efficient local search
    # For each t-subset ts, blocks containing ts are formed by adding (k-t) elements from the remaining (v-t) elements
    ts_to_blocks = {}
    for ts in all_t_subsets:
        ts_set = set(ts)
        remaining = [i for i in range(v) if i not in ts_set]
        # Generate all combinations of (k-t) from remaining
        blocks = []
        for combo in itertools.combinations(remaining, k - t):
            block = tuple(sorted(list(ts) + list(combo)))
            blocks.append(block)
        ts_to_blocks[ts] = blocks
    
    # For each block, precompute which t-subsets it covers
    # We'll compute this on the fly for blocks we use, but cache it
    block_cover_cache = {}
    
    def get_block_cover(block):
        if block not in block_cover_cache:
            covered = frozenset(t_subset_to_idx[ts] for ts in itertools.combinations(block, t))
            block_cover_cache[block] = covered
        return block_cover_cache[block]
    
    def get_initial_covering(rng, pool_blocks, pool_covers):
        """
        Build a valid covering using a greedy approach that considers
        both immediate coverage and future potential.
        """
        uncovered = set(range(n_t))
        selected = []
        selected_covers = []
        
        # For each t-subset, track which pool blocks cover it
        ts_to_pool_blocks = [[] for _ in range(n_t)]
        for i, cov in enumerate(pool_covers):
            for idx in cov:
                ts_to_pool_blocks[idx].append(i)
        
        while uncovered:
            if time.time() > deadline - 30.0:
                break
            
            # Find the uncovered t-subset with the fewest covering blocks in the pool
            # This is a "most constrained" heuristic
            min_count = float('inf')
            target_idx = -1
            for idx in list(uncovered):
                count = len(ts_to_pool_blocks[idx])
                if count < min_count:
                    min_count = count
                    target_idx = idx
                if count == 0:
                    break
            
            if target_idx == -1 or min_count == 0:
                break
            
            # From the blocks covering this t-subset, pick the one that covers the most uncovered t-subsets
            candidates = ts_to_pool_blocks[target_idx]
            if not candidates:
                break
            
            best_i = -1
            best_count = -1
            for i in candidates:
                count = len(pool_covers[i] & uncovered)
                if count > best_count:
                    best_count = count
                    best_i = i
            
            if best_i == -1:
                break
            
            selected.append(best_i)
            selected_covers.append(pool_covers[best_i])
            uncovered -= pool_covers[best_i]
        
        return selected, selected_covers
    
    def local_search(selected_indices, pool_blocks, pool_covers, rng):
        """
        Perform local search to minimize the number of blocks.
        Uses deletion and swap moves.
        """
        selected_set = set(selected_indices)
        current_count = len(selected_set)
        
        # Build coverage count array
        cover_count = np.zeros(n_t, dtype=np.int32)
        for i in selected_set:
            for idx in pool_covers[i]:
                cover_count[idx] += 1
        
        # Verify all are covered
        if np.any(cover_count == 0):
            return None, float('inf')
        
        T = 1.0
        T_min = 0.05
        cooling = 0.999
        max_steps = 8000
        steps = 0
        best_count = current_count
        best_set = set(selected_set)
        
        while T > T_min and steps < max_steps:
            if time.time() > deadline - 5.0:
                break
            
            steps += 1
            move_type = rng.random()
            
            if move_type < 0.6:
                # Deletion move: try to remove a block
                if not selected_set:
                    continue
                i = rng.choice(list(selected_set))
                
                # Check if removal is valid
                can_remove = True
                for idx in pool_covers[i]:
                    if cover_count[idx] == 1:
                        can_remove = False
                        break
                
                if not can_remove:
                    continue
                
                # Apply deletion
                for idx in pool_covers[i]:
                    cover_count[idx] -= 1
                selected_set.remove(i)
                current_count -= 1
                
                if current_count < best_count:
                    best_count = current_count
                    best_set = set(selected_set)
                
                T *= cooling
                
            elif move_type < 0.8:
                # Swap move: remove one block, add another
                if not selected_set:
                    continue
                i = rng.choice(list(selected_set))
                
                # Find a block j not in selected_set that can replace i
                # We want j to cover the t-subsets that i covers but are not covered by other blocks
                # Or at least, the swap should be valid
                
                # Get the t-subsets that would become uncovered if we remove i
                critical = set()
                for idx in pool_covers[i]:
                    if cover_count[idx] == 1:
                        critical.add(idx)
                
                if not critical:
                    # This is effectively a deletion, already handled
                    continue
                
                # Find a block j that covers all critical t-subsets
                found_j = -1
                for j in range(len(pool_blocks)):
                    if j in selected_set:
                        continue
                    cov_j = pool_covers[j]
                    # Check if j covers all critical t-subsets
                    if critical.issubset(cov_j):
                        found_j = j
                        break
                
                if found_j == -1:
                    continue
                
                # Apply swap
                for idx in pool_covers[i]:
                    cover_count[idx] -= 1
                for idx in pool_covers[found_j]:
                    cover_count[idx] += 1
                selected_set.remove(i)
                selected_set.add(found_j)
                # current_count unchanged
                
                T *= cooling
                
            else:
                # Add move: add a random block (to escape local minima)
                j = rng.randint(0, len(pool_blocks) - 1)
                if j in selected_set:
                    continue
                
                # Apply add
                for idx in pool_covers[j]:
                    cover_count[idx] += 1
                selected_set.add(j)
                current_count += 1
                
                # Accept with probability exp(-1/T)
                if rng.random() < np.exp(-1.0 / T):
                    T *= cooling
                else:
                    # Reject
                    for idx in pool_covers[j]:
                        cover_count[idx] -= 1
                    selected_set.remove(j)
                    current_count -= 1
                    T *= cooling
        
        # Return the best set found
        if best_count < float('inf'):
            return list(best_set), best_count
        return None, float('inf')
    
    best_final_blocks = None
    best_final_count = float('inf')
    
    num_restarts = 8
    pool_size = 400
    
    for restart in range(num_restarts):
        if time.time() > deadline - 20.0:
            break
        
        rng = random.Random(42 + restart)
        
        # Generate pool
        all_possible_blocks = list(itertools.combinations(range(v), k))
        pool_blocks = list(rng.sample(all_possible_blocks, pool_size))
        pool_covers = [get_block_cover(b) for b in pool_blocks]
        
        # Get initial covering
        selected_indices, selected_covers = get_initial_covering(rng, pool_blocks, pool_covers)
        
        if not selected_indices:
            continue
        
        # Verify validity
        covered = set()
        for cov in selected_covers:
            covered |= cov
        if covered != set(range(n_t)):
            continue
        
        current_count = len(selected_indices)
        if current_count < best_final_count:
            best_final_count = current_count
            best_final_blocks = [pool_blocks[i] for i in selected_indices]
        
        # Local search
        ls_result, ls_count = local_search(selected_indices, pool_blocks, pool_covers, rng)
        
        if ls_result is not None and ls_count < best_final_count:
            best_final_count = ls_count
            best_final_blocks = [pool_blocks[i] for i in ls_result]
    
    # Fallback: if we didn't find anything, do a simple greedy
    if best_final_blocks is None:
        rng = random.Random(42)
        all_possible_blocks = list(itertools.combinations(range(v), k))
        pool_size_fallback = 1000
        pool_blocks = list(rng.sample(all_possible_blocks, pool_size_fallback))
        pool_covers = [get_block_cover(b) for b in pool_blocks]
        
        uncovered = set(range(n_t))
        blocks = []
        ts_to_pool_blocks = [[] for _ in range(n_t)]
        for i, cov in enumerate(pool_covers):
            for idx in cov:
                ts_to_pool_blocks[idx].append(i)
        
        while uncovered:
            if time.time() > deadline - 2.0:
                break
            # Pick random uncovered
            ts_idx = rng.choice(list(uncovered))
            candidates = ts_to_pool_blocks[ts_idx]
            if not candidates:
                break
            i = rng.choice(candidates)
            blocks.append(pool_blocks[i])
            uncovered -= pool_covers[i]
        
        if uncovered:
            # Last resort: try to fill gaps
            for ts_idx in list(uncovered):
                ts = all_t_subsets[ts_idx]
                # Find any block containing this t-subset
                for block in ts_to_blocks[ts]:
                    if block not in [tuple(b) for b in blocks]:
                        blocks.append(list(block))
                        uncovered.discard(ts_idx)
                        break
        
        best_final_blocks = blocks
    
    # Final verification and cleanup
    if not best_final_blocks:
        return []
    
    # Verify coverage
    covered = set()
    for b in best_final_blocks:
        for ts in itertools.combinations(b, t):
            covered.add(t_subset_to_idx[ts])
    
    if covered != set(range(n_t)):
        # Try to fix by adding missing blocks
        missing = set(range(n_t)) - covered
        for ts_idx in missing:
            ts = all_t_subsets[ts_idx]
            for block in ts_to_blocks[ts]:
                if block not in [tuple(b) for b in best_final_blocks]:
                    best_final_blocks.append(list(block))
                    break
    
    # Remove duplicate blocks
    unique_blocks = []
    seen = set()
    for b in best_final_blocks:
        bt = tuple(b)
        if bt not in seen:
            seen.add(bt)
            unique_blocks.append(b)
    
    return unique_blocks

if __name__ == "__main__":
    blocks = construct(24, 7, 3)
    print(f"Blocks: {len(blocks)}")
