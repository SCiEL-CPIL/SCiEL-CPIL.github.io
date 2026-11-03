import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # 1. Precomputation
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}

    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # block_to_tsets: list of lists of tset indices
    # tset_to_blocks: list of lists of block indices
    block_to_tsets = [[] for _ in range(num_blocks)]
    tset_to_blocks = [[] for _ in range(num_tsets)]
    
    for b_idx, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            ts_idx = tset_to_idx[ts]
            block_to_tsets[b_idx].append(ts_idx)
            tset_to_blocks[ts_idx].append(b_idx)

    # Helper to get a valid solution from block indices
    def get_solution(indices):
        return [list(all_blocks[i]) for i in indices]

    def run_local_search(initial_indices, seed=0):
        """
        Takes a list of block indices, applies local search improvements.
        Returns the best list of indices found.
        """
        if not initial_indices:
            return []
        
        rng = random.Random(seed)
        current = list(initial_indices)
        
        # Calculate initial cover counts
        cover_counts = np.zeros(num_tsets, dtype=np.int32)
        for b_idx in current:
            for ts_idx in block_to_tsets[b_idx]:
                cover_counts[ts_idx] += 1
        
        best_size = len(current)
        best_solution = list(current)
        
        local_start = time.time()
        
        # Local search loop
        while time.time() - local_start < 15.0:  # Limit time per local search run
            improved = False
            
            # Phase 1: Remove redundant blocks
            # We iterate until no more redundant blocks are found
            while True:
                if time.time() - local_start > 15.0:
                    break
                found_redundant = False
                # Shuffle order to avoid bias
                order = list(range(len(current)))
                rng.shuffle(order)
                
                for i in order:
                    if i >= len(current):
                        continue
                    b_idx = current[i]
                    is_redundant = True
                    for ts_idx in block_to_tsets[b_idx]:
                        if cover_counts[ts_idx] == 1:
                            is_redundant = False
                            break
                    
                    if is_redundant:
                        # Remove block
                        for ts_idx in block_to_tsets[b_idx]:
                            cover_counts[ts_idx] -= 1
                        current.pop(i)
                        found_redundant = True
                        if len(current) < best_size:
                            best_size = len(current)
                            best_solution = list(current)
                        break # Restart pass after removal
                
                if not found_redundant:
                    break
            
            if not current:
                break

            # Phase 2: 1-for-1 Swaps
            # Try to find a swap that might enable future reductions
            swap_made = False
            # Shuffle current to try different blocks first
            current_order = list(range(len(current)))
            rng.shuffle(current_order)
            
            for i in current_order:
                if time.time() - local_start > 15.0:
                    break
                if i >= len(current):
                    continue
                
                b_out = current[i]
                
                # Find tsets uniquely covered by b_out
                unique_tsets = []
                for ts_idx in block_to_tsets[b_out]:
                    if cover_counts[ts_idx] == 1:
                        unique_tsets.append(ts_idx)
                        
                if not unique_tsets:
                    continue

                # Find candidate blocks B_in that cover all unique_tsets
                # Intersection of tset_to_blocks for each unique tset
                if len(unique_tsets) == 1:
                    candidates = set(tset_to_blocks[unique_tsets[0]])
                else:
                    # Start with the smallest list for efficiency
                    min_idx = min(unique_tsets, key=lambda x: len(tset_to_blocks[x]))
                    candidates = set(tset_to_blocks[min_idx])
                    for ts_idx in unique_tsets:
                        if ts_idx == min_idx:
                            continue
                        if not candidates:
                            break
                        candidates.intersection_update(tset_to_blocks[ts_idx])
                
                candidates.discard(b_out)
                
                if not candidates:
                    continue
                
                # Pick a random candidate
                b_in = rng.choice(list(candidates))
                
                # Perform swap
                # Remove b_out
                for ts_idx in block_to_tsets[b_out]:
                    cover_counts[ts_idx] -= 1
                # Add b_in
                for ts_idx in block_to_tsets[b_in]:
                    cover_counts[ts_idx] += 1
                
                current[i] = b_in
                swap_made = True
                break # Restart local search loop after one swap to re-evaluate redundancy
            
            if not swap_made:
                # No more swaps possible or time limit reached
                break

        return best_solution

    # Main logic
    best_solution = None
    best_size = float('inf')
    
    # Run multiple restarts
    attempts = 0
    while time.time() < deadline:
        seed = random.randint(0, 2**32 - 1)
        
        # Greedy initial solution
        rng = random.Random(seed)
        uncovered = [True] * num_tsets
        selected_blocks = []
        
        # To speed up greedy, we can maintain a list of uncovered tset indices
        uncovered_indices = list(range(num_tsets))
        
        while uncovered_indices:
            # Pick a random uncovered tset
            rand_pos = rng.randint(0, len(uncovered_indices) - 1)
            ts_idx = uncovered_indices[rand_pos]
            
            # Find best block covering this tset
            candidates = tset_to_blocks[ts_idx]
            
            best_gain = -1
            best_block_idx = -1
            
            for b_idx in candidates:
                # Count how many uncovered tsets this block covers
                gain = 0
                for t_idx in block_to_tsets[b_idx]:
                    if uncovered[t_idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            selected_blocks.append(best_block_idx)
            
            # Update uncovered
            for t_idx in block_to_tsets[best_block_idx]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = False
                    # Remove t_idx from uncovered_indices
                    # Find position
                    # This is O(N) which is slow. 
                    # Optimization: Just rebuild uncovered_indices periodically or use a different structure.
                    # For v=19, t=5, num_tsets ~ 11k. Rebuilding is 11k ops. 
                    # Greedy steps ~ 300. 300*11k = 3.3M ops. Acceptable.
                    pass
            
            # Rebuild uncovered_indices
            uncovered_indices = [i for i, u in enumerate(uncovered) if u]
            
        # Local search
        solution = run_local_search(selected_blocks, seed)
        
        if len(solution) < best_size:
            best_size = len(solution)
            best_solution = solution
            
        attempts += 1

    if best_solution is None:
        return []
        
    return get_solution(best_solution)
