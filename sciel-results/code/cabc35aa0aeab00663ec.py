"""
Covering design C(19,8,5) using a two-phase approach:
1. Fast greedy set cover to get a valid (but not minimal) covering.
2. Local search (block deletion + block swap) to reduce the number of blocks.

Key optimizations:
- Use numpy bitmasks for fast coverage checks.
- Precompute the list of 5-subsets for each 8-block.
- Use a deadline guard to return the best solution found before the time limit.
- Multiple restarts of the greedy phase with different random seeds to explore
  different basins of attraction.
- Local search alternates between deleting redundant blocks and swapping
  a block for another that covers more uncovered 5-subsets.
"""
import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Precompute all t-subsets and their indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}

    # Precompute all k-subsets
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)

    # For each k-subset, precompute the list of t-subset indices it covers
    block_tsubs_list = []
    for block in all_ksubs:
        tsubs_in_block = [tsub_index[ts] for ts in itertools.combinations(block, t)]
        block_tsubs_list.append(tsubs_in_block)
    
    block_tsubs = np.array(block_tsubs_list, dtype=np.int32)
    # Also store as list of lists for faster indexing in some cases
    block_tsubs_list = block_tsubs_list

    def run_greedy(seed):
        random.seed(seed)
        covered = np.zeros(n_tsubs, dtype=bool)
        selected = []
        
        while not covered.all():
            if time.time() > deadline:
                return selected
            
            # Calculate gains: for each block, how many currently uncovered t-subsets it covers
            uncov = (~covered).astype(np.int32)
            gains = np.take(uncov, block_tsubs).sum(axis=1)
            
            max_gain = gains.max()
            if max_gain == 0:
                break
                
            # Find all blocks with max gain and pick one randomly
            best_indices = np.where(gains == max_gain)[0]
            chosen_idx = random.choice(best_indices)
            
            selected.append(chosen_idx)
            covered[block_tsubs[chosen_idx]] = True
            
        return selected

    def polish_delete(selected):
        """Remove redundant blocks in a single backward pass."""
        if not selected:
            return selected
        cover_count = np.zeros(n_tsubs, dtype=np.int32)
        for idx in selected:
            cover_count[block_tsubs[idx]] += 1
            
        new_selected = []
        for idx in reversed(selected):
            if time.time() > deadline:
                return selected  # return original if we run out of time
            if np.all(cover_count[block_tsubs[idx]] >= 2):
                cover_count[block_tsubs[idx]] -= 1
            else:
                new_selected.append(idx)
        return list(reversed(new_selected))

    def local_search(selected, max_iters=500):
        """
        Local search: alternate between deleting redundant blocks and
        trying to swap a block for another that reduces total count or
        enables more deletions.
        """
        if not selected:
            return selected
        
        best = list(selected)
        best_len = len(best)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Phase 1: delete redundant blocks
            best = polish_delete(best)
            if len(best) < best_len:
                best_len = len(best)
            
            # Phase 2: try swapping each block with another
            improved = False
            current = list(best)
            cover_count = np.zeros(n_tsubs, dtype=np.int32)
            for idx in current:
                cover_count[block_tsubs[idx]] += 1
            
            for i, idx in enumerate(current):
                if time.time() > deadline:
                    break
                
                # Try removing this block temporarily
                cover_count[block_tsubs[idx]] -= 1
                uncovered = np.where(cover_count == 0)[0]
                
                if len(uncovered) == 0:
                    # This block is redundant (should have been caught by polish, but just in case)
                    current.pop(i)
                    improved = True
                    # Recompute cover counts
                    cover_count = np.zeros(n_tsubs, dtype=np.int32)
                    for jdx in current:
                        cover_count[block_tsubs[jdx]] += 1
                    i = -1  # restart the loop
                    break
                
                # Find the best replacement block that covers the most uncovered t-subsets
                # among blocks not currently in the solution
                in_solution = set(current)
                # For efficiency, only consider blocks that cover at least one uncovered t-subset
                # Compute gain for each candidate block
                gains = np.take((cover_count == 0).astype(np.int32), block_tsubs).sum(axis=1)
                # Exclude blocks already in the solution
                for jdx in current:
                    gains[jdx] = -1
                
                max_gain = gains.max()
                if max_gain > 0:
                    candidates = np.where(gains == max_gain)[0]
                    # Pick a random candidate for diversity
                    new_idx = random.choice(candidates)
                    
                    # Check if the swap is beneficial:
                    # After swap, we need to ensure all t-subsets are still covered.
                    # The new block covers all uncovered ones (by construction of gains),
                    # and the old block's remaining coverage is still provided by others.
                    # Actually, we need to be more careful. Let's just do the swap and verify.
                    current[i] = new_idx
                    # Recompute cover counts
                    cover_count = np.zeros(n_tsubs, dtype=np.int32)
                    for jdx in current:
                        cover_count[block_tsubs[jdx]] += 1
                    if np.all(cover_count >= 1):
                        improved = True
                        i = -1  # restart the loop
                        break
                    else:
                        # Revert
                        current[i] = idx
                        cover_count = np.zeros(n_tsubs, dtype=np.int32)
                        for jdx in current:
                            cover_count[block_tsubs[jdx]] += 1
                        i += 1
                        continue
                else:
                    # Restore cover count
                    cover_count[block_tsubs[idx]] += 1
                    i += 1
                    continue
            
            if not improved:
                break
            
            if len(best) < best_len:
                best_len = len(best)
        
        return best

    best_solution = None
    best_len = float('inf')
    
    # Run multiple greedy instances followed by local search
    num_greedy_runs = 8
    
    for i in range(num_greedy_runs):
        if time.time() > deadline - 10:
            break
        
        # Run greedy
        sol = run_greedy(i)
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = list(sol)
        
        # Run local search on this solution
        polished = polish_delete(sol)
        if len(polished) < best_len:
            best_len = len(polished)
            best_solution = list(polished)
        
        # Try local search
        searched = local_search(polished)
        if len(searched) < best_len:
            best_len = len(searched)
            best_solution = list(searched)
    
    if best_solution is None:
        best_solution = run_greedy(0)
        best_solution = polish_delete(best_solution)
    
    # Final polish
    final = polish_delete(best_solution)
    if len(final) < len(best_solution):
        best_solution = final
    
    # Convert indices to blocks
    result = [list(all_ksubs[i]) for i in best_solution]
    
    return result
