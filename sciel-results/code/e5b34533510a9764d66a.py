import itertools
import random
import time
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using WalkSAT-style stochastic local search.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v, k)) if v > k else [])
    # Fix: all_blocks = list(itertools.combinations(range(v), k))
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map each block to the indices of t-subsets it covers
    block_to_tsubs = []
    for block in all_blocks:
        covered_indices = [tsub_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_to_tsubs.append(covered_indices)
    
    # Convert to numpy arrays
    block_tsubs_arrays = [np.array(t_indices, dtype=np.int32) for t_indices in block_to_tsubs]
    
    # Inverse index: for each t-subset, which blocks cover it
    tsub_to_blocks = defaultdict(list)
    for b_idx, t_indices in enumerate(block_to_tsubs):
        for t_idx in t_indices:
            tsub_to_blocks[t_idx].append(b_idx)
    
    tsub_to_blocks_arrays = [np.array(blks, dtype=np.int32) for blks in tsub_to_blocks.values()]
    
    # Function to compute uncovered count for a set of blocks
    def get_uncovered(blocks):
        """Return a boolean mask of uncovered t-subsets"""
        covered = np.zeros(n_tsubs, dtype=bool)
        for b_idx in blocks:
            covered[block_tsubs_arrays[b_idx]] = True
        return ~covered
    
    def count_uncovered(blocks):
        """Return number of uncovered t-subsets"""
        covered = np.zeros(n_tsubs, dtype=bool)
        for b_idx in blocks:
            covered[block_tsubs_arrays[b_idx]] = True
        return np.sum(~covered)
    
    # Initial construction: simple greedy to get a valid covering
    def greedy_initial():
        uncovered = np.ones(n_tsubs, dtype=bool)
        blocks = []
        rng = np.random.RandomState(42)
        
        while np.any(uncovered):
            if time.time() > deadline - 5:
                break
            best_gain = -1
            best_b = -1
            # Sample blocks
            sample_size = min(500, n_blocks)
            for b_idx in rng.choice(n_blocks, size=sample_size, replace=False):
                gain = np.sum(uncovered[block_tsubs_arrays[b_idx]])
                if gain > best_gain:
                    best_gain = gain
                    best_b = b_idx
            if best_b == -1:
                break
            blocks.append(best_b)
            uncovered[block_tsubs_arrays[best_b]] = False
        return blocks
    
    # WalkSAT-style local search
    def walk_sat_local_search(initial_blocks, max_time=10):
        if not initial_blocks:
            return initial_blocks
        
        end_time = min(start_time + max_time, deadline - 2)
        current = list(initial_blocks)
        current_uncovered = count_uncovered(current)
        best = list(current)
        best_uncovered = current_uncovered
        
        # Coverage count for efficiency
        tsub_count = np.zeros(n_tsubs, dtype=np.int32)
        for b_idx in current:
            tsub_count[block_tsubs_arrays[b_idx]] += 1
        
        # Temperature for simulated annealing
        temp = 5.0
        cooling = 0.99
        
        rng = np.random.RandomState(123)
        
        while time.time() < end_time:
            # Randomly pick a block to remove or swap
            if len(current) == 0:
                break
            
            i = rng.randint(0, len(current) - 1)
            b_idx = current[i]
            
            # Option 1: Remove the block
            # Option 2: Swap with a random other block
            
            if rng.random() < 0.5:
                # Try removal
                t_indices = block_tsubs_arrays[b_idx]
                # How many t-subsets would become uncovered?
                newly_uncovered = np.sum(tsub_count[t_indices] == 1)
                # How many t-subsets are currently uncovered that this block covers?
                currently_uncovered_covered = np.sum((tsub_count[t_indices] == 0))
                
                delta = newly_uncovered - currently_uncovered_covered
                
                if delta <= 0 or (delta > 0 and rng.random() < np.exp(-delta / temp)):
                    # Accept
                    current.pop(i)
                    tsub_count[t_indices] -= 1
                    current_uncovered += delta
                    temp *= cooling
            else:
                # Try swap: replace current[i] with a random other block
                if len(current) < n_blocks:
                    # Pick a random block not in current
                    attempts = 0
                    new_b = None
                    while attempts < 10:
                        new_b = rng.randint(0, n_blocks - 1)
                        if new_b not in current:
                            break
                        attempts += 1
                    
                    if new_b is not None and new_b not in current:
                        old_t = block_tsubs_arrays[b_idx]
                        new_t = block_tsubs_arrays[new_b]
                        
                        # Compute delta: change in uncovered count
                        # t-subsets only in old: become uncovered if count was 1
                        # t-subsets only in new: become covered if count was 0
                        # t-subsets in both: no change
                        
                        old_only = np.setdiff1d(old_t, new_t)
                        new_only = np.setdiff1d(new_t, old_t)
                        
                        lose = np.sum(tsub_count[old_only] == 1)  # become uncovered
                        gain = np.sum(tsub_count[new_only] == 0)  # become covered
                        
                        delta = lose - gain
                        
                        if delta <= 0 or (delta > 0 and rng.random() < np.exp(-delta / temp)):
                            # Accept swap
                            current[i] = new_b
                            tsub_count[old_t] -= 1
                            tsub_count[new_t] += 1
                            current_uncovered += delta
                            temp *= cooling
            
            # Track best
            if current_uncovered < best_uncovered:
                best_uncovered = current_uncovered
                best = list(current)
                if best_uncovered == 0:
                    break  # Perfect covering found
            
            # Periodic restart if stuck
            if rng.random() < 0.01:
                # Slight perturbation: add a random block
                if len(current) < n_blocks:
                    new_b = rng.randint(0, n_blocks - 1)
                    if new_b not in current:
                        current.append(new_b)
                        tsub_count[block_tsubs_arrays[new_b]] += 1
                        # Recount uncovered
                        current_uncovered = count_uncovered(current)
        
        return best
    
    # Main: get initial greedy, then run WalkSAT local search
    initial = greedy_initial()
    
    if not initial:
        # Fallback: trivial covering
        blocks = []
        covered = np.zeros(n_tsubs, dtype=bool)
        for t_idx in range(n_tsubs):
            if not covered[t_idx]:
                for b_idx in tsub_to_blocks_arrays[t_idx]:
                    if b_idx not in blocks:
                        blocks.append(b_idx)
                        covered[block_tsubs_arrays[b_idx]] = True
                        break
        return [list(all_blocks[b_idx]) for b_idx in blocks]
    
    # Run WalkSAT local search with time budget
    time_left = deadline - 2 - time.time()
    if time_left > 5:
        result = walk_sat_local_search(initial, max_time=time_left)
        # Verify
        if count_uncovered(result) == 0:
            initial = result
    
    # Final verification
    if count_uncovered(initial) != 0:
        # Fix any uncovered t-subsets
        covered = np.zeros(n_tsubs, dtype=bool)
        for b_idx in initial:
            covered[block_tsubs_arrays[b_idx]] = True
        for t_idx in range(n_tsubs):
            if not covered[t_idx]:
                for b_idx in tsub_to_blocks_arrays[t_idx]:
                    if b_idx not in initial:
                        initial.append(b_idx)
                        covered[block_tsubs_arrays[b_idx]] = True
                        break
    
    final_blocks = [list(all_blocks[b_idx]) for b_idx in initial]
    return final_blocks
