"""New program using Local Search (Add-Remove) for Covering Design."""
import itertools
import random
import numpy as np
from collections import defaultdict
from math import comb
import time

def construct(v, k, t):
    # Generate all k-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map t-subsets to indices for coverage tracking
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute: for each block, which t-subsets it covers
    block_to_tsubs = []
    # For greedy init
    tsub_to_blocks = defaultdict(list)
    
    for b_idx, block in enumerate(all_blocks):
        tsubs_in_block = []
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[ts]
            tsub_to_blocks[idx].append(b_idx)
            tsubs_in_block.append(idx)
        block_to_tsubs.append(tsubs_in_block)
        
    def greedy_init():
        """Run standard greedy to get a starting solution."""
        covered = np.zeros(n_tsubs, dtype=bool)
        gain = np.full(n_blocks, comb(k, t), dtype=int)
        selected = []
        
        while True:
            max_gain = np.max(gain)
            if max_gain == 0:
                break
            candidates = np.where(gain == max_gain)[0]
            b_idx = candidates[0] # Deterministic for init speed
            selected.append(b_idx)
            
            for ts_idx in block_to_tsubs[b_idx]:
                if not covered[ts_idx]:
                    covered[ts_idx] = True
                    for other_b in tsub_to_blocks[ts_idx]:
                        gain[other_b] -= 1
        return selected

    # Initial solution
    initial_selected = greedy_init()
    in_solution = np.zeros(n_blocks, dtype=bool)
    for b in initial_selected:
        in_solution[b] = True
        
    # Calculate initial cover counts
    cover_count = np.zeros(n_tsubs, dtype=int)
    for b in initial_selected:
        for ts_idx in block_to_tsubs[b]:
            cover_count[ts_idx] += 1
            
    best_len = len(initial_selected)
    best_solution = initial_selected.copy()
    
    # Maintain a list of currently selected blocks for fast iteration
    selected_list = initial_selected.copy()
    
    start_time = time.time()
    time_limit = 110 # Leave 10s buffer
    
    def try_remove():
        """Remove any block that is not the unique cover for any t-subset."""
        i = 0
        while i < len(selected_list):
            b_idx = selected_list[i]
            # Check if all t-subsets in this block are covered by >1 block
            is_redundant = True
            for ts_idx in block_to_tsubs[b_idx]:
                if cover_count[ts_idx] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove block
                in_solution[b_idx] = False
                for ts_idx in block_to_tsubs[b_idx]:
                    cover_count[ts_idx] -= 1
                # Remove from selected_list
                selected_list.pop(i)
                # Do not increment i, check the new element at this index
            else:
                i += 1

    # Local Search Loop
    rng = random.Random(42)
    
    while time.time() - start_time < time_limit:
        # Pick a random block to add
        b_add = rng.randint(0, n_blocks - 1)
        
        if in_solution[b_add]:
            continue
            
        # Add it
        in_solution[b_add] = True
        selected_list.append(b_add)
        for ts_idx in block_to_tsubs[b_add]:
            cover_count[ts_idx] += 1
            
        # Try to remove redundant blocks
        try_remove()
        
        current_len = len(selected_list)
        
        if current_len < best_len:
            best_len = current_len
            best_solution = selected_list.copy()
            
        # Reset if we drift too far
        if current_len > best_len + 50:
            # Clear current
            for b in selected_list:
                in_solution[b] = False
                for ts_idx in block_to_tsubs[b]:
                    cover_count[ts_idx] -= 1
            selected_list = best_solution.copy()
            cover_count[:] = 0
            for b in best_solution:
                in_solution[b] = True
                for ts_idx in block_to_tsubs[b]:
                    cover_count[ts_idx] += 1

    # Convert best_solution indices to blocks
    return [list(all_blocks[i]) for i in best_solution]
