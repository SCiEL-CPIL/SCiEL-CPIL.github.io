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
    # For each t-subset, which blocks cover it (for potential swap targets or coverage checks)
    # Note: n_blocks is ~75k, n_tsubs is ~11628. 
    # Storing tsub_to_blocks as lists of ints is fine.
    tsub_to_blocks = defaultdict(list)
    
    for b_idx, block in enumerate(all_blocks):
        tsubs_in_block = []
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[ts]
            tsub_to_blocks[idx].append(b_idx)
            tsubs_in_block.append(idx)
        block_to_tsubs.append(tsubs_in_block)
        
    # Precompute block sets for faster intersection/union if needed, 
    # but we will rely on cover_count arrays.

    def greedy_init(rng):
        """Run randomized greedy to get a starting solution."""
        covered = np.zeros(n_tsubs, dtype=bool)
        gain = np.full(n_blocks, comb(k, t), dtype=int)
        selected = []
        in_solution = np.zeros(n_blocks, dtype=bool)
        
        while True:
            max_gain = np.max(gain)
            if max_gain == 0:
                break
            candidates = np.where(gain == max_gain)[0]
            # Randomly pick one of the candidates with max gain
            b_idx = candidates[rng.randint(0, len(candidates) - 1)]
            selected.append(b_idx)
            in_solution[b_idx] = True
            
            for ts_idx in block_to_tsubs[b_idx]:
                if not covered[ts_idx]:
                    covered[ts_idx] = True
                    # Update gains for all blocks covering this t-subset
                    for other_b in tsub_to_blocks[ts_idx]:
                        gain[other_b] -= 1
        return selected, in_solution

    def local_search(selected_list, in_solution, rng, time_limit_inner):
        """
        Local search with Remove and Swap moves.
        Returns (selected_list, in_solution, cover_count, best_len)
        """
        # Initialize cover_count
        cover_count = np.zeros(n_tsubs, dtype=int)
        for b in selected_list:
            for ts_idx in block_to_tsubs[b]:
                cover_count[ts_idx] += 1
        
        best_len = len(selected_list)
        best_solution = selected_list.copy()
        best_in_solution = in_solution.copy()
        
        start_ls_time = time.time()
        improved = True
        while improved and (time.time() - start_ls_time) < time_limit_inner:
            improved = False
            
            # 1. Try Remove moves
            # Shuffle to avoid bias
            order = list(range(len(selected_list)))
            rng.shuffle(order)
            
            for i in order:
                if i >= len(selected_list): continue
                b_idx = selected_list[i]
                
                # Check if b_idx is the unique cover for any t-subset
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
                    selected_list.pop(i)
                    improved = True
                    best_len = len(selected_list)
                    best_solution = selected_list.copy()
                    best_in_solution = in_solution.copy()
            
            # 2. Try Swap moves (1-in, 1-out)
            # We look for a block 'b_in' not in solution that, when added,
            # allows us to remove at least one block 'b_out' from solution.
            # Simplified: Try to swap a random 'b_out' with a random 'b_in'.
            # If the new set is still a cover, and we can then remove more, accept.
            # To be efficient, we only accept swaps that immediately allow a removal
            # or result in a smaller set after a quick removal pass.
            
            if len(selected_list) > 0:
                # Pick a random block to swap out
                # We limit the number of swap attempts to keep it fast
                num_swap_attempts = 50
                for _ in range(num_swap_attempts):
                    if time.time() - start_ls_time > time_limit_inner:
                        break
                    
                    b_out_idx = rng.randint(0, len(selected_list) - 1)
                    b_out = selected_list[b_out_idx]
                    
                    # Pick a random block to swap in
                    b_in = rng.randint(0, n_blocks - 1)
                    if in_solution[b_in]:
                        continue
                    
                    # Temporarily add b_in and remove b_out
                    # We need to check if the resulting set is a valid cover.
                    # Instead of full check, we can check if b_in covers any t-subset
                    # that was ONLY covered by b_out. If so, the swap is invalid 
                    # unless b_in covers other blocks' t-subsets... 
                    # Actually, simplest valid check: 
                    # 1. Add b_in, remove b_out.
                    # 2. Check if any t-subset has count 0.
                    
                    # To speed up, we can pre-compute "critical" t-subsets for b_out.
                    # Critical if cover_count[ts] == 1.
                    # If b_out has no critical t-subsets, the swap is always valid 
                    # (coverage is maintained or improved).
                    # If it has critical ones, b_in MUST cover them.
                    
                    has_critical = False
                    critical_tsubs = []
                    for ts_idx in block_to_tsubs[b_out]:
                        if cover_count[ts_idx] == 1:
                            has_critical = True
                            critical_tsubs.append(ts_idx)
                    
                    if has_critical:
                        # Check if b_in covers all critical tsubs
                        # We can do this by checking if b_in is in tsub_to_blocks for each critical tsub
                        # Or simply: intersect block_to_tsubs[b_in] with critical_tsubs
                        b_in_tsubs_set = set(block_to_tsubs[b_in])
                        if not set(critical_tsubs).issubset(b_in_tsubs_set):
                            continue # Swap invalid, b_in doesn't cover critical sets
                    
                    # Perform Swap
                    # Remove b_out
                    in_solution[b_out] = False
                    for ts_idx in block_to_tsubs[b_out]:
                        cover_count[ts_idx] -= 1
                    selected_list[b_out_idx] = b_in # Replace in list
                    
                    # Add b_in
                    in_solution[b_in] = True
                    for ts_idx in block_to_tsubs[b_in]:
                        cover_count[ts_idx] += 1
                    
                    # Now try to remove redundant blocks again
                    # This might reduce the size further
                    # We run a quick removal pass
                    removed_any = False
                    i = 0
                    while i < len(selected_list):
                        b_cur = selected_list[i]
                        is_redundant = True
                        for ts_idx in block_to_tsubs[b_cur]:
                            if cover_count[ts_idx] == 1:
                                is_redundant = False
                                break
                        if is_redundant:
                            in_solution[b_cur] = False
                            for ts_idx in block_to_tsubs[b_cur]:
                                cover_count[ts_idx] -= 1
                            selected_list.pop(i)
                            removed_any = True
                        else:
                            i += 1
                    
                    if len(selected_list) < best_len:
                        best_len = len(selected_list)
                        best_solution = selected_list.copy()
                        best_in_solution = in_solution.copy()
                        improved = True
                    else:
                        # Revert swap if it didn't improve or if we want to keep state consistent for next iteration
                        # Actually, if it didn't improve, we should revert to maintain the "current" state 
                        # or accept it as a neutral move? 
                        # Standard ILS often accepts neutral moves to escape local optima.
                        # However, for simplicity and stability, let's only keep improvements 
                        # or accept neutral swaps with a small probability?
                        # Let's revert if not improved to keep the search focused on reduction.
                        # But wait, if we revert, we don't explore new neighborhoods.
                        # Let's accept neutral swaps with 50% probability to allow wandering.
                        if rng.random() > 0.5:
                            # Revert
                            in_solution[b_in] = False
                            for ts_idx in block_to_tsubs[b_in]:
                                cover_count[ts_idx] -= 1
                            in_solution[b_out] = True
                            for ts_idx in block_to_tsubs[b_out]:
                                cover_count[ts_idx] += 1
                            # Revert removals? 
                            # This is getting complicated to revert the removals.
                            # Better strategy: Only perform swaps if they result in an improvement 
                            # after the removal pass. If not, revert the swap AND the removals.
                            # To make revert easy, we should snapshot the state before the swap.
                            # But snapshots are expensive.
                            # Alternative: Only do swaps that are "strictly better" or "equal" 
                            # and we don't revert, we just continue from the new state.
                            # This is a Random Walk.
                            # Let's do: If len(selected_list) <= best_len, keep the new state.
                            # If len(selected_list) > best_len, revert.
                            
                            # Reverting the removals is hard. 
                            # So, let's restrict swaps to those that immediately allow a removal 
                            # or result in a smaller set.
                            # If the set size didn't shrink, revert the swap.
                            # To revert, we need to undo the removals too.
                            # This is too complex for a simple edit.
                            
                            # Let's stick to: Only accept swap if it leads to a strictly smaller set 
                            # after removal. Otherwise, revert the swap and the removals.
                            # To make this possible, we must track what was removed.
                            
                            # Simpler approach for this edit: 
                            # Don't do complex swaps. Just do "Add" then "Remove".
                            # The current program does this. 
                            # The improvement is in the INITIALIZATION and ITERATION.
                            # Let's revert the swap logic to just "Add random block, then Remove redundancies".
                            # This is what the current program does.
                            # The NEW part is Randomized Greedy Init and Multiple Restarts.
                            
                            # Let's revert the swap attempt entirely for now to keep it robust.
                            # We will rely on the ILS loop with Add-Remove.
                            
                            # Revert Swap
                            in_solution[b_in] = False
                            for ts_idx in block_to_tsubs[b_in]:
                                cover_count[ts_idx] -= 1
                            in_solution[b_out] = True
                            for ts_idx in block_to_tsubs[b_out]:
                                cover_count[ts_idx] += 1
                            
                            # Revert Removals: 
                            # We lost track of what was removed. 
                            # This indicates that doing swaps with subsequent removals is hard to revert 
                            # without snapshots.
                            
                            # DECISION: Remove the Swap logic from this specific local search 
                            # and focus on a robust ILS with Add-Remove and Randomized Init.
                            # The "Swap" idea is good but implementation is tricky for reversion.
                            # We will just do Add-Remove in the local search.
                            pass

        return best_solution, best_in_solution, best_len

    # Main ILS Loop
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 110
    
    global_best_len = float('inf')
    global_best_solution = []
    
    # Run multiple restarts
    num_restarts = 0
    while time.time() - start_time < time_limit:
        num_restarts += 1
        # 1. Randomized Greedy Init
        init_sel, init_in_sol = greedy_init(rng)
        
        # 2. Local Search (Remove only, for stability)
        # We will implement a simpler local search that just does removals repeatedly
        # until no more removals are possible.
        selected_list = init_sel
        in_solution = init_in_sol
        cover_count = np.zeros(n_tsubs, dtype=int)
        for b in selected_list:
            for ts_idx in block_to_tsubs[b]:
                cover_count[ts_idx] += 1
        
        # Local Search: Iterated Removals
        improved = True
        while improved:
            improved = False
            # Shuffle to break symmetry
            order = list(range(len(selected_list)))
            rng.shuffle(order)
            for i in order:
                if i >= len(selected_list): continue
                b_idx = selected_list[i]
                is_redundant = True
                for ts_idx in block_to_tsubs[b_idx]:
                    if cover_count[ts_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    in_solution[b_idx] = False
                    for ts_idx in block_to_tsubs[b_idx]:
                        cover_count[ts_idx] -= 1
                    selected_list.pop(i)
                    improved = True
        
        current_len = len(selected_list)
        if current_len < global_best_len:
            global_best_len = current_len
            global_best_solution = selected_list.copy()
        
        # 3. Perturbation for next iteration (if we were doing full ILS)
        # For now, just restart with a new greedy init.
        # This is "Iterated Greedy".

    # Convert best_solution indices to blocks
    return [list(all_blocks[i]) for i in global_best_solution]
