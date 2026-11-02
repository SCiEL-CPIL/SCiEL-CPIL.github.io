import itertools
import random
import time
import numpy as np
from collections import defaultdict

_DEADLINE = 145.0
_start_time = None

def _elapsed():
    return time.time() - _start_time if _start_time is not None else 0.0

def _remaining():
    return max(0.0, _DEADLINE - _elapsed())

def construct(v, k, t):
    global _start_time
    _start_time = time.time()

    # Precompute all t-subsets (targets)
    all_ts = list(itertools.combinations(range(v), t))
    n_ts = len(all_ts)
    ts_index = {ts: i for i, ts in enumerate(all_ts)}

    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, precompute the list of t-subset indices it covers
    block_ts = []
    for blk in all_blocks:
        ts_list = [ts_index[ts] for ts in itertools.combinations(blk, t)]
        block_ts.append(ts_list)

    # Precompute: for each t-subset index, which blocks contain it
    ts_to_blocks = defaultdict(list)
    for b_idx, ts_list in enumerate(block_ts):
        for s_idx in ts_list:
            ts_to_blocks[s_idx].append(b_idx)

    # --- 1. Greedy Initialization (Vectorized-ish) ---
    def greedy_init(seed):
        local_rng = random.Random(seed)
        # uncovered: boolean array, True if not covered
        uncovered = np.ones(n_ts, dtype=bool)
        uncovered_count = n_ts
        
        # gain[b] = number of currently uncovered t-subsets covered by block b
        gains = np.zeros(n_blocks, dtype=np.int32)
        for b in range(n_blocks):
            gains[b] = len(block_ts[b]) # Initially all are uncovered
        
        selected = []
        
        # We need to efficiently pick the block with max gain.
        # Since n_blocks is ~50k, a linear scan is O(50k).
        # Number of steps is ~150. Total ops ~7.5M. Fast enough in Python if optimized.
        
        while uncovered_count > 0:
            if time.time() - _start_time > _DEADLINE - 5.0:
                break
                
            # Find block with max gain
            best_b = -1
            best_gain = -1
            for b in range(n_blocks):
                if gains[b] > best_gain:
                    best_gain = gains[b]
                    best_b = b
            
            if best_gain <= 0:
                break # Should not happen if problem is solvable
            
            selected.append(best_b)
            
            # Update uncovered and gains
            for s_idx in block_ts[best_b]:
                if uncovered[s_idx]:
                    uncovered[s_idx] = False
                    uncovered_count -= 1
                    # Decrement gains of all blocks covering this t-subset
                    for other_b in ts_to_blocks[s_idx]:
                        gains[other_b] -= 1
                        if gains[other_b] < 0:
                            gains[other_b] = 0 # Safety, shouldn't go negative if logic is right
        
        return selected

    # --- 2. Simulated Annealing Optimization ---
    def sa_optimize(initial_blocks, seed, time_limit):
        local_rng = random.Random(seed)
        start_time = time.time()
        
        # State:
        # in_block: boolean array
        # coverage: int array, count of blocks covering each t-subset
        # n_uncovered: int
        # n_selected: int
        
        in_block = np.zeros(n_blocks, dtype=bool)
        coverage = np.zeros(n_ts, dtype=np.int32)
        
        for b in initial_blocks:
            in_block[b] = True
            for s in block_ts[b]:
                coverage[s] += 1
        
        n_uncovered = np.sum(coverage == 0)
        n_selected = len(initial_blocks)
        
        # Objective: n_selected + P * n_uncovered
        P = 1000
        
        def get_obj():
            return n_selected + P * n_uncovered
        
        current_obj = get_obj()
        best_obj = current_obj
        best_in_block = in_block.copy()
        
        T = 10.0
        T_min = 0.01
        cooling = 0.9995
        
        # Precompute list of block indices for random selection
        block_indices = np.arange(n_blocks)
        
        while time.time() - start_time < time_limit:
            if T < T_min:
                break
                
            # Choose move type
            # 60% Remove, 20% Add, 20% Swap
            move_type = local_rng.random()
            
            delta_obj = 0
            # Track changes to revert if rejected
            changed_blocks = [] # list of (block_idx, old_in_block, old_coverage_changes)
            # Actually, simpler: just compute delta directly
            
            if move_type < 0.6:
                # Remove a random block from selected
                if n_selected == 0:
                    continue
                # Pick random selected block
                # Using a list of selected blocks is faster for random pick
                # But we have in_block array. Let's maintain a list of selected indices.
                # For now, just pick random b and check if in_block[b]
                # This is inefficient if n_selected is small.
                # Let's maintain a list.
                
                # Optimization: Maintain `selected_list`
                # This requires updating it on every accept/reject.
                
                # Let's stick to array for simplicity, but picking random from 50k is O(1).
                b = local_rng.choice(block_indices)
                if not in_block[b]:
                    continue
                
                # Tentative Remove
                in_block[b] = False
                n_selected -= 1
                
                new_uncovered = 0
                for s in block_ts[b]:
                    coverage[s] -= 1
                    if coverage[s] == 0:
                        new_uncovered += 1
                
                delta_uncovered = new_uncovered
                delta_obj = -1 + P * delta_uncovered
                
                if local_rng.random() < np.exp(-delta_obj / T) if delta_obj > 0 else True:
                    # Accept
                    n_uncovered += delta_uncovered
                    current_obj += delta_obj
                    if current_obj < best_obj:
                        best_obj = current_obj
                        best_in_block = in_block.copy()
                else:
                    # Reject: Revert
                    in_block[b] = True
                    n_selected += 1
                    for s in block_ts[b]:
                        coverage[s] += 1
                        
            elif move_type < 0.8:
                # Add a random block not in selected
                b = local_rng.choice(block_indices)
                if in_block[b]:
                    continue
                
                # Tentative Add
                in_block[b] = True
                n_selected += 1
                
                old_uncovered = 0
                for s in block_ts[b]:
                    if coverage[s] == 0:
                        old_uncovered += 1
                    coverage[s] += 1
                
                delta_uncovered = -old_uncovered
                delta_obj = 1 + P * delta_uncovered
                
                if local_rng.random() < np.exp(-delta_obj / T) if delta_obj > 0 else True:
                    # Accept
                    n_uncovered += delta_uncovered
                    current_obj += delta_obj
                    if current_obj < best_obj:
                        best_obj = current_obj
                        best_in_block = in_block.copy()
                else:
                    # Reject: Revert
                    in_block[b] = False
                    n_selected -= 1
                    for s in block_ts[b]:
                        coverage[s] -= 1
                        
            else:
                # Swap: Remove b1, Add b2
                # Pick b1 from selected, b2 from unselected
                b1 = local_rng.choice(block_indices)
                if not in_block[b1]:
                    continue
                b2 = local_rng.choice(block_indices)
                if in_block[b2]:
                    continue
                if b1 == b2:
                    continue
                
                # Tentative Swap
                # Remove b1
                in_block[b1] = False
                # Add b2
                in_block[b2] = True
                
                # n_selected unchanged
                
                # Calculate change in uncovered
                # We need to know how many sets become uncovered when removing b1
                # and how many become covered when adding b2.
                
                # It's complex to calculate delta exactly without double counting.
                # Easier: just recalculate n_uncovered? No, too slow.
                
                # Let's track:
                # sets_lost: sets covered by b1 but NOT by b2 (and no other block)
                # sets_gained: sets covered by b2 but NOT by b1 (and no other block)
                
                # This requires checking coverage counts.
                
                # Step 1: Remove b1 from coverage
                # Save old coverage for sets in b1?
                # We can just iterate through union of b1 and b2.
                
                # Sets affected are those in b1 or b2.
                # For s in b1:
                #   if coverage[s] == 1 (only b1 covers it), it becomes 0 after remove.
                #   coverage[s] -= 1
                # For s in b2:
                #   if coverage[s] == 0 (before add, after remove of b1), it becomes 1.
                #   coverage[s] += 1
                
                # Order matters.
                # 1. Remove b1.
                # 2. Add b2.
                
                new_uncovered_delta = 0
                
                # Remove b1
                for s in block_ts[b1]:
                    if coverage[s] == 1:
                        # This set will become uncovered
                        new_uncovered_delta += 1
                    coverage[s] -= 1
                
                # Add b2
                for s in block_ts[b2]:
                    if coverage[s] == 0:
                        # This set was uncovered, now covered
                        new_uncovered_delta -= 1
                    coverage[s] += 1
                
                delta_obj = P * new_uncovered_delta
                
                if local_rng.random() < np.exp(-delta_obj / T) if delta_obj > 0 else True:
                    # Accept
                    n_uncovered += new_uncovered_delta
                    current_obj += delta_obj
                    if current_obj < best_obj:
                        best_obj = current_obj
                        best_in_block = in_block.copy()
                else:
                    # Reject: Revert
                    # Reverse Add b2
                    for s in block_ts[b2]:
                        coverage[s] -= 1
                    # Reverse Remove b1
                    for s in block_ts[b1]:
                        coverage[s] += 1
                    in_block[b1] = True
                    in_block[b2] = False
                    
            T *= cooling
            
        # Return best valid solution found
        # We need to extract blocks from best_in_block
        # But best_in_block might be invalid if best_obj was achieved with uncovered > 0.
        # We must ensure the returned solution is valid.
        
        # Check validity of best_in_block
        # Reconstruct coverage for best_in_block
        best_cov = np.zeros(n_ts, dtype=np.int32)
        best_blocks_list = []
        for b in range(n_blocks):
            if best_in_block[b]:
                best_blocks_list.append(b)
                for s in block_ts[b]:
                    best_cov[s] += 1
        
        if np.any(best_cov == 0):
             # Invalid. Fall back to initial or repair.
             # For now, let's repair the current state or use initial.
             # We'll return the initial blocks if the best is invalid.
             return initial_blocks
        else:
             return best_blocks_list

    # --- 3. Repair ---
    def repair(blocks):
        # Ensure validity by adding blocks greedily for uncovered sets
        coverage = np.zeros(n_ts, dtype=np.int32)
        in_block = set(blocks)
        
        for b in blocks:
            for s in block_ts[b]:
                coverage[s] += 1
        
        # Find uncovered
        uncovered = np.where(coverage == 0)[0]
        
        # Greedily cover
        while len(uncovered) > 0 and time.time() - _start_time < _DEADLINE - 2.0:
            # Find block covering max uncovered
            best_b = -1
            best_gain = 0
            # Optimize: only check blocks that cover at least one uncovered set
            # For simplicity, scan all, but break early if possible?
            # n_blocks is 50k. len(uncovered) decreases.
            # This might be slow.
            
            # Better: use ts_to_blocks
            # Pick a random uncovered set, look at its blocks, pick best?
            # Or: for each uncovered set, update gains of its blocks.
            
            # Let's do a simple pass.
            # Since this is a fallback, it shouldn't run long.
            
            # Efficient way:
            # Maintain a list of uncovered sets.
            # For each block, gain = number of uncovered sets it covers.
            # This is expensive to recompute.
            
            # Let's just pick a random uncovered set and add a random block covering it?
            # No, greedy is better.
            
            # Simple Greedy Step:
            # 1. Pick a random uncovered set s.
            # 2. Among blocks covering s, pick one that covers the most OTHER uncovered sets.
            
            s = uncovered[0] # Just take the first one
            candidates = ts_to_blocks[s]
            
            best_cand = -1
            best_gain = 0
            for b in candidates:
                if b in in_block:
                    continue
                gain = 0
                for ss in block_ts[b]:
                    if coverage[ss] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_cand = b
            
            if best_cand == -1:
                break
                
            blocks.append(best_cand)
            in_block.add(best_cand)
            
            for ss in block_ts[best_cand]:
                coverage[ss] += 1
            
            # Update uncovered list
            # This is O(n_ts) each time.
            # Since we are in a fallback, it's okay.
            uncovered = np.where(coverage == 0)[0]
            
        return blocks

    # --- 4. Polish ---
    def polish(blocks):
        # Remove redundant blocks
        coverage = np.zeros(n_ts, dtype=np.int32)
        for b in blocks:
            for s in block_ts[b]:
                coverage[s] += 1
                
        changed = True
        while changed and time.time() - _start_time < _DEADLINE - 1.0:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                b = blocks[i]
                # Check if any set is covered ONLY by b
                redundant = True
                for s in block_ts[b]:
                    if coverage[s] == 1:
                        redundant = False
                        break
                if redundant:
                    for s in block_ts[b]:
                        coverage[s] -= 1
                    blocks.pop(i)
                    changed = True
                    break
        return blocks

    # --- Main Execution ---
    
    # 1. Greedy Init
    initial_blocks = greedy_init(42)
    
    # 2. SA Optimization
    # Allocate 80% of time to SA
    sa_time = _remaining() * 0.8
    if sa_time > 10:
        optimized_blocks = sa_optimize(initial_blocks, 12345, sa_time)
    else:
        optimized_blocks = initial_blocks
    
    # 3. Repair (just in case SA returned invalid)
    optimized_blocks = repair(optimized_blocks)
    
    # 4. Polish
    optimized_blocks = polish(optimized_blocks)
    
    # 5. Convert to list of lists of ints
    result = [list(all_blocks[b]) for b in optimized_blocks]
    
    return result
