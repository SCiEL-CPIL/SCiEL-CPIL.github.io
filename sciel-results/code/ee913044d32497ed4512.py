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

    # --- 1. Greedy Initialization with Heap ---
    def greedy_init(seed):
        local_rng = random.Random(seed)
        # uncovered: boolean array, True if not covered
        uncovered = np.ones(n_ts, dtype=bool)
        uncovered_count = n_ts
        
        # gain[b] = number of currently uncovered t-subsets covered by block b
        gains = np.zeros(n_blocks, dtype=np.int32)
        for b in range(n_blocks):
            gains[b] = len(block_ts[b])
        
        # Use a max-heap (negate gains for min-heap behavior)
        # Each entry: (-gain, block_idx)
        heap = [(-gains[b], b) for b in range(n_blocks)]
        heapq.heapify(heap)
        
        # Lazy deletion: track stale entries by checking current gain
        selected = []
        
        while uncovered_count > 0:
            if time.time() - _start_time > _DEADLINE - 5.0:
                break
            
            # Pop until we find a valid (non-stale) entry
            best_b = -1
            best_gain = -1
            while heap:
                neg_gain, b = heapq.heappop(heap)
                current_gain = gains[b]
                if -neg_gain == current_gain and current_gain > 0:
                    best_b = b
                    best_gain = current_gain
                    break
                elif -neg_gain == current_gain and current_gain <= 0:
                    break
                # else: stale entry, discard
            
            if best_gain <= 0:
                break
            
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
                            gains[other_b] = 0
                        # Push updated gain to heap (lazy: old entries will be discarded)
                        heapq.heappush(heap, (-gains[other_b], other_b))
        
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
        
        n_uncovered = int(np.sum(coverage == 0))
        n_selected = len(initial_blocks)
        
        # Maintain a list of selected block indices for efficient random selection
        selected_list = list(initial_blocks)
        
        # Objective: n_selected + P * n_uncovered
        P = 1000
        
        current_obj = n_selected + P * n_uncovered
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
            
            if move_type < 0.6:
                # Remove a random block from selected
                if n_selected == 0:
                    continue
                
                # Pick random index from selected_list
                idx = local_rng.randrange(n_selected)
                b = selected_list[idx]
                
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
                    # Remove from selected_list (swap with last for O(1))
                    selected_list[idx] = selected_list[-1]
                    selected_list.pop()
                    if current_obj < best_obj:
                        best_obj = current_obj
                        best_in_block = in_block.copy()
                else:
                    # Reject: Revert
                    in_block[b] = True
                    n_selected += 1
                    selected_list.append(b)
                    for s in block_ts[b]:
                        coverage[s] += 1
                        
            elif move_type < 0.8:
                # Add a random block not in selected
                b = int(local_rng.choice(block_indices))
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
                    selected_list.append(b)
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
                if n_selected == 0:
                    continue
                # Pick b1 from selected
                idx1 = local_rng.randrange(n_selected)
                b1 = selected_list[idx1]
                # Pick b2 from unselected
                b2 = int(local_rng.choice(block_indices))
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
                
                new_uncovered_delta = 0
                
                # Remove b1
                for s in block_ts[b1]:
                    if coverage[s] == 1:
                        new_uncovered_delta += 1
                    coverage[s] -= 1
                
                # Add b2
                for s in block_ts[b2]:
                    if coverage[s] == 0:
                        new_uncovered_delta -= 1
                    coverage[s] += 1
                
                delta_obj = P * new_uncovered_delta
                
                if local_rng.random() < np.exp(-delta_obj / T) if delta_obj > 0 else True:
                    # Accept
                    n_uncovered += new_uncovered_delta
                    current_obj += delta_obj
                    selected_list[idx1] = b2
                    if current_obj < best_obj:
                        best_obj = current_obj
                        best_in_block = in_block.copy()
                else:
                    # Reject: Revert
                    for s in block_ts[b2]:
                        coverage[s] -= 1
                    for s in block_ts[b1]:
                        coverage[s] += 1
                    in_block[b1] = True
                    in_block[b2] = False
                    
            T *= cooling
            
        # Return best valid solution found
        best_cov = np.zeros(n_ts, dtype=np.int32)
        best_blocks_list = []
        for b in range(n_blocks):
            if best_in_block[b]:
                best_blocks_list.append(b)
                for s in block_ts[b]:
                    best_cov[s] += 1
        
        if np.any(best_cov == 0):
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
        # Remove redundant blocks using count-based check
        coverage = np.zeros(n_ts, dtype=np.int32)
        for b in blocks:
            for s in block_ts[b]:
                coverage[s] += 1
                
        # Work on a copy to avoid modifying during iteration
        blocks_list = list(blocks)
        changed = True
        while changed and time.time() - _start_time < _DEADLINE - 1.0:
            changed = False
            for i in range(len(blocks_list) - 1, -1, -1):
                b = blocks_list[i]
                # Check if any set is covered ONLY by b
                redundant = True
                for s in block_ts[b]:
                    if coverage[s] == 1:
                        redundant = False
                        break
                if redundant:
                    for s in block_ts[b]:
                        coverage[s] -= 1
                    blocks_list.pop(i)
                    changed = True
                    break
        return blocks_list

    # --- Main Execution ---
    
    # 1. Multiple Greedy Inits with different seeds
    greedy_seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000]
    best_greedy = None
    best_greedy_len = float('inf')
    
    for seed in greedy_seeds:
        if time.time() - _start_time > _DEADLINE - 30.0:
            break
        gb = greedy_init(seed)
        if len(gb) < best_greedy_len:
            best_greedy_len = len(gb)
            best_greedy = gb
    
    initial_blocks = best_greedy
    
    # 2. SA Optimization on the best greedy solution
    sa_time = _remaining() * 0.85
    if sa_time > 5:
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
