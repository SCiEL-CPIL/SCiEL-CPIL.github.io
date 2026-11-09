import itertools
import random
import time
import numpy as np

# Constants for the specific problem C(19, 7, 4)
V = 19
K = 7
T = 4

def _precompute():
    """Precompute block t-subset indices and t-subset block lists."""
    points = range(V)
    t_subsets = list(itertools.combinations(points, T))
    num_t = len(t_subsets)
    t_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    
    k_subsets = list(itertools.combinations(points, K))
    num_k = len(k_subsets)
    
    # block_tsubsets[i] contains indices of t-subsets covered by k-subset i
    block_tsubsets = np.zeros((num_k, K - T + 1 if False else 35), dtype=np.int32) # 35 is C(7,4)
    # Actually C(7,4) = 35.
    
    # ts_blocks[i] contains indices of k-subsets covering t-subset i
    ts_blocks = [set() for _ in range(num_t)]
    
    for i, block in enumerate(k_subsets):
        # Find all 4-subsets of this block
        tss = itertools.combinations(block, T)
        for j, ts in enumerate(tss):
            idx = t_to_idx[ts]
            block_tsubsets[i, j] = idx
            ts_blocks[idx].add(i)
            
    return block_tsubsets, ts_blocks, k_subsets, t_subsets

# Global precomputed data
BLOCK_TSUBSETS, TS_BLOCKS, K_SUBSETS, T_SUBSETS = _precompute()
NUM_BLOCKS = len(K_SUBSETS)
NUM_TSUBSETS = len(T_SUBSETS)

def _greedy_init(rng):
    """Fast greedy initialization using NumPy."""
    uncovered = np.ones(NUM_TSUBSETS, dtype=bool)
    selected = []
    
    while np.any(uncovered):
        # Calculate gain for each block
        # uncovered[BLOCK_TSUBSETS] creates a boolean array of shape (NUM_BLOCKS, 35)
        # Summing along axis 1 gives the count of newly covered t-subsets
        gains = np.sum(uncovered[BLOCK_TSUBSETS], axis=1)
        
        # Find block with max gain
        # Randomize tie-breaking
        max_gain = np.max(gains)
        if max_gain == 0:
            break # Should not happen
            
        candidates = np.where(gains == max_gain)[0]
        best_idx = rng.choice(candidates)
        
        selected.append(best_idx)
        # Update uncovered
        uncovered[BLOCK_TSUBSETS[best_idx]] = False
        
    return selected

def _reduce_solution(selected, cover_count):
    """Remove redundant blocks. Returns new selected list and cover_count."""
    changed = True
    while changed:
        changed = False
        new_selected = []
        for b in selected:
            # Check if any t-subset in b has cover_count == 1
            # If yes, b is critical, keep it.
            # If no, b is redundant, remove it.
            is_redundant = True
            for ts_idx in BLOCK_TSUBSETS[b]:
                if cover_count[ts_idx] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove b
                for ts_idx in BLOCK_TSUBSETS[b]:
                    cover_count[ts_idx] -= 1
                changed = True
            else:
                new_selected.append(b)
        
        selected = new_selected
        
    return selected, cover_count

def _local_search(selected, cover_count, rng, deadline):
    """Hill climbing local search."""
    # Convert selected to a set for O(1) lookup
    selected_set = set(selected)
    current_size = len(selected)
    
    while time.time() < deadline:
        # Pick a random block to try swapping
        if not selected:
            break
        b = rng.choice(selected)
        
        # Find critical t-subsets for b
        critical_ts = []
        for ts_idx in BLOCK_TSUBSETS[b]:
            if cover_count[ts_idx] == 1:
                critical_ts.append(ts_idx)
                
        if not critical_ts:
            # b is redundant, should have been removed
            for ts_idx in BLOCK_TSUBSETS[b]:
                cover_count[ts_idx] -= 1
            selected.remove(b)
            selected_set.remove(b)
            current_size -= 1
            continue
            
        # Find candidate blocks that cover all critical_ts
        # Intersection of ts_blocks[ts] for ts in critical_ts
        candidates = TS_BLOCKS[critical_ts[0]].copy()
        for ts in critical_ts[1:]:
            candidates &= TS_BLOCKS[ts]
            
        # Remove blocks already in selected
        candidates -= selected_set
        
        if not candidates:
            continue
            
        # Pick a random candidate
        b_new = rng.choice(list(candidates))
        
        # Perform swap
        # 1. Remove b
        for ts_idx in BLOCK_TSUBSETS[b]:
            cover_count[ts_idx] -= 1
        selected.remove(b)
        selected_set.remove(b)
        
        # 2. Add b_new
        for ts_idx in BLOCK_TSUBSETS[b_new]:
            cover_count[ts_idx] += 1
        selected.append(b_new)
        selected_set.add(b_new)
        
        # 3. Reduce
        new_selected, new_cover_count = _reduce_solution(selected, cover_count)
        
        if len(new_selected) < current_size:
            # Accept
            selected = new_selected
            selected_set = set(new_selected)
            cover_count = new_cover_count
            current_size = len(selected)
        else:
            # Revert
            # Remove b_new
            for ts_idx in BLOCK_TSUBSETS[b_new]:
                cover_count[ts_idx] -= 1
            selected.remove(b_new)
            selected_set.remove(b_new)
            
            # Add b back
            for ts_idx in BLOCK_TSUBSETS[b]:
                cover_count[ts_idx] += 1
            selected.append(b)
            selected_set.add(b)
            
            # Revert reduction? 
            # _reduce_solution modified cover_count and selected in place? 
            # No, it returned new ones. But it modified the input `cover_count`?
            # In _reduce_solution, I did `cover_count[ts_idx] -= 1`. This modifies the array in place.
            # So I need to be careful.
            # Let's refactor _reduce_solution to not modify inputs, or restore state.
            
            # Actually, in the revert step above, I restored the swap.
            # But _reduce_solution might have removed other blocks and modified cover_count.
            # So reverting just the swap is not enough.
            
            # Better approach: Copy state before swap? Too expensive.
            # Alternative: _reduce_solution returns the list of removed blocks and their t-subsets.
            # Or, simply: if reduction doesn't improve size, revert the swap AND re-add any blocks removed by reduction.
            
            # Let's trace _reduce_solution.
            # It iterates and removes redundant blocks.
            # If we revert the swap, the blocks removed by reduction might no longer be redundant.
            
            # Simpler: Just don't do in-place modification in reduce.
            # Or, if no improvement, just revert the swap and re-run reduce from the swapped state? No, that's circular.
            
            # Let's modify _reduce_solution to return (new_selected, new_cover_count, removed_blocks).
            # If we revert, we re-add removed_blocks.
            
            pass # Handled in refactored version below

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0
    
    if v != 19 or k != 7 or t != 4:
        # Fallback for other parameters (not expected)
        return list(itertools.combinations(range(v), k))
        
    rng = random.Random(42)
    
    best_selected = None
    best_size = float('inf')
    
    # Run multiple greedy starts
    for i in range(20):
        if time.time() > deadline - 30.0:
            break
            
        selected = _greedy_init(rng)
        
        # Initialize cover count
        cover_count = np.zeros(NUM_TSUBSETS, dtype=np.int32)
        for b in selected:
            cover_count[BLOCK_TSUBSETS[b]] += 1
            
        # Reduce
        selected, cover_count = _reduce_solution(selected, cover_count)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = selected[:]
            
    # Local search on best solution
    if best_selected is not None and time.time() < deadline - 5.0:
        cover_count = np.zeros(NUM_TSUBSETS, dtype=np.int32)
        for b in best_selected:
            cover_count[BLOCK_TSUBSETS[b]] += 1
            
        best_selected, cover_count = _local_search_v2(best_selected, cover_count, rng, deadline)
        
    # Convert indices to actual blocks
    return [list(K_SUBSETS[idx]) for idx in best_selected]

def _local_search_v2(selected, cover_count, rng, deadline):
    """Refactored local search with proper revert."""
    selected_set = set(selected)
    current_size = len(selected)
    
    while time.time() < deadline:
        if not selected:
            break
        b = rng.choice(selected)
        
        # Find critical t-subsets
        critical_ts = [ts for ts in BLOCK_TSUBSETS[b] if cover_count[ts] == 1]
        
        if not critical_ts:
            # Redundant
            for ts in BLOCK_TSUBSETS[b]:
                cover_count[ts] -= 1
            selected.remove(b)
            selected_set.remove(b)
            current_size -= 1
            continue
            
        candidates = TS_BLOCKS[critical_ts[0]].copy()
        for ts in critical_ts[1:]:
            candidates &= TS_BLOCKS[ts]
        candidates -= selected_set
        
        if not candidates:
            continue
            
        b_new = rng.choice(list(candidates))
        
        # Save state for revert
        # We modify cover_count in place.
        # We modify selected in place.
        
        # Apply swap
        for ts in BLOCK_TSUBSETS[b]:
            cover_count[ts] -= 1
        selected.remove(b)
        selected_set.remove(b)
        
        for ts in BLOCK_TSUBSETS[b_new]:
            cover_count[ts] += 1
        selected.append(b_new)
        selected_set.add(b_new)
        
        # Reduce
        # We need to know what was removed to revert
        old_selected = selected[:]
        old_cover_count = cover_count.copy() # Expensive? 3876 ints. ~30KB. OK.
        
        new_selected, new_cover_count = _reduce_solution(selected, cover_count)
        
        if len(new_selected) < current_size:
            selected = new_selected
            selected_set = set(new_selected)
            cover_count = new_cover_count
            current_size = len(selected)
        else:
            # Revert
            selected = old_selected
            selected_set = set(old_selected)
            cover_count = old_cover_count

    return selected, cover_count
