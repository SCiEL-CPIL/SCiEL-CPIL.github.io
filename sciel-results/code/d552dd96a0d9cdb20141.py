import time
import random
import sys
import numpy as np

N = 256
MAX_SUM = 2 * N  # Maximum possible sum is 2*N

def _make_sum_array(sel):
    """Create a boolean array where sums[s] = True if s is a sum of two elements in sel."""
    sums = bytearray(MAX_SUM + 1)
    for i in range(len(sel)):
        a = sel[i]
        for j in range(i, len(sel)):
            s = a + sel[j]
            sums[s] = 1
    return sums

def _is_maximal_fast(sel, sums_arr):
    """
    Fast maximality check using bytearray for sums.
    Returns True if sel is inclusion-maximal (no missing element can be added).
    """
    sel_set = set(sel)
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        
        # Condition 1: 2*x is already a sum
        if sums_arr[2 * x]:
            continue
        
        # Condition 2: x + a is a sum for some a in sel
        blocked = False
        for a in sel:
            if sums_arr[x + a]:
                blocked = True
                break
        
        if not blocked:
            return False
            
    return True

def _greedy_construct_fast(order):
    """Greedy construction with bytearray sum tracking."""
    selected = []
    sums_arr = bytearray(MAX_SUM + 1)
    for x in order:
        if sums_arr[2 * x]:
            continue
        can_add = True
        for a in selected:
            if sums_arr[x + a]:
                can_add = False
                break
        if can_add:
            selected.append(x)
            sums_arr[2 * x] = 1
            for a in selected:
                sums_arr[x + a] = 1
    return selected, sums_arr

def _local_search_reduce_fast(sel, sums_arr, deadline):
    """
    Try to remove elements from sel while keeping it maximal.
    Uses bytearray for fast sum recomputation.
    """
    current_sel = list(sel)
    current_sums = sums_arr
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        current_sel_set = set(current_sel)
        
        # Try removing each element
        for i in range(len(current_sel)):
            if time.time() > deadline:
                return current_sel
            
            x_to_remove = current_sel[i]
            new_sel = current_sel[:i] + current_sel[i+1:]
            
            # Recompute sums for the new set
            new_sums = _make_sum_array(new_sel)
            
            if _is_maximal_fast(new_sel, new_sums):
                current_sel = new_sel
                current_sums = new_sums
                improved = True
                break
    return current_sel

def _try_remove_multiple(sel, sums_arr, deadline):
    """
    Try to remove multiple elements at once for faster reduction.
    """
    current_sel = list(sel)
    current_sums = sums_arr
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        n = len(current_sel)
        
        # Try all pairs of removals
        for i in range(n):
            if time.time() > deadline:
                return current_sel
            for j in range(i + 1, n):
                new_sel = current_sel[:i] + current_sel[i+1:j] + current_sel[j+1:]
                new_sums = _make_sum_array(new_sel)
                if _is_maximal_fast(new_sel, new_sums):
                    current_sel = new_sel
                    current_sums = new_sums
                    improved = True
                    break
            if improved:
                break
    return current_sel

def _bitmask_sums(sel):
    """Create a Python int bitmask where bit s is set if s is a sum of two elements."""
    mask = 0
    for i in range(len(sel)):
        a = sel[i]
        for j in range(i, len(sel)):
            s = a + sel[j]
            mask |= (1 << s)
    return mask

def _is_maximal_bitmask(sel, mask):
    """Fast maximality check using bitmask."""
    sel_set = set(sel)
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # Check if 2*x is blocked
        if (mask >> (2 * x)) & 1:
            continue
        # Check if x+a is blocked for some a in sel
        blocked = False
        for a in sel:
            if (mask >> (x + a)) & 1:
                blocked = True
                break
        if not blocked:
            return False
    return True

def _greedy_construct_bitmask(order):
    """Greedy construction with bitmask sum tracking."""
    selected = []
    mask = 0
    for x in order:
        if (mask >> (2 * x)) & 1:
            continue
        can_add = True
        for a in selected:
            if (mask >> (x + a)) & 1:
                can_add = False
                break
        if can_add:
            selected.append(x)
            mask |= (1 << (2 * x))
            for a in selected:
                mask |= (1 << (x + a))
    return selected, mask

def _local_search_bitmask(sel, mask, deadline):
    """Local search using bitmask for faster checks."""
    current_sel = list(sel)
    current_mask = mask
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(current_sel)):
            if time.time() > deadline:
                return current_sel
            new_sel = current_sel[:i] + current_sel[i+1:]
            new_mask = _bitmask_sums(new_sel)
            if _is_maximal_bitmask(new_sel, new_mask):
                current_sel = new_sel
                current_mask = new_mask
                improved = True
                break
    return current_sel

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin for 60s limit
    
    best_set = None
    best_size = float('inf')
    
    # 1. Generate initial candidates using different greedy strategies
    candidates = []
    
    # Ascending
    sel, sums = _greedy_construct_fast(list(range(1, N + 1)))
    if _is_maximal_fast(sel, sums):
        candidates.append((sel, sums))
        
    # Descending
    sel, sums = _greedy_construct_fast(list(range(N, 0, -1)))
    if _is_maximal_fast(sel, sums):
        candidates.append((sel, sums))
        
    # Randomized Greedy
    rng = random.Random(12345)
    num_initial_trials = 800
    for _ in range(num_initial_trials):
        if time.time() > deadline - 10:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        sel, sums = _greedy_construct_fast(order)
        if _is_maximal_fast(sel, sums):
            candidates.append((sel, sums))
            
    # 2. Local search on each candidate
    for sel, sums in candidates:
        if time.time() > deadline - 3:
            break
            
        # Update best if this initial set is already better
        if len(sel) < best_size:
            best_size = len(sel)
            best_set = list(sel)
            
        # Run local search to minimize size
        optimized_sel = _local_search_reduce_fast(sel, sums, deadline)
        
        if len(optimized_sel) < best_size:
            best_size = len(optimized_sel)
            best_set = optimized_sel
            
            # Try multi-removal on the improved set
            if time.time() < deadline - 5:
                further_optimized = _try_remove_multiple(optimized_sel, _make_sum_array(optimized_sel), deadline)
                if len(further_optimized) < best_size:
                    best_size = len(further_optimized)
                    best_set = further_optimized
    
    # 3. Bitmask-based search for additional improvements
    if time.time() < deadline - 5:
        rng2 = random.Random(99999)
        for _ in range(300):
            if time.time() > deadline - 3:
                break
            order = list(range(1, N + 1))
            rng2.shuffle(order)
            sel, mask = _greedy_construct_bitmask(order)
            if _is_maximal_bitmask(sel, mask):
                if len(sel) < best_size:
                    best_size = len(sel)
                    best_set = list(sel)
                optimized_sel = _local_search_bitmask(sel, mask, deadline)
                if len(optimized_sel) < best_size:
                    best_size = len(optimized_sel)
                    best_set = optimized_sel
    
    # 4. Final polish: try to remove elements from best_set
    if best_set is not None and time.time() < deadline - 2:
        final_sel = _local_search_reduce_fast(best_set, _make_sum_array(best_set), deadline)
        if len(final_sel) < best_size:
            best_size = len(final_sel)
            best_set = final_sel
    
    # 5. If no valid set found (should not happen), fallback
    if best_set is None:
        sel, sums = _greedy_construct_fast(list(range(1, N + 1)))
        if _is_maximal_fast(sel, sums):
            best_set = sel
        else:
            # Force maximality by adding elements
            sel_set = set(sel)
            sums_arr = _make_sum_array(sel)
            changed = True
            while changed and time.time() < deadline:
                changed = False
                for x in range(1, N + 1):
                    if x in sel_set:
                        continue
                    if sums_arr[2 * x]:
                        continue
                    ok = True
                    for a in sel:
                        if sums_arr[x + a]:
                            ok = False
                            break
                    if ok:
                        sel.append(x)
                        sel_set.add(x)
                        sums_arr[2 * x] = 1
                        for a in sel:
                            sums_arr[x + a] = 1
                        changed = True
                        break
            best_set = sel
            
    return best_set
