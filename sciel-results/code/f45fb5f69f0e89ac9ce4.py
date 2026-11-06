import time
import random
import sys

N = 256
MAX_SUM = 2 * N  # 512

def _compute_sums_bitmask(sel):
    """Compute sum bitmask: bit s is set if s can be expressed as a+b with a<=b from sel."""
    mask = 0
    for i in range(len(sel)):
        a = sel[i]
        for j in range(i, len(sel)):
            mask |= (1 << (a + sel[j]))
    return mask

def _is_maximal_bitmask(sel, sum_mask):
    """Check maximality using bitmask. Returns True if sel is inclusion-maximal Sidon."""
    sel_set = set(sel)
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # Condition 1: 2*x already a sum
        if (sum_mask >> (2 * x)) & 1:
            continue
        # Condition 2: x + a is a sum for some a in sel
        blocked = False
        for a in sel:
            if (sum_mask >> (x + a)) & 1:
                blocked = True
                break
        if not blocked:
            return False
    return True

def _greedy_construct_bitmask(order):
    """Greedy construction using bitmask sum tracking."""
    selected = []
    sum_mask = 0
    for x in order:
        # Check if x can be added
        if (sum_mask >> (2 * x)) & 1:
            continue
        can_add = True
        for a in selected:
            if (sum_mask >> (x + a)) & 1:
                can_add = False
                break
        if can_add:
            selected.append(x)
            # Update sum_mask
            sum_mask |= (1 << (2 * x))
            for a in selected:
                sum_mask |= (1 << (x + a))
    return selected, sum_mask

def _local_search_reduce_bitmask(sel, sum_mask, deadline):
    """
    Try to remove elements from sel while keeping it maximal, using bitmask.
    """
    current_sel = list(sel)
    current_mask = sum_mask
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
            
            # Recompute sums for new_sel
            new_mask = _compute_sums_bitmask(new_sel)
            
            if _is_maximal_bitmask(new_sel, new_mask):
                current_sel = new_sel
                current_mask = new_mask
                improved = True
                break
        
    return current_sel

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin
    
    best_set = None
    best_size = float('inf')
    
    candidates = []
    
    # Ascending
    sel, mask = _greedy_construct_bitmask(list(range(1, N + 1)))
    if _is_maximal_bitmask(sel, mask):
        candidates.append((sel, mask))
        
    # Descending
    sel, mask = _greedy_construct_bitmask(list(range(N, 0, -1)))
    if _is_maximal_bitmask(sel, mask):
        candidates.append((sel, mask))
        
    # Randomized Greedy - more iterations due to speedup
    rng = random.Random(12345)
    num_initial_trials = 200
    for trial in range(num_initial_trials):
        if time.time() > deadline - 15:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        sel, mask = _greedy_construct_bitmask(order)
        if _is_maximal_bitmask(sel, mask):
            candidates.append((sel, mask))
            
    # Local search on each candidate
    for sel, mask in candidates:
        if time.time() > deadline - 3:
            break
            
        if len(sel) < best_size:
            best_size = len(sel)
            best_set = sel
            
        optimized_sel = _local_search_reduce_bitmask(sel, mask, deadline)
        
        if len(optimized_sel) < best_size:
            best_size = len(optimized_sel)
            best_set = optimized_sel
            
    # Fallback
    if best_set is None:
        sel, mask = _greedy_construct_bitmask(list(range(1, N + 1)))
        if _is_maximal_bitmask(sel, mask):
            best_set = sel
        else:
            # Force maximality by adding elements
            sel_set = set(sel)
            changed = True
            while changed and time.time() < deadline:
                changed = False
                for x in range(1, N + 1):
                    if x in sel_set:
                        continue
                    if (mask >> (2 * x)) & 1:
                        continue
                    ok = True
                    for a in sel:
                        if (mask >> (x + a)) & 1:
                            ok = False
                            break
                    if ok:
                        sel.append(x)
                        sel_set.add(x)
                        mask |= (1 << (2 * x))
                        for a in sel:
                            mask |= (1 << (x + a))
                        changed = True
                        break
            if _is_maximal_bitmask(sel, mask):
                best_set = sel
            else:
                best_set = sel
                
    return best_set
