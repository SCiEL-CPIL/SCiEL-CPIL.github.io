"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 256
# EVOLVE-BLOCK-START

def _is_maximal_fast(selected, N):
    """Check maximality using bytearray for sum presence."""
    sel_set = set(selected)
    max_sum = 2 * N
    sum_arr = bytearray(max_sum + 1)
    n = len(selected)
    for i in range(n):
        a = selected[i]
        for j in range(i, n):
            s = a + selected[j]
            sum_arr[s] = 1
    
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        if sum_arr[2 * x]:
            continue
        blocked = False
        for a in selected:
            if sum_arr[x + a]:
                blocked = True
                break
        if not blocked:
            return False
    return True

def _greedy_maximal_ordered(order, N):
    """Generate a maximal Sidon set using greedy saturation in a specific order."""
    selected = []
    max_sum = 2 * N
    sum_arr = bytearray(max_sum + 1)
    
    for x in order:
        ok = True
        if sum_arr[2 * x]:
            ok = False
        else:
            for a in selected:
                if sum_arr[x + a]:
                    ok = False
                    break
        if not ok:
            continue
            
        # Add x
        sum_arr[2 * x] = 1
        for a in selected:
            sum_arr[x + a] = 1
        selected.append(x)
        
    return selected

def _shrink_maximal_fast(selected, N, deadline):
    """Try to remove elements while maintaining maximality, using fast check."""
    selected = list(selected)
    
    while time.time() < deadline:
        improved = False
        indices = list(range(len(selected)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
            x = selected[idx]
            candidate = selected[:idx] + selected[idx+1:]
            
            if _is_maximal_fast(candidate, N):
                selected = candidate
                improved = True
                break
        
        if not improved:
            break
            
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best = None
    best_size = float('inf')
    
    num_restarts = 50000 # High limit, time is the constraint
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        rng = random.Random(i * 12345 + 67890)
        
        # Diversify start: Random, Ascending, Descending
        mode = i % 3
        if mode == 0:
            order = list(range(1, N + 1))
            rng.shuffle(order)
        elif mode == 1:
            order = list(range(1, N + 1))
        else:
            order = list(range(N, 0, -1))
            
        sel = _greedy_maximal_ordered(order, N)
        
        if not sel:
            continue
            
        sel = _shrink_maximal_fast(sel, N, deadline)
        
        if len(sel) < best_size:
            best = sel
            best_size = len(sel)
            
            if best_size <= 16:
                break

    if best is None:
        # Fallback
        sel = _greedy_maximal_ordered(list(range(1, N + 1)), N)
        return sel
        
    return best

# EVOLVE-BLOCK-END
