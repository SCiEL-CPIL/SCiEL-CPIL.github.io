import time
import random
import numpy as np

N = 16
DEADLINE = 325.0
N_TOTAL = N * N

# Precompute all points and pairwise squared distances
_ALL_POINTS = [(x, y) for y in range(N) for x in range(N)]
_POINT_INDEX = {p: i for i, p in enumerate(_ALL_POINTS)}
_D2 = np.zeros((N_TOTAL, N_TOTAL), dtype=np.int32)
for i, p1 in enumerate(_ALL_POINTS):
    for j, p2 in enumerate(_ALL_POINTS):
        _D2[i, j] = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def _can_add_vec(idx_p, chosen_indices):
    """Check if point idx_p can be added to chosen without creating isosceles triangle."""
    if len(chosen_indices) < 2:
        return True
    ch = np.array(chosen_indices, dtype=np.int32)
    # For each pair (a,b) in chosen, check if (p,a,b) is isosceles
    # d(p,a) == d(p,b) or d(p,a) == d(a,b) or d(p,b) == d(a,b)
    dp = _D2[idx_p, ch]  # distances from p to each chosen point
    for i in range(len(ch)):
        for j in range(i+1, len(ch)):
            da = dp[i]
            db = dp[j]
            dab = _D2[ch[i], ch[j]]
            if da == db or da == dab or db == dab:
                return False
    return True

def _blocked_by_vec(idx_p, chosen_indices):
    """Check if point idx_p is blocked by chosen_indices (forms isosceles with some pair)."""
    if len(chosen_indices) < 2:
        return False
    ch = np.array(chosen_indices, dtype=np.int32)
    dp = _D2[idx_p, ch]
    for i in range(len(ch)):
        for j in range(i+1, len(ch)):
            da = dp[i]
            db = dp[j]
            dab = _D2[ch[i], ch[j]]
            if da == db or da == dab or db == dab:
                return True
    return False

def _is_no_isosceles_vec(indices):
    """Check if set of point indices forms no-isosceles set."""
    n = len(indices)
    if n < 3:
        return True
    idx = np.array(indices, dtype=np.int32)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                d1 = _D2[idx[i], idx[j]]
                d2 = _D2[idx[i], idx[k]]
                d3 = _D2[idx[j], idx[k]]
                if d1 == d2 or d1 == d3 or d2 == d3:
                    return False
    return True

def _is_maximal_vec(chosen_indices, time_limit=None):
    """Check if chosen set is maximal (no point can be added)."""
    chosen_set = set(chosen_indices)
    for i in range(N_TOTAL):
        if i in chosen_set:
            continue
        if not _blocked_by_vec(i, chosen_indices):
            return False
    return True

def _reduce_to_minimal_vec(chosen_indices):
    """Iteratively remove points if remaining set is still maximal."""
    S = list(chosen_indices)
    changed = True
    while changed:
        changed = False
        for i in range(len(S)):
            if len(S) - 1 < 2:
                continue
            S_test = S[:i] + S[i+1:]
            if _is_maximal_vec(S_test):
                S = S_test
                changed = True
                break
    return S

def _greedy_build_from_core(core_indices, deadline_margin=10):
    """
    Greedily build maximal set from a no-isosceles core.
    Returns list of point indices, or None on timeout.
    """
    chosen = list(core_indices)
    chosen_set = set(chosen)
    
    while True:
        # Find all addable points
        addable = []
        for i in range(N_TOTAL):
            if i in chosen_set:
                continue
            if _can_add_vec(i, chosen):
                addable.append(i)
        
        if not addable:
            break
        
        # Check if already maximal
        if _is_maximal_vec(chosen):
            return chosen
        
        # Score each addable point by how many currently-unblocked points it would block
        # First find unblocked points
        unblocked = []
        for i in range(N_TOTAL):
            if i in chosen_set:
                continue
            if not _blocked_by_vec(i, chosen):
                unblocked.append(i)
        
        best_p = None
        best_gain = -1
        for p in addable:
            gain = 0
            for q in unblocked:
                if q == p:
                    gain += 1
                    continue
                # Check if q would be blocked by adding p
                # q blocked if exists a in chosen s.t. (q,p,a) isosceles
                blocked = False
                for a in chosen:
                    dpq = _D2[p, q]
                    daq = _D2[a, q]
                    dap = _D2[a, p]
                    if dpq == daq or dpq == dap or daq == dap:
                        blocked = True
                        break
                if blocked:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_p = p
        
        if best_p is None:
            break
        
        chosen.append(best_p)
        chosen_set.add(best_p)
        
        if time.time() > DEADLINE - deadline_margin:
            return None
    
    return chosen

def construct():
    start_time = time.time()
    
    best_set = None
    best_size = N_TOTAL + 1
    
    rng = random.Random(42)
    
    # Strategy 1: Try small cores and build up with greedy coverage
    core_sizes = [2, 3, 4, 5, 6, 7, 8]
    
    for core_size in core_sizes:
        if time.time() - start_time > DEADLINE - 40:
            break
        
        # Try random cores
        n_attempts = 200
        for attempt in range(n_attempts):
            if time.time() - start_time > DEADLINE - 15:
                break
            
            # Pick random core
            core_indices = rng.sample(range(N_TOTAL), core_size)
            
            # Check if core is no-isosceles
            if not _is_no_isosceles_vec(core_indices):
                continue
            
            # Try to build maximal set from this core
            maximal = _greedy_build_from_core(core_indices, deadline_margin=10)
            
            if maximal is None:
                continue
            
            # Reduce to minimal
            reduced = _reduce_to_minimal_vec(maximal)
            
            if len(reduced) < best_size:
                best_size = len(reduced)
                best_set = reduced[:]
            
            if best_size <= 8:
                return [_ALL_POINTS[i] for i in best_set]
    
    # Strategy 2: Random greedy + reduction
    for attempt in range(500):
        if time.time() - start_time > DEADLINE - 20:
            break
            
        order = list(range(N_TOTAL))
        rng.shuffle(order)
        
        chosen = []
        for i in order:
            if _can_add_vec(i, chosen):
                chosen.append(i)
        
        reduced = _reduce_to_minimal_vec(chosen)
        
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced[:]
            
        if best_size <= 8:
            break
    
    # Strategy 3: Local search on best set - try swaps
    if best_set is not None and best_size > 8:
        for attempt in range(100):
            if time.time() - start_time > DEADLINE - 10:
                break
                
            S_list = list(best_set)
            n_S = len(S_list)
            
            improved = False
            for i in range(n_S):
                S_without = S_list[:i] + S_list[i+1:]
                
                # Try to add a different point
                for p_add in range(N_TOTAL):
                    if p_add in S_without:
                        continue
                    
                    if not _can_add_vec(p_add, S_without):
                        continue
                    
                    S_new = S_without + [p_add]
                    
                    if not _is_maximal_vec(S_new):
                        continue
                    
                    # Try to reduce
                    reduced = _reduce_to_minimal_vec(S_new)
                    
                    if len(reduced) < best_size:
                        best_size = len(reduced)
                        best_set = reduced[:]
                        improved = True
                        break
                    
                    if time.time() - start_time > DEADLINE - 10:
                        break
                if improved or time.time() - start_time > DEADLINE - 10:
                    break
            
            if not improved:
                break
            
            if best_size <= 8:
                break
    
    if best_set is None:
        # Fallback: greedy construction
        chosen = []
        for i in range(N_TOTAL):
            if _can_add_vec(i, chosen):
                chosen.append(i)
        best_set = _reduce_to_minimal_vec(chosen)
        
    return [_ALL_POINTS[i] for i in best_set]
