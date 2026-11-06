import time
import random
import numpy as np

N = 16
DEADLINE = 320.0
N_TOTAL = N * N

_ALL_POINTS = [(x, y) for y in range(N) for x in range(N)]

# Precompute squared distances
_D2 = np.zeros((N_TOTAL, N_TOTAL), dtype=np.int32)
for i, p1 in enumerate(_ALL_POINTS):
    for j, p2 in enumerate(_ALL_POINTS):
        _D2[i, j] = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

# Precompute isosceles relation masks
# iso_mask[i][j] = bitmask of points k such that (i,j,k) forms an isosceles triangle
# (excluding i and j themselves)
iso_mask = [[0] * N_TOTAL for _ in range(N_TOTAL)]

for i in range(N_TOTAL):
    for j in range(i + 1, N_TOTAL):
        di_j = _D2[i, j]
        mask = 0
        for k in range(N_TOTAL):
            if k == i or k == j:
                continue
            di_k = _D2[i, k]
            dj_k = _D2[j, k]
            if di_k == dj_k or di_k == di_j or dj_k == di_j:
                mask |= (1 << k)
        iso_mask[i][j] = mask
        iso_mask[j][i] = mask

def _is_no_isosceles(S_mask, S_list):
    """Check if set S is no-isosceles."""
    n = len(S_list)
    if n < 3:
        return True
    for i in range(n):
        for j in range(i + 1, n):
            a = S_list[i]
            b = S_list[j]
            if iso_mask[a][b] & S_mask:
                return False
    return True

def _get_blocked_mask(S_list):
    """
    Return a bitmask of all points blocked by pairs in S_list.
    A point p is blocked if it forms an isosceles triangle with some pair (a,b) in S_list.
    """
    blocked = 0
    for i in range(len(S_list)):
        for j in range(i + 1, len(S_list)):
            a = S_list[i]
            b = S_list[j]
            blocked |= iso_mask[a][b]
    return blocked

def _is_maximal(S_mask, S_list):
    """
    Check if S is maximal: no point outside S can be added without violating no-isosceles.
    This is equivalent to saying every point outside S is blocked by S.
    """
    if len(S_list) < 2:
        return False # A set with < 2 points cannot block anything (except trivially if N=1, but N=16)
    
    blocked = _get_blocked_mask(S_list)
    # All points not in S must be in blocked
    # blocked | S_mask should cover all points
    # Actually, blocked might contain points in S, that's fine.
    # We need: (blocked & ~S_mask) == (~S_mask)
    # Or: (blocked | S_mask) == (1 << N_TOTAL) - 1
    
    full_mask = (1 << N_TOTAL) - 1
    return (blocked | S_mask) == full_mask

def _can_add(p, S_mask, S_list):
    """
    Check if point p can be added to S while maintaining no-isosceles.
    p can be added if it doesn't form an isosceles triangle with any pair in S.
    This is equivalent to: p is not blocked by S.
    """
    if S_mask & (1 << p):
        return False
    blocked = _get_blocked_mask(S_list)
    return not (blocked & (1 << p))

def _local_search_reduce(S_list, deadline_margin=10):
    """
    Iteratively try to remove points from S while maintaining maximality.
    Tries single removals, then pair removals.
    """
    S = list(S_list)
    S_mask = 0
    for x in S:
        S_mask |= (1 << x)
    
    changed = True
    while changed:
        changed = False
        if time.time() > DEADLINE - deadline_margin:
            break
            
        n_S = len(S)
        
        # Try single removals
        for i in range(n_S):
            S_test = S[:i] + S[i+1:]
            S_mask_test = S_mask & ~(1 << S[i])
            
            # Validity is preserved by subset
            # Check maximality
            if _is_maximal(S_mask_test, S_test):
                S = S_test
                S_mask = S_mask_test
                changed = True
                break
        
        if changed:
            continue
            
        # Try pair removals
        for i in range(n_S):
            if time.time() > DEADLINE - deadline_margin:
                break
            for j in range(i + 1, n_S):
                S_test = S[:i] + S[i+1:j] + S[j+1:]
                S_mask_test = S_mask & ~(1 << S[i]) & ~(1 << S[j])
                
                if _is_maximal(S_mask_test, S_test):
                    S = S_test
                    S_mask = S_mask_test
                    changed = True
                    break
            if changed:
                break
                
    return S

def _repair_maximality(S_list, deadline_margin=10):
    """
    Given a valid set S (not necessarily maximal), greedily add points to make it maximal.
    """
    S = list(S_list)
    S_mask = 0
    for x in S:
        S_mask |= (1 << x)
        
    if _is_maximal(S_mask, S):
        return S
        
    while True:
        if time.time() > DEADLINE - deadline_margin:
            break
            
        # Find unblocked points
        blocked = _get_blocked_mask(S)
        unblocked = []
        for p in range(N_TOTAL):
            if S_mask & (1 << p):
                continue
            if not (blocked & (1 << p)):
                unblocked.append(p)
                
        if not unblocked:
            break
            
        # Find the point that blocks the most unblocked points
        best_p = None
        best_gain = -1
        
        for p in unblocked:
            # If we add p, which new points become blocked?
            # Points q are blocked if there exists a in S such that (p, a, q) is isosceles.
            # Or if q is p itself.
            # Actually, adding p to S means S_new = S U {p}.
            # New blocked points = (blocked by S) U (blocked by pairs involving p).
            # Pairs involving p are (p, a) for a in S.
            # So new blocked = blocked | (OR_{a in S} iso_mask[p][a])
            
            new_blocked = blocked
            for a in S:
                new_blocked |= iso_mask[p][a]
                
            gain = 0
            for q in unblocked:
                if new_blocked & (1 << q):
                    gain += 1
                    
            if gain > best_gain:
                best_gain = gain
                best_p = p
                
        if best_p is None:
            break
            
        S.append(best_p)
        S_mask |= (1 << best_p)
        
    return S

def _generate_initial_set(deadline_margin=10):
    """
    Generate a random maximal set.
    Start with random points, ensure validity, then repair maximality.
    """
    S = []
    S_mask = 0
    # Start with a small random core
    core_size = random.randint(3, 5)
    candidates = random.sample(range(N_TOTAL), core_size)
    for p in candidates:
        if _can_add(p, S_mask, S):
            S.append(p)
            S_mask |= (1 << p)
            
    # Repair to maximal
    S = _repair_maximality(S, deadline_margin)
    return S

def construct():
    start_time = time.time()
    
    best_set = None
    best_size = N_TOTAL + 1
    
    rng = random.Random(42)
    
    # Strategy 1: Multiple initializations with ILS
    for attempt in range(100):
        if time.time() - start_time > DEADLINE - 30:
            break
            
        # Generate initial maximal set
        S = _generate_initial_set(deadline_margin=10)
        
        # Local search to reduce size
        S = _local_search_reduce(S, deadline_margin=10)
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = S[:]
            
        # Iterated Local Search: Perturb and Repair
        if best_size <= 8:
            break
            
        # Perturbation: Remove k random points
        k = random.randint(2, 4)
        if len(S) > k:
            remove_indices = rng.sample(range(len(S)), k)
            remove_set = set(S[i] for i in remove_indices)
            
            S_perturbed = [x for x in S if x not in remove_set]
            
            # Repair maximality
            S_repaired = _repair_maximality(S_perturbed, deadline_margin=10)
            
            # Local search again
            S_repaired = _local_search_reduce(S_repaired, deadline_margin=10)
            
            if len(S_repaired) < best_size:
                best_size = len(S_repaired)
                best_set = S_repaired[:]
            elif len(S_repaired) == best_size and rng.random() < 0.5:
                # Diversify
                best_set = S_repaired[:]
                
    # Final check
    if best_set is None:
        # Fallback: Greedy construction
        S = []
        S_mask = 0
        for p in range(N_TOTAL):
            if _can_add(p, S_mask, S):
                S.append(p)
                S_mask |= (1 << p)
        best_set = _local_search_reduce(S, deadline_margin=5)
        
    return [_ALL_POINTS[i] for i in best_set]
