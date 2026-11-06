import time
import random
import numpy as np

N = 256

def _precompute():
    """Precompute AP structures for fast delta calculations."""
    # 3-APs: (a, b, c) with a < b < c, b-a = c-b
    all_3aps = []
    for d in range(1, N):
        for a in range(1, N - 2*d + 1):
            b = a + d
            c = a + 2*d
            all_3aps.append((a, b, c))
    
    # 4-APs: (a, b, c, d) with a < b < c < d, b-a = c-b = d-c
    all_4aps = []
    for d in range(1, N):
        for a in range(1, N - 3*d + 1):
            b = a + d
            c = a + 2*d
            dd = a + 3*d
            all_4aps.append((a, b, c, dd))
    
    # Index 3-APs by element: for each x, list of (other1, other2, ap_index)
    # where the AP contains x and the other two elements
    aps_3_by_elem = {i: [] for i in range(1, N+1)}
    for idx, ap in enumerate(all_3aps):
        for x in ap:
            others = [y for y in ap if y != x]
            aps_3_by_elem[x].append((others[0], others[1], idx))
    
    # Index 4-APs by element: for each x, list of (other1, other2, other3)
    aps_4_by_elem = {i: [] for i in range(1, N+1)}
    for ap in all_4aps:
        for x in ap:
            others = [y for y in ap if y != x]
            aps_4_by_elem[x].append((others[0], others[1], others[2]))
    
    return all_3aps, all_4aps, aps_3_by_elem, aps_4_by_elem

# Global precomputed data
_all_3aps, _all_4aps, _aps_3_by_elem, _aps_4_by_elem = _precompute()

def _has_4ap_with_mask(mask, x):
    """Check if adding x to set represented by mask creates a 4-AP."""
    for o1, o2, o3 in _aps_4_by_elem[x]:
        if mask[o1-1] and mask[o2-1] and mask[o3-1]:
            return True
    return False

def _delta_3ap_add(mask, x):
    """Count new 3-APs created by adding x to set represented by mask."""
    count = 0
    for o1, o2, _ in _aps_3_by_elem[x]:
        if mask[o1-1] and mask[o2-1]:
            count += 1
    return count

def _delta_3ap_remove(mask, x):
    """Count 3-APs destroyed by removing x from set represented by mask."""
    count = 0
    for o1, o2, _ in _aps_3_by_elem[x]:
        if mask[o1-1] and mask[o2-1]:
            count += 1
    return count

def _count_3aps(mask):
    """Count total 3-APs in set represented by mask."""
    count = 0
    for ap in _all_3aps:
        if mask[ap[0]-1] and mask[ap[1]-1] and mask[ap[2]-1]:
            count += 1
    return count

def _greedy_seed():
    """Greedy construction: pick elements that maximize 3-APs while maintaining 4-AP-free."""
    mask = np.zeros(N, dtype=bool)
    A = set()
    candidates = list(range(1, N+1))
    random.shuffle(candidates)
    
    # First pass: add all elements that don't create 4-APs, prioritizing those with high potential
    # Actually, let's do a smarter greedy: iterate and pick the best valid element each time
    remaining = set(range(1, N+1))
    
    while remaining:
        best_x = None
        best_gain = -1
        
        for x in remaining:
            if _has_4ap_with_mask(mask, x):
                continue
            gain = _delta_3ap_add(mask, x)
            if gain > best_gain:
                best_gain = gain
                best_x = x
        
        if best_x is None:
            break
        
        mask[best_x-1] = True
        A.add(best_x)
        remaining.remove(best_x)
    
    return mask, A

def _strict_local_search(mask, A, time_limit):
    """Local search: try all single additions, removals, and swaps."""
    start = time.time()
    improved = True
    current_score = _count_3aps(mask)
    
    while improved and time.time() < time_limit:
        improved = False
        best_move = None
        best_move_score = current_score
        move_type = None
        move_args = None
        
        # Try additions
        for x in range(1, N+1):
            if mask[x-1]:
                continue
            if _has_4ap_with_mask(mask, x):
                continue
            gain = _delta_3ap_add(mask, x)
            if gain > 0:
                new_score = current_score + gain
                if new_score > best_move_score:
                    best_move_score = new_score
                    best_move = x
                    move_type = 'add'
        
        # Try removals (only if we want to enable better additions later, but for now skip)
        # Actually removals reduce score, so only useful in swaps
        
        # Try swaps: remove a from A, add b not in A
        # This is expensive: O(|A| * (N-|A|))
        # Optimization: only try swaps if they might help
        for a in A:
            # Temporarily remove a
            mask[a-1] = False
            loss_a = _delta_3ap_remove(mask, a)  # This is wrong - we need to compute before removing
            
            # Actually, let's compute properly
            # loss_a = number of 3-APs that contain a and have both other elements in A\{a}
            # We need to check after removing a
            # Let me restructure: compute gain of adding b to A\{a}
            
            for b in range(1, N+1):
                if mask[b-1]:
                    continue
                if _has_4ap_with_mask(mask, b):
                    continue
                gain_b = _delta_3ap_add(mask, b)
                new_score = current_score - loss_a + gain_b
                if new_score > best_move_score:
                    best_move_score = new_score
                    best_move = (a, b)
                    move_type = 'swap'
            
            # Restore a
            mask[a-1] = True
        
        if move_type == 'add':
            mask[best_move-1] = True
            A.add(best_move)
            current_score = best_move_score
            improved = True
        elif move_type == 'swap':
            a, b = best_move
            mask[a-1] = False
            A.remove(a)
            mask[b-1] = True
            A.add(b)
            current_score = best_move_score
            improved = True
    
    return mask, A, current_score

def _simulated_annealing(mask, A, time_limit, T0=1.0, T_end=0.01, cooling=0.999):
    """Simulated annealing with single moves."""
    start = time.time()
    current_score = _count_3aps(mask)
    best_mask = mask.copy()
    best_A = A.copy()
    best_score = current_score
    T = T0
    
    while time.time() < time_limit and T > T_end:
        # Choose a random move: add, remove, or swap
        move_type = random.choice(['add', 'remove', 'swap'])
        
        if move_type == 'add':
            # Pick random element not in A
            not_in_A = [x for x in range(1, N+1) if not mask[x-1]]
            if not not_in_A:
                continue
            x = random.choice(not_in_A)
            if _has_4ap_with_mask(mask, x):
                continue
            gain = _delta_3ap_add(mask, x)
            delta = gain
            # Accept if delta >= 0 or with probability exp(delta/T)
            if delta >= 0 or random.random() < np.exp(delta / T):
                mask[x-1] = True
                A.add(x)
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_mask = mask.copy()
                    best_A = A.copy()
        elif move_type == 'remove':
            # Pick random element in A
            if not A:
                continue
            x = random.choice(list(A))
            loss = _delta_3ap_remove(mask, x)
            delta = -loss
            if delta >= 0 or random.random() < np.exp(delta / T):
                mask[x-1] = False
                A.remove(x)
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_mask = mask.copy()
                    best_A = A.copy()
        elif move_type == 'swap':
            # Pick random a in A, b not in A
            if not A:
                continue
            a = random.choice(list(A))
            not_in_A = [x for x in range(1, N+1) if not mask[x-1]]
            if not not_in_A:
                continue
            b = random.choice(not_in_A)
            
            # Compute delta: remove a, add b
            # Need to check validity of adding b to A\{a}
            mask[a-1] = False
            if _has_4ap_with_mask(mask, b):
                mask[a-1] = True
                continue
            gain_b = _delta_3ap_add(mask, b)
            # loss_a was computed with a still in A, need to recompute
            # Actually, _delta_3ap_remove(mask, a) with a removed from mask gives 0
            # We need the number of 3-APs containing a where both others are in A\{a}
            # Let's compute it properly
            loss_a = 0
            for o1, o2, _ in _aps_3_by_elem[a]:
                if mask[o1-1] and mask[o2-1]:
                    loss_a += 1
            delta = gain_b - loss_a
            mask[a-1] = True  # restore for now
            
            if delta >= 0 or random.random() < np.exp(delta / T):
                mask[a-1] = False
                A.remove(a)
                mask[b-1] = True
                A.add(b)
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_mask = mask.copy()
                    best_A = A.copy()
        
        T *= cooling
    
    return best_mask, best_A, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    # Greedy seed
    mask, A = _greedy_seed()
    
    # Strict local search
    time_for_ls = min(15, deadline - time.time() - 5)
    if time_for_ls > 2:
        mask, A, score = _strict_local_search(mask, A, time.time() + time_for_ls)
    
    # Simulated annealing with multiple restarts
    best_mask = mask.copy()
    best_A = A.copy()
    best_score = _count_3aps(mask)
    
    remaining_time = deadline - time.time() - 2
    if remaining_time > 5:
        # Run SA for the remaining time
        mask_sa, A_sa, score_sa = _simulated_annealing(mask, A, time.time() + remaining_time)
        if score_sa > best_score:
            best_mask = mask_sa
            best_A = A_sa
            best_score = score_sa
    
    return sorted(best_A)
