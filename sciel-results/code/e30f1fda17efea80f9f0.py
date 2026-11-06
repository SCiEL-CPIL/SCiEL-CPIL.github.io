import time
import random
import numpy as np
from collections import Counter

N = 16
DEADLINE = 325.0
N_TOTAL = N * N
EPS = 1e-9

_ALL_POINTS = [(x, y) for y in range(N) for x in range(N)]
_X = np.array([p[0] for p in _ALL_POINTS], dtype=np.float64)
_Y = np.array([p[1] for p in _ALL_POINTS], dtype=np.float64)

# Precompute squared distances
_D2 = np.zeros((N_TOTAL, N_TOTAL), dtype=np.int32)
for i in range(N_TOTAL):
    for j in range(N_TOTAL):
        _D2[i, j] = (i % N - j % N) ** 2 + (i // N - j // N) ** 2

# Precompute isosceles masks
# iso_c_mask[i][j] = bitmask of points k such that (i, j, k) is isosceles
iso_c_mask = [[0] * N_TOTAL for _ in range(N_TOTAL)]
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
        iso_c_mask[i][j] = mask
        iso_c_mask[j][i] = mask

# For each point p, precompute which pairs (i, j) make (i, j, p) isosceles
# blocked_pairs[p] = list of (i, j) such that (i, j, p) is isosceles
blocked_pairs = [[] for _ in range(N_TOTAL)]
for p in range(N_TOTAL):
    bit_p = 1 << p
    for i in range(N_TOTAL):
        for j in range(i + 1, N_TOTAL):
            if i == p or j == p:
                continue
            if iso_c_mask[i][j] & bit_p:
                blocked_pairs[p].append((i, j))

def _is_no_isosceles(S_list):
    """Check if set S is no-isosceles using precomputed masks."""
    n = len(S_list)
    if n < 3:
        return True
    S_mask = 0
    for idx in S_list:
        S_mask |= (1 << idx)
    for i in range(n):
        for j in range(i + 1, n):
            a = S_list[i]
            b = S_list[j]
            if iso_c_mask[a][b] & S_mask:
                return False
    return True

def _can_add(p, S_list):
    """Check if point p can be added to S without creating isosceles triangle."""
    n = len(S_list)
    if n < 2:
        return True
    S_mask = 0
    for idx in S_list:
        S_mask |= (1 << idx)
    for a in S_list:
        if iso_c_mask[p][a] & S_mask:
            return False
    return True

def _is_blocked(p, S_list):
    """Check if point p is blocked by set S (forms isosceles with some pair in S)."""
    n = len(S_list)
    if n < 2:
        return False
    S_mask = 0
    for idx in S_list:
        S_mask |= (1 << idx)
    for a, b in blocked_pairs[p]:
        if (S_mask & (1 << a)) and (S_mask & (1 << b)):
            return True
    return False

def _is_maximal(S_list):
    """Check if S is maximal: no point can be added."""
    S_set = set(S_list)
    for p in range(N_TOTAL):
        if p in S_set:
            continue
        if not _is_blocked(p, S_list):
            return False
    return True

def _reduce_to_minimal(S_list, time_limit=None):
    """Iteratively remove points if remaining set is still maximal."""
    S = list(S_list)
    changed = True
    while changed:
        changed = False
        for i in range(len(S)):
            if len(S) - 1 < 2:
                continue
            if time_limit and time.time() > time_limit:
                return S
            S_test = S[:i] + S[i+1:]
            if _is_maximal(S_test):
                S = S_test
                changed = True
                break
    return S

def _greedy_cover(start_time, deadline_margin=15):
    """
    Greedy set-cover approach to build a maximal no-isosceles set.
    """
    chosen = []
    chosen_set = set()
    blocked = [False] * N_TOTAL
    
    while True:
        if time.time() - start_time > DEADLINE - deadline_margin:
            return None
        
        # Check if already maximal
        if _is_maximal(chosen):
            return chosen
        
        # Find all unblocked, non-chosen points
        unblocked = []
        for q in range(N_TOTAL):
            if q in chosen_set:
                continue
            if not blocked[q]:
                unblocked.append(q)
        
        if not unblocked:
            break
        
        # Find addable points
        addable = []
        for p in range(N_TOTAL):
            if p in chosen_set:
                continue
            if _can_add(p, chosen):
                addable.append(p)
        
        if not addable:
            break
        
        # For each addable point, compute how many unblocked points it would block
        best_p = None
        best_gain = -1
        
        for p in addable:
            gain = 0
            for q in unblocked:
                if q == p:
                    continue
                # q is blocked by p if there exists a in chosen such that (p, a, q) is isosceles
                # i.e., q in iso_c_mask[p][a]
                for a in chosen:
                    if iso_c_mask[p][a] & (1 << q):
                        gain += 1
                        break
            if gain > best_gain:
                best_gain = gain
                best_p = p
        
        if best_p is None:
            break
        
        chosen.append(best_p)
        chosen_set.add(best_p)
        
        # Update blocked array
        for q in range(N_TOTAL):
            if q in chosen_set:
                continue
            if not blocked[q]:
                for a in chosen:
                    if iso_c_mask[best_p][a] & (1 << q):
                        blocked[q] = True
                        break
    
    return chosen

def _local_search(S, start_time, deadline_margin=10):
    """Try to improve set S by swapping one point for another."""
    S_list = list(S)
    n_S = len(S_list)
    best = S_list[:]
    best_size = n_S
    
    for i in range(n_S):
        if time.time() - start_time > DEADLINE - deadline_margin:
            break
        
        S_without = S_list[:i] + S_list[i + 1:]
        
        # Try adding each possible point
        for p_add in range(N_TOTAL):
            if p_add in set(S_without):
                continue
            
            if not _can_add(p_add, S_without):
                continue
            
            S_new = S_without + [p_add]
            
            if not _is_maximal(S_new):
                continue
            
            reduced = _reduce_to_minimal(S_new)
            
            if len(reduced) < best_size:
                best_size = len(reduced)
                best = reduced[:]
    
    return best

def _simulated_annealing(start_time, deadline_margin=10):
    """
    Simulated Annealing for minimizing the size of a maximal no-isosceles set.
    State: a set S (list of indices).
    Energy: |S| + lambda * (number of unblocked points not in S)
    Moves:
      1. Remove a point from S (if S remains no-isosceles).
      2. Add a point to S (if S remains no-isosceles).
      3. Swap a point in S with a point not in S (if S remains no-isosceles).
    """
    # Initial state: greedy cover
    S = _greedy_cover(start_time, deadline_margin=5)
    if S is None:
        return None
    
    S = _reduce_to_minimal(S)
    best_S = S[:]
    best_E = len(S)
    
    T = 1.0
    T_min = 0.01
    cooling = 0.995
    
    while time.time() - start_time < DEADLINE - deadline_margin:
        if T < T_min:
            T = 1.0 # Reheat occasionally? Or just stop. Let's stop for simplicity unless we want to keep going.
            # Actually, let's just break if T is too low.
            break
        
        # Generate a neighbor
        move_type = random.random()
        S_new = S[:]
        valid_move = False
        
        if move_type < 0.4 and len(S) > 0:
            # Remove
            idx = random.randint(0, len(S) - 1)
            p = S[idx]
            S_new = S[:idx] + S[idx+1:]
            if _is_no_isosceles(S_new):
                valid_move = True
        elif move_type < 0.7:
            # Add
            candidates = [p for p in range(N_TOTAL) if p not in set(S)]
            if candidates:
                p = random.choice(candidates)
                if _can_add(p, S):
                    S_new.append(p)
                    valid_move = True
        else:
            # Swap
            if len(S) > 0:
                idx = random.randint(0, len(S) - 1)
                p_remove = S[idx]
                candidates = [p for p in range(N_TOTAL) if p not in set(S)]
                if candidates:
                    p_add = random.choice(candidates)
                    S_test = S[:idx] + S[idx+1:]
                    if _can_add(p_add, S_test):
                        S_new = S_test + [p_add]
                        valid_move = True
        
        if not valid_move:
            continue
        
        # Calculate Energy
        # E = |S| + penalty for non-maximality
        # Penalty = number of unblocked points not in S
        S_set = set(S_new)
        unblocked_count = 0
        for p in range(N_TOTAL):
            if p in S_set:
                continue
            if not _is_blocked(p, S_new):
                unblocked_count += 1
        
        E_new = len(S_new) + 5.0 * unblocked_count # Weight on maximality
        
        # Current Energy
        S_set_old = set(S)
        unblocked_count_old = 0
        for p in range(N_TOTAL):
            if p in S_set_old:
                continue
            if not _is_blocked(p, S):
                unblocked_count_old += 1
        E_old = len(S) + 5.0 * unblocked_count_old
        
        delta_E = E_new - E_old
        
        if delta_E < 0 or random.random() < np.exp(-delta_E / T):
            S = S_new
            if unblocked_count == 0: # If maximal
                if len(S) < best_E:
                    best_E = len(S)
                    best_S = S[:]
        
        T *= cooling
    
    return best_S

def construct():
    start_time = time.time()
    
    best_set = None
    best_size = N_TOTAL + 1
    
    rng = random.Random(42)
    
    # Strategy 1: Greedy cover
    result = _greedy_cover(start_time, deadline_margin=20)
    if result is not None:
        reduced = _reduce_to_minimal(result)
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced[:]
        
        # Local search on the result
        if time.time() - start_time < DEADLINE - 10:
            improved = _local_search(reduced, start_time, deadline_margin=10)
            if len(improved) < best_size:
                best_size = len(improved)
                best_set = improved[:]
    
    # Strategy 2: Simulated Annealing
    if time.time() - start_time < DEADLINE - 30:
        sa_result = _simulated_annealing(start_time, deadline_margin=20)
        if sa_result is not None:
            if len(sa_result) < best_size:
                best_size = len(sa_result)
                best_set = sa_result[:]
    
    # Strategy 3: Random restarts with greedy
    for attempt in range(50):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Randomize the greedy order by shuffling points?
        # Or just run greedy again? Greedy is deterministic.
        # Let's try a randomized greedy: pick best among top K candidates with some probability.
        
        # For now, let's just do more local search on the best found
        if best_set is not None:
            improved = _local_search(best_set, start_time, deadline_margin=5)
            if len(improved) < best_size:
                best_size = len(improved)
                best_set = improved[:]
    
    if best_set is None:
        # Fallback
        chosen = []
        for i in range(N_TOTAL):
            if _can_add(i, chosen):
                chosen.append(i)
        best_set = _reduce_to_minimal(chosen)
    
    return [_ALL_POINTS[i] for i in best_set]
