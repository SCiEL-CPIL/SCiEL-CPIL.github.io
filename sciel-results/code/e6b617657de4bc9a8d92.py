"""Combined approach: Greedy + Minimality Reduction + Coverage + Simulated Annealing with precomputed distances."""
import time
import random
import itertools
import numpy as np

N = 16
DEADLINE = 320.0  # Safety margin before 330s kill

# Precompute all points and distance matrix
ALL_POINTS = [(x, y) for y in range(N) for x in range(N)]
N_TOTAL = len(ALL_POINTS)
POINT_INDEX = {p: i for i, p in enumerate(ALL_POINTS)}
# Dist[i, j] = squared distance between ALL_POINTS[i] and ALL_POINTS[j]
DIST = np.zeros((N_TOTAL, N_TOTAL), dtype=np.int32)
for i in range(N_TOTAL):
    for j in range(i + 1, N_TOTAL):
        d = (ALL_POINTS[i][0] - ALL_POINTS[j][0])**2 + (ALL_POINTS[i][1] - ALL_POINTS[j][1])**2
        DIST[i, j] = d
        DIST[j, i] = d

def _dist2_idx(i, j):
    """Get squared distance by index."""
    return DIST[i, j]

def _is_maximal_idx(S_idx):
    """
    Check if set of indices S_idx is a maximal no-isosceles set.
    Uses precomputed DIST for speed.
    """
    S_set = set(S_idx)
    n_S = len(S_set)
    
    if n_S < 2:
        return False
    
    S_list = list(S_set)
    
    # For each point not in S, check if it's blocked by some pair in S
    for p_idx in range(N_TOTAL):
        if p_idx in S_set:
            continue
        
        blocked = False
        for i in range(n_S):
            if blocked:
                break
            a_idx = S_list[i]
            da = DIST[p_idx, a_idx]
            for j in range(i + 1, n_S):
                b_idx = S_list[j]
                dpb = DIST[p_idx, b_idx]
                dab = DIST[a_idx, b_idx]
                
                if da == dpb or da == dab or dpb == dab:
                    blocked = True
                    break
                    
        if not blocked:
            return False
            
    return True

def _reduce_to_minimal_idx(S_idx):
    """
    Iteratively remove points from S if the remaining set is still maximal.
    """
    S_list = list(S_idx)
    changed = True
    while changed:
        changed = False
        # Try removing from the end first (often more effective)
        for i in range(len(S_list) - 1, -1, -1):
            S_test = S_list[:i] + S_list[i+1:]
            
            if len(S_test) < 2:
                continue
                
            if _is_maximal_idx(set(S_test)):
                S_list = S_test
                changed = True
                break
                
    return S_list

def _can_add_idx(p_idx, chosen_idx):
    """Check if point p_idx can be added to chosen_idx without creating an isosceles triangle."""
    for i in range(len(chosen_idx)):
        a_idx = chosen_idx[i]
        da = DIST[p_idx, a_idx]
        for j in range(i + 1, len(chosen_idx)):
            b_idx = chosen_idx[j]
            dpb = DIST[p_idx, b_idx]
            dab = DIST[a_idx, b_idx]
            if da == dpb or da == dab or dpb == dab:
                return False
    return True

def _coverage_score_idx(p_idx, chosen_idx, S_set):
    """
    Score how many points p_idx would block if added to chosen_idx.
    """
    if not _can_add_idx(p_idx, chosen_idx):
        return 0
    
    new_blocked = 0
    n_chosen = len(chosen_idx)
    
    for q_idx in range(N_TOTAL):
        if q_idx in S_set or q_idx == p_idx:
            continue
        
        # Check if q is currently unblocked
        already_blocked = False
        for i in range(n_chosen):
            if already_blocked:
                break
            a_idx = chosen_idx[i]
            dqa = DIST[q_idx, a_idx]
            for j in range(i + 1, n_chosen):
                b_idx = chosen_idx[j]
                dq_b = DIST[q_idx, b_idx]
                d_ab = DIST[a_idx, b_idx]
                if dqa == dq_b or dqa == d_ab or dq_b == d_ab:
                    already_blocked = True
                    break
        if already_blocked:
            continue
        
        # Check if q would be blocked by adding p
        for a_idx in chosen_idx:
            dpa = DIST[p_idx, a_idx]
            dqa = DIST[q_idx, a_idx]
            dpq = DIST[p_idx, q_idx]
            if dpa == dqa or dpa == dpq or dqa == dpq:
                new_blocked += 1
                break
                
    return new_blocked

def _greedy_coverage_idx(rng):
    """
    Greedy construction using coverage heuristic (index-based).
    """
    remaining = set(range(N_TOTAL))
    chosen = []
    
    start = rng.choice(list(remaining))
    remaining.remove(start)
    chosen.append(start)
    S_set = {start}
    
    while remaining:
        best_p = None
        best_score = -1
        
        candidates = list(remaining)
        if len(candidates) > 150:
            candidates = rng.sample(candidates, 150)
        
        for p_idx in candidates:
            score = _coverage_score_idx(p_idx, chosen, S_set)
            if score > best_score:
                best_score = score
                best_p = p_idx
        
        if best_p is None or best_score <= 0:
            break
            
        remaining.remove(best_p)
        chosen.append(best_p)
        S_set.add(best_p)
        
    return chosen

def _extend_to_maximal(chosen_idx):
    """Extend a no-isosceles set to be maximal by adding more points."""
    S_set = set(chosen_idx)
    chosen = list(chosen_idx)
    
    for p_idx in range(N_TOTAL):
        if p_idx in S_set:
            continue
        if _can_add_idx(p_idx, chosen):
            chosen.append(p_idx)
            S_set.add(p_idx)
            
    return chosen

def _simulated_annealing(start_set, rng, max_iters=5000, T_start=10.0, T_end=0.1, alpha=0.999):
    """
    Simulated annealing to find smaller maximal no-isosceles sets.
    State: a set of indices.
    Moves: swap one point with another, or remove a point.
    Cost: size of set + penalty for non-maximality.
    """
    current_set = set(start_set)
    current_cost = len(current_set) + (1000 if not _is_maximal_idx(current_set) else 0)
    best_set = set(current_set)
    best_cost = current_cost
    
    T = T_start
    
    for it in range(max_iters):
        if time.time() - _sa_start_time > DEADLINE:
            break
            
        # Generate a random move
        move_type = rng.random()
        
        if move_type < 0.7 and len(current_set) > 2:
            # Remove a random point
            to_remove = rng.choice(list(current_set))
            new_set = current_set - {to_remove}
        elif move_type < 0.9 and len(current_set) >= 2:
            # Swap: remove one, add another
            to_remove = rng.choice(list(current_set))
            candidates = [p for p in range(N_TOTAL) if p not in current_set and p != to_remove]
            if not candidates:
                continue
            to_add = rng.choice(candidates)
            new_set = (current_set - {to_remove}) | {to_add}
        else:
            # Add a random point
            candidates = [p for p in range(N_TOTAL) if p not in current_set]
            if not candidates:
                continue
            to_add = rng.choice(candidates)
            new_set = current_set | {to_add}
        
        # Check if new_set is no-isosceles
        new_list = list(new_set)
        valid = True
        for i in range(len(new_list)):
            if not valid:
                break
            for j in range(i + 1, len(new_list)):
                if not valid:
                    break
                for k in range(j + 1, len(new_list)):
                    a, b, c = new_list[i], new_list[j], new_list[k]
                    d1 = DIST[a, b]
                    d2 = DIST[a, c]
                    d3 = DIST[b, c]
                    if d1 == d2 or d1 == d3 or d2 == d3:
                        valid = False
                        break
        
        if not valid:
            T *= alpha
            continue
        
        new_maximal = _is_maximal_idx(new_set)
        new_cost = len(new_set) + (1000 if not new_maximal else 0)
        
        delta = new_cost - current_cost
        
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            current_set = new_set
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_set = set(current_set)
        
        T *= alpha
        
    return list(best_set)

_sa_start_time = 0.0

def _local_search_idx(S_idx, all_indices):
    """
    Try to improve S by swapping one point for another.
    """
    S_set = set(S_idx)
    S_list = list(S_set)
    n_S = len(S_list)
    
    for i in range(n_S):
        p_remove = S_list[i]
        S_without = S_list[:i] + S_list[i+1:]
        S_without_set = set(S_without)
        
        for p_add in all_indices:
            if p_add in S_without_set:
                continue
            if p_add == p_remove:
                continue
            
            if not _can_add_idx(p_add, S_without):
                continue
            
            S_new = S_without + [p_add]
            
            if _is_maximal_idx(set(S_new)):
                reduced = _reduce_to_minimal_idx(set(S_new))
                if len(reduced) < n_S:
                    return reduced
                    
    return None

def construct():
    global _sa_start_time
    start_time = time.time()
    _sa_start_time = start_time
    
    best_set = None
    best_size = N_TOTAL + 1
    
    rng = random.Random(42)
    
    # Strategy 1: Random greedy + reduction
    for attempt in range(150):
        if time.time() - start_time > DEADLINE:
            break
            
        order = list(range(N_TOTAL))
        rng.shuffle(order)
        
        chosen = []
        for p_idx in order:
            if _can_add_idx(p_idx, chosen):
                chosen.append(p_idx)
        
        if len(chosen) < 2:
            continue
            
        # Extend to maximal
        chosen = _extend_to_maximal(chosen)
        
        reduced = _reduce_to_minimal_idx(set(chosen))
        
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced[:]
            
        if best_size <= 6:
            break
    
    # Strategy 2: Coverage greedy + reduction
    for attempt in range(80):
        if time.time() - start_time > DEADLINE:
            break
            
        chosen = _greedy_coverage_idx(rng)
        
        if len(chosen) < 2:
            continue
            
        # Extend to maximal
        chosen = _extend_to_maximal(chosen)
        
        reduced = _reduce_to_minimal_idx(set(chosen))
        
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced[:]
            
        if best_size <= 6:
            break
    
    # Strategy 3: Simulated annealing from best found
    if best_set is not None and time.time() - start_time < DEADLINE - 60:
        _sa_start_time = time.time()
        improved = _simulated_annealing(best_set, rng, max_iters=3000, T_start=5.0, T_end=0.05, alpha=0.9995)
        improved = _reduce_to_minimal_idx(set(improved))
        if len(improved) < best_size:
            best_size = len(improved)
            best_set = improved[:]
    
    # Strategy 4: Local search on best found
    if best_set is not None and time.time() - start_time < DEADLINE - 30:
        improved = _local_search_idx(best_set, range(N_TOTAL))
        if improved is not None and len(improved) < best_size:
            best_size = len(improved)
            best_set = improved[:]
    
    if best_set is None:
        # Fallback: simple deterministic greedy
        chosen = []
        for p_idx in range(N_TOTAL):
            if _can_add_idx(p_idx, chosen):
                chosen.append(p_idx)
        best_set = chosen
    
    # Convert indices back to points
    return [ALL_POINTS[i] for i in best_set]
