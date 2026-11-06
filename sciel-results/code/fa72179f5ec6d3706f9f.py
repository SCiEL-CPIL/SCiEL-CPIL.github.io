import itertools
import time
import random
import numpy as np

N = 16
V = N * N
FULL_MASK = (1 << V) - 1

def _precompute():
    points = list(itertools.product(range(N), repeat=2))
    pts = np.array(points, dtype=np.int32)
    
    # Compute squared distances
    dist = np.zeros((V, V), dtype=np.int32)
    for i in range(V):
        diff = pts[i] - pts
        dist[i] = np.sum(diff**2, axis=1)
        
    # Precompute conflict masks
    conflict_mask = [[0] * V for _ in range(V)]
    
    for i in range(V):
        for j in range(i + 1, V):
            dij = dist[i][j]
            mask = 0
            for k in range(V):
                if k == i or k == j:
                    continue
                dik = dist[i][k]
                djk = dist[j][k]
                if dik == dij or djk == dij or dik == djk:
                    mask |= (1 << k)
            conflict_mask[i][j] = mask
            conflict_mask[j][i] = mask
            
    return points, conflict_mask, dist

_points, _conflict_mask, _dist = None, None, None

def _init():
    global _points, _conflict_mask, _dist
    if _points is None:
        _points, _conflict_mask, _dist = _precompute()

def _is_valid_set(S_mask, S_list):
    """Check if S is a valid no-isosceles set using bitmask."""
    for i in range(len(S_list)):
        a = S_list[i]
        for j in range(i + 1, len(S_list)):
            b = S_list[j]
            if S_mask & _conflict_mask[a][b]:
                return False
    return True

def _is_maximal_set(S_mask, S_list):
    """Check if S is maximal."""
    blocked = 0
    for i in range(len(S_list)):
        a = S_list[i]
        for j in range(i + 1, len(S_list)):
            b = S_list[j]
            blocked |= _conflict_mask[a][b]
    return (S_mask | blocked) == FULL_MASK

def _backtrack_search(target_size, seed, time_limit=5.0):
    """
    Backtracking search for a valid maximal set of exactly target_size points.
    Uses constraint propagation to prune the search tree.
    """
    rng = random.Random(seed)
    start_time = time.time()
    
    # Order points to try first (heuristic: points that block many others)
    point_order = list(range(V))
    # Precompute for each point, how many other points it conflicts with (as part of some pair)
    # Actually, let's just use random order with some structure
    
    # For backtracking, we build the set incrementally
    # State: current set S, current blocked mask, remaining candidates
    
    best_found = None
    
    def backtrack(S_list, S_mask, blocked_mask, depth, max_depth):
        nonlocal best_found
        if time.time() - start_time > time_limit:
            return False
            
        if len(S_list) == target_size:
            if _is_maximal_set(S_mask, S_list):
                best_found = list(S_list)
                return True
            return False
            
        if len(S_list) > target_size:
            return False
            
        # Check if we can still achieve maximality
        # Points that are already in S or blocked are "covered"
        covered = S_mask | blocked_mask
        uncovered = FULL_MASK & ~covered
        
        # If uncovered has too many points to cover with remaining slots, prune
        remaining_slots = target_size - len(S_list)
        # Each new point can block at most some number of points
        # This is a loose bound; we'll rely on validity checks
        
        # Find candidate points: not in S, not blocked
        candidates = []
        cm = uncovered
        while cm:
            lsb = cm & -cm
            p = lsb.bit_length() - 1
            cm ^= lsb
            candidates.append(p)
            
        if not candidates:
            # No candidates but not at target size - can't continue
            return False
            
        # Shuffle candidates for randomization
        rng.shuffle(candidates)
        
        # Limit candidates to explore (for performance)
        max_candidates_to_try = min(len(candidates), 50)
        
        for p in candidates[:max_candidates_to_try]:
            # Check if adding p is valid
            valid = True
            for a in S_list:
                if _conflict_mask[p][a] & S_mask:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Compute new blocked mask
            new_blocked = blocked_mask
            for a in S_list:
                new_blocked |= _conflict_mask[p][a]
                
            new_S_mask = S_mask | (1 << p)
            new_S_list = S_list + [p]
            
            if backtrack(new_S_list, new_S_mask, new_blocked, depth + 1, max_depth):
                return True
                
    backtrack([], 0, 0, 0, target_size)
    return best_found

def _greedy_with_backtrack_refinement(seed):
    """
    Hybrid approach: use greedy to get a large maximal set, then use backtracking
    to find smaller maximal subsets.
    """
    rng = random.Random(seed)
    
    # First, build a maximal set using greedy
    S = []
    S_mask = 0
    blocked = 0
    
    while (S_mask | blocked) != FULL_MASK:
        candidates_mask = FULL_MASK & ~(S_mask | blocked)
        if candidates_mask == 0:
            break
            
        best_p = -1
        best_score = -1
        cm = candidates_mask
        while cm:
            lsb = cm & -cm
            p = lsb.bit_length() - 1
            cm ^= lsb
            
            valid = True
            new_blocked = 0
            for a in S:
                if _conflict_mask[p][a] & S_mask:
                    valid = False
                    break
                new_blocked |= _conflict_mask[p][a]
            
            if not valid:
                continue
                
            new_coverage = new_blocked & ~(blocked | S_mask)
            score = new_coverage.bit_count()
            
            if score > best_score:
                best_score = score
                best_p = p
            elif score == best_score and best_p != -1 and rng.random() < 0.3:
                best_p = p
        
        if best_p == -1:
            break
            
        S.append(best_p)
        S_mask |= (1 << best_p)
        for a in S[:-1]:
            blocked |= _conflict_mask[best_p][a]
    
    # Now try to reduce the set using local search
    S = _local_search_aggressive(S, seed, max_iters=50)
    
    return S

def _local_search_aggressive(S, seed, max_iters=50):
    """
    More aggressive local search with multiple move types.
    """
    rng = random.Random(seed)
    S = list(S)
    
    for iteration in range(max_iters):
        improved = False
        
        # Try removal
        for i in range(len(S)):
            trial = S[:i] + S[i+1:]
            S_mask = 0
            for p in trial:
                S_mask |= (1 << p)
            if _is_valid_set(S_mask, trial) and _is_maximal_set(S_mask, trial):
                S = trial
                improved = True
                break
        
        if improved:
            continue
            
        # Try swap: remove one, add one
        for i in range(len(S)):
            base = S[:i] + S[i+1:]
            base_mask = 0
            for p in base:
                base_mask |= (1 << p)
            
            # Find candidates to add
            candidates = [p for p in range(V) if not (base_mask >> p) & 1]
            rng.shuffle(candidates)
            
            for p in candidates[:30]:
                trial = base + [p]
                S_mask = 0
                for q in trial:
                    S_mask |= (1 << q)
                if _is_valid_set(S_mask, trial) and _is_maximal_set(S_mask, trial):
                    S = trial
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            break
            
    return S

def _simulated_annealing(start_S, seed, time_limit=10.0):
    """
    Simulated annealing to minimize set size while maintaining maximality.
    """
    rng = random.Random(seed)
    start_time = time.time()
    
    S = list(start_S)
    current_size = len(S)
    best_S = list(S)
    best_size = current_size
    
    temperature = 10.0
    cooling_rate = 0.95
    
    while time.time() - start_time < time_limit and temperature > 0.01:
        # Generate a neighbor
        move_type = rng.random()
        neighbor = None
        
        if move_type < 0.3 and len(S) > 1:
            # Removal
            i = rng.randrange(len(S))
            neighbor = S[:i] + S[i+1:]
        elif move_type < 0.7 and len(S) > 1:
            # Swap
            i = rng.randrange(len(S))
            base = S[:i] + S[i+1:]
            base_mask = 0
            for p in base:
                base_mask |= (1 << p)
            candidates = [p for p in range(V) if not (base_mask >> p) & 1]
            if candidates:
                p = rng.choice(candidates)
                neighbor = base + [p]
        else:
            # Add a random point
            S_mask = 0
            for p in S:
                S_mask |= (1 << p)
            candidates = [p for p in range(V) if not (S_mask >> p) & 1]
            if candidates:
                p = rng.choice(candidates)
                neighbor = S + [p]
        
        if neighbor is None:
            temperature *= cooling_rate
            continue
            
        neighbor_mask = 0
        for p in neighbor:
            neighbor_mask |= (1 << p)
            
        if not _is_valid_set(neighbor_mask, neighbor):
            temperature *= cooling_rate
            continue
            
        # Check maximality
        if not _is_maximal_set(neighbor_mask, neighbor):
            temperature *= cooling_rate
            continue
            
        new_size = len(neighbor)
        delta = new_size - current_size
        
        if delta < 0:
            S = neighbor
            current_size = new_size
            if new_size < best_size:
                best_size = new_size
                best_S = list(neighbor)
        else:
            # Accept with probability exp(-delta / temperature)
            if rng.random() < np.exp(-delta / temperature):
                S = neighbor
                current_size = new_size
        
        temperature *= cooling_rate
    
    return best_S

def construct():
    _init()
    
    best_size = float('inf')
    best_set = None
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Phase 1: Greedy + local search with multiple seeds
    trial_count = 0
    while time.time() < deadline - 60:  # Leave 60s for phase 2
        seed = trial_count * 7919 + 12345
        
        S = _greedy_with_backtrack_refinement(seed)
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = list(S)
            
            if best_size <= 8:
                break
                
        trial_count += 1
        
    # Phase 2: Simulated annealing starting from best found
    if best_set is not None and time.time() < deadline - 10:
        for sa_seed in range(5):
            if time.time() > deadline - 10:
                break
            sa_seed_val = sa_seed * 31337 + 999
            S_sa = _simulated_annealing(best_set, sa_seed_val, time_limit=8.0)
            if len(S_sa) < best_size:
                best_size = len(S_sa)
                best_set = list(S_sa)
                
    # Phase 3: Backtracking search for small sizes
    if best_size > 10 and time.time() < deadline - 5:
        for target in range(8, min(best_size, 14)):
            if time.time() > deadline - 5:
                break
            for bt_seed in range(3):
                if time.time() > deadline - 5:
                    break
                bt_seed_val = target * 1000 + bt_seed * 777
                result = _backtrack_search(target, bt_seed_val, time_limit=3.0)
                if result is not None and len(result) < best_size:
                    best_size = len(result)
                    best_set = list(result)
                    
    if best_set is None:
        return []
        
    return [_points[i] for i in best_set]
