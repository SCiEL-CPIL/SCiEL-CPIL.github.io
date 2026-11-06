"""Improved program for minimum maximal no-isosceles set."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np
from collections import defaultdict

def _precompute():
    """Precompute all necessary data for fast isosceles checking."""
    points = list(itertools.product(range(N), repeat=2))
    V = len(points)
    pts = np.array(points, dtype=np.int32)
    
    # Compute squared distances
    dist = np.zeros((V, V), dtype=np.int32)
    for i in range(V):
        diff = pts[i] - pts
        dist[i] = np.sum(diff**2, axis=1)
    
    # For each pair (a, b), compute the set of points c such that {a,b,c} is isosceles
    # This means d(a,b)==d(a,c) or d(a,b)==d(b,c) or d(a,c)==d(b,c)
    iso_conflicts = [[set() for _ in range(V)] for __ in range(V)]
    
    for i in range(V):
        for j in range(i+1, V):
            dij = dist[i, j]
            # Points c where d(i,c) == dij
            mask1 = (dist[i] == dij)
            # Points c where d(j,c) == dij
            mask2 = (dist[j] == dij)
            # Points c where d(i,c) == d(j,c)
            mask3 = (dist[i] == dist[j])
            
            mask = mask1 | mask2 | mask3
            mask[i] = False
            mask[j] = False
            
            ks = np.where(mask)[0]
            for k in ks:
                iso_conflicts[i][j].add(k)
                iso_conflicts[j][i].add(k)
    
    return points, dist, iso_conflicts

_points, _dist, _iso_conflicts = None, None, None

def _init():
    global _points, _dist, _iso_conflicts
    if _points is None:
        _points, _dist, _iso_conflicts = _precompute()

def _check_valid(S):
    """Check if S is a valid no-isosceles set."""
    S_set = set(S)
    for i in range(len(S)):
        for j in range(i+1, len(S)):
            a, b = S[i], S[j]
            # Check if any other point in S is in iso_conflicts[a][b]
            for c in S:
                if c != a and c != b and c in _iso_conflicts[a][b]:
                    return False
    return True

def _get_blocked(S):
    """Get the set of points blocked by S (points that can't be added)."""
    blocked = set()
    S_list = list(S)
    for i in range(len(S_list)):
        for j in range(i+1, len(S_list)):
            a, b = S_list[i], S_list[j]
            blocked |= _iso_conflicts[a][b]
    return blocked

def _is_maximal(S, V):
    """Check if S is maximal."""
    S_set = set(S)
    blocked = _get_blocked(S)
    # Every point not in S must be blocked
    for p in range(V):
        if p not in S_set and p not in blocked:
            return False
    return True

def _greedy_build(seed, V, iso_conflicts, dist):
    """Greedy construction with randomized tie-breaking."""
    rng = random.Random(seed)
    S = []
    U = set(range(V))
    
    # Precompute for each point, how many points it would block with each other point
    # But we need to be careful about validity
    
    while U:
        # Compute current blocked set
        blocked = _get_blocked(S)
        
        # Find candidates that can be added (not in blocked)
        candidates = [p for p in U if p not in blocked]
        
        if not candidates:
            break
        
        # Score each candidate by how many points it would block
        best_score = -1
        best_candidate = None
        best_block_set = set()
        
        # Randomize order for diversity
        cand_list = candidates[:]
        rng.shuffle(cand_list)
        
        for p in cand_list:
            # Check validity: adding p shouldn't create isosceles with any pair in S
            valid = True
            for i in range(len(S)):
                for j in range(i+1, len(S)):
                    if p in iso_conflicts[S[i]][S[j]]:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Compute how many new points would be blocked
            new_blocked = set()
            for a in S:
                new_blocked |= iso_conflicts[p][a]
            new_blocked &= U
            new_blocked -= {p}
            
            score = len(new_blocked)
            
            if score > best_score:
                best_score = score
                best_candidate = p
                best_block_set = new_blocked
            elif score == best_score and best_candidate is not None:
                # Random tie-breaking
                if rng.random() < 0.5:
                    best_candidate = p
                    best_block_set = new_blocked
        
        if best_candidate is None:
            break
        
        S.append(best_candidate)
        U.remove(best_candidate)
        U -= best_block_set
    
    return S

def _local_search(S, V, iso_conflicts, max_iters=100, seed=0):
    """Local search to try to reduce set size while maintaining maximality."""
    rng = random.Random(seed)
    S = list(S)
    S_set = set(S)
    
    for it in range(max_iters):
        improved = False
        
        # Try removing a point and see if we can still maintain maximality
        # by adding back other points
        for idx in range(len(S)):
            trial = S[:idx] + S[idx+1:]
            trial_set = set(trial)
            
            if not trial:
                continue
            
            # Check if trial is still valid (no isosceles)
            valid = True
            for i in range(len(trial)):
                for j in range(i+1, len(trial)):
                    for k in trial:
                        if k != trial[i] and k != trial[j] and k in iso_conflicts[trial[i]][trial[j]]:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check if trial is maximal
            blocked = _get_blocked(trial)
            is_max = True
            for p in range(V):
                if p not in trial_set and p not in blocked:
                    is_max = False
                    break
            
            if is_max:
                S = trial
                S_set = set(S)
                improved = True
                break
        
        if not improved:
            # Try swapping: remove one point, add another
            for idx in range(len(S)):
                removed = S[idx]
                trial = S[:idx] + S[idx+1:]
                trial_set = set(trial)
                
                if not trial:
                    continue
                
                # Check validity of trial
                valid = True
                for i in range(len(trial)):
                    for j in range(i+1, len(trial)):
                        for k in trial:
                            if k != trial[i] and k != trial[j] and k in iso_conflicts[trial[i]][trial[j]]:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                # Find a point to add that makes it maximal
                blocked = _get_blocked(trial)
                missing = [p for p in range(V) if p not in trial_set and p not in blocked]
                
                if not missing:
                    # Already maximal with smaller set
                    S = trial
                    S_set = set(S)
                    improved = True
                    break
                
                # Try adding one of the missing points
                for p in missing:
                    # Check if adding p keeps validity
                    can_add = True
                    for a in trial:
                        if p in iso_conflicts[a][removed]:
                            pass  # This doesn't matter, we're not using removed
                    for i in range(len(trial)):
                        for j in range(i+1, len(trial)):
                            if p in iso_conflicts[trial[i]][trial[j]]:
                                can_add = False
                                break
                        if not can_add:
                            break
                    
                    if not can_add:
                        continue
                    
                    new_set = trial + [p]
                    new_set_set = set(new_set)
                    
                    # Check if new_set is maximal
                    blocked2 = _get_blocked(new_set)
                    is_max = True
                    for q in range(V):
                        if q not in new_set_set and q not in blocked2:
                            is_max = False
                            break
                    
                    if is_max:
                        S = new_set
                        S_set = set(S)
                        improved = True
                        break
                
                if improved:
                    break
        
        if not improved:
            break
    
    return S

def construct():
    _init()
    points = _points
    V = len(points)
    iso_conflicts = _iso_conflicts
    dist = _dist
    
    best_size = float('inf')
    best_set = None
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    trial_count = 0
    no_improve_count = 0
    
    while time.time() < deadline:
        seed = trial_count * 7919 + 12345
        
        S = _greedy_build(seed, V, iso_conflicts, dist)
        
        if len(S) < best_size:
            # Try to improve with local search
            S_improved = _local_search(S, V, iso_conflicts, max_iters=50, seed=seed)
            if len(S_improved) < len(S):
                S = S_improved
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = list(S)
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        trial_count += 1
        
        # Early termination if we've found a very small set
        if best_size <= 5:
            break
        
        # If no improvement for a while, try different approach
        if no_improve_count > 50:
            no_improve_count = 0
            # Try a different greedy strategy: start with random points
            for _ in range(5):
                if time.time() >= deadline:
                    break
                seed2 = trial_count * 13 + 99999
                rng = random.Random(seed2)
                S_rand = []
                U = set(range(V))
                
                # Start with a few random points
                for _ in range(3):
                    if not U:
                        break
                    p = rng.choice(list(U))
                    S_rand.append(p)
                    U.remove(p)
                
                # Then greedily add
                while U:
                    blocked = _get_blocked(S_rand)
                    candidates = [p for p in U if p not in blocked]
                    if not candidates:
                        break
                    
                    valid_cands = []
                    for p in candidates:
                        valid = True
                        for i in range(len(S_rand)):
                            for j in range(i+1, len(S_rand)):
                                if p in iso_conflicts[S_rand[i]][S_rand[j]]:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            valid_cands.append(p)
                    
                    if not valid_cands:
                        break
                    
                    p = rng.choice(valid_cands)
                    S_rand.append(p)
                    U.remove(p)
                    
                    # Remove blocked points
                    blocked = _get_blocked(S_rand)
                    U -= blocked
                
                S_improved = _local_search(S_rand, V, iso_conflicts, max_iters=30, seed=seed2)
                if len(S_improved) < best_size:
                    best_size = len(S_improved)
                    best_set = list(S_improved)
                    no_improve_count = 0
        
        if best_size <= 4:
            break
    
    if best_set is None:
        # Fallback: return an empty set or a trivial set
        return []
    
    return [points[i] for i in best_set]

# EVOLVE-BLOCK-END
