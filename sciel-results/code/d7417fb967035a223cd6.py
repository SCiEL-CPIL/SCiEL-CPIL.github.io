import numpy as np
import random
import time

N = 16
GRID_POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = len(GRID_POINTS)

# Precompute squared distances
coords = np.array(GRID_POINTS, dtype=np.int32)
diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
dist_sq = np.sum(diffs * diffs, axis=-1)  # shape (256, 256)


def check_no_isosceles(S):
    """Check if set S (list of indices) is no-isosceles. Vectorized."""
    if len(S) < 3:
        return True
    S_arr = np.array(S, dtype=np.int32)
    # dists[i,j] = dist_sq[S[i], S[j]]
    dists = dist_sq[np.ix_(S_arr, S_arr)]
    # For each point i, check if all distances to other points are unique
    # A triangle (i,j,k) is isosceles if any two of d(i,j), d(i,k), d(j,k) are equal
    # Equivalent: for each i, the set {d(i,j) for j != i} must have no duplicates
    # AND for each pair (j,k), d(i,j) != d(i,k) is already covered
    # Actually the proper check: no three points form isosceles.
    # d(i,j) == d(i,k) for i,j,k distinct -> isosceles with apex i
    # d(i,j) == d(j,k) for i,j,k distinct -> isosceles with apex j
    # So: for each i, all d(i,j) for j in S\{i} must be distinct.
    # This is sufficient because if d(i,j)==d(i,k) that's caught, 
    # and d(i,j)==d(j,k) means from j's perspective d(j,i)==d(j,k), also caught.
    for i in range(len(S)):
        other_dists = np.delete(dists[i], i)
        if len(np.unique(other_dists)) != len(other_dists):
            return False
    return True


def get_addable(S_set, S_list):
    """Get set of points that can be added to S while maintaining no-isosceles. Vectorized."""
    S_arr = np.array(S_list, dtype=np.int32)
    addable = []
    
    if len(S_list) == 0:
        return set(range(NUM_POINTS))
    
    # For each candidate q not in S, check if adding q keeps no-isosceles
    # Conditions for q to be addable:
    # 1. d(q, a) must be distinct for all a in S (no two points in S equidistant from q)
    # 2. For each a in S, d(q, a) must not equal d(a, b) for any b in S, b != a
    #    (i.e., d(q,a) not in {d(a,b) for b in S, b != a})
    
    candidates = [q for q in range(NUM_POINTS) if q not in S_set]
    if not candidates:
        return set()
    
    cand_arr = np.array(candidates, dtype=np.int32)
    
    # d_cand_s[i, j] = dist_sq[candidates[i], S[j]]
    d_cand_s = dist_sq[np.ix_(cand_arr, S_arr)]
    
    # Condition 1: for each candidate i, all d_cand_s[i, :] must be unique
    cond1 = np.ones(len(candidates), dtype=bool)
    for i in range(len(candidates)):
        row = d_cand_s[i]
        if len(np.unique(row)) != len(row):
            cond1[i] = False
    
    # Condition 2: for each candidate i and each S[j], d_cand_s[i,j] not in {d(S[j], S[k]) for k != j}
    # Precompute for each S[j], the set of distances from S[j] to other S points
    s_dists = dist_sq[np.ix_(S_arr, S_arr)]
    # forbidden[i, j] = is d_cand_s[i,j] in {s_dists[j, k] for k != j}?
    cond2 = np.ones((len(candidates), len(S_list)), dtype=bool)
    for j in range(len(S_list)):
        other_s_dists = set(s_dists[j].tolist()) - {0}  # distances from S[j] to other S points
        for i in range(len(candidates)):
            if d_cand_s[i, j] in other_s_dists:
                cond2[i, j] = False
    
    valid = cond1 & cond2.all(axis=1)
    addable = set(candidates[valid])
    return addable


def check_maximal(S_set, S_list):
    """Check if S is maximal (no point can be added). Returns list of addable points."""
    addable = get_addable(S_set, S_list)
    return addable


def greedy_repair(S_list, S_set, addable):
    """Greedily add points from addable to make S maximal."""
    S_list = list(S_list)
    S_set = set(S_set)
    
    # We need to iteratively add points until no more can be added
    # But adding a point may remove others from addable
    # Strategy: repeatedly pick a point to add that maximizes coverage
    # For simplicity, just add greedily in random order until maximal
    
    current_addable = list(addable)
    random.shuffle(current_addable)
    
    for q in current_addable:
        if q in S_set:
            continue
        # Check if q is still addable
        S_arr = np.array(S_list, dtype=np.int32)
        if len(S_list) == 0:
            S_list.append(q)
            S_set.add(q)
            continue
        
        d_qs = dist_sq[q, S_arr]
        # Condition 1: all distances from q to S must be unique
        if len(np.unique(d_qs)) != len(d_qs):
            continue
        # Condition 2: for each a in S, d(q,a) not in distances from a to other S points
        ok = True
        for idx, a in enumerate(S_list):
            other_dists = set(dist_sq[a, S_arr].tolist()) - {0, d_qs[idx]}
            if d_qs[idx] in other_dists:
                ok = False
                break
        if ok:
            S_list.append(q)
            S_set.add(q)
    
    return S_list, S_set


def construct():
    start_time = time.time()
    deadline = start_time + 290  # safety margin
    
    best_set = None
    best_size = NUM_POINTS + 1
    
    # Phase 1: Generate diverse initial solutions via random greedy
    # Phase 2: Local search via random removals + repair
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Generate initial solution via random greedy
        random_order = list(range(NUM_POINTS))
        random.shuffle(random_order)
        
        S_list = []
        S_set = set()
        for p in random_order:
            if p in S_set:
                continue
            if len(S_list) == 0:
                S_list.append(p)
                S_set.add(p)
                continue
            
            S_arr = np.array(S_list, dtype=np.int32)
            d_ps = dist_sq[p, S_arr]
            
            # Check condition 1: unique distances
            if len(np.unique(d_ps)) != len(d_ps):
                continue
            
            # Check condition 2
            ok = True
            for idx, a in enumerate(S_list):
                other_dists = set(dist_sq[a, S_arr].tolist()) - {0, d_ps[idx]}
                if d_ps[idx] in other_dists:
                    ok = False
                    break
            
            if ok:
                S_list.append(p)
                S_set.add(p)
        
        # Try to minimize by random removals
        current_S = list(S_list)
        current_S_set = set(S_list)
        current_size = len(current_S)
        
        # Local search: try removing points and repairing
        max_local_steps = 50
        for step in range(max_local_steps):
            if time.time() > deadline:
                break
            
            if current_size <= best_size:
                break
            
            # Try removing 1-3 random points
            k = random.randint(1, min(3, current_size - 1)) if current_size > 1 else 0
            if k == 0:
                break
            
            to_remove = random.sample(current_S, k)
            new_S = [x for x in current_S if x not in set(to_remove)]
            new_S_set = set(new_S)
            
            # Check if new_S is still no-isosceles (it should be, since subset)
            # Repair to maximality
            addable = get_addable(new_S_set, new_S)
            repaired_S, repaired_S_set = greedy_repair(new_S, new_S_set, addable)
            
            # Check if repaired is no-isosceles
            if not check_no_isosceles(repaired_S):
                continue
            
            if len(repaired_S) < current_size:
                current_S = repaired_S
                current_S_set = repaired_S_set
                current_size = len(repaired_S)
            
            # If we're at a local minimum, try a bigger perturbation
            if step > 10 and current_size >= best_size:
                # Remove more points
                k2 = random.randint(2, min(5, current_size))
                to_remove2 = random.sample(current_S, k2)
                new_S2 = [x for x in current_S if x not in set(to_remove2)]
                new_S_set2 = set(new_S2)
                addable2 = get_addable(new_S_set2, new_S2)
                repaired_S2, repaired_S_set2 = greedy_repair(new_S2, new_S_set2, addable2)
                
                if check_no_isosceles(repaired_S2) and len(repaired_S2) < current_size:
                    current_S = repaired_S2
                    current_S_set = repaired_S_set2
                    current_size = len(repaired_S2)
        
        # Check if current solution is valid and record
        if check_no_isosceles(current_S):
            # Verify maximality
            addable_final = get_addable(set(current_S), current_S)
            if len(addable_final) == 0:
                if len(current_S) < best_size:
                    best_size = len(current_S)
                    best_set = list(current_S)
        
        # If we've found a good solution, do more focused search
        if best_set is not None and best_size <= 15:
            # Try to improve further with more aggressive perturbations
            if time.time() < deadline - 30:
                for _ in range(20):
                    if time.time() > deadline:
                        break
                    k = random.randint(3, min(7, best_size))
                    to_remove = random.sample(best_set, k)
                    new_S = [x for x in best_set if x not in set(to_remove)]
                    new_S_set = set(new_S)
                    addable = get_addable(new_S_set, new_S)
                    repaired_S, repaired_S_set = greedy_repair(new_S, new_S_set, addable)
                    
                    if check_no_isosceles(repaired_S) and len(repaired_S) < best_size:
                        # Verify maximality
                        addable_check = get_addable(set(repaired_S), repaired_S)
                        if len(addable_check) == 0:
                            best_size = len(repaired_S)
                            best_set = list(repaired_S)
    
    # Fallback
    if best_set is None:
        random_order = list(range(NUM_POINTS))
        random.shuffle(random_order)
        S_list = []
        S_set = set()
        for p in random_order:
            if p in S_set:
                continue
            if len(S_list) == 0:
                S_list.append(p)
                S_set.add(p)
                continue
            S_arr = np.array(S_list, dtype=np.int32)
            d_ps = dist_sq[p, S_arr]
            if len(np.unique(d_ps)) != len(d_ps):
                continue
            ok = True
            for idx, a in enumerate(S_list):
                other_dists = set(dist_sq[a, S_arr].tolist()) - {0, d_ps[idx]}
                if d_ps[idx] in other_dists:
                    ok = False
                    break
            if ok:
                S_list.append(p)
                S_set.add(p)
        best_set = S_list
    
    return [GRID_POINTS[i] for i in best_set]
