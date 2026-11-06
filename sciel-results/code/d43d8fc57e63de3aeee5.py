import time
import random
import itertools

N = 16
DEADLINE = 320.0

def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def _is_isosceles(a, b, c):
    d1 = _dist2(a, b)
    d2 = _dist2(a, c)
    d3 = _dist2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3

def _precompute_iso_pairs():
    """For each point p, precompute the set of pairs (i,j) of other points such that {p, i, j} is isosceles.
    Return a dict: p -> set of frozenset({i,j}) where i,j are indices into all_points."""
    all_points = [(x, y) for x in range(N) for y in range(N)]
    n = len(all_points)
    idx_to_point = {i: all_points[i] for i in range(n)}
    point_to_idx = {all_points[i]: i for i in range(n)}
    
    # iso_pairs[p_idx] = set of (i, j) with i < j, i != p, j != p, such that {p, i, j} is isosceles
    iso_pairs = [set() for _ in range(n)]
    
    for p_idx in range(n):
        p = all_points[p_idx]
        for i in range(n):
            if i == p_idx:
                continue
            for j in range(i + 1, n):
                if j == p_idx:
                    continue
                if _is_isosceles(p, all_points[i], all_points[j]):
                    iso_pairs[p_idx].add((i, j))
    
    return all_points, point_to_idx, iso_pairs

def _is_no_isosceles(indices, all_points):
    for a, b, c in itertools.combinations(indices, 3):
        if _is_isosceles(all_points[a], all_points[b], all_points[c]):
            return False
    return True

def _is_maximal(indices, all_points, point_to_idx, iso_pairs):
    """Check if the set of indices is maximal."""
    idx_set = set(indices)
    n = len(all_points)
    for p_idx in range(n):
        if p_idx in idx_set:
            continue
        # Can p_idx be added? Check if there's any pair (i,j) in idx_set such that {p_idx, i, j} is isosceles
        # If NO such pair exists, then p_idx can be added -> not maximal
        blocked = False
        for pair in iso_pairs[p_idx]:
            i, j = pair
            if i in idx_set and j in idx_set:
                blocked = True
                break
        if not blocked:
            return False
    return True

def _greedy_maximal(all_points, point_to_idx, iso_pairs, seed=0):
    rng = random.Random(seed)
    order = list(range(len(all_points)))
    rng.shuffle(order)
    chosen = []
    chosen_set = set()
    
    for idx in order:
        # Check if adding idx creates any isosceles
        ok = True
        for i in chosen:
            for j in chosen:
                if i == j:
                    continue
                if _is_isosceles(all_points[idx], all_points[i], all_points[j]):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            chosen.append(idx)
            chosen_set.add(idx)
    
    # Make maximal: add any point that can be added
    changed = True
    while changed:
        changed = False
        for idx in range(len(all_points)):
            if idx in chosen_set:
                continue
            ok = True
            for i in chosen:
                for j in chosen:
                    if i == j:
                        continue
                    if _is_isosceles(all_points[idx], all_points[i], all_points[j]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                chosen.append(idx)
                chosen_set.add(idx)
                changed = True
    
    return chosen

def _try_remove(indices, all_points, point_to_idx, iso_pairs):
    """Try to remove a point while maintaining maximality. Return (success, new_indices)."""
    for i, idx in enumerate(indices):
        reduced = indices[:i] + indices[i+1:]
        if _is_maximal(reduced, all_points, point_to_idx, iso_pairs):
            return True, reduced
    return False, indices

def _local_search(indices, all_points, point_to_idx, iso_pairs, time_limit=5.0):
    start = time.time()
    best = list(indices)
    best_len = len(best)
    
    while time.time() - start < time_limit:
        success, new_indices = _try_remove(best, all_points, point_to_idx, iso_pairs)
        if success:
            best = new_indices
            best_len = len(best)
            continue
        break
    
    return best

def construct():
    start_time = time.time()
    all_points, point_to_idx, iso_pairs = _precompute_iso_pairs()
    
    best = None
    best_len = float('inf')
    
    seed = 0
    while time.time() - start_time < DEADLINE - 10:
        greedy_set = _greedy_maximal(all_points, point_to_idx, iso_pairs, seed)
        
        if time.time() - start_time < DEADLINE - 5:
            improved = _local_search(greedy_set, all_points, point_to_idx, iso_pairs, 
                                     time_limit=min(5, DEADLINE - 5 - (time.time() - start_time)))
        else:
            improved = greedy_set
        
        if len(improved) < best_len:
            best = improved
            best_len = len(improved)
        
        seed += 1
    
    if best is None:
        best = _greedy_maximal(all_points, point_to_idx, iso_pairs, 0)
    
    return [all_points[i] for i in best]

