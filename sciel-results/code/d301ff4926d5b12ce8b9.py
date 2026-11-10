"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import random
import time


def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _precompute_pairwise_dists(all_points):
    """Precompute squared distances between all pairs of points."""
    n = len(all_points)
    dists = [[0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = all_points[i]
        for j in range(i + 1, n):
            xj, yj = all_points[j]
            d = (xi - xj) ** 2 + (yi - yj) ** 2
            dists[i][j] = d
            dists[j][i] = d
    return dists


def _build_conflict_masks(all_points, dists):
    """
    For each point p, compute a bitmask of points q such that adding both p and q
    to a set would create an isosceles triangle with some third point r.
    
    Actually, we need: for a given set S, can we add p? We need to check:
    1. For all a, b in S: d(p,a) != d(p,b)  [apex at p]
    2. For all a, b in S: d(p,a) != d(a,b) and d(p,b) != d(a,b)  [apex at a or b]
    
    Let's precompute for each pair (a,b): the set of points p such that
    d(p,a) == d(a,b) or d(p,b) == d(a,b) or d(p,a) == d(p,b).
    """
    n = len(all_points)
    # conflict[p] is a set of indices q such that {p, q} cannot both be in a valid set
    # without also checking the third point. This is complex.
    # 
    # Alternative: precompute for each p, the distance to every other point.
    # Then _can_add can be fast.
    pass


def _can_add_fast(p_idx, chosen_indices, dists):
    """
    Check if point p_idx can be added to the set of chosen_indices
    without forming an isosceles triangle.
    
    dists[i][j] = squared distance between point i and point j.
    """
    if not chosen_indices:
        return True
    
    p = p_idx
    seen_dists = set()
    for a in chosen_indices:
        d = dists[p][a]
        if d in seen_dists:
            return False
        seen_dists.add(d)
    
    # Check: for any pair (a, b) in chosen, is d(p,a) == d(a,b) or d(p,b) == d(a,b)?
    k = len(chosen_indices)
    for i in range(k):
        a = chosen_indices[i]
        dp_a = dists[p][a]
        for j in range(i + 1, k):
            b = chosen_indices[j]
            d_ab = dists[a][b]
            if dp_a == d_ab or dists[p][b] == d_ab:
                return False
    return True


def _greedy_maximal(order, chosen_indices, dists, deadline):
    """Greedy maximal construction given an ordering of point indices."""
    for p_idx in order:
        if time.time() > deadline:
            return None
        if _can_add_fast(p_idx, chosen_indices, dists):
            chosen_indices.append(p_idx)
    return chosen_indices


def _is_maximal(chosen_indices, all_point_indices, dists):
    """Check if no point can be added to chosen without creating an isosceles triangle."""
    chosen_set = set(chosen_indices)
    for p_idx in all_point_indices:
        if p_idx in chosen_set:
            continue
        if _can_add_fast(p_idx, chosen_indices, dists):
            return False
    return True


def _prune(chosen_indices, all_point_indices, dists, deadline):
    """Try to remove points that are not essential for maximality."""
    if time.time() > deadline:
        return chosen_indices
    chosen_indices = list(chosen_indices)
    i = 0
    while i < len(chosen_indices):
        if time.time() > deadline:
            break
        candidate = chosen_indices[:i] + chosen_indices[i+1:]
        if _is_maximal(candidate, all_point_indices, dists):
            chosen_indices = candidate
        else:
            i += 1
    return chosen_indices


def _local_search_improve(chosen_indices, all_point_indices, dists, deadline):
    """
    Try to improve the set by swapping: remove one point, add another.
    Also try to remove points if the set remains maximal.
    """
    chosen_indices = list(chosen_indices)
    chosen_set = set(chosen_indices)
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        # Try removing each point
        for i in range(len(chosen_indices)):
            if time.time() > deadline:
                break
            candidate = chosen_indices[:i] + chosen_indices[i+1:]
            if _is_maximal(candidate, all_point_indices, dists):
                chosen_indices = candidate
                chosen_set = set(chosen_indices)
                improved = True
                break
        if not improved:
            # Try swapping: remove one, add one
            for i in range(len(chosen_indices)):
                if time.time() > deadline:
                    break
                removed = chosen_indices[i]
                base = chosen_indices[:i] + chosen_indices[i+1:]
                for p_idx in all_point_indices:
                    if p_idx in chosen_set:
                        continue
                    if _can_add_fast(p_idx, base, dists):
                        # Check if the new set is maximal
                        new_set = base + [p_idx]
                        if _is_maximal(new_set, all_point_indices, dists):
                            chosen_indices = new_set
                            chosen_set = set(chosen_indices)
                            improved = True
                            break
                if improved:
                    break
    return chosen_indices


def construct():
    deadline = time.time() + 280  # safety margin before 330s kill
    all_points = list(itertools.product(range(N), repeat=2))
    n_points = len(all_points)
    all_point_indices = list(range(n_points))
    
    # Precompute distances
    dists = _precompute_pairwise_dists(all_points)
    
    best = None
    best_size = float('inf')
    
    # First: deterministic row-major greedy
    result = _greedy_maximal(all_point_indices, [], dists, deadline)
    if result is not None:
        result = _prune(result, all_point_indices, dists, deadline)
        result = _local_search_improve(result, all_point_indices, dists, deadline)
        best = [all_points[i] for i in result]
        best_size = len(best)
    
    # Randomized restarts with different orderings
    rng = random.Random(42)
    
    for trial in range(10000):
        if time.time() > deadline:
            break
        # Shuffle the order
        order = all_point_indices[:]
        rng.shuffle(order)
        chosen = []
        result = _greedy_maximal(order, chosen, dists, deadline)
        if result is None:
            break
        # Prune the result
        result = _prune(result, all_point_indices, dists, deadline)
        result = _local_search_improve(result, all_point_indices, dists, deadline)
        if len(result) < best_size:
            best = [all_points[i] for i in result]
            best_size = len(best)
            if best_size <= 10:
                break
    
    return best if best is not None else []
# EVOLVE-BLOCK-END
