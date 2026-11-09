"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

# Deadline guard: must finish well before 330s wall-clock
_DEADLINE = time.time() + 280


def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _is_isosceles(a, b, c):
    """Check if triangle (a,b,c) is isosceles (two equal sides)."""
    d1 = _d2(a, b)
    d2 = _d2(a, c)
    d3 = _d2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3


def _precompute_conflict_pairs():
    """
    For each grid point p, compute the set of pairs (i, j) of other grid points
    such that (p, point_i, point_j) forms an isosceles triangle.
    This allows fast checking when trying to add p to a chosen set.
    """
    all_points = [(x, y) for x in range(N) for y in range(N)]
    n_pts = len(all_points)
    # conflicts[p_idx] = set of frozenset({i, j}) or a set of (i,j) tuples
    # Actually, for efficiency, store as a set of (min(i,j), max(i,j)) tuples
    conflicts = [set() for _ in range(n_pts)]
    
    for p_idx in range(n_pts):
        p = all_points[p_idx]
        for i in range(n_pts):
            if i == p_idx:
                continue
            a = all_points[i]
            for j in range(i + 1, n_pts):
                if j == p_idx:
                    continue
                b = all_points[j]
                if _is_isosceles(p, a, b):
                    conflicts[p_idx].add((i, j))
    
    return all_points, conflicts


def _is_valid_no_isosceles(chosen_indices, all_points, conflicts):
    """Check that the set of chosen indices forms a no-isosceles set."""
    # For every triple of chosen points, check not isosceles
    # More efficiently: for each chosen point p, check no pair in chosen is in conflicts[p]
    chosen_set = set(chosen_indices)
    for p_idx in chosen_indices:
        for (i, j) in conflicts[p_idx]:
            if i in chosen_set and j in chosen_set:
                return False
    return True


def _is_maximal(chosen_indices, all_points, conflicts):
    """Check that no additional grid point can be added."""
    chosen_set = set(chosen_indices)
    for p_idx in range(len(all_points)):
        if p_idx in chosen_set:
            continue
        # Check if p_idx can be added: no pair in chosen_set is in conflicts[p_idx]
        can_add = True
        for (i, j) in conflicts[p_idx]:
            if i in chosen_set and j in chosen_set:
                can_add = False
                break
        if can_add:
            return False
    return True


def _greedy_construct():
    """Greedy construction in row-major order."""
    all_points = [(x, y) for x in range(N) for y in range(N)]
    chosen = []
    chosen_set = set()
    for p_idx, p in enumerate(all_points):
        # Check if adding p creates any isosceles with two existing points
        can_add = True
        for (i, j) in _conflicts[p_idx]:
            if i in chosen_set and j in chosen_set:
                can_add = False
                break
        if can_add:
            chosen.append(p_idx)
            chosen_set.add(p_idx)
    return chosen


def _random_greedy(order_seed):
    """Random greedy construction."""
    rng = random.Random(order_seed)
    all_points = [(x, y) for x in range(N) for y in range(N)]
    order = list(range(len(all_points)))
    rng.shuffle(order)
    chosen = []
    chosen_set = set()
    for p_idx in order:
        can_add = True
        for (i, j) in _conflicts[p_idx]:
            if i in chosen_set and j in chosen_set:
                can_add = False
                break
        if can_add:
            chosen.append(p_idx)
            chosen_set.add(p_idx)
    return chosen


def _local_search(chosen_indices, all_points, conflicts, max_iters=200):
    """
    Try to reduce the set size by removing points that are not "essential"
    for maximality, while maintaining no-isosceles property.
    """
    chosen_set = set(chosen_indices)
    current = list(chosen_indices)
    
    for _ in range(max_iters):
        if time.time() > _DEADLINE:
            break
        improved = False
        for p_idx in list(current):
            # Try removing p_idx
            new_set = set(current)
            new_set.discard(p_idx)
            
            # Check no-isosceles still holds (it will, since removing points can't create isosceles)
            # Check maximality: no point can be added
            can_add_any = False
            for q_idx in range(len(all_points)):
                if q_idx in new_set:
                    continue
                # Check if q_idx can be added
                q_can_add = True
                for (i, j) in conflicts[q_idx]:
                    if i in new_set and j in new_set:
                        q_can_add = False
                        break
                if q_can_add:
                    can_add_any = True
                    break
            
            if not can_add_any:
                # Removing p_idx still gives a maximal no-isosceles set
                current.remove(p_idx)
                improved = True
                break
        
        if not improved:
            break
    
    return current


def construct():
    global _conflicts
    
    # Precompute conflicts
    all_points, _conflicts = _precompute_conflict_pairs()
    
    # Start with greedy
    best = _greedy_construct()
    best_size = len(best)
    
    # Try multiple random greedy starts
    for seed in range(50):
        if time.time() > _DEADLINE:
            break
        candidate = _random_greedy(seed)
        if len(candidate) < best_size:
            best = candidate
            best_size = len(candidate)
    
    # Local search to reduce size
    if time.time() < _DEADLINE - 30:
        improved = _local_search(best, all_points, _conflicts)
        if len(improved) < best_size:
            best = improved
            best_size = len(best)
    
    # Convert indices to points
    return [all_points[i] for i in best]
# EVOLVE-BLOCK-END
