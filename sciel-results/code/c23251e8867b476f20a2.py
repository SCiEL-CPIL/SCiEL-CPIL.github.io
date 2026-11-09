import itertools
import time
import random
import sys
from collections import defaultdict

N = 16
_DEADLINE = time.time() + 280

def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def _precompute(all_points):
    """
    Precompute:
    1. dist_matrix[i][j] = squared distance
    2. iso_pairs_with[x] = set of frozenset({p,q}) such that {x,p,q} is isosceles
    3. For each x, dist_to_points[x][d] = set of points y with d(x,y)=d
    """
    n = len(all_points)
    dist_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _d2(all_points[i], all_points[j])
            dist_matrix[i][j] = d
            dist_matrix[j][i] = d
    
    # Precompute iso_pairs_with: for each x, which pairs (p,q) form isosceles with x
    iso_pairs_with = [set() for _ in range(n)]
    
    # For each triple (i,j,k), check if isosceles
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                d_ij = dist_matrix[i][j]
                d_ik = dist_matrix[i][k]
                d_jk = dist_matrix[j][k]
                
                # Check if any two distances are equal
                if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                    # This triple is isosceles
                    iso_pairs_with[i].add(frozenset((j, k)))
                    iso_pairs_with[j].add(frozenset((i, k)))
                    iso_pairs_with[k].add(frozenset((i, j)))
    
    # For faster checking, also precompute for each x and each distance d,
    # which points have that distance from x
    dist_to_points = [defaultdict(set) for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                d = dist_matrix[i][j]
                dist_to_points[i][d].add(j)
    
    return dist_matrix, iso_pairs_with, dist_to_points

def _can_add_fast(x, chosen_set, iso_pairs_with, dist_matrix, dist_to_points):
    """
    Check if x can be added to chosen_set.
    Uses precomputed iso_pairs_with for fast checking.
    """
    if not chosen_set:
        return True
    
    # Check 1: All distances from x to points in chosen must be distinct
    # This is equivalent to: no two points p,q in chosen have d(x,p) = d(x,q)
    # Which means: no pair (p,q) in chosen is in iso_pairs_with[x] due to equal distances from x
    # But iso_pairs_with[x] includes ALL isosceles triples with x, not just those where x is the apex
    
    # Let me use a different approach: directly check the two conditions
    # Condition 1: all d(x,p) for p in chosen are distinct
    seen_distances = set()
    for p in chosen_set:
        d = dist_matrix[x][p]
        if d in seen_distances:
            return False
        seen_distances.add(d)
    
    # Condition 2: for each p in chosen, d(x,p) must not equal d(p,q) for any q in chosen, q≠p
    for p in chosen_set:
        d_xp = dist_matrix[x][p]
        # Check if any q in chosen, q≠p, has d(p,q) == d_xp
        for q in chosen_set:
            if q == p:
                continue
            if dist_matrix[p][q] == d_xp:
                return False
    
    return True

def _is_maximal_fast(chosen_set, all_indices, iso_pairs_with, dist_matrix, dist_to_points):
    """
    Check if chosen_set is maximal.
    """
    for x in all_indices:
        if x in chosen_set:
            continue
        if _can_add_fast(x, chosen_set, iso_pairs_with, dist_matrix, dist_to_points):
            return False
    return True

def _construct_greedy(all_indices, iso_pairs_with, dist_matrix, dist_to_points, rng):
    """
    Greedy construction with random ordering.
    """
    order = list(all_indices)
    rng.shuffle(order)
    chosen = []
    chosen_set = set()
    for p in order:
        if _can_add_fast(p, chosen_set, iso_pairs_with, dist_matrix, dist_to_points):
            chosen.append(p)
            chosen_set.add(p)
    return chosen, chosen_set

def _local_search_reduce(chosen, chosen_set, all_indices, iso_pairs_with, dist_matrix, dist_to_points, rng, deadline):
    """
    Local search: try to remove 2 points and add 1 to reduce size.
    """
    s = list(chosen)
    s_set = set(s)
    n = len(s)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        # Try all pairs to remove
        for i in range(n):
            if time.time() > deadline:
                return s, s_set
            for j in range(i + 1, n):
                if time.time() > deadline:
                    return s, s_set
                
                p, q = s[i], s[j]
                # Create base set without p and q
                base_set = s_set - {p, q}
                
                # Try to add a point r
                for r in all_indices:
                    if r in base_set:
                        continue
                    
                    if not _can_add_fast(r, base_set, iso_pairs_with, dist_matrix, dist_to_points):
                        continue
                    
                    candidate_set = base_set | {r}
                    if _is_maximal_fast(candidate_set, all_indices, iso_pairs_with, dist_matrix, dist_to_points):
                        s = list(candidate_set)
                        s_set = candidate_set
                        n = len(s)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
                
    return s, s_set

def _local_search_swap(chosen, chosen_set, all_indices, iso_pairs_with, dist_matrix, dist_to_points, rng, deadline):
    """
    Local search: try to swap one point for another.
    """
    s = list(chosen)
    s_set = set(s)
    n = len(s)
    attempts = 0
    max_attempts = 50
    while attempts < max_attempts and time.time() < deadline:
        attempts += 1
        moved = False
        i = rng.randrange(n)
        p = s[i]
        base_set = s_set - {p}
        
        for _ in range(20):
            r = rng.choice(all_indices)
            if r in base_set:
                continue
            
            if not _can_add_fast(r, base_set, iso_pairs_with, dist_matrix, dist_to_points):
                continue
            
            candidate_set = base_set | {r}
            if _is_maximal_fast(candidate_set, all_indices, iso_pairs_with, dist_matrix, dist_to_points):
                s = list(candidate_set)
                s_set = candidate_set
                n = len(s)
                moved = True
                break
        
        if not moved:
            break
            
    return s, s_set

def construct():
    all_points = list(itertools.product(range(N), repeat=2))
    all_indices = list(range(len(all_points)))
    dist_matrix, iso_pairs_with, dist_to_points = _precompute(all_points)
    
    best = None
    best_size = float('inf')
    rng = random.Random(42)
    
    while time.time() < _DEADLINE:
        # Greedy construction
        chosen, chosen_set = _construct_greedy(all_indices, iso_pairs_with, dist_matrix, dist_to_points, rng)
        
        # Local search reduce
        pruned, pruned_set = _local_search_reduce(chosen, chosen_set, all_indices, iso_pairs_with, dist_matrix, dist_to_points, rng, _DEADLINE)
        
        # If no improvement, try swap
        if len(pruned) == len(chosen):
            pruned, pruned_set = _local_search_swap(pruned, pruned_set, all_indices, iso_pairs_with, dist_matrix, dist_to_points, rng, _DEADLINE)
        
        # Update best
        if len(pruned) < best_size:
            best_size = len(pruned)
            best = pruned
    
    if best is None:
        best, _ = _construct_greedy(all_indices, iso_pairs_with, dist_matrix, dist_to_points, rng)
        
    return [all_points[i] for i in best]
