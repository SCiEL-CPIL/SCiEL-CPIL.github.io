import itertools
import time
import random
import sys
from functools import lru_cache

N = 16
_DEADLINE = time.time() + 280
TOTAL_POINTS = N * N  # 256

def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def _precompute_distances(all_points):
    n = len(all_points)
    dist_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _d2(all_points[i], all_points[j])
            dist_matrix[i][j] = d
            dist_matrix[j][i] = d
    return dist_matrix

def _precompute_iso_triples(all_points, dist_matrix):
    """
    Find all isosceles triples (i, j, k) where i, j, k are distinct and
    the triangle is isosceles (two sides equal).
    A triangle is isosceles if d(i,j) == d(i,k) or d(i,j) == d(j,k) or d(j,k) == d(i,k).
    """
    n = len(all_points)
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dij = dist_matrix[i][j]
                dik = dist_matrix[i][k]
                djk = dist_matrix[j][k]
                if dij == dik or dij == djk or dik == djk:
                    triples.append((i, j, k))
    return triples

def _can_add(x, chosen_indices, dist_matrix, chosen_dist_sets):
    """
    Check if point x can be added to chosen.
    """
    k = len(chosen_indices)
    if k == 0:
        return True
        
    distances_from_x = []
    
    # 1. Distances from x to all p in chosen must be distinct
    for p in chosen_indices:
        d = dist_matrix[x][p]
        if d in distances_from_x:
            return False
        distances_from_x.append(d)
        
    # 2. For each p in chosen, d(x, p) must not be a distance already used by p to other points in chosen
    for idx, p in enumerate(chosen_indices):
        d_xp = dist_matrix[x][p]
        if d_xp in chosen_dist_sets[idx]:
            return False
            
    return True

def _update_dist_sets_on_add(x, chosen_indices, dist_matrix, chosen_dist_sets):
    """
    Update chosen_dist_sets after adding x to chosen_indices.
    """
    k = len(chosen_indices)
    new_dist_sets = [set(s) for s in chosen_dist_sets]
    for i in range(k):
        p = chosen_indices[i]
        d = dist_matrix[x][p]
        new_dist_sets[i].add(d)
    new_set_for_x = set()
    for p in chosen_indices:
        new_set_for_x.add(dist_matrix[x][p])
    new_dist_sets.append(new_set_for_x)
    return new_dist_sets

def _is_maximal(chosen_indices, all_indices, dist_matrix, chosen_dist_sets):
    """
    Check if chosen is maximal.
    """
    chosen_set = set(chosen_indices)
    for x in all_indices:
        if x in chosen_set:
            continue
        if _can_add(x, chosen_indices, dist_matrix, chosen_dist_sets):
            return False
    return True

def _construct_greedy(all_indices, dist_matrix, rng):
    """
    Construct a maximal set by adding points in random order.
    """
    order = list(all_indices)
    rng.shuffle(order)
    chosen_indices = []
    chosen_dist_sets = []
    for p in order:
        if _can_add(p, chosen_indices, dist_matrix, chosen_dist_sets):
            chosen_indices.append(p)
            chosen_dist_sets = _update_dist_sets_on_add(p, chosen_indices[:-1], dist_matrix, chosen_dist_sets)
    return chosen_indices, chosen_dist_sets

def _branch_and_bound_find_minimal_no_iso(all_indices, dist_matrix, iso_triples, current_best, deadline):
    """
    Branch and bound to find a minimal no-isosceles set.
    We want to find a set S such that:
    1. No three points in S form an isosceles triangle.
    2. S is as small as possible.
    
    We use a recursive approach with bitsets for efficiency.
    """
    n = len(all_indices)
    
    # Build conflict graph: for each point, which other points form an iso triple with it?
    # Actually, we need to track which triples are "active" (all 3 points chosen).
    
    # For B&B, we decide for each point whether to include it or not.
    # State: (index, chosen_set, forbidden_triples)
    # This is too complex for direct B&B with 256 points.
    
    # Alternative: Use a greedy heuristic to get a good initial bound,
    # then use local search to improve.
    
    # For now, this function is a placeholder for future implementation.
    # The main improvement will come from better local search.
    pass

def _local_search_reduce(chosen_indices, chosen_dist_sets, all_indices, dist_matrix, rng, deadline):
    """
    Try to reduce the size of the maximal set.
    Move: Remove p, q and add r.
    If (S \ {p, q}) U {r} is maximal, accept.
    """
    s = list(chosen_indices)
    ds = [set(x) for x in chosen_dist_sets]
    n = len(s)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        # Try all pairs (p, q) to remove
        for i in range(n):
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                
                p, q = s[i], s[j]
                
                # Construct base set without p, q
                base_indices = []
                base_dist_sets = []
                for k in range(n):
                    if k == i or k == j:
                        continue
                    base_indices.append(s[k])
                
                # Compute dist sets for base
                m = len(base_indices)
                base_dist_sets = [set() for _ in range(m)]
                for a in range(m):
                    for b in range(a+1, m):
                        d = dist_matrix[base_indices[a]][base_indices[b]]
                        base_dist_sets[a].add(d)
                        base_dist_sets[b].add(d)
                        
                base_set = set(base_indices)
                
                # Try to add a point r not in base
                for r in all_indices:
                    if r in base_set:
                        continue
                    
                    # Check if r can be added to base
                    if not _can_add(r, base_indices, dist_matrix, base_dist_sets):
                        continue
                    
                    # Candidate set
                    candidate_indices = base_indices + [r]
                    candidate_dist_sets = _update_dist_sets_on_add(r, base_indices, dist_matrix, base_dist_sets)
                    
                    # Check maximality
                    if _is_maximal(candidate_indices, all_indices, dist_matrix, candidate_dist_sets):
                        s = candidate_indices
                        ds = candidate_dist_sets
                        n = len(s)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
                
    return s, ds

def _local_search_swap(chosen_indices, chosen_dist_sets, all_indices, dist_matrix, rng, deadline):
    """
    Try to swap points to reduce size or maintain size with better structure.
    Move: Remove p, add q. If (S \ {p}) U {q} is maximal and |S| stays same, accept with some probability.
    """
    s = list(chosen_indices)
    ds = [set(x) for x in chosen_dist_sets]
    n = len(s)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            p = s[i]
            
            # Construct base set without p
            base_indices = []
            base_dist_sets = []
            for k in range(n):
                if k == i:
                    continue
                base_indices.append(s[k])
            
            m = len(base_indices)
            base_dist_sets = [set() for _ in range(m)]
            for a in range(m):
                for b in range(a+1, m):
                    d = dist_matrix[base_indices[a]][base_indices[b]]
                    base_dist_sets[a].add(d)
                    base_dist_sets[b].add(d)
                    
            base_set = set(base_indices)
            
            # Try to add a point q not in base
            for q in all_indices:
                if q in base_set:
                    continue
                
                if not _can_add(q, base_indices, dist_matrix, base_dist_sets):
                    continue
                
                candidate_indices = base_indices + [q]
                candidate_dist_sets = _update_dist_sets_on_add(q, base_indices, dist_matrix, base_dist_sets)
                
                if _is_maximal(candidate_indices, all_indices, dist_matrix, candidate_dist_sets):
                    # Accept if it leads to a different configuration (for diversity)
                    # or if we want to explore the space
                    s = candidate_indices
                    ds = candidate_dist_sets
                    improved = True
                    break
            if improved:
                break
                
    return s, ds

def construct():
    all_points = list(itertools.product(range(N), repeat=2))
    all_indices = list(range(len(all_points)))
    dist_matrix = _precompute_distances(all_points)
    
    best_indices = None
    best_size = float('inf')
    rng = random.Random(42)
    
    # Run multiple restarts
    while time.time() < _DEADLINE:
        # 1. Greedy Construction
        chosen_indices, chosen_dist_sets = _construct_greedy(all_indices, dist_matrix, rng)
        
        # 2. Local Search: Try to reduce size
        pruned_indices, pruned_dist_sets = _local_search_reduce(
            chosen_indices, chosen_dist_sets, all_indices, dist_matrix, rng, _DEADLINE
        )
        
        # 3. Local Search: Swap to explore space
        swapped_indices, swapped_dist_sets = _local_search_swap(
            pruned_indices, pruned_dist_sets, all_indices, dist_matrix, rng, _DEADLINE
        )
        
        # 4. Update best
        if len(swapped_indices) < best_size:
            best_size = len(swapped_indices)
            best_indices = swapped_indices
            
    if best_indices is None:
        best_indices, _ = _construct_greedy(all_indices, dist_matrix, rng)
        
    return [all_points[i] for i in best_indices]
