import itertools
import time
import random
import sys

N = 16
_DEADLINE = time.time() + 280

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

def _can_add(x, chosen_indices, dist_matrix, chosen_dist_sets):
    """
    Check if point x can be added to chosen.
    chosen_indices: list of indices in chosen set.
    chosen_dist_sets: list of sets, where chosen_dist_sets[i] is the set of distances 
                       from chosen_indices[i] to all other points in chosen_indices.
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
        # chosen_dist_sets[idx] contains distances from p to all q in chosen, q != p
        if d_xp in chosen_dist_sets[idx]:
            return False
            
    return True

def _update_dist_sets_on_add(x, chosen_indices, dist_matrix, chosen_dist_sets):
    """
    Update chosen_dist_sets after adding x to chosen_indices.
    Returns new lists.
    """
    k = len(chosen_indices)
    new_dist_sets = [set(s) for s in chosen_dist_sets]
    # Add distance from x to each existing point in their sets
    for i in range(k):
        p = chosen_indices[i]
        d = dist_matrix[x][p]
        new_dist_sets[i].add(d)
    # Add set for x: distances from x to all existing points
    new_set_for_x = set()
    for p in chosen_indices:
        new_set_for_x.add(dist_matrix[x][p])
    new_dist_sets.append(new_set_for_x)
    return new_dist_sets

def _update_dist_sets_on_remove(idx_to_remove, chosen_indices, chosen_dist_sets):
    """
    Update chosen_dist_sets after removing chosen_indices[idx_to_remove].
    """
    k = len(chosen_indices)
    removed_point = chosen_indices[idx_to_remove]
    new_dist_sets = []
    new_indices = []
    for i in range(k):
        if i == idx_to_remove:
            continue
        new_indices.append(chosen_indices[i])
        new_set = set(chosen_dist_sets[i])
        # Remove distance to the removed point
        d = dist_matrix[chosen_indices[i]][removed_point]
        new_set.discard(d)
        new_dist_sets.append(new_set)
    return new_indices, new_dist_sets

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
                    # We need to recompute dist sets for the base?
                    # It's expensive to do this for every pair.
                    # Alternative: Just use the slow _can_add for the candidate check?
                    # Or, since n is small (~10), recomputing is O(n^2) which is fine.
                    pass
                
                # Actually, let's just build the base sets properly.
                # This is O(n^2) per pair. Pairs are O(n^2). Total O(n^4).
                # n=10, n^4 = 10000. Very fast.
                
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
                # We can restrict r to points that were "blocked" by p or q?
                # For now, try all points not in base.
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
        
        # 3. Update best
        if len(pruned_indices) < best_size:
            best_size = len(pruned_indices)
            best_indices = pruned_indices
            
    if best_indices is None:
        best_indices, _ = _construct_greedy(all_indices, dist_matrix, rng)
        
    return [all_points[i] for i in best_indices]
