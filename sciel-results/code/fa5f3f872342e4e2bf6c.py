import itertools
import time
import random
import sys

N = 16
DEADLINE = time.time() + 280
TOTAL_POINTS = N * N

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

def _can_add(p, chosen_set, chosen_indices, dist_matrix, dist_sets):
    """
    Check if point p can be added to the current set.
    """
    # 1. Distances from p to all chosen must be distinct
    seen_dists = set()
    for idx in chosen_indices:
        d = dist_matrix[p][idx]
        if d in seen_dists:
            return False
        seen_dists.add(d)
        
    # 2. For each chosen idx, d(p, idx) must not be in dist_sets[idx]
    for idx in chosen_indices:
        d = dist_matrix[p][idx]
        if d in dist_sets[idx]:
            return False
            
    return True

def _make_maximal(chosen_indices, chosen_set, dist_sets, all_indices, dist_matrix, rng):
    """
    Greedily add points to make the set maximal.
    """
    candidates = [i for i in all_indices if i not in chosen_set]
    rng.shuffle(candidates)
    
    for p in candidates:
        if _can_add(p, chosen_set, chosen_indices, dist_matrix, dist_sets):
            # Add p
            chosen_indices.append(p)
            chosen_set.add(p)
            
            # Update dist_sets
            for idx in chosen_indices[:-1]:
                dist_sets[idx].add(dist_matrix[p][idx])
            
            # Create new set for p
            new_set = set()
            for idx in chosen_indices[:-1]:
                new_set.add(dist_matrix[p][idx])
            dist_sets.append(new_set)
            
    return chosen_indices, chosen_set, dist_sets

def _remove_point(idx_to_remove, chosen_indices, chosen_set, dist_sets, all_indices, dist_matrix, rng):
    """
    Remove a point, then re-maximize.
    """
    pos = chosen_indices.index(idx_to_remove)
    chosen_indices.pop(pos)
    chosen_set.remove(idx_to_remove)
    
    # Rebuild dist_sets
    new_dist_sets = [set() for _ in chosen_indices]
    for i in range(len(chosen_indices)):
        for j in range(i+1, len(chosen_indices)):
            d = dist_matrix[chosen_indices[i]][chosen_indices[j]]
            new_dist_sets[i].add(d)
            new_dist_sets[j].add(d)
            
    # Re-maximize
    new_indices, new_set, new_dist_sets = _make_maximal(
        chosen_indices, chosen_set, new_dist_sets, all_indices, dist_matrix, rng
    )
    
    return new_indices, new_set, new_dist_sets

def construct():
    all_points = list(itertools.product(range(N), repeat=2))
    all_indices = list(range(len(all_points)))
    dist_matrix = _precompute_distances(all_points)
    
    best_size = float('inf')
    best_indices = None
    
    rng = random.Random(42)
    
    while time.time() < DEADLINE:
        curr_indices = []
        curr_set = set()
        curr_dist_sets = []
        
        curr_indices, curr_set, curr_dist_sets = _make_maximal(
            curr_indices, curr_set, curr_dist_sets, all_indices, dist_matrix, rng
        )
        
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            indices_to_try = list(curr_indices)
            rng.shuffle(indices_to_try)
            
            for idx in indices_to_try:
                if time.time() > DEADLINE:
                    break
                
                new_indices, new_set, new_dist_sets = _remove_point(
                    idx, curr_indices, curr_set, curr_dist_sets, all_indices, dist_matrix, rng
                )
                
                if len(new_indices) < len(curr_indices):
                    curr_indices = new_indices
                    curr_set = new_set
                    curr_dist_sets = new_dist_sets
                    improved = True
                    break
        
        if len(curr_indices) < best_size:
            best_size = len(curr_indices)
            best_indices = curr_indices[:]
            
    if best_indices is None:
        curr_indices, curr_set, curr_dist_sets = _make_maximal([], set(), [], all_indices, dist_matrix, rng)
        best_indices = curr_indices
        
    return [all_points[i] for i in best_indices]
