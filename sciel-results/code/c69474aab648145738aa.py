import time
import random
import math
from collections import defaultdict

N = 64

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin
    
    num_points = N * N
    coords = [(i // N, i % N) for i in range(num_points)]
    
    # Precompute squared distances using numpy
    try:
        import numpy as np
        xs = np.array([c[0] for c in coords], dtype=np.int32)
        ys = np.array([c[1] for c in coords], dtype=np.int32)
        dist2 = (xs[:, None] - xs[None, :])**2 + (ys[:, None] - ys[None, :])**2
    except ImportError:
        dist2 = [[0]*num_points for _ in range(num_points)]
        for i in range(num_points):
            xi, yi = coords[i]
            for j in range(num_points):
                xj, yj = coords[j]
                dx = xi - xj
                dy = yi - yj
                dist2[i][j] = dx*dx + dy*dy
    
    def get_dist(i, j):
        return dist2[i][j]
    
    # Build conflict graph: edge (i,j) exists if adding both i and j to a set
    # would create an isosceles triangle with some third point k.
    # Actually, the constraint is: no three points form an isosceles triangle.
    # So we need to ensure that for any three selected points a,b,c, 
    # d(a,b) != d(a,c), d(a,b) != d(b,c), d(a,c) != d(b,c).
    # 
    # The conflict graph approach: Two points i and j are "conflicting" if there
    # exists a point k such that {i,j,k} forms an isosceles triangle. But this
    # depends on the third point, so it's not a simple pairwise conflict.
    #
    # Better approach: Use a different structural method.
    # Let's use a greedy algorithm with multiple restarts and different ordering strategies.
    # The key insight: we want to maximize the number of points such that no three form
    # an isosceles triangle. This is equivalent to: for each point p in the set, all
    # distances from p to other points in the set must be distinct.
    #
    # New strategy: Use a randomized greedy algorithm with multiple different point
    # orderings, and track the best solution found.
    
    rng = random.Random(42)
    
    def validate_set(point_set):
        """Check if a set of points has no isosceles triangles."""
        if len(point_set) <= 2:
            return True
        
        # For each point, check that all distances to other points are distinct
        for p in point_set:
            dists = {}
            for q in point_set:
                if p == q:
                    continue
                d = get_dist(p, q)
                if d in dists:
                    return False
                dists[d] = True
        return True
    
    def greedy_construction(order):
        """
        Greedily add points in the given order if they don't create an isosceles triangle.
        Returns the resulting set.
        """
        selected = set()
        # For efficiency, maintain for each selected point the set of distances to other selected points
        dist_sets = {}  # dist_sets[p] = set of distances from p to other selected points
        
        for p in order:
            # Check if adding p would create an isosceles triangle
            can_add = True
            
            # First check: distances from p to existing points must all be distinct
            new_dists = set()
            for q in selected:
                d = get_dist(p, q)
                if d in new_dists:
                    can_add = False
                    break
                new_dists.add(d)
            
            if not can_add:
                continue
            
            # Second check: for each existing point q, the distance d(p,q) must not already
            # be in dist_sets[q]
            for q in selected:
                d = get_dist(p, q)
                if d in dist_sets[q]:
                    can_add = False
                    break
            
            if can_add:
                selected.add(p)
                dist_sets[p] = new_dists
                for q in selected:
                    if q != p:
                        dist_sets[q].add(get_dist(p, q))
        
        return selected
    
    def greedy_with_local_improvement(order, max_iterations=100):
        """
        Start with greedy construction, then try local improvements by swapping points.
        """
        selected = greedy_construction(order)
        if not selected:
            return selected
        
        # Build distance sets
        dist_sets = {}
        for p in selected:
            dist_sets[p] = set()
        for p in selected:
            for q in selected:
                if p != q:
                    dist_sets[p].add(get_dist(p, q))
        
        improved = True
        iteration = 0
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # Try to add any unselected point
            for p in order:
                if p in selected:
                    continue
                
                # Check if p can be added
                new_dists = set()
                can_add = True
                for q in selected:
                    d = get_dist(p, q)
                    if d in new_dists:
                        can_add = False
                        break
                    new_dists.add(d)
                    if d in dist_sets[q]:
                        can_add = False
                        break
                
                if can_add:
                    selected.add(p)
                    dist_sets[p] = new_dists
                    for q in selected:
                        if q != p:
                            dist_sets[q].add(get_dist(p, q))
                    improved = True
            
            # Try to swap: remove one point, add another
            selected_list = list(selected)
            for i, p_remove in enumerate(selected_list):
                if len(selected) <= 2:
                    break
                
                # Temporarily remove p_remove
                saved_dists = {q: set(dist_sets[q]) for q in selected}
                saved_dist_sets = {q: set(dist_sets[q]) for q in selected}
                
                for q in selected:
                    if q != p_remove:
                        dist_sets[q].discard(get_dist(p_remove, q))
                del dist_sets[p_remove]
                selected.discard(p_remove)
                
                # Try to add a different point
                for p_add in order:
                    if p_add in selected or p_add == p_remove:
                        continue
                    
                    new_dists = set()
                    can_add = True
                    for q in selected:
                        d = get_dist(p_add, q)
                        if d in new_dists:
                            can_add = False
                            break
                        new_dists.add(d)
                        if d in dist_sets[q]:
                            can_add = False
                            break
                    
                    if can_add:
                        selected.add(p_add)
                        dist_sets[p_add] = new_dists
                        for q in selected:
                            if q != p_add:
                                dist_sets[q].add(get_dist(p_add, q))
                        improved = True
                        break
                
                # If we didn't find a swap, restore
                if not improved:
                    selected.add(p_remove)
                    dist_sets[p_remove] = saved_dist_sets[p_remove]
                    for q in selected:
                        if q != p_remove:
                            dist_sets[q] = saved_dist_sets[q]
        
        return selected
    
    # Try multiple different orderings
    best_set = set()
    
    # Ordering 1: Random
    order1 = list(range(num_points))
    rng.shuffle(order1)
    set1 = greedy_with_local_improvement(order1, max_iterations=50)
    if len(set1) > len(best_set):
        best_set = set1
    
    # Ordering 2: Sort by x-coordinate, then y
    order2 = sorted(range(num_points), key=lambda i: (coords[i][0], coords[i][1]))
    set2 = greedy_with_local_improvement(order2, max_iterations=50)
    if len(set2) > len(best_set):
        best_set = set2
    
    # Ordering 3: Sort by distance from center
    cx, cy = N/2.0, N/2.0
    order3 = sorted(range(num_points), key=lambda i: (coords[i][0]-cx)**2 + (coords[i][1]-cy)**2)
    set3 = greedy_with_local_improvement(order3, max_iterations=50)
    if len(set3) > len(best_set):
        best_set = set3
    
    # Ordering 4: Random with bias toward corners
    order4 = list(range(num_points))
    rng.shuffle(order4)
    # Bias: points near corners get higher priority
    corner_scores = []
    for i in range(num_points):
        x, y = coords[i]
        # Distance to nearest corner
        min_corner_dist = min(
            x*x + y*y,  # (0,0)
            (N-1-x)**2 + y*y,  # (N-1,0)
            x*x + (N-1-y)**2,  # (0,N-1)
            (N-1-x)**2 + (N-1-y)**2  # (N-1,N-1)
        )
        corner_scores.append((min_corner_dist, i))
    corner_scores.sort()
    order4 = [i for _, i in corner_scores]
    set4 = greedy_with_local_improvement(order4, max_iterations=50)
    if len(set4) > len(best_set):
        best_set = set4
    
    # Ordering 5: Spiral from center
    order5 = []
    visited = set()
    cx, cy = N//2, N//2
    for r in range(max(N//2, 1) + 1):
        # Add points at distance r from center in spiral order
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                if max(abs(dx), abs(dy)) == r:
                    x, y = cx+dx, cy+dy
                    if 0 <= x < N and 0 <= y < N:
                        idx = y*N + x
                        if idx not in visited:
                            order5.append(idx)
                            visited.add(idx)
    # Add any remaining points
    for i in range(num_points):
        if i not in visited:
            order5.append(i)
    set5 = greedy_with_local_improvement(order5, max_iterations=50)
    if len(set5) > len(best_set):
        best_set = set5
    
    # Validate the best set
    if not validate_set(best_set):
        # Fallback to a small valid set
        best_set = {0, N+2, 4*N+1}  # (0,0), (1,2), (4,1)
    
    return [(i // N, i % N) for i in best_set]

# EVOLVE-BLOCK-END
