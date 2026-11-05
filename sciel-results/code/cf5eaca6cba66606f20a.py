import math
import time
import itertools

def construct():
    deadline = time.time() + 140
    
    def dist_sq(p, q):
        da = p[0] - q[0]
        db = p[1] - q[1]
        return da*da + db*db + da*db

    def count_edges(points):
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if dist_sq(points[i], points[j]) == 1:
                    count += 1
        return count

    # Hexagon of radius 2: 19 points
    R_core = 2
    core_points = []
    for a in range(-R_core, R_core+1):
        for b in range(-R_core, R_core+1):
            if abs(a+b) <= R_core:
                core_points.append([a, b])
    
    # Hexagon of radius 3: 37 points, boundary = 18 points
    R_full = 3
    full_points = []
    for a in range(-R_full, R_full+1):
        for b in range(-R_full, R_full+1):
            if abs(a+b) <= R_full:
                full_points.append([a, b])
                
    core_set = set(map(tuple, core_points))
    boundary_points = [p for p in full_points if tuple(p) not in core_set]
    
    n_boundary = len(boundary_points)  # 18
    target_count = 13  # 19 + 13 = 32
    
    core_edges = count_edges(core_points)
    
    # Pre-compute boundary-core edges and boundary-boundary adjacency
    b_core_edges = [0] * n_boundary
    for i, bp in enumerate(boundary_points):
        for cp in core_points:
            if dist_sq(bp, cp) == 1:
                b_core_edges[i] += 1
                
    b_b_adj = [[0]*n_boundary for _ in range(n_boundary)]
    for i in range(n_boundary):
        for j in range(i+1, n_boundary):
            if dist_sq(boundary_points[i], boundary_points[j]) == 1:
                b_b_adj[i][j] = 1
                b_b_adj[j][i] = 1
                
    best_edges = -1
    best_combo = None
    
    for combo in itertools.combinations(range(n_boundary), target_count):
        if time.time() > deadline - 5:
            break
        edges = core_edges
        for i in combo:
            edges += b_core_edges[i]
        for idx in range(len(combo)):
            for jdx in range(idx+1, len(combo)):
                edges += b_b_adj[combo[idx]][combo[jdx]]
        if edges > best_edges:
            best_edges = edges
            best_combo = combo
            
    if best_combo is None:
        best_combo = tuple(range(target_count))
    
    final_points = core_points + [boundary_points[i] for i in best_combo]
    
    if len(final_points) != 32:
        return [[a, b] for a in range(16) for b in range(2)]
        
    return final_points
