import time
import math
from itertools import combinations

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 110.0  # Safety margin for 120s limit

    # Distance squared on triangular lattice
    # Points are a + b*omega, omega = exp(i*pi/3)
    # |z|^2 = da^2 + db^2 + da*db
    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da*da + db*db + da*db

    def count_edges(points):
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if dist_sq(points[i], points[j]) == 1:
                    count += 1
        return count

    def hex_dist(p):
        a, b = p
        return (abs(a) + abs(b) + abs(a+b)) // 2

    # 1. Generate the 37-point hexagon (Ring 0 to 3)
    hex_points = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if hex_dist([a, b]) <= 3:
                hex_points.append([a, b])
    
    # Verify we have 37 points
    # 1 + 6 + 12 + 18 = 37
    
    # 2. Identify the outer ring (Ring 3)
    outer_ring = [p for p in hex_points if hex_dist(p) == 3]
    # There should be 18 points in the outer ring
    
    # 3. We need to remove 5 points from the outer ring to get 32 points.
    # We want to maximize the number of edges in the remaining 32 points.
    # This is equivalent to minimizing the number of edges lost.
    
    # Pre-compute adjacency or just count edges for each candidate
    # Since 37 points is small, we can just count edges directly.
    
    best_edges = -1
    best_config = None
    
    # Enumerate all combinations of 5 points to remove from the 18 outer ring points
    # C(18, 5) = 8568
    # For each, create the 32-point set and count edges.
    
    # Optimization: 
    # Edges in full hexagon = 90.
    # Edges in subset = 90 - (edges incident to removed points) + (edges between removed points)
    # Wait, if we remove a set R, the edges remaining are those not incident to R.
    # Edges remaining = Total Edges - Edges(incident to R)
    # Edges(incident to R) = Sum(deg(p) for p in R) - Edges(inside R)
    # So Edges(remaining) = 90 - (Sum(deg(p) for p in R) - Edges(R))
    # To maximize Edges(remaining), we minimize (Sum(deg(p) for p in R) - Edges(R))
    
    # Let's pre-compute degrees in the full hexagon and adjacency between outer ring points.
    point_index = {tuple(p): i for i, p in enumerate(hex_points)}
    n_hex = len(hex_points)
    
    # Adjacency matrix or list
    adj = [[False]*n_hex for _ in range(n_hex)]
    degrees = [0]*n_hex
    
    for i in range(n_hex):
        for j in range(i+1, n_hex):
            if dist_sq(hex_points[i], hex_points[j]) == 1:
                adj[i][j] = True
                adj[j][i] = True
                degrees[i] += 1
                degrees[j] += 1
                
    total_edges_hex = sum(degrees) // 2 # Should be 90
    
    # Map outer ring points to their indices in hex_points
    outer_indices = [point_index[tuple(p)] for p in outer_ring]
    
    # We want to choose 5 indices from outer_indices
    # Cost = Sum(degrees[i] for i in chosen) - (number of edges between chosen points)
    
    min_cost = float('inf')
    best_removed_indices = None
    
    for combo in combinations(outer_indices, 5):
        cost = 0
        for i in combo:
            cost += degrees[i]
        
        # Subtract edges within the combo
        for i in range(5):
            for j in range(i+1, 5):
                if adj[combo[i]][combo[j]]:
                    cost -= 1
        
        if cost < min_cost:
            min_cost = cost
            best_removed_indices = combo
            
    # Construct the best 32-point configuration
    removed_set = set(best_removed_indices)
    best_config = [hex_points[i] for i in range(n_hex) if i not in removed_set]
    
    # Verify edge count
    edges = count_edges(best_config)
    
    # If for some reason this heuristic (subset of hexagon) didn't yield 76,
    # or if there's a bug, fall back to a local search on a spiral.
    # But this method is theoretically sound for "dense" configurations.
    
    if edges < 76:
        # Fallback: Local search
        # Use the best_config as start
        result = best_config
        point_set = set((p[0], p[1]) for p in result)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(result)):
                if time.time() > deadline:
                    break
                ax, ay = result[i][0], result[i][1]
                # Try moving to nearby empty spots
                for da in range(-3, 4):
                    for db in range(-3, 4):
                        nx, ny = ax + da, ay + db
                        if (nx, ny) == (ax, ay) or (nx, ny) in point_set:
                            continue
                        if abs(nx) > 64 or abs(ny) > 64:
                            continue
                        
                        old_p = [ax, ay]
                        new_p = [nx, ny]
                        edges_lost = 0
                        edges_gained = 0
                        for j in range(len(result)):
                            if j == i: continue
                            if dist_sq(old_p, result[j]) == 1:
                                edges_lost += 1
                            if dist_sq(new_p, result[j]) == 1:
                                edges_gained += 1
                            
                        delta = edges_gained - edges_lost
                        if delta > 0:
                            result[i] = [nx, ny]
                            point_set.remove((ax, ay))
                            point_set.add((nx, ny))
                            edges += delta
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return result

    return best_config
