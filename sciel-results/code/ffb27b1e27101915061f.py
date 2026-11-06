import itertools
import time

def construct():
    # Generate the 37-point hexagon (Rings 0, 1, 2)
    points_37 = []
    for a in range(-2, 3):
        for b in range(-2, 3):
            if max(abs(a), abs(b), abs(a+b)) <= 2:
                points_37.append((a, b))
    
    # We need to remove 5 points from these 37 to get 32 points.
    # We want to maximize edges in the remaining 32.
    # Equivalently, minimize edges lost.
    
    # Precompute adjacency for 37 points
    point_set = set(points_37)
    neighbors = {}
    for p in points_37:
        a, b = p
        ns = []
        for da, db in [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]:
            q = (a+da, b+db)
            if q in point_set:
                ns.append(q)
        neighbors[p] = ns

    # Map points to indices for bitmask or list manipulation
    # 37 points, remove 5. C(37,5) is 435897. Manageable.
    # But we can just iterate over combinations of 5 points to remove.
    
    best_edges = -1
    best_points = None
    
    # To speed up, we can start from the current best (Parent/Child) and do local search.
    # Or just brute force if fast enough.
    # 435k combinations * edge count check.
    # Edge count check: sum of degrees in remaining set / 2.
    # We can precompute total edges (90) and subtract edges incident to removed points.
    
    total_edges_37 = 0
    for p in points_37:
        total_edges_37 += len(neighbors[p])
    total_edges_37 //= 2
    
    # Iterate over all combinations of 5 points to remove
    # Optimization: Only consider removing points from Ring 2?
    # Removing Ring 1 or 0 is likely worse.
    ring2_points = [p for p in points_37 if max(abs(p[0]), abs(p[1]), abs(p[0]+p[1])) == 2]
    
    # If we restrict to removing 5 from Ring 2 (12 points), C(12,5) = 792. Very fast.
    for combo in itertools.combinations(ring2_points, 5):
        removed_set = set(combo)
        # Count edges lost
        edges_lost = 0
        # Edges within removed set
        for p in combo:
            for q in neighbors[p]:
                if q in removed_set:
                    edges_lost += 1
        edges_lost //= 2 # Each internal edge counted twice
        
        # Edges between removed and kept
        for p in combo:
            for q in neighbors[p]:
                if q not in removed_set:
                    edges_lost += 1
        
        remaining_edges = total_edges_37 - edges_lost
        if remaining_edges > best_edges:
            best_edges = remaining_edges
            best_points = [p for p in points_37 if p not in removed_set]
            
    # If best_points is None, fallback to hardcoded
    if best_points is None:
        best_points = [
            [0, 0],
            [1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1],
            [2, 0], [1, 1], [0, 2], [-1, 2], [-2, 2], [-2, 1],
            [-2, 0], [-1, -1], [0, -2], [1, -2], [2, -2], [2, -1],
            [3, 0], [2, 1], [1, 2], [0, 3], [-1, 3], [-2, 3],
            [-3, 3], [-3, 2], [-3, 1], [-3, 0], [-2, -1], [-1, -2], [-1, -3]
        ]
        
    return best_points
