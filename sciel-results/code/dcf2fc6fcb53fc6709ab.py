import time

def construct():
    """
    Constructs 32 points on the triangular lattice maximizing the number of edges.
    Uses Breadth-First Search from the origin to generate a compact hexagonal cluster.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # Target number of points
    n_points = 32
    
    # BFS setup
    visited = set()
    queue = [(0, 0)]
    visited.add((0, 0))
    points = []
    
    # Neighbor offsets for the triangular lattice
    # (da, db) such that da^2 + db^2 + da*db == 1
    offsets = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, -1),
        (-1, 1)
    ]
    
    while queue and len(points) < n_points:
        # Check time
        if time.time() > deadline:
            break
            
        a, b = queue.pop(0)
        points.append([a, b])
        
        for da, db in offsets:
            na, nb = a + da, b + db
            if (na, nb) not in visited:
                visited.add((na, nb))
                queue.append((na, nb))
                
    # If for some reason we didn't get 32 (should not happen), pad with distant points
    # But BFS from 0 on infinite lattice will always provide enough.
    # Just in case, add arbitrary distinct points if len < 32
    if len(points) < n_points:
        # This block is likely never executed
        a = 10
        b = 10
        while len(points) < n_points:
            p = [a, b]
            if p not in points:
                points.append(p)
            a += 1
            if a > 64:
                a = -64
                b += 1
                
    return points
