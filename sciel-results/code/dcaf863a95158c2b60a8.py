"""Triangular lattice packing: maximize edges for 32 points."""

import time

def _dist_sq(p1, p2):
    """Squared distance between two lattice points on triangular lattice."""
    a1, b1 = p1
    a2, b2 = p2
    da = a1 - a2
    db = b1 - b2
    # Distance squared: |a + b*omega|^2 where omega = exp(i*pi/3)
    # = a^2 + b^2 + a*b (for the triangular lattice metric)
    return da*da + db*db + da*db

def _edge_count(points):
    """Count pairs at exactly unit distance."""
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            d = _dist_sq(points[i], points[j])
            if d == 1:
                count += 1
    return count

def _valid(points):
    """Check all pairs are at distance >= 1 and within bounds."""
    n = len(points)
    seen = set()
    for p in points:
        a, b = p
        if abs(a) > 64 or abs(b) > 64:
            return False
        if (a, b) in seen:
            return False
        seen.add((a, b))
    for i in range(n):
        for j in range(i+1, n):
            d = _dist_sq(points[i], points[j])
            if d < 1:
                return False
    return True

def _local_search(points, deadline):
    """Local search to maximize edge count."""
    points = [list(p) for p in points]
    best_edges = _edge_count(points)
    n = len(points)
    
    # Neighbor offsets for triangular lattice (unit distance)
    offsets = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                break
            for da, db in offsets:
                new_a = points[i][0] + da
                new_b = points[i][1] + db
                if abs(new_a) > 64 or abs(new_b) > 64:
                    continue
                # Check if position is already occupied
                if any(points[j][0] == new_a and points[j][1] == new_b for j in range(n) if j != i):
                    continue
                # Check minimum distance constraint
                valid = True
                for j in range(n):
                    if j == i:
                        continue
                    d = _dist_sq([new_a, new_b], points[j])
                    if d < 1:
                        valid = False
                        break
                if not valid:
                    continue
                # Try the move
                old = points[i][:]
                points[i] = [new_a, new_b]
                edges = _edge_count(points)
                if edges > best_edges:
                    best_edges = edges
                    improved = True
                else:
                    points[i] = old
    
    return points, best_edges

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s deadline with safety margin
    
    # A 5x6 block (30 points) plus 2 additional points yields 74 edges.
    # Base block: a in 0..5, b in 0..4
    points = [[a, b] for a in range(6) for b in range(5)]
    # Add (6, 2) and (6, 3) to maximize connections
    points.append([6, 2])
    points.append([6, 3])
    
    best_points = [list(p) for p in points]
    best_edges = _edge_count(best_points)
    
    # Run local search
    if time.time() < deadline:
        search_points, search_edges = _local_search(points, deadline)
        if search_edges > best_edges:
            best_points = search_points
            best_edges = search_edges
    
    return best_points
