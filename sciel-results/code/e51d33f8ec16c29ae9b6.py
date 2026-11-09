import time
import random

def _count_edges(points):
    """
    Count the number of pairs of points at distance exactly 1.
    Distance squared in triangular lattice (basis 1, omega) for diff (da, db) is da^2 + db^2 + da*db.
    Unit distance requires da^2 + db^2 + da*db == 1.
    Unique undirected neighbors to check for each point:
    (1, 0), (0, 1), (1, -1)
    """
    point_set = set((p[0], p[1]) for p in points)
    edges = 0
    neighbors = [(1, 0), (0, 1), (1, -1)]
    
    for a, b in point_set:
        for da, db in neighbors:
            if (a + da, b + db) in point_set:
                edges += 1
    return edges

def _greedy_construct(n_points=32):
    """
    Greedily construct a set of n_points on the triangular lattice
    maximizing the number of unit-distance edges.
    """
    points = set()
    # Start with origin
    points.add((0, 0))
    
    # Directions to check for neighbors
    neighbors = [(1, 0), (0, 1), (1, -1), (-1, 0), (0, -1), (-1, 1)]
    
    for _ in range(n_points - 1):
        best_point = None
        best_score = -1
        best_dist = float('inf')
        
        # Generate candidates: neighbors of existing points
        # This keeps the cluster compact
        candidates = set()
        for p in points:
            for da, db in neighbors:
                candidates.add((p[0] + da, p[1] + db))
        
        # Filter out already selected points
        candidates -= points
        
        # Evaluate each candidate
        for c in candidates:
            # Count how many neighbors c has in the current points set
            added_edges = 0
            for da, db in neighbors:
                if (c[0] + da, c[1] + db) in points:
                    added_edges += 1
            
            # Calculate distance from origin to prefer compact shapes
            # Using Euclidean distance in the lattice coordinates is a proxy
            # or we can use the lattice norm. Let's use simple a^2+b^2+ab
            dist = c[0]*c[0] + c[1]*c[1] + c[0]*c[1]
            
            # We want to maximize added_edges, and minimize dist
            if added_edges > best_score or (added_edges == best_score and dist < best_dist):
                best_score = added_edges
                best_dist = dist
                best_point = c
        
        if best_point is not None:
            points.add(best_point)
        else:
            # Fallback if no neighbors found (should not happen for small n)
            # Just add a random point far away? Or break?
            # For n=32, this won't happen.
            break
            
    return list(points)

def construct():
    start_time = time.time()
    deadline = start_time + 10.0
    
    # Generate the initial configuration using greedy method
    best_points = _greedy_construct(32)
    best_score = _count_edges(best_points)
    
    # Optional: Local search to ensure we haven't missed a better configuration
    # Given the greedy method is quite strong for this specific geometry,
    # and the time limit is tight for complex searches, we might just return.
    # However, a simple random restart or perturbation might help if greedy gets stuck.
    
    # Let's try a few random restarts of the greedy process with different starting points
    # or slight variations to see if we can beat the first result.
    
    for _ in range(20): # Small number of restarts
        if time.time() > deadline:
            break
            
        # Random start point
        start = (random.randint(-5, 5), random.randint(-5, 5))
        points = {start}
        
        neighbors = [(1, 0), (0, 1), (1, -1), (-1, 0), (0, -1), (-1, 1)]
        
        for _ in range(31):
            candidates = set()
            for p in points:
                for da, db in neighbors:
                    candidates.add((p[0] + da, p[1] + db))
            
            candidates -= points
            
            best_point = None
            best_score = -1
            best_dist = float('inf')
            
            for c in candidates:
                added_edges = 0
                for da, db in neighbors:
                    if (c[0] + da, c[1] + db) in points:
                        added_edges += 1
                
                dist = c[0]*c[0] + c[1]*c[1] + c[0]*c[1]
                
                # Add a bit of randomness to tie-breaking to explore different shapes
                # But primarily maximize edges
                if added_edges > best_score or (added_edges == best_score and dist < best_dist):
                    best_score = added_edges
                    best_dist = dist
                    best_point = c
            
            if best_point:
                points.add(best_point)
        
        current_score = _count_edges(list(points))
        if current_score > best_score:
            best_score = current_score
            best_points = list(points)
            
    return best_points
