import math
import itertools

def construct():
    # Generate points in axial coordinates (q, r)
    # Hexagonal distance: (|q| + |r| + |q+r|) // 2
    
    def hex_dist(q, r):
        return (abs(q) + abs(r) + abs(q + r)) // 2

    # Get all points with dist <= 3 (37 points)
    points = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            if hex_dist(q, r) <= 3:
                points.append((q, r))
                
    n_points = len(points)
    if n_points != 37:
        # Fallback if generation changes
        return [[p[0], p[1]] for p in points[:32]]

    # Build adjacency list
    # Map point to index
    point_to_idx = {p: i for i, p in enumerate(points)}
    adj = {i: set() for i in range(n_points)}
    
    for i in range(n_points):
        q1, r1 = points[i]
        for dq, dr in [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]:
            q2, r2 = q1 + dq, r1 + dr
            if (q2, r2) in point_to_idx:
                j = point_to_idx[(q2, r2)]
                adj[i].add(j)
                adj[j].add(i)
                
    total_edges = sum(len(v) for v in adj.values()) // 2
    # Should be 96
    
    # We need to remove 5 points to get 32 points.
    # We want to minimize the number of edges removed.
    # Target edges = 76.
    # Edges to remove = total_edges - 76 = 20.
    
    target_removed_edges = total_edges - 76
    
    # Backtracking search for 5 points to remove
    # State: (current_removed_set, current_edges_removed)
    # Prune: if current_edges_removed > target_removed_edges, stop.
    # Also, if max possible edges we can still remove (by removing remaining points) 
    # is not enough to reach target? No, we want EXACTLY or AT MOST?
    # We want to MAXIMIZE kept edges. So we want to MINIMIZE removed edges.
    # If we find a set with removed_edges < target, it's better than 76?
    # But 76 is the theoretical max. So removed_edges cannot be < 20.
    # So we search for removed_edges == 20.
    
    best_removed_edges = total_edges # Initialize high
    best_set = None
    
    def backtrack(start, removed, removed_edges, depth):
        nonlocal best_removed_edges, best_set
        
        if depth == 5:
            if removed_edges < best_removed_edges:
                best_removed_edges = removed_edges
                best_set = removed[:]
            return
            
        # Pruning
        if removed_edges > best_removed_edges:
            return
            
        # If even removing all remaining points doesn't beat best, prune?
        # Max edges we can add to removed_edges by removing k more points.
        # Each point has max degree 6.
        # This bound is loose.
        
        for i in range(start, n_points):
            # Calculate new removed edges
            # New edges removed = edges from i to (all points) - edges from i to (already removed)
            # Wait, edges removed = edges incident to removed points that are NOT internal to removed set?
            # No.
            # Total edges in original graph = E.
            # Edges in remaining graph = E - (Edges incident to Removed) + (Edges within Removed).
            # Removed Edges = (Edges incident to Removed) - (Edges within Removed).
            
            # Let R be the set of removed points.
            # Edges incident to R = sum(deg(v) for v in R).
            # Edges within R = sum(1 for u,v in R if u~v).
            # Removed = sum(deg(v) for v in R) - 2 * EdgesWithinR.
            
            # When adding i to R:
            # New Removed = Old Removed + deg(i) - 2 * (edges between i and R)
            
            deg_i = len(adj[i])
            edges_to_removed = sum(1 for j in removed if j in adj[i])
            
            new_removed_edges = removed_edges + deg_i - 2 * edges_to_removed
            
            if new_removed_edges > best_removed_edges:
                continue
                
            removed.append(i)
            backtrack(i + 1, removed, new_removed_edges, depth + 1)
            removed.pop()
            
    backtrack(0, [], 0, 0)
    
    if best_set is None:
        # Fallback
        return [[p[0], p[1]] for p in points[:32]]
        
    # Construct final points
    removed_set = set(best_set)
    final_points = [points[i] for i in range(n_points) if i not in removed_set]
    
    return [[p[0], p[1]] for p in final_points]
