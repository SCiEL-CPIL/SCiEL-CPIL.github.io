import random
import time
import itertools
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110 seconds safety margin
    
    n = 20
    # Search space: grid [-20, 20] for better density
    coords = list(range(-20, 21))
    
    def is_valid(pts):
        """Check if no 4 points are collinear."""
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                count = 2
                for k in range(n):
                    if k == i or k == j:
                        continue
                    x3, y3 = pts[k]
                    if dx * (y3 - y1) - dy * (x3 - x1) == 0:
                        count += 1
                if count >= 4:
                    return False
        return True

    def get_triples(pts):
        """Get list of all 3-collinear triples as tuples of indices."""
        triples = []
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                for k in range(j+1, n):
                    x3, y3 = pts[k]
                    if dx * (y3 - y1) - dy * (x3 - x1) == 0:
                        triples.append((i, j, k))
        return triples

    def min_vertex_cover(triples):
        """
        Find minimum vertex cover of 3-uniform hypergraph.
        Returns the size of the cover.
        """
        if not triples:
            return 0
        
        # Use B&B with pruning
        best_cover_size = n
        
        # Sort triples by "density" to improve pruning
        # We'll just use the first edge for simplicity in B&B
        
        def bnb(edges, depth):
            nonlocal best_cover_size
            if depth >= best_cover_size:
                return
            if not edges:
                best_cover_size = depth
                return
                
            # Pick the first edge
            v1, v2, v3 = edges[0]
            
            # Branch 1: Include v1
            new_edges_1 = [e for e in edges[1:] if v1 not in e]
            bnb(new_edges_1, depth + 1)
            
            # Branch 2: Include v2
            new_edges_2 = [e for e in edges[1:] if v2 not in e]
            bnb(new_edges_2, depth + 1)
            
            # Branch 3: Include v3
            new_edges_3 = [e for e in edges[1:] if v3 not in e]
            bnb(new_edges_3, depth + 1)

        bnb(triples, 0)
        return best_cover_size

    def calculate_score(pts):
        triples = get_triples(pts)
        cover_size = min_vertex_cover(triples)
        return n - cover_size

    def random_points():
        pts = []
        while len(pts) < n:
            x = random.choice(coords)
            y = random.choice(coords)
            p = (x, y)
            if p not in pts:
                pts.append(p)
        return pts

    best_score = 21
    best_pts = None
    
    # Initial random search
    for _ in range(100):
        if time.time() > deadline:
            break
        pts = random_points()
        if is_valid(pts):
            score = calculate_score(pts)
            if score < best_score:
                best_score = score
                best_pts = pts[:]
                if best_score <= 12:
                    break

    # Local Search from best found
    if best_pts is None:
        # Fallback: return a valid configuration
        return [[x, x*x] for x in range(-10, 10)]

    current_pts = best_pts[:]
    current_score = best_score
    
    while time.time() < deadline:
        improved = False
        # Try moving each point
        for i in range(n):
            if time.time() > deadline:
                break
            original_p = current_pts[i]
            # Try a few random moves for this point
            for _ in range(20):
                new_x = random.choice(coords)
                new_y = random.choice(coords)
                new_p = (new_x, new_y)
                if new_p == original_p:
                    continue
                if new_p in current_pts:
                    continue
                
                # Temporarily replace
                old_p = current_pts[i]
                current_pts[i] = new_p
                
                if is_valid(current_pts):
                    score = calculate_score(current_pts)
                    if score < current_score:
                        current_score = score
                        best_score = score
                        best_pts = current_pts[:]
                        improved = True
                    else:
                        current_pts[i] = old_p
                else:
                    current_pts[i] = old_p
            
            if improved:
                break # Restart loop from improved state
        
        if not improved:
            # Try a different random restart if stuck
            if best_score <= 11:
                break

    return [list(p) for p in best_pts]
