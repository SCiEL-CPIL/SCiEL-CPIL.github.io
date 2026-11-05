import random
import time
import itertools
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 45.0  # Safety margin before 60s limit
    
    n = 30
    # Initial configuration: 6x5 grid
    # Points: (x, y) where x in [0, 5], y in [0, 4]
    # This is a known good configuration for equal area problems.
    points = [(x, y) for x in range(6) for y in range(5)]
    
    # Precompute indices for all triples
    # We store (i, j, k) for i < j < k
    triples = list(itertools.combinations(range(n), 3))
    num_triples = len(triples)
    
    # Helper to calculate doubled area (absolute determinant)
    def calc_area(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        # 2 * Area = |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
        val = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        return val
    
    # Initialize area counts
    # Using a dictionary for sparse counts, or a list if areas are small.
    # Max area in [-30, 30]^2 is roughly 30*30*2 = 1800? 
    # Actually, max det for points in [-30,30] is around 3600.
    # Let's use a dict to be safe and memory efficient.
    area_counts = {}
    
    # Helper to add a triple's area to counts
    def add_area_counts(pts, tris, counts):
        for i, j, k in tris:
            a = calc_area(pts[i], pts[j], pts[k])
            if a > 0:
                counts[a] = counts.get(a, 0) + 1
    
    # Helper to remove a triple's area from counts
    def remove_area_counts(pts, tris, counts):
        for i, j, k in tris:
            a = calc_area(pts[i], pts[j], pts[k])
            if a > 0:
                counts[a] -= 1
                if counts[a] == 0:
                    del counts[a]
                
    # Initial counts
    add_area_counts(points, triples, area_counts)
    
    # Get current best score
    current_best_score = max(area_counts.values()) if area_counts else 0
    best_points = points.copy()
    
    # For efficient updates, we need to know which triples involve a specific point.
    # Precompute list of triples involving each index
    triples_involving = [[] for _ in range(n)]
    for idx, (i, j, k) in enumerate(triples):
        triples_involving[i].append(idx)
        triples_involving[j].append(idx)
        triples_involving[k].append(idx)
        
    # Local Search Loop
    while time.time() - start_time < deadline:
        # Pick a random point to move
        idx = random.randint(0, n - 1)
        
        # Save old coordinates
        old_x, old_y = points[idx]
        
        # Pick a new random coordinate in [-30, 30]
        new_x = random.randint(-30, 30)
        new_y = random.randint(-30, 30)
        
        # Check distinctness
        new_point = (new_x, new_y)
        is_valid = True
        for p in points:
            if p == new_point:
                is_valid = False
                break
        
        if not is_valid:
            continue
            
        # Remove old areas for triples involving idx
        tris_idx = triples_involving[idx]
        remove_area_counts(points, [triples[t] for t in tris_idx], area_counts)
        
        # Apply move
        points[idx] = new_point
        
        # Add new areas for triples involving idx
        add_area_counts(points, [triples[t] for t in tris_idx], area_counts)
        
        # Calculate new score
        new_score = max(area_counts.values()) if area_counts else 0
        
        if new_score > current_best_score:
            current_best_score = new_score
            best_points = points.copy()
        else:
            # Revert move
            points[idx] = (old_x, old_y)
            # Recalculate areas for revert? 
            # It's faster to just re-add old and remove new if we didn't accept, 
            # but since we already updated area_counts, we must revert area_counts too.
            
            # Remove new areas
            remove_area_counts(points, [triples[t] for t in tris_idx], area_counts)
            # Revert point
            points[idx] = (old_x, old_y)
            # Add old areas
            add_area_counts(points, [triples[t] for t in tris_idx], area_counts)
            
    return best_points
# EVOLVE-BLOCK-END
