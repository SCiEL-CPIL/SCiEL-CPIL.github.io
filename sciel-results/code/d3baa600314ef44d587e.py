import time
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Helper to calculate doubled area
    def calc_area(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

    # Generate initial grid
    def make_grid(w, h):
        pts = []
        for x in range(w):
            for y in range(h):
                pts.append((x - w//2, y - h//2))
        return pts

    candidates = [
        make_grid(6, 5),
        make_grid(5, 6),
        make_grid(10, 3),
        make_grid(3, 10),
        make_grid(15, 2),
        make_grid(2, 15)
    ]
    
    best_score = 0
    best_points = candidates[0]
    
    # Evaluate initial candidates
    for pts in candidates:
        # Check validity
        if len(set(pts)) != 30:
            continue
        if not all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts):
            continue
            
        counts = defaultdict(int)
        for i, j, k in itertools.combinations(range(30), 3):
            d = calc_area(pts[i], pts[j], pts[k])
            if d > 0:
                counts[d] += 1
        
        if counts:
            max_c = max(counts.values())
            if max_c > best_score:
                best_score = max_c
                best_points = pts[:]

    # Local Search
    # We need a fast update mechanism
    # Precompute pairs for each point?
    # When point i moves, triangles (i, j, k) change.
    # There are 29*28/2 = 406 such triangles.
    
    points = best_points[:]
    counts = defaultdict(int)
    for i, j, k in itertools.combinations(range(30), 3):
        d = calc_area(points[i], points[j], points[k])
        if d > 0:
            counts[d] += 1
            
    current_max = max(counts.values()) if counts else 0
    current_best_area = max(counts, key=counts.get) if counts else 0

    # Optimization: Store all triples and their current areas?
    # Or just recompute the 406 affected ones.
    
    def get_current_max():
        if not counts:
            return 0, 0
        return max(counts.values()), max(counts, key=counts.get)

    # Random seed for reproducibility in local search? 
    # The task doesn't require deterministic output, just valid and good.
    # But using a fixed seed helps debugging.
    random.seed(42)

    while time.time() < deadline:
        i = random.randint(0, 29)
        old_p = points[i]
        
        # Generate new position
        # Try random nearby first, then random global?
        # Global random might be too far. Nearby is better for local search.
        # Let's mix: 50% nearby, 50% random in bounds.
        
        if random.random() < 0.5:
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
            new_x = old_p[0] + dx
            new_y = old_p[1] + dy
        else:
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
            
        new_p = (new_x, new_y)
        
        # Check bounds and collision
        if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
            continue
        if new_p == old_p:
            continue
        if any(p == new_p for p in points if p != old_p):
            continue
            
        # Apply move
        points[i] = new_p
        
        # Update counts for triangles involving i
        # Triangles are (i, j, k) where j < k, j,k != i
        # We need to subtract old areas and add new areas.
        
        # To do this efficiently, we iterate over all pairs (j, k) from the other 29 points.
        # This is 406 iterations.
        
        # Store changes to revert if necessary
        # changes: list of (area_old, area_new)
        # But we need to update counts immediately to check max?
        # We can just update counts, check max, and if worse, revert.
        
        # However, calculating max(counts.values()) every time is O(K) where K is number of distinct areas.
        # K can be large.
        # Alternative: Keep a histogram of counts? i.e. count_of_counts[area_count] = number_of_areas_with_that_count.
        # Then max count is just the largest key in count_of_counts with value > 0.
        
        # Let's implement count_of_counts.
        # count_of_counts[c] = number of areas d such that counts[d] == c.
        
        # We need to initialize this.
        # And update it when counts[d] changes.
        
        # This adds complexity. Given 406 updates, and max() over ~100-500 keys is fast, 
        # maybe just using max(counts.values()) is fine?
        # 406 * 2 (dec/inc) + max() 
        # max() over dict values.
        # Let's try it. If too slow, optimize.
        
        affected_areas = []
        for j in range(30):
            if j == i: continue
            for k in range(j+1, 30):
                if k == i: continue
                # Triangle (i, j, k)
                old_d = calc_area(old_p, points[j], points[k])
                new_d = calc_area(new_p, points[j], points[k])
                
                if old_d != new_d:
                    affected_areas.append((old_d, new_d))
                    # Update counts
                    if old_d > 0:
                        counts[old_d] -= 1
                        if counts[old_d] == 0:
                            del counts[old_d]
                    if new_d > 0:
                        counts[new_d] = counts.get(new_d, 0) + 1
        
        new_max, new_area = get_current_max()
        
        if new_max > current_max:
            current_max = new_max
            current_best_area = new_area
            best_points = points[:]
            best_score = new_max
        else:
            # Revert
            points[i] = old_p
            for old_d, new_d in affected_areas:
                if new_d > 0:
                    counts[new_d] -= 1
                    if counts[new_d] == 0:
                        del counts[new_d]
                if old_d > 0:
                    counts[old_d] = counts.get(old_d, 0) + 1

    return best_points
