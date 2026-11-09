import numpy as np
import itertools
from collections import Counter

def construct():
    # Pool of candidate points
    pool = [(x, y) for x in range(-10, 11) for y in range(-10, 11)]
    
    # Initial configuration: 6x5 grid
    init_pts = [(x, y) for x in range(-2, 4) for y in range(-2, 3)]
    
    def compute_areas(pts):
        """Compute all positive doubled areas for a set of points."""
        n = len(pts)
        areas = []
        for i in range(n):
            x1, y1 = pts[i]
            for j in range(i+1, n):
                x2, y2 = pts[j]
                for k in range(j+1, n):
                    x3, y3 = pts[k]
                    det = abs((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))
                    if det > 0:
                        areas.append(det)
        return areas
    
    def get_best_area(areas):
        """Find the area with highest multiplicity."""
        if not areas:
            return 0, 0
        counts = Counter(areas)
        return max(counts.items(), key=lambda x: x[1])
    
    # Precompute area counts for initial config
    current_pts = init_pts[:]
    areas = compute_areas(current_pts)
    best_area, best_count = get_best_area(areas)
    
    # Track best solution
    global_best_count = best_count
    global_best_pts = current_pts[:]
    global_best_area = best_area
    
    # Simulated annealing
    import random
    import math
    
    T = 100.0
    T_min = 0.01
    cooling = 0.995
    max_iters = 3000
    
    for it in range(max_iters):
        if T < T_min:
            break
        
        # Pick a random point to move
        idx = random.randint(0, 29)
        old_pt = current_pts[idx]
        
        # Pick a random new point from pool (not already in set)
        candidates = [p for p in pool if p not in current_pts]
        if not candidates:
            break
        new_pt = random.choice(candidates)
        
        # Compute new areas efficiently
        # Remove old_pt, add new_pt
        new_pts = current_pts[:]
        new_pts[idx] = new_pt
        
        new_areas = compute_areas(new_pts)
        new_best_area, new_best_count = get_best_area(new_areas)
        
        # Acceptance criterion
        delta = new_best_count - best_count
        if delta > 0 or random.random() < math.exp(delta / T):
            current_pts = new_pts
            best_area, best_count = new_best_area, new_best_count
            
            if best_count > global_best_count:
                global_best_count = best_count
                global_best_pts = current_pts[:]
                global_best_area = best_area
        
        T *= cooling
    
    # Final local search to polish
    for _ in range(100):
        improved = False
        areas = compute_areas(global_best_pts)
        best_area, best_count = get_best_area(areas)
        
        # Try swapping each point with each candidate
        for i in range(30):
            old_pt = global_best_pts[i]
            candidates = [p for p in pool if p not in global_best_pts]
            for new_pt in candidates:
                new_pts = global_best_pts[:]
                new_pts[i] = new_pt
                new_areas = compute_areas(new_pts)
                _, new_count = get_best_area(new_areas)
                if new_count > best_count:
                    global_best_pts = new_pts
                    best_count = new_count
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return global_best_pts
