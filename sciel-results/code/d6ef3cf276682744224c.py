"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations
from collections import Counter

def _score(points):
    """Compute the maximum multiplicity of a positive doubled area."""
    n = len(points)
    if n < 3:
        return 0
    pts = np.array(points, dtype=np.int64)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                det = abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
                if det > 0:
                    areas.append(det)
    if not areas:
        return 0
    return max(Counter(areas).values())

def _score_fast(points):
    """Faster scoring using vectorized computation."""
    n = len(points)
    if n < 3:
        return 0
    pts = np.array(points, dtype=np.int64)
    # Compute all pairwise differences
    # For each triple (i,j,k), doubled area = |cross(pts[j]-pts[i], pts[k]-pts[i])|
    # We can compute this efficiently
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                dx1 = pts[j,0] - pts[i,0]
                dy1 = pts[j,1] - pts[i,1]
                dx2 = pts[k,0] - pts[i,0]
                dy2 = pts[k,1] - pts[i,1]
                det = abs(dx1*dy2 - dx2*dy1)
                if det > 0:
                    areas.append(det)
    if not areas:
        return 0
    return max(Counter(areas).values())

def _local_search(points, max_iter=200, neighborhood=3, deadline=None):
    """Local search to improve the configuration."""
    best = list(points)
    best_score = _score(best)
    n = len(best)
    
    for iteration in range(max_iter):
        if deadline and time.time() > deadline - 5:
            break
        
        improved = False
        # Try moving each point in the neighborhood
        for idx in range(n):
            for dx in range(-neighborhood, neighborhood+1):
                for dy in range(-neighborhood, neighborhood+1):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = best[idx][0] + dx
                    new_y = best[idx][1] + dy
                    if -30 <= new_x <= 30 and -30 <= new_y <= 30:
                        # Check distinctness
                        new_point = (new_x, new_y)
                        if new_point not in best:
                            trial = list(best)
                            trial[idx] = new_point
                            s = _score(trial)
                            if s > best_score:
                                best = trial
                                best_score = s
                                improved = True
                                break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return best, best_score

def construct():
    deadline = time.time() + 80  # Safety margin before 90s limit
    
    # Start with a good initial configuration
    # Use a grid-like pattern that tends to produce equal areas
    # A 5x6 grid
    initial = [(x, y) for x in range(5) for y in range(6)]
    
    # Scale to fit in [-30, 30]
    # Current range: x in [0,4], y in [0,5]
    # Scale by 6 and center
    scaled = [(x*6 - 12, y*6 - 15) for x in range(5) for y in range(6)]
    
    best_points = scaled
    best_score = _score(best_points)
    
    # Try several random restarts with local search
    random.seed(42)
    for restart in range(50):
        if time.time() > deadline - 10:
            break
        
        # Generate a random initial configuration
        pts = []
        used = set()
        while len(pts) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            if (x, y) not in used:
                pts.append((x, y))
                used.add((x, y))
        
        # Run local search
        found, score = _local_search(pts, max_iter=100, neighborhood=3, deadline=deadline)
        if score > best_score:
            best_points = found
            best_score = score
    
    # Final polish with larger neighborhood on the best found
    if time.time() < deadline - 15:
        found, score = _local_search(best_points, max_iter=50, neighborhood=5, deadline=deadline)
        if score > best_score:
            best_points = found
            best_score = score
    
    return best_points
# EVOLVE-BLOCK-END
