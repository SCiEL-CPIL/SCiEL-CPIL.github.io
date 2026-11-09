import time
import math
import random

# Pre-define neighbors for fast edge counting
# Triangular lattice neighbors for (a, b) are:
# (a+1, b), (a-1, b), (a, b+1), (a, b-1), (a+1, b-1), (a-1, b+1)
_NEIGHBORS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]

def _hexdist(a, b):
    """Calculate hexagonal distance from origin (0,0)"""
    return (abs(a) + abs(b) + abs(a + b)) // 2

def _count_edges(points):
    """Count the number of pairs at distance exactly 1"""
    point_set = set(points)
    count = 0
    for p in points:
        a, b = p
        for da, db in _NEIGHBORS:
            if (a + da, b + db) in point_set:
                count += 1
    return count // 2  # Each edge is counted twice

def _generate_optimal_spiral(n, angle_offset=0.0):
    """
    Generate the optimal hexagonal cluster of size n.
    This is done by sorting all points in a large enough bounding box
    by (hex_distance, angle + offset) and taking the first n.
    """
    candidates = []
    radius_limit = 10 
    for a in range(-radius_limit, radius_limit + 1):
        for b in range(-radius_limit, radius_limit + 1):
            d = _hexdist(a, b)
            if d <= radius_limit:
                x = a + 0.5 * b
                y = math.sqrt(3) / 2 * b
                angle = math.atan2(y, x) + angle_offset
                if angle < 0:
                    angle += 2 * math.pi
                candidates.append((d, angle, a, b))
    
    candidates.sort(key=lambda x: (x[0], x[1]))
    result = [(a, b) for _, _, a, b in candidates[:n]]
    return result

def _simulated_annealing(points, n, deadline, rng):
    current_points = list(points)
    current_set = set(current_points)
    current_edges = _count_edges(current_points)
    
    pool = set()
    for p in current_points:
        a, b = p
        for da in range(-5, 6):
            for db in range(-5, 6):
                if _hexdist(a+da, b+db) <= 10:
                    pool.add((a+da, b+db))
    pool = list(pool)
    
    best_points = current_points[:]
    best_edges = current_edges
    
    T = 1.0
    cooling_rate = 0.95
    
    while time.time() < deadline and T > 0.01:
        steps = 1000
        for _ in range(steps):
            if time.time() > deadline:
                break
            
            i = rng.randrange(n)
            p_i = current_points[i]
            
            if not pool:
                break
            
            c_idx = rng.randrange(len(pool))
            p_c = pool[c_idx]
            
            if p_c == p_i or p_c in current_set:
                continue
            
            lost_edges = 0
            for da, db in _NEIGHBORS:
                if (p_i[0] + da, p_i[1] + db) in current_set:
                    lost_edges += 1
            
            gained_edges = 0
            for da, db in _NEIGHBORS:
                nb = (p_c[0] + da, p_c[1] + db)
                if nb in current_set and nb != p_i:
                    gained_edges += 1
            
            delta = gained_edges - lost_edges
            new_edges = current_edges + delta
            
            if new_edges > current_edges:
                current_points[i] = p_c
                current_set.discard(p_i)
                current_set.add(p_c)
                current_edges = new_edges
                
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_points = current_points[:]
                    
                if best_edges >= 76:
                    return best_points
            else:
                if delta < 0:
                    prob = math.exp(delta / T)
                    if rng.random() < prob:
                        current_points[i] = p_c
                        current_set.discard(p_i)
                        current_set.add(p_c)
                        current_edges = new_edges
        
        T *= cooling_rate
        
    return best_points

def construct():
    deadline = time.time() + 100
    n = 32
    rng = random.Random(42)
    
    best_points = []
    best_edges = 0
    
    # Try multiple spiral variations to find an optimal one quickly
    # The choice of which 13 points from radius 3 are included depends on angle sorting.
    # By varying the angle offset, we sample different subsets.
    for offset in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        initial_points = _generate_optimal_spiral(n, offset)
        edges = _count_edges(initial_points)
        if edges > best_edges:
            best_edges = edges
            best_points = initial_points
        if edges >= 76:
            return [[a, b] for a, b in initial_points]
            
    # If no spiral reached 76, refine the best one with SA
    if best_edges < 76:
        refined_points = _simulated_annealing(best_points, n, deadline, rng)
        final_edges = _count_edges(refined_points)
        if final_edges > best_edges:
            best_points = refined_points
            best_edges = final_edges
            
    return [[a, b] for a, b in best_points]
