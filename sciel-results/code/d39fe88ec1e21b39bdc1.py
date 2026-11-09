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

def _generate_hex_spiral(n):
    """
    Generate optimal hexagonal cluster by adding points in order of
    increasing hex distance, breaking ties by angle.
    For n=32, this gives 1+6+12+13=32 points.
    """
    # Collect all points up to sufficient radius
    max_radius = 5  # Enough for n=32
    candidates = []
    for a in range(-max_radius, max_radius + 1):
        for b in range(-max_radius, max_radius + 1):
            d = _hexdist(a, b)
            if d <= max_radius:
                x = a + 0.5 * b
                y = math.sqrt(3) / 2 * b
                angle = math.atan2(y, x)
                if angle < 0:
                    angle += 2 * math.pi
                candidates.append((d, angle, a, b))
    
    candidates.sort(key=lambda x: (x[0], x[1]))
    result = [(a, b) for _, _, a, b in candidates[:n]]
    return result

def construct():
    n = 32
    deadline = time.time() + 120
    
    # Try the hex spiral with various angle offsets
    best_points = []
    best_edges = 0
    
    for offset in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5]:
        if time.time() > deadline - 10:
            break
        
        # Generate spiral with this offset
        max_radius = 5
        candidates = []
        for a in range(-max_radius, max_radius + 1):
            for b in range(-max_radius, max_radius + 1):
                d = _hexdist(a, b)
                if d <= max_radius:
                    x = a + 0.5 * b
                    y = math.sqrt(3) / 2 * b
                    angle = math.atan2(y, x) + offset
                    if angle < 0:
                        angle += 2 * math.pi
                    candidates.append((d, angle, a, b))
        
        candidates.sort(key=lambda x: (x[0], x[1]))
        points = [(a, b) for _, _, a, b in candidates[:n]]
        edges = _count_edges(points)
        
        if edges > best_edges:
            best_edges = edges
            best_points = points
        
        if edges >= 76:
            return [[a, b] for a, b in best_points]
    
    # If no spiral reached 76, use simulated annealing
    if best_edges < 76 and time.time() < deadline - 5:
        rng = random.Random(42)
        current_points = list(best_points)
        current_set = set(current_points)
        current_edges = best_edges
        
        # Build pool of candidate positions
        pool = set()
        for p in current_points:
            a, b = p
            for da in range(-3, 4):
                for db in range(-3, 4):
                    if _hexdist(a+da, b+db) <= 5:
                        pool.add((a+da, b+db))
        pool = list(pool)
        
        T = 1.0
        cooling_rate = 0.95
        
        while time.time() < deadline - 5 and T > 0.01:
            for _ in range(500):
                if time.time() > deadline - 5:
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
                        return [[a, b] for a, b in best_points]
                else:
                    if delta < 0:
                        prob = math.exp(delta / T)
                        if rng.random() < prob:
                            current_points[i] = p_c
                            current_set.discard(p_i)
                            current_set.add(p_c)
                            current_edges = new_edges
            
            T *= cooling_rate
    
    return [[a, b] for a, b in best_points]
