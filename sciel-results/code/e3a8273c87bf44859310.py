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

def _generate_optimal_spiral(n):
    """
    Generate the optimal hexagonal cluster of size n.
    This is done by sorting all points in a large enough bounding box
    by (hex_distance, angle) and taking the first n.
    This creates the most compact shape, which is known to maximize edges.
    """
    candidates = []
    # Bounding box large enough to contain n points
    # For n=32, radius 4 is sufficient (1+6+12+18 = 37)
    radius_limit = 10 
    for a in range(-radius_limit, radius_limit + 1):
        for b in range(-radius_limit, radius_limit + 1):
            d = _hexdist(a, b)
            if d <= radius_limit:
                # Convert to cartesian for angle sorting
                # x = a + 0.5*b, y = sqrt(3)/2 * b
                x = a + 0.5 * b
                y = math.sqrt(3) / 2 * b
                # Use atan2 for angle in [0, 2*pi)
                angle = math.atan2(y, x)
                if angle < 0:
                    angle += 2 * math.pi
                candidates.append((d, angle, a, b))
    
    # Sort by distance, then angle
    candidates.sort(key=lambda x: (x[0], x[1]))
    
    # Take the first n points
    result = [(a, b) for _, _, a, b in candidates[:n]]
    return result

def _simulated_annealing(points, n, deadline, rng):
    """
    Refine the points using Simulated Annealing.
    This is a structural change from the previous Hill Climbing.
    It allows neutral and sometimes worse moves to escape local optima.
    """
    current_points = list(points)
    current_set = set(current_points)
    current_edges = _count_edges(current_points)
    
    # Pool of candidate points
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
    
    # Temperature parameters
    T = 1.0
    cooling_rate = 0.95
    
    while time.time() < deadline and T > 0.01:
        improved = False
        # Number of steps per temperature
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
            
            # Calculate new edge count
            # Remove p_i, add p_c
            # We can do this incrementally for speed, but n=32 is small.
            # Let's do incremental to be safe on time.
            
            # Edges lost by removing p_i
            lost_edges = 0
            for da, db in _NEIGHBORS:
                if (p_i[0] + da, p_i[1] + db) in current_set:
                    lost_edges += 1
            
            # Edges gained by adding p_c
            # Note: p_i is still in current_set for this check, so we must exclude it
            gained_edges = 0
            for da, db in _NEIGHBORS:
                nb = (p_c[0] + da, p_c[1] + db)
                if nb in current_set and nb != p_i:
                    gained_edges += 1
            
            delta = gained_edges - lost_edges
            new_edges = current_edges + delta
            
            if new_edges > current_edges:
                # Accept
                current_points[i] = p_c
                current_set.discard(p_i)
                current_set.add(p_c)
                current_edges = new_edges
                improved = True
                
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_points = current_points[:]
                    
                if best_edges >= 76:
                    return best_points
            else:
                # Metropolis criterion
                # Accept with probability exp(-delta / T)
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
    deadline = time.time() + 100  # Safety margin for 120s limit
    n = 32
    rng = random.Random(42)
    
    # 1. Generate the optimal hexagonal spiral
    # This is known to be optimal or very close to optimal for triangular lattices
    initial_points = _generate_optimal_spiral(n)
    
    # Check if already optimal
    edges = _count_edges(initial_points)
    if edges >= 76:
        return [[a, b] for a, b in initial_points]
    
    # 2. Refine with Simulated Annealing
    # SA is a different structural approach compared to the previous Hill Climbing
    refined_points = _simulated_annealing(initial_points, n, deadline, rng)
    final_edges = _count_edges(refined_points)
    
    if final_edges >= 76:
        return [[a, b] for a, b in refined_points]
    
    # 3. If SA didn't reach 76, try a few random restarts with SA
    # This is unlikely given the quality of the spiral start, but adds robustness
    for _ in range(5):
        if time.time() > deadline - 5:
            break
        # Perturb the current best
        perturbed = refined_points[:]
        for _ in range(3):
            i = rng.randrange(n)
            # Random valid replacement
            for _attempt in range(20):
                c_idx = rng.randrange(len(_generate_optimal_spiral(100))) # Use a large pool
                p_c = _generate_optimal_spiral(100)[c_idx]
                if p_c in perturbed:
                    continue
                # Check validity (distance >= 1)
                valid = True
                for other in perturbed:
                    if other == perturbed[i]:
                        continue
                    da = p_c[0] - other[0]
                    db = p_c[1] - other[1]
                    dist_sq = da*da + db*db + da*db
                    if dist_sq < 1:
                        valid = False
                        break
                if valid:
                    perturbed[i] = p_c
                    break
        
        refined_points = _simulated_annealing(perturbed, n, deadline - 5, rng)
        final_edges = _count_edges(refined_points)
        if final_edges >= 76:
            break
            
    return [[a, b] for a, b in refined_points]

if __name__ == "__main__":
    # Simple self-test
    pts = construct()
    edges = _count_edges(pts)
    print(f"Points: {pts}")
    print(f"Edges: {edges}")
    # Expected 76
