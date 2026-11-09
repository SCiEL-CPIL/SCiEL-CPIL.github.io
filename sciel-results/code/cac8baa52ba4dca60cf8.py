import random
import time
from itertools import combinations

def _check_validity(points):
    """Check if no 4 points are collinear."""
    n = len(points)
    # Map from line key to list of point indices
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            # Normalize direction vector
            g = __import__('math').gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            if dx < 0:
                dx = -dx
                dy = -dy
            elif dx == 0 and dy < 0:
                dy = -dy
            
            # Line equation: ax + by = c
            # Normal vector (-dy, dx)
            a = -dy
            b = dx
            c = a * p1[0] + b * p1[1]
            
            key = (a, b, c)
            if key not in line_map:
                line_map[key] = []
            line_map[key].append(i)
            line_map[key].append(j)
            
    # Check for any line with 4 or more unique points
    for key, indices in line_map.items():
        unique_indices = set(indices)
        if len(unique_indices) >= 4:
            return False
    return True

def _count_3_collinear(points):
    """Count number of collinear triples."""
    n = len(points)
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            g = __import__('math').gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            if dx < 0:
                dx = -dx
                dy = -dy
            elif dx == 0 and dy < 0:
                dy = -dy
            
            a = -dy
            b = dx
            c = a * p1[0] + b * p1[1]
            
            key = (a, b, c)
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
            
    count = 0
    for key, indices in line_map.items():
        k = len(indices)
        if k >= 3:
            count += __import__('math').comb(k, 3)
    return count

def _estimate_max_3_free(points):
    """
    Greedy upper bound for max 3-free subset.
    This is an upper bound on the score. Lower is better for the final score,
    so we want to minimize this estimate.
    """
    n = len(points)
    # Build conflict graph: edge if 3 collinear? No, 3-free means no 3 collinear.
    # So we want an independent set in the 3-uniform hypergraph of collinear triples.
    # Greedy: Pick points that are part of the most triples first?
    # Or simply: Randomized greedy.
    
    # For speed, let's just use a simple heuristic: 
    # The max 3-free subset size is at least the size of the largest set of points
    # such that no 3 are collinear.
    
    # Actually, let's just return the number of points for now as a dummy,
    # or a better bound.
    # A better bound: If we have T triples, and each point is in at most d triples,
    # then the independence number is at least n / (d+1)?
    
    # Let's just use the exact B&B for the final result, and a proxy for SA.
    # Proxy: Minimize the max degree in the triple hypergraph?
    # Or just maximize the number of triples.
    
    # For the SA objective, maximizing triples is a good proxy for minimizing the 3-free set.
    return 0 # Placeholder

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    random.seed(42)
    
    # Initial configuration: Random points
    best_points = None
    best_score = -10**9 # We maximize (triples - 1000*invalid)
    
    # Try a few random starts
    for _ in range(5):
        points = [[random.randint(-100, 100), random.randint(-100, 100)] for _ in range(20)]
        
        # Ensure distinct
        while len(set(tuple(p) for p in points)) < 20:
            idx = random.randint(0, 19)
            points[idx] = [random.randint(-100, 100), random.randint(-100, 100)]
        
        # Simulated Annealing
        T = 1000.0
        T_min = 0.1
        cooling = 0.999
        
        current_points = [p[:] for p in points]
        current_valid = _check_validity(current_points)
        current_triples = _count_3_collinear(current_points) if current_valid else 0
        current_score = current_triples - (10000 if not current_valid else 0)
        
        if current_score > best_score:
            best_score = current_score
            best_points = [p[:] for p in current_points]
            
        while T > T_min and time.time() < deadline:
            # Move: Pick a random point, move to new random location
            idx = random.randint(0, 19)
            old_p = current_points[idx][:]
            new_p = [random.randint(-100, 100), random.randint(-100, 100)]
            
            # Avoid duplicates
            if new_p in current_points and current_points.index(new_p) != idx:
                continue
                
            current_points[idx] = new_p
            
            new_valid = _check_validity(current_points)
            new_triples = _count_3_collinear(current_points) if new_valid else 0
            new_score = new_triples - (10000 if not new_valid else 0)
            
            delta = new_score - current_score
            if delta > 0 or random.random() < __import__('math').exp(delta / T):
                current_points[idx] = new_p
                current_score = new_score
                current_valid = new_valid
                current_triples = new_triples
            else:
                current_points[idx] = old_p
                
            if current_score > best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
                
            T *= cooling
            
            if time.time() > deadline:
                break

    # Final check: ensure best_points is valid
    if best_points is None:
        # Fallback to parabola if SA failed
        return [[x, x * x] for x in range(-10, 10)]
        
    if not _check_validity(best_points):
        # If best found is invalid, fallback
        return [[x, x * x] for x in range(-10, 10)]

    return best_points
