import time
import random
import itertools
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    N = 20
    # Initial random points
    # Ensure distinct
    while True:
        points = []
        seen = set()
        for _ in range(N):
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in seen:
                seen.add((x, y))
                points.append([x, y])
        if is_valid(points):
            break
        # If not valid, try again or perturb
    
    best_points = [p[:] for p in points]
    best_triples = count_triples(points)
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    while T > T_min and time.time() < deadline:
        # Pick a point to move
        idx = random.randint(0, N-1)
        old_x, old_y = points[idx]
        
        # Generate new point
        # Small perturbation
        dx = random.randint(-5, 5)
        dy = random.randint(-5, 5)
        if dx == 0 and dy == 0:
            continue
        new_x = old_x + dx
        new_y = old_y + dy
        
        if new_x < -100 or new_x > 100 or new_y < -100 or new_y > 100:
            continue
            
        # Check if new point is distinct
        if (new_x, new_y) in [(p[0], p[1]) for i, p in enumerate(points) if i != idx]:
            continue
            
        points[idx] = [new_x, new_y]
        
        if is_valid(points):
            new_triples = count_triples(points)
            # Objective: Maximize triples
            delta = new_triples - best_triples
            if delta > 0 or random.random() < math.exp(delta / T): # Note: delta is negative if worse
                # Actually, if we want to MAXIMIZE triples, and delta = new - old
                # If new > old, delta > 0, accept.
                # If new < old, delta < 0, accept with prob exp(delta/T)
                # Wait, standard SA minimizes energy. Here we maximize triples.
                # So energy = -triples.
                # delta_energy = -new_triples - (-old_triples) = old_triples - new_triples
                # Accept if delta_energy < 0 or prob exp(-delta_energy/T)
                # Let's just use the probability directly.
                if new_triples >= best_triples:
                    best_points = [p[:] for p in points]
                    best_triples = new_triples
                else:
                    # Calculate probability
                    # We are going "downhill" in triple count
                    p_accept = math.exp((new_triples - best_triples) / T)
                    if random.random() < p_accept:
                        pass # Keep the move
                    else:
                        # Revert
                        points[idx] = [old_x, old_y]
        else:
            # Revert
            points[idx] = [old_x, old_y]
            
        T *= cooling
        
    # Final check and return
    return [[p[0], p[1]] for p in best_points]

def is_valid(points):
    # Check no 4 collinear
    N = len(points)
    for i in range(N):
        xi, yi = points[i]
        counts = {}
        for j in range(N):
            if i == j: continue
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            # Normalize direction
            g = math.gcd(dx, dy)
            if g == 0: continue
            dx //= g
            dy //= g
            # Handle negative zero
            if dx < 0:
                dx = -dx
                dy = -dy
            key = (dx, dy)
            counts[key] = counts.get(key, 0) + 1
        if any(v >= 3 for v in counts.values()):
            return False
    return True

def count_triples(points):
    N = len(points)
    total = 0
    for i in range(N):
        xi, yi = points[i]
        counts = {}
        for j in range(N):
            if i == j: continue
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            g = math.gcd(dx, dy)
            if g == 0: continue
            dx //= g
            dy //= g
            if dx < 0:
                dx = -dx
                dy = -dy
            key = (dx, dy)
            counts[key] = counts.get(key, 0) + 1
        for k in counts.values():
            if k >= 2:
                total += math.comb(k, 2)
    return total // 3
