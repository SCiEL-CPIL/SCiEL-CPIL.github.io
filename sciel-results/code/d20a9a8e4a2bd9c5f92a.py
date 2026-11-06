"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 145  # Leave 5s margin for evaluation overhead
    
    # 1. Generate initial random points ensuring no 4 collinear
    def is_valid_config(points):
        n = len(points)
        # Check no 4 collinear
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    p1 = points[i]
                    p2 = points[j]
                    p3 = points[k]
                    # Check if any other point p4 is collinear
                    # Vector p2-p1
                    dx1 = p2[0] - p1[0]
                    dy1 = p2[1] - p1[1]
                    # Vector p3-p1
                    dx2 = p3[0] - p1[0]
                    dy2 = p3[1] - p1[1]
                    # Cross product for collinearity of p1, p2, p3
                    # We don't need p1,p2,p3 to be collinear, we need to check if p4 is on the line defined by any pair?
                    # No, "no four collinear" means for ANY set of 4 points, they are not all on one line.
                    # Equivalent: No line contains 4 or more points.
                    pass
        
        # Efficient check: For each pair, count how many other points are collinear.
        # If any pair has >= 2 other points collinear with it, we have 4 collinear.
        for i in range(n):
            for j in range(i + 1, n):
                p1 = points[i]
                p2 = points[j]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                count = 0
                for k in range(n):
                    if k == i or k == j:
                        continue
                    p3 = points[k]
                    dx3 = p3[0] - p1[0]
                    dy3 = p3[1] - p1[1]
                    if dx * dy3 - dy * dx3 == 0:
                        count += 1
                if count >= 2:
                    return False
        return True

    def count_collinear_triples(points):
        n = len(points)
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                p1 = points[i]
                p2 = points[j]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                # Normalize direction to avoid double counting lines?
                # We just want the number of triples.
                # A line with k points has C(k,3) triples.
                # It is easier to count lines and sum C(k,3).
                pass
        
        # Better: Count lines with >= 3 points.
        # Use a set of normalized lines.
        lines = set()
        for i in range(n):
            for j in range(i + 1, n):
                p1 = points[i]
                p2 = points[j]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                # Normalize (dx, dy)
                if dx == 0:
                    norm = (0, 1)
                elif dy == 0:
                    norm = (1, 0)
                else:
                    g = math.gcd(abs(dx), abs(dy))
                    dx //= g
                    dy //= g
                    if dx < 0:
                        dx = -dx
                        dy = -dy
                    norm = (dx, dy)
                
                # Calculate line constant c for x*dy - y*dx = c
                # Using p1
                c = p1[0] * norm[1] - p1[1] * norm[0]
                line_id = (norm[0], norm[1], c)
                lines.add(line_id)
        
        # This counts unique lines. We need to know how many points are on each line to count triples.
        # Re-do: Map line_id to set of points.
        line_points = {}
        for i in range(n):
            for j in range(i + 1, n):
                p1 = points[i]
                p2 = points[j]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                if dx == 0:
                    norm = (0, 1)
                elif dy == 0:
                    norm = (1, 0)
                else:
                    g = math.gcd(abs(dx), abs(dy))
                    dx //= g
                    dy //= g
                    if dx < 0:
                        dx = -dx
                        dy = -dy
                    norm = (dx, dy)
                c = p1[0] * norm[1] - p1[1] * norm[0]
                line_id = (norm[0], norm[1], c)
                if line_id not in line_points:
                    line_points[line_id] = set()
                line_points[line_id].add(i)
                line_points[line_id].add(j)
        
        total_triples = 0
        for pts in line_points.values():
            k = len(pts)
            if k >= 3:
                total_triples += math.comb(k, 3)
        return total_triples

    def generate_random_config():
        while True:
            points = []
            used = set()
            for _ in range(20):
                while True:
                    x = random.randint(-100, 100)
                    y = random.randint(-100, 100)
                    if (x, y) not in used:
                        used.add((x, y))
                        points.append([x, y])
                        break
            if is_valid_config(points):
                return points

    # 2. Local Search to maximize collinear triples
    best_points = generate_random_config()
    best_triples = count_collinear_triples(best_points)
    
    current_points = best_points
    current_triples = best_triples
    
    while time.time() - start_time < timeout:
        # Try a random swap
        i, j = random.sample(range(20), 2)
        new_points = [p[:] for p in current_points]
        # Move point i to a new random location
        old_x, old_y = new_points[i]
        while True:
            nx = random.randint(-100, 100)
            ny = random.randint(-100, 100)
            if (nx, ny) not in [(p[0], p[1]) for p in new_points if p is not new_points[i]]:
                new_points[i] = [nx, ny]
                break
        
        if not is_valid_config(new_points):
            continue
            
        new_triples = count_collinear_triples(new_points)
        
        if new_triples > current_triples:
            current_points = new_points
            current_triples = new_triples
            if current_triples > best_triples:
                best_points = current_points
                best_triples = current_triples
        else:
            # Accept with some probability (Simulated Annealing style) or just reject
            # For simplicity and speed, reject unless better, or accept if equal to explore
            if new_triples == current_triples and random.random() < 0.5:
                current_points = new_points
                current_triples = new_triples

    return best_points
# EVOLVE-BLOCK-END
