import math
import random
import time

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Initial configuration: Hexagonal lattice
    # 7 points in center (radius 1), 4 points in outer ring (radius 2)
    # Lattice spacing is sqrt(3)
    d = math.sqrt(3)
    
    # Generate initial points
    points = []
    # Center
    points.append([0.0, 0.0])
    # Ring 1 (6 points)
    for i in range(6):
        angle = i * math.pi / 3
        x = d * math.cos(angle)
        y = d * math.sin(angle)
        points.append([x, y])
    # Ring 2 (4 points, pick 4 from 12)
    # The 12 points are at distance 2d.
    # We pick 4 that are roughly symmetric to minimize bounding box.
    # Indices in the 12-point ring: 0, 3, 6, 9 (every 90 degrees? No, hex symmetry is 60)
    # Let's just pick 4 consecutive or spread out.
    # To minimize max radius, we want them close to center, but they are at 2d.
    # Let's pick 4 points at angles 0, 60, 120, 180? No, that's 4 points.
    # Let's use angles: 0, 2pi/3, pi, 4pi/3?
    # Actually, let's just pick 4 arbitrary ones and let the optimizer fix it.
    # Better: pick 4 points that are "corners" of a square-ish shape?
    # Let's pick indices 0, 2, 5, 7 of the 12 points?
    # 12 points at distance 2d.
    # Angles: k * pi/6.
    # Let's pick k=0, 2, 5, 7?
    # Let's just use a simple heuristic: pick 4 points from the ring.
    ring2_indices = [0, 2, 5, 7] # Arbitrary start
    for k in ring2_indices:
        angle = k * math.pi / 6
        x = 2 * d * math.cos(angle)
        y = 2 * d * math.sin(angle)
        points.append([x, y])
        
    # Ensure we have exactly N points
    # If N < 11, we would remove points. If N > 11, we would add.
    # For N=11, this is exact.
    
    def calc_S(points):
        max_s = 0.0
        for x, y in points:
            s1 = abs(y)
            s2 = abs(x + math.sqrt(3)*y) / math.sqrt(3)
            s3 = abs(x - math.sqrt(3)*y) / math.sqrt(3)
            s = max(s1, s2, s3)
            if s > max_s:
                max_s = s
        return 1.0 + max_s

    def check_overlap(points, tol=1e-6):
        min_dist = math.sqrt(3) - tol
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < min_dist:
                    return False
        return True

    current_S = calc_S(points)
    best_points = [p[:] for p in points]
    best_S = current_S
    
    # Local Search / Simulated Annealing
    # Parameters
    T = 0.5
    cooling = 0.999
    steps = 0
    
    while time.time() < deadline:
        steps += 1
        # Pick a random point to move
        idx = random.randint(0, N-1)
        old_x, old_y = points[idx]
        
        # Perturbation
        step_size = 0.1 * T + 0.01
        new_x = old_x + random.uniform(-step_size, step_size)
        new_y = old_y + random.uniform(-step_size, step_size)
        
        points[idx] = [new_x, new_y]
        
        if check_overlap(points):
            new_S = calc_S(points)
            delta = new_S - current_S
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_S = new_S
                if current_S < best_S:
                    best_S = current_S
                    best_points = [p[:] for p in points]
            else:
                # Revert
                points[idx] = [old_x, old_y]
        else:
            # Revert
            points[idx] = [old_x, old_y]
            
        T *= cooling
        if T < 0.01:
            T = 0.01

    # Return the best found configuration
    # The task requires: inner rows (x, y, angle_degrees)
    # All angles are 0.0
    inner = [[p[0], p[1], 0.0] for p in best_points]
    # Outer hexagon center (0,0), side length best_S, rotation 0
    return inner, (0.0, 0.0), best_S, 0.0

# EVOLVE-BLOCK-END
