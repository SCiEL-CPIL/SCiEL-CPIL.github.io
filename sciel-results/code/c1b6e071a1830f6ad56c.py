import random
import time
import math
from collections import defaultdict
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Helper to get canonical line key
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        
        if A == 0 and B == 0:
            return None # Should not happen for distinct points
            
        # Normalize by GCD
        g = gcd(gcd(abs(A), abs(B)), abs(C)) if C != 0 else gcd(abs(A), abs(B))
        if g == 0:
            return None
        A //= g
        B //= g
        C //= g
        
        # Ensure unique sign: first non-zero of (A, B, C) is positive
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
            
        return (A, B, C)

    def evaluate(points):
        # Check validity: distinct and bounds
        if len(set(tuple(p) for p in points)) != 16:
            return -1000000
        for x, y in points:
            if abs(x) > 1000 or abs(y) > 1000:
                return -1000000
        
        line_counts = defaultdict(int)
        
        for i in range(16):
            for j in range(i + 1, 16):
                key = get_line_key(points[i], points[j])
                if key:
                    line_counts[key] += 1
        
        n4_lines = 0
        for count in line_counts.values():
            # count is number of pairs
            # k points -> k*(k-1)/2 pairs
            # k=4 -> 6 pairs
            # k=5 -> 10 pairs
            if count == 6:
                n4_lines += 1
            elif count >= 10:
                # 5 or more collinear
                return -1000000 + n4_lines # Penalize heavily
        
        return n4_lines

    def get_initial_configurations():
        configs = []
        
        # 1. 4x4 Grid
        grid = [[x, y] for x in range(4) for y in range(4)]
        configs.append([p[:] for p in grid])
        
        # 2. Affine transforms of 4x4 grid
        # Try some simple affine transformations: x' = ax+by, y' = cx+dy
        # Keep them within bounds.
        transforms = [
            (1, 1, 1, 1), # Identity (already covered)
            (1, 0, 0, 2), # Stretch Y
            (2, 0, 0, 1), # Stretch X
            (1, 1, 1, -1), # Shear
            (1, -1, 1, 1), # Shear
        ]
        for a, b, c, d in transforms:
            pts = []
            valid = True
            for x in range(4):
                for y in range(4):
                    nx = a * x + b * y
                    ny = c * x + d * y
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        valid = False
                        break
                    pts.append([nx, ny])
                if not valid:
                    break
            if valid and len(set(tuple(p) for p in pts)) == 16:
                configs.append(pts)
                
        # 3. Random configurations
        for _ in range(5):
            pts = []
            while len(pts) < 16:
                x = random.randint(-50, 50)
                y = random.randint(-50, 50)
                if [x, y] not in pts:
                    pts.append([x, y])
            configs.append(pts)
            
        return configs

    best_score = -1
    best_pts = None
    
    initial_configs = get_initial_configurations()
    
    # Evaluate initial configs to find a good starting point
    for cfg in initial_configs:
        score = evaluate(cfg)
        if score > best_score:
            best_score = score
            best_pts = [p[:] for p in cfg]
            
    # Simulated Annealing
    current_pts = [p[:] for p in best_pts]
    current_score = best_score
    temperature = 20.0
    min_temperature = 0.1
    cooling_rate = 0.995
    
    # Keep a record of best found so far
    global_best_pts = [p[:] for p in best_pts]
    global_best_score = best_score
    
    while time.time() < deadline:
        # Generate neighbor
        new_pts = [p[:] for p in current_pts]
        
        # Randomly choose move type
        move_type = random.random()
        
        if move_type < 0.7:
            # Single point shift
            idx = random.randint(0, 15)
            step = random.randint(-5, 5)
            if step != 0:
                if random.random() < 0.5:
                    new_pts[idx][0] += step
                else:
                    new_pts[idx][1] += step
        elif move_type < 0.9:
            # Two point swap (exchange coordinates)
            i, j = random.sample(range(16), 2)
            new_pts[i], new_pts[j] = new_pts[j][:], new_pts[i][:]
        else:
            # Random jump for one point
            idx = random.randint(0, 15)
            new_pts[idx] = [random.randint(-100, 100), random.randint(-100, 100)]
            
        new_score = evaluate(new_pts)
        
        # Metropolis acceptance
        delta = new_score - current_score
        if delta > 0:
            current_pts = new_pts
            current_score = new_score
            
            if current_score > global_best_score:
                global_best_score = current_score
                global_best_pts = [p[:] for p in current_pts]
        else:
            # Accept with probability exp(delta / T)
            if temperature > 0 and random.random() < math.exp(delta / temperature):
                current_pts = new_pts
                current_score = new_score
        
        # Cool down
        temperature *= cooling_rate
        if temperature < min_temperature:
            temperature = min_temperature
            
        # Restart logic: if stuck for a while or temperature is low, restart from best
        if temperature == min_temperature:
            # Restart from global best or a random initial config
            if random.random() < 0.5:
                current_pts = [p[:] for p in global_best_pts]
                current_score = global_best_score
            else:
                # Pick a random initial config and perturb it
                cfg = random.choice(initial_configs)
                current_pts = [p[:] for p in cfg]
                for _ in range(10): # Perturb
                    idx = random.randint(0, 15)
                    current_pts[idx][0] += random.randint(-5, 5)
                    current_pts[idx][1] += random.randint(-5, 5)
                current_score = evaluate(current_pts)
            temperature = 10.0 # Reheat slightly

    return global_best_pts

if __name__ == "__main__":
    # Quick test
    pts = construct()
    print(pts)
