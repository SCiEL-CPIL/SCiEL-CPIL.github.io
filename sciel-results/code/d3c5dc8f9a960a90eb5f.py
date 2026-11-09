import random
import time
import math
from collections import defaultdict
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        
        if A == 0 and B == 0:
            return None
            
        # Normalize by GCD
        # Use absolute values for GCD
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

    def get_seeds():
        seeds = []
        
        # 1. 4x4 Grid
        grid = [[x, y] for x in range(4) for y in range(4)]
        seeds.append([p[:] for p in grid])
        
        # 2. Affine Shear of Grid
        shear = [[x + y, y] for x in range(4) for y in range(4)]
        seeds.append([p[:] for p in shear])
        
        # 3. Affine Shear (Other direction)
        shear2 = [[x, x + y] for x in range(4) for y in range(4)]
        seeds.append([p[:] for p in shear2])
        
        # 4. Random
        for _ in range(3):
            pts = []
            while len(pts) < 16:
                x = random.randint(-20, 20)
                y = random.randint(-20, 20)
                if [x, y] not in pts:
                    pts.append([x, y])
            seeds.append(pts)
            
        return seeds

    best_score = -1
    best_pts = None
    
    seeds = get_seeds()
    
    # Evaluate seeds
    for seed in seeds:
        score = evaluate(seed)
        if score > best_score:
            best_score = score
            best_pts = [p[:] for p in seed]
            
    current_pts = [p[:] for p in best_pts]
    current_score = best_score
    
    global_best_pts = [p[:] for p in best_pts]
    global_best_score = best_score
    
    temperature = 50.0
    min_temperature = 0.5
    cooling_rate = 0.99
    
    # We will run multiple restarts if time permits
    restarts = 0
    max_restarts = 5
    
    while time.time() < deadline:
        # Generate neighbor
        new_pts = [p[:] for p in current_pts]
        
        move_type = random.random()
        
        if move_type < 0.6:
            # Single point shift (small)
            idx = random.randint(0, 15)
            step_x = random.randint(-10, 10)
            step_y = random.randint(-10, 10)
            if step_x == 0 and step_y == 0:
                step_x = 1
            new_pts[idx][0] += step_x
            new_pts[idx][1] += step_y
        elif move_type < 0.8:
            # Single point jump (large)
            idx = random.randint(0, 15)
            new_pts[idx] = [random.randint(-100, 100), random.randint(-100, 100)]
        elif move_type < 0.9:
            # Two point swap
            i, j = random.sample(range(16), 2)
            new_pts[i], new_pts[j] = new_pts[j][:], new_pts[i][:]
        else:
            # "Line completion" heuristic move
            # Pick a random pair, see if we can move a 3rd point to make a 4-line
            # This is complex to implement correctly in a simple move, so we stick to random
            # But we can try to move a point to a coordinate that is "aligned" with two others
            i, j, k = random.sample(range(16), 3)
            p1 = new_pts[i]
            p2 = new_pts[j]
            # Calculate line p1-p2
            # Try to move p_k to a point on this line that is not p1 or p2
            # And ideally not creating a 5-collinear
            # For simplicity, we just do a small random move here too
            idx = k
            new_pts[idx][0] += random.randint(-5, 5)
            new_pts[idx][1] += random.randint(-5, 5)
            
        new_score = evaluate(new_pts)
        
        # Metropolis acceptance
        delta = new_score - current_score
        if delta > 0:
            current_pts = new_pts
            current_score = new_score
            
            if current_score > global_best_score:
                global_best_score = current_score
                global_best_pts = [p[:] for p in current_pts]
                # If we find a very good score, we can reset temperature slightly to explore around it
                # temperature = max(temperature, 5.0) 
        else:
            # Accept with probability exp(delta / T)
            if temperature > 0 and random.random() < math.exp(delta / temperature):
                current_pts = new_pts
                current_score = new_score
        
        # Cool down
        temperature *= cooling_rate
        if temperature < min_temperature:
            # Restart from global best or a random seed
            restarts += 1
            if restarts > max_restarts:
                # If we've restarted too many times, we might just stop or keep restarting from best
                # Let's just restart from global best to polish
                current_pts = [p[:] for p in global_best_pts]
                current_score = global_best_score
                temperature = 10.0
            else:
                if random.random() < 0.5:
                    current_pts = [p[:] for p in global_best_pts]
                    current_score = global_best_score
                else:
                    # Pick a random seed
                    seed = random.choice(seeds)
                    current_pts = [p[:] for p in seed]
                    current_score = evaluate(current_pts)
                    # Perturb slightly
                    for _ in range(5):
                        idx = random.randint(0, 15)
                        current_pts[idx][0] += random.randint(-5, 5)
                        current_pts[idx][1] += random.randint(-5, 5)
                    current_score = evaluate(current_pts)
                
                temperature = 30.0 # Reheat

    return global_best_pts

if __name__ == "__main__":
    # Quick test
    pts = construct()
    print(pts)
