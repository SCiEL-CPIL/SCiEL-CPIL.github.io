import math
import random
import time
from collections import defaultdict

def count_four_lines(points):
    n = len(points)
    line_counts = defaultdict(int)
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                slope_key = (0, 1)
            elif dy == 0:
                slope_key = (1, 0)
            else:
                g = math.gcd(abs(dx), abs(dy))
                dx_n = dx // g
                dy_n = dy // g
                if dx_n < 0:
                    dx_n = -dx_n
                    dy_n = -dy_n
                slope_key = (dx_n, dy_n)
            
            c = slope_key[1] * x1 - slope_key[0] * y1
            line_id = (slope_key[0], slope_key[1], c)
            line_counts[line_id] += 1
            
    four_count = 0
    five_or_more = 0
    for count in line_counts.values():
        if count == 4:
            four_count += 1
        elif count >= 5:
            five_or_more += 1
            
    # Penalize 5+ collinear points heavily
    score = four_count - 100 * five_or_more
    return score

def construct():
    start_time = time.time()
    time_limit = 90  # seconds
    
    random.seed(42)
    
    # Initial configuration: 4x4 grid
    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score = count_four_lines(best_points)
    
    current_points = [list(p) for p in best_points]
    current_score = best_score
    
    T = 50.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            T = T_min
            
        # Pick a random point to move
        i = random.randint(0, 15)
        orig_x, orig_y = current_points[i]
        
        # Generate a new random position
        # Bias towards keeping points somewhat close to original to maintain structure?
        # Or completely random?
        # Let's try a mix: 50% small move, 50% large move
        if random.random() < 0.5:
            nx = orig_x + random.randint(-20, 20)
            ny = orig_y + random.randint(-20, 20)
        else:
            nx = random.randint(-1000, 1000)
            ny = random.randint(-1000, 1000)
            
        if abs(nx) > 1000 or abs(ny) > 1000:
            continue
            
        # Check collision
        collision = False
        for j in range(16):
            if j != i and current_points[j][0] == nx and current_points[j][1] == ny:
                collision = True
                break
                
        if collision:
            continue
            
        new_points = [list(p) for p in current_points]
        new_points[i] = [nx, ny]
        new_score = count_four_lines(new_points)
        
        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / T):
            current_points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = [list(p) for p in current_points]
                
        T *= cooling
        
    return best_points
