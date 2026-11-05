import time
import random
import math
import sys

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit (150s wall-clock kill)

    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return None
        g = math.gcd(abs(dx), abs(dy))
        dx //= g
        dy //= g
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        c = dy * x1 - dx * y1
        return (dx, dy, c)

    def evaluate(points):
        n = len(points)
        if len(set(tuple(p) for p in points)) != n:
            return -10000
        line_counts = {}
        for i in range(n):
            xi, yi = points[i]
            for j in range(i + 1, n):
                xj, yj = points[j]
                dx = xj - xi
                dy = yj - yi
                if dx == 0 and dy == 0:
                    return -10000
                g = math.gcd(abs(dx), abs(dy))
                dx //= g
                dy //= g
                if dx < 0 or (dx == 0 and dy < 0):
                    dx = -dx
                    dy = -dy
                c = dy * xi - dx * yi
                key = (dx, dy, c)
                line_counts[key] = line_counts.get(key, 0) + 1
        score = 0
        invalid = False
        for count in line_counts.values():
            if count == 6:
                score += 1
            elif count >= 10:
                invalid = True
                break
        if invalid:
            return -1000 + score
        return score

    # Generate affine plane of order 4: points are (x, y) in GF(4)^2
    # GF(4) = {0, 1, alpha, alpha+1} where alpha^2 = alpha + 1
    # We'll use a specific representation: elements 0,1,2,3 with 2^2=2+1=3, 3^2=0
    # Actually let's use the standard: GF(4) with primitive element w where w^2+w+1=0
    # Elements: 0, 1, w, w+1
    # We'll label them 0,1,2,3 where 2=w, 3=w+1
    # Multiplication table for GF(4):
    #   0 1 2 3
    # 0 0 0 0 0
    # 1 0 1 2 3
    # 2 0 2 3 1
    # 3 0 3 1 2
    
    gf4_mult = {
        (0,0):0, (0,1):0, (0,2):0, (0,3):0,
        (1,0):0, (1,1):1, (1,2):2, (1,3):3,
        (2,0):0, (2,1):2, (2,2):3, (2,3):1,
        (3,0):0, (3,1):3, (3,2):1, (3,3):2
    }
    
    def gf4_add(a, b):
        return a ^ b  # XOR works for GF(4) with this labeling
    
    def gf4_mul(a, b):
        return gf4_mult[(a, b)]
    
    # Generate all 16 points and 20 lines of the affine plane
    # Points: (x, y) where x, y in {0,1,2,3}
    # Lines: for each slope m in {0,1,2,3} (vertical) and intercept b, 
    #         plus the 4 vertical lines
    # Total: 4 slopes * 4 intercepts + 4 vertical = 20 lines
    
    abstract_points = [(x, y) for x in range(4) for y in range(4)]
    
    # Generate lines
    lines = []
    # Vertical lines: x = c for c in {0,1,2,3}
    for c in range(4):
        line = [(c, y) for y in range(4)]
        lines.append(line)
    
    # Non-vertical lines: y = m*x + b for m in {0,1,2,3}, b in {0,1,2,3}
    for m in range(4):
        for b in range(4):
            line = []
            for x in range(4):
                y = gf4_add(gf4_mul(m, x), b)
                line.append((x, y))
            lines.append(line)
    
    # Now we need to map these 16 abstract points to integer coordinates
    # Strategy: use a linear embedding with small integers
    # Map (x, y) in GF(4)^2 to integer coordinates
    # We'll try several mappings and pick the best
    
    # Mapping 1: simple grid-like
    # Map GF(4) element to integer: 0->0, 1->1, 2->2, 3->3
    # Then (x,y) -> (x, y) gives 4x4 grid
    
    # Mapping 2: use different base
    # Try to find a mapping that creates more 4-point lines
    
    # Let's try a systematic approach: assign integer coordinates to the 16 points
    # such that the 20 lines of the affine plane become actual lines in Z^2
    
    # This is a constraint satisfaction problem. Let's use a greedy approach.
    
    # First, let's try the simple 4x4 grid and see its score
    simple_grid = [[x, y] for x in range(4) for y in range(4)]
    simple_score = evaluate(simple_grid)
    
    # Now try to improve by permuting or transforming
    # The key insight: we need the 20 affine plane lines to be actual geometric lines
    
    # Let's try a different approach: use the structure to guide coordinate assignment
    # We'll assign coordinates one point at a time, trying to satisfy as many lines as possible
    
    # For now, let's start with the simple grid and do local search
    best_points = [p[:] for p in simple_grid]
    best_score = simple_score
    current_points = [p[:] for p in simple_grid]
    current_score = simple_score
    
    # Try several initial configurations
    initial_configs = [
        [[x, y] for x in range(4) for y in range(4)],  # 4x4 grid
        [[x*2, y*2] for x in range(4) for y in range(4)],  # scaled
        [[x, y*2] for x in range(4) for y in range(4)],  # stretched
        [[x*2, y] for x in range(4) for y in range(4)],  # stretched
        # Add some structured configurations
        [[x*3, y*3] for x in range(4) for y in range(4)],
        [[x*5, y*2] for x in range(4) for y in range(4)],
        [[x*2, y*5] for x in range(4) for y in range(4)],
    ]
    
    for config in initial_configs:
        s = evaluate(config)
        if s > best_score:
            best_score = s
            best_points = [p[:] for p in config]
            current_points = [p[:] for p in config]
            current_score = s
    
    # Local search with simulated annealing
    T = 5.0
    T_min = 0.1
    
    def random_move(points):
        idx = random.randint(0, 15)
        if random.random() < 0.7:
            new_x = points[idx][0] + random.randint(-3, 3)
            new_y = points[idx][1] + random.randint(-3, 3)
        else:
            new_x = points[idx][0] + random.randint(-10, 10)
            new_y = points[idx][1] + random.randint(-10, 10)
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            return None
        new_point = [new_x, new_y]
        for k in range(16):
            if k != idx and points[k] == new_point:
                return None
        return idx, new_point
    
    no_improve = 0
    max_no_improve = 500
    
    while time.time() - start_time < time_limit:
        move = random_move(current_points)
        if move is None:
            continue
        idx, new_point = move
        old_point = current_points[idx][:]
        current_points[idx] = new_point
        new_score = evaluate(current_points)
        
        if new_score > current_score:
            current_score = new_score
            no_improve = 0
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
        else:
            delta = current_score - new_score
            if delta < 0:
                accept = True
            else:
                accept = random.random() < math.exp(-delta / T)
            if accept:
                pass
            else:
                current_points[idx] = old_point
            no_improve += 1
        
        T *= 0.998
        if T < T_min:
            T = T_min
        
        if no_improve > max_no_improve:
            # Restart with a random configuration
            idx = random.randint(0, 15)
            new_x = random.randint(-50, 50)
            new_y = random.randint(-50, 50)
            is_distinct = True
            for k in range(16):
                if k != idx and current_points[k] == [new_x, new_y]:
                    is_distinct = False
                    break
            if is_distinct:
                current_points[idx] = [new_x, new_y]
                current_score = evaluate(current_points)
            no_improve = 0
            T = 5.0  # Reheat
    
    return best_points
