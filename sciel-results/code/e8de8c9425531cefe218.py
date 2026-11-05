import time
import random
import math

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit

    # Initial configuration: 4x4 grid
    # This has score 10.
    initial_points = [[x, y] for x in range(4) for y in range(4)]
    
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
        line_counts = {}
        
        for i in range(n):
            xi, yi = points[i]
            for j in range(i + 1, n):
                xj, yj = points[j]
                dx = xj - xi
                dy = yj - yi
                if dx == 0 and dy == 0:
                    return -10000 # Distinctness check
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
            if count == 6: # 4 points -> C(4,2)=6 pairs
                score += 1
            elif count >= 10: # 5 points -> C(5,2)=10 pairs
                invalid = True
                break
        
        if invalid:
            return -1000 + score
        return score

    best_points = [p[:] for p in initial_points]
    best_score = evaluate(best_points)
    current_points = [p[:] for p in initial_points]
    current_score = best_score
    
    # Local Search with Random Restarts
    # We will use a simple hill climbing with random kicks.
    
    def random_move(points):
        idx = random.randint(0, 15)
        # Try a small move first
        if random.random() < 0.5:
            new_x = points[idx][0] + random.randint(-5, 5)
            new_y = points[idx][1] + random.randint(-5, 5)
        else:
            # Larger move
            new_x = points[idx][0] + random.randint(-20, 20)
            new_y = points[idx][1] + random.randint(-20, 20)
        
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            return None
            
        new_point = [new_x, new_y]
        # Check distinctness
        for k in range(16):
            if k != idx and points[k] == new_point:
                return None
        return idx, new_point

    no_improve_count = 0
    max_no_improve = 1000
    
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
            no_improve_count = 0
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
        else:
            # Revert
            current_points[idx] = old_point
            no_improve_count += 1
            
        if no_improve_count > max_no_improve:
            # Kick: move a random point to a random location in a larger range
            # to escape local optimum
            idx = random.randint(0, 15)
            new_x = random.randint(-100, 100)
            new_y = random.randint(-100, 100)
            if abs(new_x) <= 1000 and abs(new_y) <= 1000:
                # Check distinctness
                is_distinct = True
                for k in range(16):
                    if k != idx and current_points[k] == [new_x, new_y]:
                        is_distinct = False
                        break
                if is_distinct:
                    current_points[idx] = [new_x, new_y]
                    current_score = evaluate(current_points)
            no_improve_count = 0

    return best_points
