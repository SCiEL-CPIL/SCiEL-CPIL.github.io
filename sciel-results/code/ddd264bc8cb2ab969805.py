import time
import random
import math

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit

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

    def incremental_eval(points, moved_idx, old_point, new_point):
        """Evaluate score change when moving point moved_idx from old_point to new_point"""
        n = len(points)
        # Check distinctness
        for k in range(n):
            if k != moved_idx and points[k] == new_point:
                return None  # Invalid move
        
        # Count lines through old_point (excluding pairs with other points)
        old_line_counts = {}
        for j in range(n):
            if j == moved_idx:
                continue
            xj, yj = points[j]
            dx = xj - old_point[0]
            dy = yj - old_point[1]
            if dx == 0 and dy == 0:
                return None
            g = math.gcd(abs(dx), abs(dy))
            dx //= g
            dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            c = dy * old_point[0] - dx * old_point[1]
            key = (dx, dy, c)
            old_line_counts[key] = old_line_counts.get(key, 0) + 1
        
        # Count lines through new_point
        new_line_counts = {}
        for j in range(n):
            if j == moved_idx:
                continue
            xj, yj = points[j]
            dx = xj - new_point[0]
            dy = yj - new_point[1]
            if dx == 0 and dy == 0:
                return None
            g = math.gcd(abs(dx), abs(dy))
            dx //= g
            dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            c = dy * new_point[0] - dx * new_point[1]
            key = (dx, dy, c)
            new_line_counts[key] = new_line_counts.get(key, 0) + 1
        
        # For each line, count how many points (other than moved_idx) are on it
        # We need to know the full line count to determine if it's a 4-point line
        # This requires knowing all points on each line, so incremental eval is complex
        # Fall back to full evaluation for correctness
        return None

    # Initial configurations to try
    initial_configs = []
    
    # 4x4 grid
    initial_configs.append([[x, y] for x in range(4) for y in range(4)])
    
    # 4x4 grid shifted
    initial_configs.append([[x + 1, y + 1] for x in range(4) for y in range(4)])
    
    # 4x4 grid with different spacing
    initial_configs.append([[x * 2, y * 2] for x in range(4) for y in range(4)])
    
    # 4x4 grid rotated
    initial_configs.append([[x + y, x - y] for x in range(4) for y in range(4)])
    
    # 4x4 grid scaled
    initial_configs.append([[x * 3, y * 3] for x in range(4) for y in range(4)])
    
    # 4x4 grid with offset
    initial_configs.append([[x + 5, y + 3] for x in range(4) for y in range(4)])
    
    # 4x4 grid with different pattern
    initial_configs.append([[x * 2 + (y % 2), y * 2] for x in range(4) for y in range(4)])
    
    # 4x4 grid with diagonal offset
    initial_configs.append([[x + y, y] for x in range(4) for y in range(4)])
    
    # 4x4 grid with modular pattern
    initial_configs.append([[x * 3, (x * 2 + y) % 7] for x in range(4) for y in range(4)])
    
    # 4x4 grid with prime spacing
    initial_configs.append([[x * 5, y * 7] for x in range(4) for y in range(4)])
    
    # 4x4 grid with larger spacing
    initial_configs.append([[x * 10, y * 10] for x in range(4) for y in range(4)])
    
    # 4x4 grid with mixed spacing
    initial_configs.append([[x * 4, y * 6] for x in range(4) for y in range(4)])
    
    # 4x4 grid with alternating offset
    initial_configs.append([[x * 3 + (x % 2), y * 3 + (y % 2)] for x in range(4) for y in range(4)])
    
    # 4x4 grid with cross pattern
    initial_configs.append([[x, y] for x in range(4) for y in range(4)])
    
    # 4x4 grid with shifted rows
    initial_configs.append([[x + (y % 2), y] for x in range(4) for y in range(4)])
    
    # 4x4 grid with larger scale
    initial_configs.append([[x * 7, y * 7] for x in range(4) for y in range(4)])

    best_points = None
    best_score = -10000
    
    # Evaluate initial configurations and pick the best
    for config in initial_configs:
        # Check bounds
        valid = True
        for p in config:
            if abs(p[0]) > 1000 or abs(p[1]) > 1000:
                valid = False
                break
        if not valid:
            continue
        
        # Check distinctness
        seen = set()
        for p in config:
            if (p[0], p[1]) in seen:
                valid = False
                break
            seen.add((p[0], p[1]))
        if not valid:
            continue
        
        score = evaluate(config)
        if score > best_score:
            best_score = score
            best_points = [p[:] for p in config]
    
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(4)]
        best_score = evaluate(best_points)
    
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    def random_move(points):
        idx = random.randint(0, 15)
        # Try a small move first
        if random.random() < 0.3:
            new_x = points[idx][0] + random.randint(-3, 3)
            new_y = points[idx][1] + random.randint(-3, 3)
        elif random.random() < 0.6:
            new_x = points[idx][0] + random.randint(-10, 10)
            new_y = points[idx][1] + random.randint(-10, 10)
        else:
            # Larger move
            new_x = points[idx][0] + random.randint(-30, 30)
            new_y = points[idx][1] + random.randint(-30, 30)
        
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            return None
            
        new_point = [new_x, new_y]
        # Check distinctness
        for k in range(16):
            if k != idx and points[k] == new_point:
                return None
        return idx, new_point

    def targeted_move(points):
        """Try to create a 4-point line by moving a point onto a line with 3 existing points"""
        n = len(points)
        
        # Find lines with exactly 3 points
        line_points = {}
        for i in range(n):
            xi, yi = points[i]
            for j in range(i + 1, n):
                xj, yj = points[j]
                dx = xj - xi
                dy = yj - yi
                g = math.gcd(abs(dx), abs(dy))
                dx //= g
                dy //= g
                if dx < 0 or (dx == 0 and dy < 0):
                    dx = -dx
                    dy = -dy
                c = dy * xi - dx * yi
                key = (dx, dy, c)
                if key not in line_points:
                    line_points[key] = set()
                line_points[key].add(i)
                line_points[key].add(j)
        
        # For lines with 3 points, try to find a 4th point position
        candidates = []
        for key, pts in line_points.items():
            if len(pts) == 3:
                # Get the 3 points
                pts_list = list(pts)
                p1 = points[pts_list[0]]
                p2 = points[pts_list[1]]
                p3 = points[pts_list[2]]
                
                # The line is determined by any two points
                # Find integer points on this line within bounds
                x1, y1 = p1
                x2, y2 = p2
                dx = x2 - x1
                dy = y2 - y1
                g = math.gcd(abs(dx), abs(dy))
                dx //= g
                dy //= g
                
                # Parametrize: (x1 + t*dx, y1 + t*dy) for integer t
                # Find t values that give integer points within bounds
                # and are not already occupied
                occupied = set((p[0], p[1]) for p in points)
                
                # Try t values in a reasonable range
                for t in range(-20, 21):
                    if t == 0:
                        continue  # Skip p1
                    new_x = x1 + t * dx
                    new_y = y1 + t * dy
                    if abs(new_x) > 1000 or abs(new_y) > 1000:
                        continue
                    if (new_x, new_y) in occupied:
                        continue
                    # Check if this point is on the line with p3
                    # It should be since we're on the same line
                    # Add to candidates
                    candidates.append((new_x, new_y))
        
        if not candidates:
            return None
        
        # Pick a random candidate
        new_x, new_y = random.choice(candidates)
        new_point = [new_x, new_y]
        
        # Pick a random point to move
        idx = random.randint(0, 15)
        
        # Check distinctness
        for k in range(16):
            if k != idx and points[k] == new_point:
                return None
        
        return idx, new_point

    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() - start_time < time_limit:
        # Choose move type
        if random.random() < 0.4:
            move = targeted_move(current_points)
        else:
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
            # Kick: structured perturbation
            # Move 2-3 points to random locations
            num_kicks = random.randint(2, 3)
            for _ in range(num_kicks):
                idx = random.randint(0, 15)
                new_x = random.randint(-50, 50)
                new_y = random.randint(-50, 50)
                if abs(new_x) <= 1000 and abs(new_y) <= 1000:
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
