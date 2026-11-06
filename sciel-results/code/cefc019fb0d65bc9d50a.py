import time
import random
from collections import defaultdict
from math import gcd

def get_line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = gcd(abs(dx), abs(dy))
    if g > 0:
        dx //= g
        dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def compute_line_counts(points):
    counts = defaultdict(int)
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            key = get_line_key(points[i], points[j])
            if key:
                counts[key] += 1
    return counts

def evaluate_score(line_counts):
    four_count = 0
    five_or_more = 0
    for key, pair_count in line_counts.items():
        if pair_count == 6:
            four_count += 1
        elif pair_count >= 10:
            five_or_more += 1
    return 1000 * four_count - 10000 * five_or_more

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    initial_points = [[x, y] for x in range(4) for y in range(4)]
    
    best_points = [p[:] for p in initial_points]
    best_score = evaluate_score(compute_line_counts(best_points))
    
    random.seed(42)
    
    # Define a function to run a local search from a start
    def local_search(start_pts, time_limit):
        current_points = [p[:] for p in start_pts]
        if len(set(tuple(p) for p in current_points)) < 16:
            return None, -1000000
        for p in current_points:
            if abs(p[0]) > 1000 or abs(p[1]) > 1000:
                return None, -1000000
                
        current_line_counts = compute_line_counts(current_points)
        current_score = evaluate_score(current_line_counts)
        
        # Greedy Phase
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for idx in range(16):
                old_point = current_points[idx][:]
                # Try moving to nearby points
                # We limit the neighborhood to keep it fast
                # A 21x21 box around the current point
                best_move = None
                best_move_score = current_score
                for dx in range(-10, 11):
                    for dy in range(-10, 11):
                        if dx == 0 and dy == 0:
                            continue
                        nx = old_point[0] + dx
                        ny = old_point[1] + dy
                        if abs(nx) > 1000 or abs(ny) > 1000:
                            continue
                        new_point = [nx, ny]
                        
                        # Check dup
                        dup = False
                        for i in range(16):
                            if i != idx and current_points[i][0] == nx and current_points[i][1] == ny:
                                dup = True
                                break
                        if dup:
                            continue
                        
                        # Calculate score change
                        # This is expensive if done naively. 
                        # We need incremental update.
                        
                        # Remove old
                        for j in range(16):
                            if j == idx: continue
                            key = get_line_key(old_point, current_points[j])
                            if key:
                                current_line_counts[key] -= 1
                                if current_line_counts[key] == 0: del current_line_counts[key]
                        # Add new
                        for j in range(16):
                            if j == idx: continue
                            key = get_line_key(new_point, current_points[j])
                            if key:
                                current_line_counts[key] = current_line_counts.get(key, 0) + 1
                                
                        new_score = evaluate_score(current_line_counts)
                        
                        # Revert for next iteration
                        for j in range(16):
                            if j == idx: continue
                            key = get_line_key(new_point, current_points[j])
                            if key:
                                current_line_counts[key] -= 1
                                if current_line_counts[key] == 0: del current_line_counts[key]
                        for j in range(16):
                            if j == idx: continue
                            key = get_line_key(old_point, current_points[j])
                            if key:
                                current_line_counts[key] = current_line_counts.get(key, 0) + 1
                                
                        if new_score > best_move_score:
                            best_move_score = new_score
                            best_move = new_point
                            
                if best_move:
                    # Apply best move
                    current_points[idx] = best_move
                    # Update counts permanently
                    for j in range(16):
                        if j == idx: continue
                        key = get_line_key(old_point, current_points[j])
                        if key:
                            current_line_counts[key] -= 1
                            if current_line_counts[key] == 0: del current_line_counts[key]
                    for j in range(16):
                        if j == idx: continue
                        key = get_line_key(best_move, current_points[j])
                        if key:
                            current_line_counts[key] = current_line_counts.get(key, 0) + 1
                            
                    current_score = best_move_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_points = [p[:] for p in current_points]
                        
        # SA Phase
        T = 100.0
        while T > 1.0 and time.time() < time_limit:
            idx = random.randint(0, 15)
            old_point = current_points[idx][:]
            dx = random.randint(-20, 20)
            dy = random.randint(-20, 20)
            nx = old_point[0] + dx
            ny = old_point[1] + dy
            if abs(nx) > 1000 or abs(ny) > 1000:
                T *= 0.99
                continue
            new_point = [nx, ny]
            dup = False
            for i in range(16):
                if i != idx and current_points[i][0] == nx and current_points[i][1] == ny:
                    dup = True
                    break
            if dup:
                T *= 0.99
                continue
                
            for j in range(16):
                if j == idx: continue
                key = get_line_key(old_point, current_points[j])
                if key:
                    current_line_counts[key] -= 1
                    if current_line_counts[key] == 0: del current_line_counts[key]
            for j in range(16):
                if j == idx: continue
                key = get_line_key(new_point, current_points[j])
                if key:
                    current_line_counts[key] = current_line_counts.get(key, 0) + 1
                    
            current_points[idx] = new_point
            new_score = evaluate_score(current_line_counts)
            
            if new_score > current_score or random.random() < pow(2.71828, (new_score - current_score) / T):
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = [p[:] for p in current_points]
            else:
                for j in range(16):
                    if j == idx: continue
                    key = get_line_key(new_point, current_points[j])
                    if key:
                        current_line_counts[key] -= 1
                        if current_line_counts[key] == 0: del current_line_counts[key]
                for j in range(16):
                    if j == idx: continue
                    key = get_line_key(old_point, current_points[j])
                    if key:
                        current_line_counts[key] = current_line_counts.get(key, 0) + 1
                current_points[idx] = old_point
                
            T *= 0.99
            
        return current_points, current_score

    # Run multiple restarts
    start_configs = []
    start_configs.append([p[:] for p in initial_points])
    for _ in range(5):
        pts = [p[:] for p in initial_points]
        idx = random.randint(0, 15)
        pts[idx] = [random.randint(-10, 10), random.randint(-10, 10)]
        start_configs.append(pts)
        
    for start_pts in start_configs:
        if time.time() > deadline:
            break
        pts, score = local_search(start_pts, deadline)
        if pts and score > best_score:
            best_score = score
            best_points = pts

    # Final check
    final_counts = compute_line_counts(best_points)
    for key, count in final_counts.items():
        if count >= 10:
            return [[x, y] for x in range(4) for y in range(4)]
            
    return best_points
