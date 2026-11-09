import time
import random
from math import gcd
from collections import defaultdict

def _count_ordinary_lines(points):
    """Count the number of lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 3:
        return 0
    
    line_map = defaultdict(set)
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            a = dy
            b = -dx
            c = a * x1 + b * y1
            g = gcd(abs(a), abs(b))
            if g > 0:
                a //= g
                b //= g
                c //= g
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
                c = -c
            line_map[(a, b, c)].add(i)
            line_map[(a, b, c)].add(j)
    
    ordinary = 0
    for line, pts in line_map.items():
        if len(pts) == 2:
            ordinary += 1
    
    return ordinary

def _validate(points):
    """Check if points are valid: 12 distinct integer points, |x|,|y| <= 10^9, not all collinear."""
    if len(points) != 12:
        return False
    if len(set(tuple(p) for p in points)) != 12:
        return False
    for p in points:
        if abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
            return False
    # Not all collinear
    for i in range(1, 12):
        dx = points[i][0] - points[0][0]
        dy = points[i][1] - points[0][1]
        for j in range(i + 1, 12):
            dx2 = points[j][0] - points[0][0]
            dy2 = points[j][1] - points[0][1]
            if dx * dy2 != dy * dx2:
                return True
    return False

def _snap_to_line(points, idx):
    """Try to move point idx to make it collinear with two other points, or extend a line."""
    best_pos = None
    best_gain = 0
    
    x, y = points[idx]
    
    # For each pair of other points, try to place idx on that line
    for i in range(12):
        if i == idx:
            continue
        for j in range(i + 1, 12):
            if j == idx:
                continue
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            
            # Try positions along the line: (x1 + k*dx, y1 + k*dy) for small k
            for k in range(-5, 6):
                nx = x1 + k * dx
                ny = y1 + k * dy
                if abs(nx) > 10**9 or abs(ny) > 10**9:
                    continue
                if (nx, ny) == (x, y):
                    continue
                # Check distinctness
                if any((nx, ny) == tuple(points[m]) for m in range(12) if m != idx):
                    continue
                
                # Count how many points would be on this line
                trial = [list(p) for p in points]
                trial[idx] = [nx, ny]
                
                # Count collinearities with this new point
                line_count = 0
                for m in range(12):
                    if m == idx:
                        continue
                    mx, my = points[m]
                    if (mx - nx) * dy == (my - ny) * dx:
                        line_count += 1
                
                if line_count >= 2:
                    gain = line_count - 1  # Extra points beyond the 2 we used
                    if gain > best_gain:
                        best_gain = gain
                        best_pos = [nx, ny]
    
    return best_pos, best_gain

def _greedy_improve(points):
    """Greedily try to reduce ordinary lines by moving points to create collinearities."""
    points = [list(p) for p in points]
    count = _count_ordinary_lines(points)
    
    for iteration in range(50):
        improved = False
        for idx in range(12):
            new_pos, gain = _snap_to_line(points, idx)
            if new_pos is not None and gain > 0:
                trial = [list(p) for p in points]
                trial[idx] = new_pos
                if _validate(trial):
                    new_count = _count_ordinary_lines(trial)
                    if new_count < count:
                        points = trial
                        count = new_count
                        improved = True
        if not improved:
            break
    
    return points, count

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Known good configurations to try
    candidates = []
    
    # Config: 11 on x-axis + 1 at (0,1) - baseline 11 ordinary lines
    candidates.append([[i, 0] for i in range(11)] + [[0, 1]])
    
    # Config: 4x3 grid
    candidates.append([[x, y] for x in range(4) for y in range(3)])
    
    # Config: 3x4 grid  
    candidates.append([[x, y] for x in range(3) for y in range(4)])
    
    # Config: Two lines of 6 each
    candidates.append([[i, 0] for i in range(6)] + [[i, 1] for i in range(6)])
    
    # Config: Cross - 6 on x-axis, 6 on y-axis
    candidates.append([[i, 0] for i in range(6)] + [[0, i] for i in range(1, 7)])
    
    # Config: 10 on x-axis + 2 off
    candidates.append([[i, 0] for i in range(10)] + [[0, 1], [1, 1]])
    
    # Config: 9 on x-axis + 3 off on y=1
    candidates.append([[i, 0] for i in range(9)] + [[0, 1], [1, 1], [2, 1]])
    
    # Config: 8 on x-axis + 4 off on y=1
    candidates.append([[i, 0] for i in range(8)] + [[0, 1], [1, 1], [2, 1], [3, 1]])
    
    # Config: 7 on x-axis + 5 off
    candidates.append([[i, 0] for i in range(7)] + [[0, 1], [1, 1], [2, 1], [3, 1], [4, 1]])
    
    # Config: 6 on x-axis + 6 off on y=1, shifted
    candidates.append([[i, 0] for i in range(6)] + [[i+1, 1] for i in range(6)])
    
    # Config: 6 on x-axis + 6 off on y=2
    candidates.append([[i, 0] for i in range(6)] + [[i, 2] for i in range(6)])
    
    # Config: 3 concurrent lines through origin, 4 points each (minus origin counted once)
    pts = [[0, 0]]
    for i in range(1, 5):
        pts.append([i, 0])
        pts.append([i, i])
        pts.append([i, 2*i])
    candidates.append(pts[:12])
    
    # Config: 2 concurrent lines through origin, 5 points each + 1 more
    pts = [[0, 0]]
    for i in range(1, 6):
        pts.append([i, 0])
        pts.append([i, i])
    pts.append([1, 2])
    candidates.append(pts[:12])
    
    # Config: 3 lines, 4 points each, not concurrent
    # Line 1: y=0, x=0,1,2,3
    # Line 2: y=1, x=0,1,2,3
    # Line 3: y=2, x=0,1,2,3
    candidates.append([[x, y] for x in range(4) for y in range(3)])
    
    # Config: 4 lines, 3 points each
    # Lines: y=0 (x=0,1,2), y=1 (x=0,1,2), y=2 (x=0,1,2), plus one more
    # Actually that's 9 points, need 3 more
    candidates.append([[x, y] for x in range(3) for y in range(4)])
    
    # Find best starting point
    best_points = None
    best_count = float('inf')
    
    for pts in candidates:
        if time.time() > deadline - 10:
            break
        if _validate(pts):
            count = _count_ordinary_lines(pts)
            if count < best_count:
                best_count = count
                best_points = [list(p) for p in pts]
    
    if best_points is None:
        best_points = [[i, 0] for i in range(11)] + [[0, 1]]
        best_count = 11
    
    # Simulated annealing with line snapping
    def _sa_search(start_points, time_budget, seed=42):
        rng = random.Random(seed)
        points = [list(p) for p in start_points]
        count = _count_ordinary_lines(points)
        best_points = [list(p) for p in points]
        best_count = count
        
        t_end = time.time() + time_budget
        temp = 5.0
        cool_rate = 0.95
        
        while time.time() < t_end and temp > 0.1:
            improved = False
            
            # Try to move a random point to create collinearity
            idx = rng.randint(0, 11)
            new_pos, gain = _snap_to_line(points, idx)
            
            if new_pos is not None and gain > 0:
                trial = [list(p) for p in points]
                trial[idx] = new_pos
                if _validate(trial):
                    new_count = _count_ordinary_lines(trial)
                    delta = new_count - count
                    if delta < 0:
                        points = trial
                        count = new_count
                        improved = True
                        if count < best_count:
                            best_count = count
                            best_points = [list(p) for p in points]
                    elif rng.random() < (10 ** (-delta / temp)):
                        points = trial
                        count = new_count
                        improved = True
            else:
                # Random perturbation
                x, y = points[idx]
                dx = rng.randint(-3, 3)
                dy = rng.randint(-3, 3)
                if dx == 0 and dy == 0:
                    dx = 1
                nx, ny = x + dx, y + dy
                if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                    if not any((nx, ny) == tuple(points[j]) for j in range(12) if j != idx):
                        trial = [list(p) for p in points]
                        trial[idx] = [nx, ny]
                        if _validate(trial):
                            new_count = _count_ordinary_lines(trial)
                            delta = new_count - count
                            if delta < 0:
                                points = trial
                                count = new_count
                                improved = True
                                if count < best_count:
                                    best_count = count
                                    best_points = [list(p) for p in points]
                            elif rng.random() < (10 ** (-delta / temp)):
                                points = trial
                                count = new_count
                                improved = True
            
            if not improved:
                temp *= cool_rate
        
        return best_points, best_count
    
    # Run SA from multiple seeds
    if time.time() < deadline - 15:
        seeds = [best_points]
        # Add some varied seeds
        for pts in candidates[:5]:
            if _validate(pts):
                seeds.append([list(p) for p in pts])
        
        num_runs = min(8, max(3, int((deadline - time.time() - 15) / 12)))
        time_per_run = (deadline - time.time() - 15) / max(num_runs, 1)
        
        for run in range(num_runs):
            if time.time() >= deadline - 5:
                break
            
            if run < len(seeds):
                sa_seed = [list(p) for p in seeds[run]]
            else:
                sa_seed = [list(p) for p in best_points]
            
            result, result_count = _sa_search(sa_seed, time_per_run, seed=100 + run)
            
            if result_count < best_count:
                best_count = result_count
                best_points = result
    
    # Final greedy polish
    if time.time() < deadline - 5 and best_count > 6:
        for attempt in range(30):
            if time.time() > deadline - 3:
                break
            improved = False
            for idx in range(12):
                if time.time() > deadline - 3:
                    break
                new_pos, gain = _snap_to_line(best_points, idx)
                if new_pos is not None and gain > 0:
                    trial = [list(p) for p in best_points]
                    trial[idx] = new_pos
                    if _validate(trial):
                        new_count = _count_ordinary_lines(trial)
                        if new_count < best_count:
                            best_points = trial
                            best_count = new_count
                            improved = True
            if not improved:
                break
    
    return best_points
