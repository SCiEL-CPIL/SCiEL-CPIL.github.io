import random
import time
import math
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    def score_fast(points):
        """Count lines with exactly 4 points using vectorized approach."""
        n = len(points)
        pts = np.array(points, dtype=np.float64)
        line_counts = {}
        
        for i in range(n):
            for j in range(i + 1, n):
                dx = pts[j, 0] - pts[i, 0]
                dy = pts[j, 1] - pts[i, 1]
                if dx == 0 and dy == 0:
                    continue
                g = math.gcd(int(abs(dx)), int(abs(dy)))
                dx_r = int(dx // g)
                dy_r = int(dy // g)
                # Normalize direction
                if dx_r < 0 or (dx_r == 0 and dy_r < 0):
                    dx_r = -dx_r
                    dy_r = -dy_r
                # Line key: (a, b, c) where ax + by = c
                a = dy_r
                b = -dx_r
                c = a * int(pts[i, 0]) + b * int(pts[i, 1])
                # Normalize sign
                if a < 0 or (a == 0 and b < 0):
                    a = -a
                    b = -b
                    c = -c
                key = (a, b, c)
                line_counts[key] = line_counts.get(key, 0) + 1
        
        score_val = 0
        invalid = False
        for count in line_counts.values():
            # k(k-1)/2 = count => k = (1+sqrt(1+8*count))/2
            disc = 1 + 8 * count
            sqrt_disc = math.isqrt(disc)
            if sqrt_disc * sqrt_disc == disc:
                k = (1 + sqrt_disc) // 2
                if k * (k - 1) // 2 == count:
                    if k == 4:
                        score_val += 1
                    elif k >= 5:
                        invalid = True
                        break
        return -1 if invalid else score_val
    
    def get_lines_with_counts(points):
        """Return dict of line_key -> list of point indices on that line."""
        n = len(points)
        line_map = defaultdict(list)
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0 and dy == 0:
                    continue
                g = math.gcd(abs(dx), abs(dy))
                dx_r = dx // g
                dy_r = dy // g
                if dx_r < 0 or (dx_r == 0 and dy_r < 0):
                    dx_r = -dx_r
                    dy_r = -dy_r
                a = dy_r
                b = -dx_r
                c = a * x1 + b * y1
                if a < 0 or (a == 0 and b < 0):
                    a = -a
                    b = -b
                    c = -c
                key = (a, b, c)
                if i not in line_map[key]:
                    line_map[key].append(i)
                if j not in line_map[key]:
                    line_map[key].append(j)
        return line_map
    
    def find_3point_lines(points):
        """Find lines that have exactly 3 points - candidates for extension to 4."""
        line_map = get_lines_with_counts(points)
        candidates = []
        for key, idxs in line_map.items():
            if len(idxs) == 3:
                candidates.append((key, idxs))
        return candidates
    
    def try_extend_line(points, line_key, idxs):
        """Try to find a 4th point on a 3-point line within bounds."""
        a, b, c = line_key
        # Line: ax + by = c
        # Need to find integer (x,y) with |x|,|y|<=1000, ax+by=c, not already in points
        existing = set(tuple(p) for p in points)
        
        if a == 0:
            # by = c => y = c/b
            if c % b != 0:
                return None
            y = c // b
            if abs(y) > 1000:
                return None
            # x can be anything, try nearby values
            xs = set()
            for idx in idxs:
                xs.add(points[idx][0])
            # Try x values near existing points
            for x in range(-1000, 1001):
                if (x, y) not in existing:
                    return [x, y]
            return None
        elif b == 0:
            # ax = c => x = c/a
            if c % a != 0:
                return None
            x = c // a
            if abs(x) > 1000:
                return None
            ys = set()
            for idx in idxs:
                ys.add(points[idx][1])
            for y in range(-1000, 1001):
                if (x, y) not in existing:
                    return [x, y]
            return None
        else:
            # ax + by = c
            # Find one solution, then parametrize
            # gcd(a,b) must divide c
            g_ab = math.gcd(abs(a), abs(b))
            if c % g_ab != 0:
                return None
            
            # Find particular solution
            # Use extended Euclidean
            def egcd(a, b):
                if b == 0:
                    return (a, 1, 0)
                g, x1, y1 = egcd(b, a % b)
                return (g, y1, x1 - (a // b) * y1)
            
            g, x0, y0 = egcd(a, b)
            # Particular solution: (x0 * c/g, y0 * c/g)
            xp = x0 * (c // g)
            yp = y0 * (c // g)
            
            # General solution: x = xp + (b/g)*t, y = yp - (a/g)*t
            bx = b // g
            ay = -a // g
            
            existing = set(tuple(p) for p in points)
            # Try t values to get into bounds
            for t in range(-2000, 2001):
                x = xp + bx * t
                y = yp + ay * t
                if abs(x) <= 1000 and abs(y) <= 1000 and (x, y) not in existing:
                    return [x, y]
            return None
    
    def systematic_local_search(points, max_iters=500):
        """Try targeted moves to create 4-point lines."""
        current = [p[:] for p in points]
        current_score = score_fast(current)
        best = [p[:] for p in current]
        best_score = current_score
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Find 3-point lines
            three_lines = find_3point_lines(current)
            
            moved = False
            for line_key, idxs in three_lines:
                if time.time() > deadline - 10:
                    break
                # Try to extend this line by moving one of the 3 points or finding a 4th
                # Option 1: Find a new point on this line
                new_pt = try_extend_line(current, line_key, idxs)
                if new_pt is not None:
                    # Try replacing one of the existing points with new_pt
                    for i in range(16):
                        if i in idxs:
                            continue
                        new_points = [p[:] for p in current]
                        new_points[i] = new_pt[:]
                        # Check distinctness
                        if len(set(tuple(p) for p in new_points)) != 16:
                            continue
                        s = score_fast(new_points)
                        if s > current_score:
                            current = new_points
                            current_score = s
                            if s > best_score:
                                best_score = s
                                best = [p[:] for p in current]
                            moved = True
                            break
                if moved:
                    break
            
            # Also try small random shifts
            if not moved:
                i = random.randint(0, 15)
                dx = random.randint(-3, 3)
                dy = random.randint(-3, 3)
                if dx == 0 and dy == 0:
                    continue
                nx, ny = current[i][0] + dx, current[i][1] + dy
                if abs(nx) > 1000 or abs(ny) > 1000:
                    continue
                new_points = [p[:] for p in current]
                new_points[i] = [nx, ny]
                if len(set(tuple(p) for p in new_points)) != 16:
                    continue
                s = score_fast(new_points)
                if s > current_score:
                    current = new_points
                    current_score = s
                    if s > best_score:
                        best_score = s
                        best = [p[:] for p in current]
        
        return best, best_score
    
    # Initial configurations to try
    def make_grid_config(offset=0, scale=1):
        return [[(x * scale) + offset, (y * scale) + offset] for x in range(4) for y in range(4)]
    
    def make_diagonal_config():
        # 4x4 grid but with some diagonal structure
        pts = []
        for x in range(4):
            for y in range(4):
                pts.append([x, y])
        return pts
    
    # Try multiple initial configurations
    best_points = None
    best_score = -1
    
    initial_configs = [
        make_grid_config(0, 1),      # Standard 4x4 grid
        make_grid_config(-1, 1),     # Shifted grid
        make_grid_config(0, 2),      # Scaled grid (may have issues with bounds)
    ]
    
    # Add a non-grid configuration
    non_grid = [
        [0, 0], [1, 0], [2, 0], [3, 0],
        [0, 1], [1, 1], [2, 1], [3, 1],
        [0, 2], [1, 2], [2, 2], [3, 2],
        [0, 3], [1, 3], [2, 3], [3, 3]
    ]
    initial_configs.append(non_grid)
    
    # Try a configuration with more diagonal lines
    diag_config = []
    for i in range(4):
        for j in range(4):
            diag_config.append([i, j])
    # Perturb some points to create diagonal 4-point lines
    initial_configs.append(diag_config)
    
    for config in initial_configs:
        if time.time() > deadline - 30:
            break
        # Verify bounds and distinctness
        if len(set(tuple(p) for p in config)) != 16:
            continue
        if any(abs(p[0]) > 1000 or abs(p[1]) > 1000 for p in config):
            continue
        
        s = score_fast(config)
        if s > best_score:
            best_score = s
            best_points = [p[:] for p in config]
    
    if best_points is None:
        best_points = make_grid_config(0, 1)
        best_score = score_fast(best_points)
    
    # Run systematic local search from best initial
    best_points, best_score = systematic_local_search(best_points, max_iters=300)
    
    # Now run simulated annealing for further improvement
    current = [p[:] for p in best_points]
    current_score = best_score
    
    T = 1.0
    T_min = 0.05
    alpha = 0.9995
    no_improve = 0
    restart_threshold = 1500
    
    while time.time() < deadline - 5:
        # Propose move
        move_type = random.random()
        new_points = [p[:] for p in current]
        
        if move_type < 0.4:
            # Small shift
            i = random.randint(0, 15)
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            if dx == 0 and dy == 0:
                continue
            nx, ny = current[i][0] + dx, current[i][1] + dy
            if abs(nx) > 1000 or abs(ny) > 1000:
                continue
            new_points[i] = [nx, ny]
        elif move_type < 0.7:
            # Medium shift
            i = random.randint(0, 15)
            dx = random.randint(-8, 8)
            dy = random.randint(-8, 8)
            if dx == 0 and dy == 0:
                continue
            nx, ny = current[i][0] + dx, current[i][1] + dy
            if abs(nx) > 1000 or abs(ny) > 1000:
                continue
            new_points[i] = [nx, ny]
        elif move_type < 0.85:
            # Targeted: try to create a 4-point line
            three_lines = find_3point_lines(current)
            if three_lines:
                line_key, idxs = random.choice(three_lines)
                new_pt = try_extend_line(current, line_key, idxs)
                if new_pt is not None:
                    # Replace a random non-line point
                    non_line_idx = [i for i in range(16) if i not in idxs]
                    if non_line_idx:
                        i = random.choice(non_line_idx)
                        new_points[i] = new_pt[:]
                    else:
                        continue
                else:
                    # Fall back to random shift
                    i = random.randint(0, 15)
                    dx = random.randint(-5, 5)
                    dy = random.randint(-5, 5)
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = current[i][0] + dx, current[i][1] + dy
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    new_points[i] = [nx, ny]
            else:
                # No 3-point lines, do random shift
                i = random.randint(0, 15)
                dx = random.randint(-5, 5)
                dy = random.randint(-5, 5)
                if dx == 0 and dy == 0:
                    continue
                nx, ny = current[i][0] + dx, current[i][1] + dy
                if abs(nx) > 1000 or abs(ny) > 1000:
                    continue
                new_points[i] = [nx, ny]
        else:
            # Swap two points
            i, j = random.sample(range(16), 2)
            new_points[i], new_points[j] = new_points[j][:], new_points[i][:]
        
        # Check distinctness
        if len(set(tuple(p) for p in new_points)) != 16:
            continue
        
        new_score = score_fast(new_points)
        
        if new_score == -1:
            continue
        
        if new_score > current_score:
            current = new_points
            current_score = new_score
            no_improve = 0
            if current_score > best_score:
                best_score = current_score
                best_points = [p[:] for p in current]
        else:
            delta = new_score - current_score
            if delta == 0:
                if random.random() < 0.3:
                    current = new_points
            else:
                prob = math.exp(delta / max(T, 0.01))
                if random.random() < prob:
                    current = new_points
                    current_score = new_score
            no_improve += 1
        
        T = max(T * alpha, T_min)
        
        # Restart if stuck
        if no_improve > restart_threshold:
            # Try a fresh random configuration
            random_pts = []
            seen = set()
            while len(random_pts) < 16:
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                if (x, y) not in seen:
                    seen.add((x, y))
                    random_pts.append([x, y])
            
            rs = score_fast(random_pts)
            if rs > current_score:
                current = random_pts
                current_score = rs
                if rs > best_score:
                    best_score = rs
                    best_points = [p[:] for p in current]
            else:
                # Keep current but reset temperature
                T = 1.0
            no_improve = 0
    
    return best_points

