import time
import random
from math import gcd
from itertools import combinations

def count_exact_four_lines_fast(points):
    """Count lines containing exactly 4 of the 16 points. Returns (four_count, five_or_more_count)."""
    n = len(points)
    if n < 4:
        return 0, 0
    
    line_counts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 1
            if g > 0:
                dx //= g
                dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            c = dy * x1 - dx * y1
            key = (dx, dy, c)
            line_counts[key] = line_counts.get(key, 0) + 1
    
    four_count = 0
    five_or_more = 0
    for pair_count in line_counts.values():
        if pair_count == 6:  # C(4,2) = 6
            four_count += 1
        elif pair_count >= 10:  # C(5,2) = 10
            five_or_more += 1
    
    return four_count, five_or_more

def get_three_point_lines(points):
    """Return a list of lines (dx, dy, c) that contain exactly 3 points."""
    n = len(points)
    line_counts = {}
    line_points = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 1
            if g > 0:
                dx //= g
                dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            c = dy * x1 - dx * y1
            key = (dx, dy, c)
            line_counts[key] = line_counts.get(key, 0) + 1
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
            
    three_lines = []
    for key, count in line_counts.items():
        if count == 3: # C(3,2) = 3
            three_lines.append(key)
    return three_lines

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_points = None
    best_score = 0
    
    random.seed(42)
    
    # Initial configuration: 4x4 grid
    initial_grid = [[x, y] for x in range(4) for y in range(4)]
    s, f = count_exact_four_lines_fast(initial_grid)
    if f == 0:
        best_score = s
        best_points = [p[:] for p in initial_grid]
        
    # Try a few structured starts
    starts = [best_points]
    
    # Sheared grids
    for shear in range(-2, 3):
        pts = []
        for i in range(4):
            for j in range(4):
                x = 4 * i + shear * j
                y = 4 * j
                pts.append([x, y])
        if len(set(tuple(p) for p in pts)) == 16:
            if all(abs(p[0]) <= 1000 and abs(p[1]) <= 1000 for p in pts):
                s, f = count_exact_four_lines_fast(pts)
                if f == 0:
                    starts.append(pts)
                    if s > best_score:
                        best_score = s
                        best_points = [p[:] for p in pts]

    # Simulated Annealing
    for start_idx in range(len(starts)):
        if time.time() > deadline:
            break
            
        current = [p[:] for p in starts[start_idx]]
        current_score = count_exact_four_lines_fast(current)[0]
        
        temperature = 10.0
        cooling = 0.9995
        no_improve = 0
        
        while time.time() < deadline:
            if no_improve > 500:
                # Restart from best
                current = [p[:] for p in best_points]
                current_score = best_score
                temperature = 10.0
                no_improve = 0
                # Perturb
                for _ in range(5):
                    idx = random.randint(0, 15)
                    dx = random.randint(-10, 10)
                    dy = random.randint(-10, 10)
                    new_x = current[idx][0] + dx
                    new_y = current[idx][1] + dy
                    if abs(new_x) <= 1000 and abs(new_y) <= 1000:
                        tmp = current[idx]
                        current[idx] = [new_x, new_y]
                        if len(set(tuple(p) for p in current)) == 16:
                            break
                        else:
                            current[idx] = tmp
                            
            # Generate Move
            move_type = random.random()
            new_pts = [p[:] for p in current]
            
            if move_type < 0.5:
                # Smart Move: Line Completion
                three_lines = get_three_point_lines(current)
                if three_lines:
                    # Pick a random 3-point line
                    dx, dy, c = random.choice(three_lines)
                    # Pick a point to move. Prefer points not on 4-lines? 
                    # For simplicity, pick any point, but avoid moving points that are part of existing 4-lines if possible?
                    # Let's just pick a random point index
                    idx = random.randint(0, 15)
                    
                    # Find a point on the line (dx, dy, c)
                    # ax + by = c => dx*x + dy*y = c ? No, key is (dx, dy, c) where c = dy*x1 - dx*y1
                    # Equation: dy*x - dx*y = c
                    # We need integer solution.
                    # We know 3 points are on it. Let's pick one of them as base.
                    # Find indices of points on this line
                    line_idx = []
                    for i, p in enumerate(current):
                        if dy*p[0] - dx*p[1] == c:
                            line_idx.append(i)
                    
                    if len(line_idx) >= 3:
                        # Pick a base point from the line
                        base_idx = random.choice(line_idx)
                        x0, y0 = current[base_idx]
                        
                        # Generate a new point on the line
                        # x = x0 + dy*t, y = y0 + dx*t
                        # Note: standard param for ax+by=c is x=x0+bt, y=y0-at.
                        # Here a=dy, b=-dx. So x = x0 - dx*t, y = y0 + dy*t.
                        # Let's verify: dy(x0 - dx*t) - dx(y0 + dy*t) = dy*x0 - dx*dy*t - dx*y0 - dx*dy*t = dy*x0 - dx*y0 - 2dx*dy*t.
                        # Wait. c = dy*x0 - dx*y0.
                        # LHS = c - 2dx*dy*t. This is not constant.
                        # Correct param: 
                        # Line: A x + B y = C.
                        # A=dy, B=-dx, C=c.
                        # x = x0 + B*t = x0 - dx*t
                        # y = y0 - A*t = y0 - dy*t
                        # Check: dy(x0 - dx*t) - dx(y0 - dy*t) = dy*x0 - dx*dy*t - dx*y0 + dx*dy*t = dy*x0 - dx*y0 = c. Correct.
                        
                        # Choose t randomly
                        # t=0 gives the base point. We want a different point.
                        # t should be integer.
                        # Pick t in a range that keeps point in bounds.
                        # x = x0 - dx*t, y = y0 - dy*t
                        # |x| <= 1000 => |x0 - dx*t| <= 1000
                        # |y| <= 1000 => |y0 - dy*t| <= 1000
                        
                        min_t = -100
                        max_t = 100
                        t = random.randint(min_t, max_t)
                        if t == 0:
                            t = 1 if t >= 0 else -1
                            
                        new_x = x0 - dx * t
                        new_y = y0 - dy * t
                        
                        if abs(new_x) <= 1000 and abs(new_y) <= 1000:
                            new_pts[idx] = [new_x, new_y]
                            # Check distinctness
                            if len(set(tuple(p) for p in new_pts)) != 16:
                                # If collision, try a few more t's
                                found = False
                                for _ in range(10):
                                    t = random.randint(min_t, max_t)
                                    new_x = x0 - dx * t
                                    new_y = y0 - dy * t
                                    if abs(new_x) <= 1000 and abs(new_y) <= 1000:
                                        new_pts[idx] = [new_x, new_y]
                                        if len(set(tuple(p) for p in new_pts)) == 16:
                                            found = True
                                            break
                                if not found:
                                    new_pts = [p[:] for p in current] # Revert
                                    # Fall through to random move? Or just skip this move
                                    # Let's just skip this move iteration
                                    continue 
                        else:
                            new_pts = [p[:] for p in current]
                            continue
                else:
                    # No 3-point lines, do random move
                    idx = random.randint(0, 15)
                    dx = random.randint(-5, 5)
                    dy = random.randint(-5, 5)
                    new_pts[idx] = [current[idx][0] + dx, current[idx][1] + dy]
                    if abs(new_pts[idx][0]) > 1000 or abs(new_pts[idx][1]) > 1000:
                        continue
                    if len(set(tuple(p) for p in new_pts)) != 16:
                        continue
            else:
                # Random Move
                idx = random.randint(0, 15)
                # Larger jumps to escape local optima
                dx = random.randint(-20, 20)
                dy = random.randint(-20, 20)
                new_pts[idx] = [current[idx][0] + dx, current[idx][1] + dy]
                if abs(new_pts[idx][0]) > 1000 or abs(new_pts[idx][1]) > 1000:
                    continue
                if len(set(tuple(p) for p in new_pts)) != 16:
                    continue
            
            s, f = count_exact_four_lines_fast(new_pts)
            
            if f > 0:
                delta = s - current_score - 100 * f
            else:
                delta = s - current_score
                
            if delta > 0 or random.random() < (2.718281828 ** (delta / max(temperature, 0.1))):
                current = new_pts
                current_score = s
                if s > best_score:
                    best_score = s
                    best_points = [p[:] for p in current]
                    no_improve = 0
                    temperature = 10.0
                else:
                    temperature *= cooling
                    no_improve += 1
            else:
                temperature *= cooling
                no_improve += 1

    # Final validation
    final = [p[:] for p in best_points]
    assert len(set(tuple(p) for p in final)) == 16, "Points not distinct"
    assert all(abs(p[0]) <= 1000 and abs(p[1]) <= 1000 for p in final), "Points out of bounds"
    s, f = count_exact_four_lines_fast(final)
    assert f == 0, f"Five or more collinear: {f}"
    
    return final
