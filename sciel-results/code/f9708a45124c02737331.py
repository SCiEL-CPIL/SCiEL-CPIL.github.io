import random
import time
import math
from itertools import combinations

def _count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    line_points = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A, B, C = -A, -B, -C
            g = _gcd3(A, B, C)
            if g > 1:
                A //= g
                B //= g
                C //= g
            key = (A, B, C)
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    
    ordinary = 0
    for key, pts in line_points.items():
        if len(pts) == 2:
            ordinary += 1
    return ordinary

def _gcd3(a, b, c):
    from math import gcd
    g = gcd(abs(a), abs(b))
    g = gcd(g, abs(c))
    return g

def _all_collinear(points):
    if len(points) < 3:
        return False
    (x0, y0), (x1, y1), (x2, y2) = points[0], points[1], points[2]
    cross = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    if cross != 0:
        return False
    for i in range(3, len(points)):
        xi, yi = points[i]
        if (x1 - x0) * (yi - y0) - (y1 - y0) * (xi - x0) != 0:
            return False
    return True

def _line_key(x1, y1, x2, y2):
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    if A == 0 and B == 0:
        return None
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    g = _gcd3(A, B, C)
    if g > 1:
        A //= g
        B //= g
        C //= g
    return (A, B, C)

def _get_line_structure(points):
    """Return dict mapping line_key -> set of point indices on that line."""
    n = len(points)
    line_points = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i][0], points[i][1], points[j][0], points[j][1])
            if key is None:
                continue
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    return line_points

def _count_ordinary_from_structure(line_points):
    count = 0
    for key, pts in line_points.items():
        if len(pts) == 2:
            count += 1
    return count

def _try_move_point_to_line(points, pi, line_key_target, exclude_idx):
    """Try to move point pi onto line line_key_target.
    Return new position or None if not possible with integer coords."""
    A, B, C = line_key_target
    # Point must satisfy A*x + B*y + C = 0
    # We need to find an integer point on this line not already occupied
    # and distinct from all other points
    existing = set()
    for j in range(len(points)):
        if j != pi:
            existing.add((points[j][0], points[j][1]))
    
    # Find a point on the line
    # A*x + B*y + C = 0
    if A == 0 and B == 0:
        return None
    if A == 0:
        # B*y + C = 0 -> y = -C/B
        if B == 0:
            return None
        if C % B != 0:
            return None
        y = -C // B
        # Try various x values
        for dx in range(-20, 21):
            x = dx
            if abs(x) > 10**9 or abs(y) > 10**9:
                continue
            if (x, y) not in existing:
                return [x, y]
        return None
    elif B == 0:
        # A*x + C = 0 -> x = -C/A
        if C % A != 0:
            return None
        x = -C // A
        for dy in range(-20, 21):
            y = dy
            if abs(x) > 10**9 or abs(y) > 10**9:
                continue
            if (x, y) not in existing:
                return [x, y]
        return None
    else:
        # A*x + B*y + C = 0
        # Find one particular solution
        # Extended gcd
        from math import gcd
        g = gcd(abs(A), abs(B))
        if C % g != 0:
            return None
        # A/g * x + B/g * y = -C/g
        Ag = A // g
        Bg = B // g
        Cg = -C // g
        # Find particular solution using extended gcd
        def egcd(a, b):
            if b == 0:
                return (a, 1, 0)
            g, x, y = egcd(b, a % b)
            return (g, y, x - (a // b) * y)
        
        g2, x0, y0 = egcd(Ag, Bg)
        # g2 should be 1
        if g2 != 1:
            return None
        # Particular solution: x = x0 * Cg, y = y0 * Cg
        px = x0 * Cg
        py = y0 * Cg
        # General solution: x = px + Bg*t, y = py - Ag*t
        for t in range(-30, 31):
            x = px + Bg * t
            y = py - Ag * t
            if abs(x) > 10**9 or abs(y) > 10**9:
                continue
            if (x, y) not in existing:
                return [x, y]
        return None

def _local_search_structural(points, time_limit):
    """Local search that moves points onto existing lines to reduce ordinary lines."""
    current = [p[:] for p in points]
    current_score = _count_ordinary_lines(current)
    best = [p[:] for p in current]
    best_score = current_score
    
    n = len(current)
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() - start_time < time_limit and no_improve_count < max_no_improve:
        line_points = _get_line_structure(current)
        current_score = _count_ordinary_from_structure(line_points)
        
        # Find all ordinary lines (size 2)
        ordinary_lines = [key for key, pts in line_points.items() if len(pts) == 2]
        
        if not ordinary_lines:
            break
        
        # For each ordinary line, try to move one of its points onto another line
        improved = False
        random.shuffle(ordinary_lines)
        
        for oline in ordinary_lines:
            if time.time() - start_time > time_limit:
                break
            if no_improve_count >= max_no_improve:
                break
            
            pts_on_oline = line_points[oline]
            for pi in list(pts_on_oline):
                # Try moving point pi onto some other line
                other_lines = [key for key in line_points if key != oline]
                random.shuffle(other_lines)
                
                for target_line in other_lines[:10]:  # Try up to 10 target lines
                    new_pos = _try_move_point_to_line(current, pi, target_line, None)
                    if new_pos is None:
                        continue
                    
                    # Check it's not the same as current position
                    if new_pos[0] == current[pi][0] and new_pos[1] == current[pi][1]:
                        continue
                    
                    # Apply move
                    orig = current[pi][:]
                    current[pi] = new_pos
                    
                    new_score = _count_ordinary_lines(current)
                    
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        no_improve_count = 0
                        if new_score < best_score:
                            best_score = new_score
                            best = [p[:] for p in current]
                    else:
                        current[pi] = orig
                    
                    if improved:
                        break
                
                if improved:
                    break
            
            if improved:
                break
        
        if not improved:
            no_improve_count += 1
    
    return best, best_score

def _generate_grid(rows, cols):
    pts = []
    for i in range(rows):
        for j in range(cols):
            pts.append([j, i])
    return pts

def _generate_pencil():
    """11 points on a line + 1 off."""
    pts = []
    for i in range(11):
        pts.append([i, 0])
    pts.append([0, 1])
    return pts

def _generate_configurations():
    """Generate diverse starting configurations."""
    configs = []
    
    # 3x4 grid
    configs.append(_generate_grid(3, 4))
    configs.append(_generate_grid(4, 3))
    
    # Near-pencil
    configs.append(_generate_pencil())
    
    # 2x6 grid
    configs.append(_generate_grid(2, 6))
    configs.append(_generate_grid(6, 2))
    
    # Random perturbations of grids
    random.seed(123)
    for base_rows, base_cols in [(3,4), (4,3), (2,6)]:
        for _ in range(3):
            base = _generate_grid(base_rows, base_cols)
            for p in base:
                p[0] += random.randint(-5, 5)
                p[1] += random.randint(-5, 5)
            configs.append(base)
    
    # Random configurations
    for _ in range(5):
        pts = []
        for _ in range(12):
            pts.append([random.randint(-10, 10), random.randint(-10, 10)])
        configs.append(pts)
    
    return configs

def construct():
    deadline = time.time() + 130  # Safety margin before 150s kill
    best_points = None
    best_score = float('inf')
    
    random.seed(42)
    configs = _generate_configurations()
    
    for config in configs:
        if time.time() > deadline:
            break
        
        # Validate
        seen = set()
        valid = True
        for p in config:
            tp = tuple(p)
            if tp in seen or abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
                valid = False
                break
            seen.add(tp)
        if not valid or len(seen) != 12:
            continue
        if _all_collinear(config):
            continue
        
        # Run structural local search
        time_left = deadline - time.time()
        if time_left < 3:
            continue
        
        pts, score = _local_search_structural(config, time_limit=time_left * 0.85)
        if score < best_score:
            best_score = score
            best_points = pts
            
        if best_score <= 6:
            break
    
    if best_points is None:
        return _generate_pencil()
        
    return best_points
