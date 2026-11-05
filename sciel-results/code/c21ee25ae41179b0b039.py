<<<<<<< SEARCH
import random
import time
from itertools import combinations
import math

def count_ordinary_lines(points):
    n = len(points)
    if n < 3:
        return 0
    
    lines = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            
            if A == 0 and B == 0:
                continue
            
            if A < 0 or (A == 0 and B < 0):
                A, B, C = -A, -B, -C
            
            key = (A, B, C)
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
            
    ordinary_count = 0
    for pts in lines.values():
        if len(pts) == 2:
            ordinary_count += 1
            
    return ordinary_count

def count_ordinary_lines_fast(points):
    """Faster counting using a different approach."""
    n = len(points)
    if n < 3:
        return 0
    
    # Count lines through each pair, track which lines have exactly 2 points
    line_points = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            
            if A == 0 and B == 0:
                continue
            
            if A < 0 or (A == 0 and B < 0):
                A, B, C = -A, -B, -C
            
            key = (A, B, C)
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    
    return sum(1 for pts in line_points.values() if len(pts) == 2)

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    best_score = float('inf')
    best_points = None
    
    def try_update(pts):
        nonlocal best_score, best_points
        if len(pts) != 12:
            return
        if len(set(map(tuple, pts))) != 12:
            return
        if any(abs(x) > 10**9 or abs(y) > 10**9 for x, y in pts):
            return
        score = count_ordinary_lines_fast(pts)
        if score < best_score:
            best_score = score
            best_points = [list(p) for p in pts]
    
    # Generate candidate points in a bounded region
    # Try multiple random constructions with greedy + local search
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Start with a random seed
        pts = []
        while len(pts) < 12:
            x = random.randint(-20, 20)
            y = random.randint(-20, 20)
            if (x, y) not in pts:
                pts.append((x, y))
        
        # Local search: try to improve by moving points to intersections
        current = [list(p) for p in pts]
        current_score = count_ordinary_lines_fast(current)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try moving each point to intersection of two lines
            for idx in range(12):
                if time.time() > deadline:
                    break
                    
                old_x, old_y = current[idx]
                
                # Get all lines through pairs of other points
                other_idx = [i for i in range(12) if i != idx]
                lines_dict = {}
                
                for a_pos in range(len(other_idx)):
                    for b_pos in range(a_pos + 1, len(other_idx)):
                        a = other_idx[a_pos]
                        b = other_idx[b_pos]
                        x1, y1 = current[a]
                        x2, y2 = current[b]
                        
                        A = y1 - y2
                        B = x2 - x1
                        C = x1 * y2 - x2 * y1
                        
                        if A == 0 and B == 0:
                            continue
                        
                        if A < 0 or (A == 0 and B < 0):
                            A, B, C = -A, -B, -C
                        
                        key = (A, B, C)
                        if key not in lines_dict:
                            lines_dict[key] = set()
                        lines_dict[key].add(a)
                        lines_dict[key].add(b)
                
                # Try intersections of pairs of lines
                line_keys = list(lines_dict.keys())
                random.shuffle(line_keys)
                
                for l1_pos in range(min(50, len(line_keys))):
                    if time.time() > deadline:
                        break
                    l1 = line_keys[l1_pos]
                    A1, B1, C1 = l1
                    
                    for l2_pos in range(l1_pos + 1, min(50, len(line_keys))):
                        l2 = line_keys[l2_pos]
                        A2, B2, C2 = l2
                        
                        det = A1 * B2 - A2 * B1
                        if det == 0:
                            continue
                        
                        x_num = B1 * C2 - B2 * C1
                        y_num = A2 * C1 - A1 * C2
                        
                        if x_num % det != 0 or y_num % det != 0:
                            continue
                        
                        new_x = x_num // det
                        new_y = y_num // det
                        
                        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                            continue
                        
                        if (new_x, new_y) == (old_x, old_y):
                            continue
                        
                        # Check not colliding with other points
                        collision = False
                        for j in range(12):
                            if j != idx and current[j][0] == new_x and current[j][1] == new_y:
                                collision = True
                                break
                        if collision:
                            continue
                        
                        # Try this move
                        current[idx] = [new_x, new_y]
                        new_score = count_ordinary_lines_fast(current)
                        
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                            try_update(current)
                            break
                        else:
                            current[idx] = [old_x, old_y]
                    
                    if improved:
                        break
                
                if improved:
                    break
            
            # If no improvement, try random perturbation
            if not improved and random.random() < 0.3:
                idx = random.randint(0, 11)
                old_x, old_y = current[idx]
                new_x = old_x + random.randint(-3, 3)
                new_y = old_y + random.randint(-3, 3)
                
                if (new_x, new_y) != (old_x, old_y) and abs(new_x) <= 10**9 and abs(new_y) <= 10**9:
                    collision = False
                    for j in range(12):
                        if j != idx and current[j][0] == new_x and current[j][1] == new_y:
                            collision = True
                            break
                    if not collision:
                        current[idx] = [new_x, new_y]
                        new_score = count_ordinary_lines_fast(current)
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                            try_update(current)
                        else:
                            current[idx] = [old_x, old_y]
        
        try_update(current)
    
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(3)]
    
    return best_points
=======
import random
import time
from itertools import combinations
import math

def count_ordinary_lines(points):
    """Count lines containing exactly 2 points."""
    n = len(points)
    if n < 3:
        return 0
    
    lines = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            
            if A == 0 and B == 0:
                continue
            
            if A < 0 or (A == 0 and B < 0):
                A, B, C = -A, -B, -C
            
            key = (A, B, C)
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
            
    ordinary_count = 0
    for pts in lines.values():
        if len(pts) == 2:
            ordinary_count += 1
            
    return ordinary_count

def count_ordinary_lines_fast(points):
    """Faster counting using a different approach."""
    n = len(points)
    if n < 3:
        return 0
    
    line_points = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            
            if A == 0 and B == 0:
                continue
            
            if A < 0 or (A == 0 and B < 0):
                A, B, C = -A, -B, -C
            
            key = (A, B, C)
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    
    return sum(1 for pts in line_points.values() if len(pts) == 2)

def get_line_structure(points):
    """Return dict mapping line_key -> set of point indices on that line."""
    n = len(points)
    line_points = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            if A == 0 and B == 0:
                continue
            if A < 0 or (A == 0 and B < 0):
                A, B, C = -A, -B, -C
            key = (A, B, C)
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    return line_points

def line_intersection_int(l1, l2):
    """Compute integer intersection of two lines (A,B,C) and (A2,B2,C2).
    Returns (x,y) if integer, else None."""
    A1, B1, C1 = l1
    A2, B2, C2 = l2
    det = A1 * B2 - A2 * B1
    if det == 0:
        return None
    x_num = B1 * C2 - B2 * C1
    y_num = A2 * C1 - A1 * C2
    if x_num % det != 0 or y_num % det != 0:
        return None
    return (x_num // det, y_num // det)

def try_place_on_line(point_idx, points, line_key, line_points_dict):
    """Try to move point point_idx onto line line_key by placing it at an
    integer intersection with another line through existing points.
    Returns new point position or None."""
    A, B, C = line_key
    
    # For point to be on this line, we need A*x + B*y + C = 0
    # Try intersections with lines through pairs of other points
    other_indices = [i for i in range(len(points)) if i != point_idx]
    
    # Collect unique lines through pairs of other points
    other_lines = {}
    for a in range(len(other_indices)):
        for b in range(a + 1, len(other_indices)):
            i, j = other_indices[a], other_indices[b]
            x1, y1 = points[i]
            x2, y2 = points[j]
            A2 = y1 - y2
            B2 = x2 - x1
            C2 = x1 * y2 - x2 * y1
            if A2 == 0 and B2 == 0:
                continue
            if A2 < 0 or (A2 == 0 and B2 < 0):
                A2, B2, C2 = -A2, -B2, -C2
            key2 = (A2, B2, C2)
            if key2 not in other_lines:
                other_lines[key2] = set()
            other_lines[key2].add(i)
            other_lines[key2].add(j)
    
    # Try intersecting the target line with each other line
    for l2_key in other_lines:
        pt = line_intersection_int(line_key, l2_key)
        if pt is None:
            continue
        new_x, new_y = pt
        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
            continue
        # Check collision
        collision = False
        for j in range(len(points)):
            if j != point_idx and points[j][0] == new_x and points[j][1] == new_y:
                collision = True
                break
        if collision:
            continue
        # Check it's not the same position
        if points[point_idx][0] == new_x and points[point_idx][1] == new_y:
            continue
        return (new_x, new_y)
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    best_score = float('inf')
    best_points = None
    
    def try_update(pts):
        nonlocal best_score, best_points
        if len(pts) != 12:
            return
        if len(set(map(tuple, pts))) != 12:
            return
        if any(abs(x) > 10**9 or abs(y) > 10**9 for x, y in pts):
            return
        score = count_ordinary_lines_fast(pts)
        if score < best_score:
            best_score = score
            best_points = [list(p) for p in pts]
    
    # Strategy: Start with structured configurations, then use line-completion
    # heuristic to eliminate ordinary lines by moving points onto 3+ point lines.
    
    # Initial structured candidates
    initial_configs = []
    
    # 4x3 grid
    initial_configs.append([(x, y) for x in range(4) for y in range(3)])
    
    # 3x4 grid  
    initial_configs.append([(x, y) for x in range(3) for y in range(4)])
    
    # 6x2 grid
    initial_configs.append([(x, y) for x in range(6) for y in range(2)])
    
    # 2x6 grid
    initial_configs.append([(x, y) for x in range(2) for y in range(6)])
    
    # 12 points on a line (will be modified)
    initial_configs.append([(i, 0) for i in range(12)])
    
    # 3x4 with offset
    initial_configs.append([(x, y) for x in range(-1, 2) for y in range(-1, 3)])
    
    # 4x3 with offset
    initial_configs.append([(x, y) for x in range(-1, 3) for y in range(-1, 2)])
    
    # Random structured: points on few lines
    for _ in range(20):
        # Pick 2-4 lines, distribute 12 points among them
        num_lines = random.choice([2, 3, 4])
        pts = []
        for line_idx in range(num_lines):
            # Random line
            if line_idx == 0:
                # Horizontal
                y = random.randint(-5, 5)
                xs = random.sample(range(-10, 10), min(4, 12 - len(pts)))
                pts.extend([(x, y) for x in xs])
            elif line_idx == 1:
                # Vertical
                x = random.randint(-5, 5)
                ys = random.sample(range(-10, 10), min(4, 12 - len(pts)))
                pts.extend([(x, y) for y in ys])
            else:
                # Diagonal-ish
                x0 = random.randint(-5, 5)
                y0 = random.randint(-5, 5)
                slope_num = random.choice([1, 2, 3])
                slope_den = random.choice([1, 2, 3])
                count_needed = 12 - len(pts)
                if count_needed <= 0:
                    break
                for t in range(count_needed):
                    x = x0 + t * slope_den
                    y = y0 + t * slope_num
                    if abs(x) <= 10 and abs(y) <= 10 and (x, y) not in pts:
                        pts.append((x, y))
        
        while len(pts) < 12:
            x = random.randint(-10, 10)
            y = random.randint(-10, 10)
            if (x, y) not in pts:
                pts.append((x, y))
        
        if len(pts) == 12:
            initial_configs.append(pts[:12])
    
    # For each initial config, apply line-completion local search
    for init_pts in initial_configs:
        if time.time() > deadline:
            break
        
        current = [list(p) for p in init_pts]
        current_score = count_ordinary_lines_fast(current)
        try_update(current)
        
        improved = True
        max_iters = 100
        iter_count = 0
        
        while improved and iter_count < max_iters and time.time() < deadline:
            improved = False
            iter_count += 1
            
            # Get current line structure
            line_points_dict = get_line_structure(current)
            
            # Find ordinary lines (lines with exactly 2 points)
            ordinary_lines = [(key, pts) for key, pts in line_points_dict.items() if len(pts) == 2]
            
            # Sort by how "removable" they are - prefer lines where moving one point
            # can join a 3+ line
            for line_key, line_pts in ordinary_lines:
                if time.time() > deadline:
                    break
                
                # For each point on this ordinary line, try to move it to another line
                for pt_idx in list(line_pts):
                    # Find lines with 3+ points that we could move to
                    rich_lines = [(k, p) for k, p in line_points_dict.items() if len(p) >= 3]
                    
                    # Sort rich lines by size (prefer larger)
                    rich_lines.sort(key=lambda x: -len(x[1]))
                    
                    for rich_key, rich_pts in rich_lines[:10]:
                        # Try to place pt_idx on rich_key
                        new_pos = try_place_on_line(pt_idx, current, rich_key, line_points_dict)
                        if new_pos is None:
                            continue
                        
                        old_pos = (current[pt_idx][0], current[pt_idx][1])
                        current[pt_idx] = [new_pos[0], new_pos[1]]
                        new_score = count_ordinary_lines_fast(current)
                        
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                            try_update(current)
                            break
                        else:
                            current[pt_idx] = [old_pos[0], old_pos[1]]
                    
                    if improved:
                        break
                
                if improved:
                    break
            
            # If no improvement via line completion, try random moves
            if not improved:
                # Try moving a point that's on an ordinary line to a random nearby position
                line_points_dict = get_line_structure(current)
                ordinary_lines = [(key, pts) for key, pts in line_points_dict.items() if len(pts) == 2]
                
                if ordinary_lines:
                    # Pick a random ordinary line and one of its points
                    _, pts = random.choice(ordinary_lines)
                    pt_idx = random.choice(list(pts))
                    old_x, old_y = current[pt_idx]
                    
                    # Try several random nearby positions
                    for _ in range(20):
                        new_x = old_x + random.randint(-3, 3)
                        new_y = old_y + random.randint(-3, 3)
                        
                        if (new_x, new_y) == (old_x, old_y):
                            continue
                        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                            continue
                        
                        collision = False
                        for j in range(12):
                            if j != pt_idx and current[j][0] == new_x and current[j][1] == new_y:
                                collision = True
                                break
                        if collision:
                            continue
                        
                        current[pt_idx] = [new_x, new_y]
                        new_score = count_ordinary_lines_fast(current)
                        
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                            try_update(current)
                            break
                        else:
                            current[pt_idx] = [old_x, old_y]
        
        try_update(current)
    
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(3)]
    
    return best_points
>>>>>>> REPLACE
