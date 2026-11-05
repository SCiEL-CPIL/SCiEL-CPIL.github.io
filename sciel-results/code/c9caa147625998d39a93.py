import time
import random
from itertools import combinations

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
            
    return sum(1 for pts in lines.values() if len(pts) == 2)

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    best_score = float('inf')
    best_points = None
    
    def try_update(pts):
        nonlocal best_score, best_points
        if len(pts) != 12:
            return False
        if len(set(map(tuple, pts))) != 12:
            return False
        if any(abs(x) > 10**9 or abs(y) > 10**9 for x, y in pts):
            return False
        
        # Check not all collinear
        if len(pts) >= 3:
            x1, y1 = pts[0]
            x2, y2 = pts[1]
            x3, y3 = pts[2]
            if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                # Check if all are collinear
                all_collinear = True
                for i in range(3, 12):
                    xi, yi = pts[i]
                    if (x2 - x1) * (yi - y1) != (y2 - y1) * (xi - x1):
                        all_collinear = False
                        break
                if all_collinear:
                    return False

        score = count_ordinary_lines(pts)
        if score < best_score:
            best_score = score
            best_points = [list(p) for p in pts]
            if score == 6: # Theoretical minimum
                return True
        return False

    # 1. Try specific known good symmetric configurations
    # These are structured to have many 3-point lines
    
    # Config 1: 3x3 Grid + 3 points forming a triangle around it or extending lines
    # Standard 3x3 grid has 8 lines with 3 points (3 rows, 3 cols, 2 diags).
    # Total pairs = 66. Pairs on 3-point lines = 8*3=24. Pairs on 2-point lines?
    # If we add 3 points carefully, we can reduce ordinary lines.
    
    candidates = []
    
    # Candidate 1: 3x3 grid centered at (0,0)
    base_grid = [(x, y) for x in [-1, 0, 1] for y in [-1, 0, 1]]
    # Add 3 points that extend the lines or form new 3-point lines
    # e.g., (2,0), (-2,0), (0,2) -> extends horizontal and vertical
    c1 = base_grid + [(2, 0), (-2, 0), (0, 2)]
    candidates.append(c1)
    
    # Candidate 2: 3x3 grid + (0, -2), (2, 2), (-2, -2)
    c2 = base_grid + [(0, -2), (2, 2), (-2, -2)]
    candidates.append(c2)
    
    # Candidate 3: Hexagonal symmetry
    # Points: (0,0), (1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,1), (1,-1), (-1,-1)
    # This is 9 points. Add 3 more.
    # Let's try a denser cluster.
    
    # Candidate 4: Two parallel lines of 4 points, and 4 points forming a rectangle?
    # Let's rely on local search from good seeds.
    
    for c in candidates:
        if try_update(c):
            return best_points

    # 2. Local Search with Structured Restarts
    
    # We will perform multiple restarts.
    # Each restart starts from a random configuration or a perturbed symmetric one.
    # The local search moves a point to the intersection of two lines formed by other points.
    
    for restart in range(200):
        if time.time() > deadline:
            break
            
        # Generate initial points
        if restart % 3 == 0:
            # Start from a random grid-like structure
            pts = []
            # 3x3 grid with random offset and scale
            scale = random.randint(1, 5)
            ox, oy = random.randint(-10, 10), random.randint(-10, 10)
            for i in range(3):
                for j in range(3):
                    pts.append((ox + i * scale, oy + j * scale))
            # Add 3 random points
            while len(pts) < 12:
                x = random.randint(-20, 20)
                y = random.randint(-20, 20)
                if (x, y) not in pts:
                    pts.append((x, y))
        else:
            # Random start
            pts = []
            while len(pts) < 12:
                x = random.randint(-15, 15)
                y = random.randint(-15, 15)
                if (x, y) not in pts:
                    pts.append((x, y))
        
        current = [list(p) for p in pts]
        current_score = count_ordinary_lines(current)
        
        if try_update(current):
            return best_points
            
        improved = True
        local_stall_count = 0
        
        while improved and time.time() < deadline:
            improved = False
            local_stall_count += 1
            
            # If stalled too long, do a bigger random jump
            if local_stall_count > 10:
                idx = random.randint(0, 11)
                old_x, old_y = current[idx]
                new_x = old_x + random.randint(-10, 10)
                new_y = old_y + random.randint(-10, 10)
                if (new_x, new_y) != (old_x, old_y) and abs(new_x) <= 10**9 and abs(new_y) <= 10**9:
                    collision = False
                    for j in range(12):
                        if j != idx and current[j][0] == new_x and current[j][1] == new_y:
                            collision = True
                            break
                    if not collision:
                        current[idx] = [new_x, new_y]
                        new_score = count_ordinary_lines(current)
                        # Accept worse moves sometimes (simulated annealing style)
                        if new_score < current_score or random.random() < 0.1:
                            current_score = new_score
                            improved = True
                            local_stall_count = 0
                            if try_update(current):
                                return best_points
                        else:
                            current[idx] = [old_x, old_y]
                continue

            # Standard local search: move point to intersection of two lines
            for idx in range(12):
                if time.time() > deadline:
                    break
                    
                old_x, old_y = current[idx]
                other_idx = [i for i in range(12) if i != idx]
                
                # Collect lines from other points
                lines_dict = {}
                for a in other_idx:
                    for b in other_idx:
                        if a >= b: continue
                        x1, y1 = current[a]
                        x2, y2 = current[b]
                        A = y1 - y2
                        B = x2 - x1
                        C = x1 * y2 - x2 * y1
                        if A == 0 and B == 0: continue
                        if A < 0 or (A == 0 and B < 0):
                            A, B, C = -A, -B, -C
                        key = (A, B, C)
                        if key not in lines_dict:
                            lines_dict[key] = set()
                        lines_dict[key].add(a)
                        lines_dict[key].add(b)
                        
                line_keys = list(lines_dict.keys())
                # Prioritize lines with more points? Or just shuffle.
                random.shuffle(line_keys)
                
                # Try moving to intersection of l1 and l2
                # Limit number of pairs to check to keep it fast
                checked_pairs = 0
                max_pairs = 100
                
                for l1_pos in range(len(line_keys)):
                    if time.time() > deadline or checked_pairs >= max_pairs:
                        break
                    l1 = line_keys[l1_pos]
                    A1, B1, C1 = l1
                    # Skip lines that already contain idx? No, idx is removed.
                    # But if l1 is a line with 2 points, moving to its intersection with another
                    # might create a 3-point line.
                    
                    for l2_pos in range(l1_pos + 1, len(line_keys)):
                        if time.time() > deadline or checked_pairs >= max_pairs:
                            break
                        l2 = line_keys[l2_pos]
                        A2, B2, C2 = l2
                        
                        det = A1 * B2 - A2 * B1
                        if det == 0: continue
                        
                        x_num = B1 * C2 - B2 * C1
                        y_num = A2 * C1 - A1 * C2
                        
                        if x_num % det != 0 or y_num % det != 0: continue
                        
                        new_x = x_num // det
                        new_y = y_num // det
                        
                        if abs(new_x) > 10**9 or abs(new_y) > 10**9: continue
                        if (new_x, new_y) == (old_x, old_y): continue
                        
                        # Check collision
                        collision = False
                        for j in range(12):
                            if j != idx and current[j][0] == new_x and current[j][1] == new_y:
                                collision = True
                                break
                        if collision: continue
                        
                        current[idx] = [new_x, new_y]
                        new_score = count_ordinary_lines(current)
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                            local_stall_count = 0
                            if try_update(current):
                                return best_points
                            break
                        else:
                            current[idx] = [old_x, old_y]
                        
                        checked_pairs += 1
                        if checked_pairs >= max_pairs:
                            break
                    
                    if improved:
                        break
                
                if improved:
                    break

    if best_points is None:
        # Fallback: return a valid configuration, even if not optimal
        best_points = [[x, y] for x in range(4) for y in range(3)]
        
    return best_points
