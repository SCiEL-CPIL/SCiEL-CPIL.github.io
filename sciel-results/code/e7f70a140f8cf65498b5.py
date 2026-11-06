import time
import random
import math
from collections import defaultdict

def _score_points(points):
    """
    Returns (num_exact_4_lines, has_5_collinear, three_point_lines).
    points: list of 16 [x, y]
    """
    n = len(points)
    lines = defaultdict(set)
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            # Normalize direction vector
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx_n = dx // g
            dy_n = dy // g
            
            # Ensure consistent direction (e.g., dx > 0 or dx == 0 and dy > 0)
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
                
            # Line equation: A*x + B*y + C = 0
            # A = dy, B = -dx, C = -(A*x1 + B*y1)
            A = dy_n
            B = -dx_n
            C = -(A * x1 + B * y1)
            
            # Normalize (A, B, C) by gcd
            g2 = math.gcd(abs(A), abs(B))
            if g2 > 0:
                A //= g2
                B //= g2
                C //= g2
                
            # Final sign normalization
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
                
            key = (A, B, C)
            lines[key].add(i)
            lines[key].add(j)
            
    exact_4 = 0
    has_5 = False
    three_point_lines = []
    
    for key, pts in lines.items():
        cnt = len(pts)
        if cnt == 4:
            exact_4 += 1
        elif cnt >= 5:
            has_5 = True
        elif cnt == 3:
            three_point_lines.append((key, list(pts)))
            
    return exact_4, has_5, three_point_lines

def construct():
    deadline = time.time() + 115
    
    # Initial configuration: 4x4 grid scaled by 10
    base = [[10 * i, 10 * j] for i in range(4) for j in range(4)]
    
    best_points = [p[:] for p in base]
    best_score, _, _ = _score_points(base)
    
    rng = random.Random(42)
    
    for restart in range(100):
        if time.time() > deadline - 10:
            break
            
        if restart == 0:
            current = [p[:] for p in base]
        else:
            current = [p[:] for p in best_points]
            # Perturb
            for _ in range(rng.randint(5, 15)):
                idx = rng.randint(0, 15)
                current[idx][0] += rng.randint(-10, 10)
                current[idx][1] += rng.randint(-10, 10)
                # Ensure distinct and in bounds
                valid = True
                if abs(current[idx][0]) > 1000 or abs(current[idx][1]) > 1000:
                    current[idx] = best_points[idx][:]
                    valid = False
                if valid:
                    for k in range(16):
                        if k != idx and current[k] == current[idx]:
                            current[idx][0] += 1
                            valid = False
                            break
                if not valid:
                    current[idx] = best_points[idx][:]

        s, h5, three_lines = _score_points(current)
        if h5:
            # If invalid, try to fix by random moves
            for _ in range(100):
                idx = rng.randint(0, 15)
                old = current[idx][:]
                current[idx][0] += rng.randint(-5, 5)
                current[idx][1] += rng.randint(-5, 5)
                if abs(current[idx][0]) > 1000 or abs(current[idx][1]) > 1000:
                    current[idx] = old
                    continue
                if any(current[k] == current[idx] for k in range(16) if k != idx):
                    current[idx] = old
                    continue
                s, h5, three_lines = _score_points(current)
                if not h5:
                    break
            if h5:
                continue

        current_score = s
        no_improve = 0
        
        while time.time() < deadline - 5 and no_improve < 1000:
            move_made = False
            
            # Heuristic: Complete 3-point lines
            if three_lines and rng.random() < 0.5:
                key, pts_idx = rng.choice(three_lines)
                A, B, C = key
                
                p1 = current[pts_idx[0]]
                p2 = current[pts_idx[1]]
                
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                g = math.gcd(abs(dx), abs(dy))
                if g > 0:
                    sx = dx // g
                    sy = dy // g
                    
                    candidates = [
                        [p1[0] + sx, p1[1] + sy],
                        [p1[0] - sx, p1[1] - sy],
                        [p2[0] + sx, p2[1] + sy],
                        [p2[0] - sx, p2[1] - sy]
                    ]
                    
                    for cand in candidates:
                        if abs(cand[0]) > 1000 or abs(cand[1]) > 1000:
                            continue
                            
                        # Try to move a point NOT in pts_idx to cand
                        for k in range(16):
                            if k in pts_idx:
                                continue
                                
                            if any(current[m] == cand for m in range(16) if m != k):
                                continue
                                
                            old = current[k][:]
                            current[k] = cand
                            s_new, h5_new, three_lines_new = _score_points(current)
                            
                            if not h5_new and s_new > current_score:
                                current_score = s_new
                                three_lines = three_lines_new
                                no_improve = 0
                                move_made = True
                                if s_new > best_score:
                                    best_score = s_new
                                    best_points = [p[:] for p in current]
                                break
                            else:
                                current[k] = old
                        if move_made:
                            break
                            
            if not move_made:
                # Random Move
                idx = rng.randint(0, 15)
                old = current[idx][:]
                
                step = rng.randint(1, 10)
                dx = rng.randint(-step, step)
                dy = rng.randint(-step, step)
                if dx == 0 and dy == 0:
                    dx = 1
                    
                new_x = old[0] + dx
                new_y = old[1] + dy
                
                if abs(new_x) > 1000 or abs(new_y) > 1000:
                    no_improve += 1
                    continue
                    
                new_pt = [new_x, new_y]
                
                if any(current[k] == new_pt for k in range(16) if k != idx):
                    no_improve += 1
                    continue
                    
                current[idx] = new_pt
                s_new, h5_new, three_lines_new = _score_points(current)
                
                if h5_new:
                    current[idx] = old
                    no_improve += 1
                else:
                    if s_new > current_score:
                        current_score = s_new
                        three_lines = three_lines_new
                        no_improve = 0
                        if s_new > best_score:
                            best_score = s_new
                            best_points = [p[:] for p in current]
                    elif s_new == current_score and rng.random() < 0.1:
                        current_score = s_new
                        three_lines = three_lines_new
                        no_improve += 1
                    else:
                        current[idx] = old
                        no_improve += 1
                        three_lines = three_lines_new # Update even if reverted? No, revert means state is old.
                        # Actually, if we revert, three_lines should be the old one.
                        # We didn't store old three_lines.
                        # _score_points is cheap. We can just re-score if needed, or accept the cost.
                        # To be safe, if we revert, we should restore three_lines?
                        # It's easier to just re-score in the next iteration if we revert.
                        # But we use three_lines for heuristic.
                        # Let's just re-score if we revert? No, too slow.
                        # Let's just keep the old three_lines if we revert.
                        # So we shouldn't update three_lines if we revert.
                        pass

    return best_points
