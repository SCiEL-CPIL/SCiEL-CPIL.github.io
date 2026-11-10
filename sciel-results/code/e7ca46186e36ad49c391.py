import numpy as np
import time
import random
from math import gcd

def normalize_line(A, B, C):
    if A == 0 and B == 0 and C == 0:
        return (0, 0, 0)
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g > 0:
        A //= g
        B //= g
        C //= g
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def count_incidences(points, lines):
    if len(points) == 0 or len(lines) == 0:
        return 0
    px = points[:, 0]
    py = points[:, 1]
    lA = lines[:, 0]
    lB = lines[:, 1]
    lC = lines[:, 2]
    vals = px[:, None] * lA[None, :] + py[:, None] * lB[None, :] + lC[None, :]
    return int(np.sum(vals == 0))

def find_point_on_line_near(A, B, C, x0, y0, search_radius=5):
    """Find an integer point (x, y) on Ax+By+C=0 close to (x0, y0)."""
    best_pt = None
    best_dist = float('inf')
    
    for dx in range(-search_radius, search_radius + 1):
        for dy in range(-search_radius, search_radius + 1):
            x = x0 + dx
            y = y0 + dy
            if A * x + B * y + C == 0:
                dist = dx * dx + dy * dy
                if dist < best_dist:
                    best_dist = dist
                    best_pt = (x, y)
    return best_pt

def generate_grid_config():
    k = 3
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    rows = []
    for a in range(1, k + 1):
        for b in range(1, k * k + 1):
            rows.append((a, -1, b))
    lines = np.array(rows, dtype=np.int64)
    return points, lines

def hill_climb_targeted(points, lines, max_iters=50000, time_limit=10.0):
    current_score = count_incidences(points, lines)
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 2000 # Allow more random walks
    
    n_pts = len(points)
    n_lines = len(lines)
    
    for i in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        if random.random() < 0.5:
            # Move a point to a nearby point on a random line
            idx = random.randint(0, n_pts - 1)
            line_idx = random.randint(0, n_lines - 1)
            
            A, B, C = lines[line_idx]
            x0, y0 = points[idx]
            
            new_pt = find_point_on_line_near(A, B, C, x0, y0, search_radius=8)
            if new_pt is None:
                # Fallback to random move
                new_pt = (x0 + random.randint(-2, 2), y0 + random.randint(-2, 2))
            
            new_x, new_y = new_pt
            
            # Check collision
            valid = True
            for j in range(n_pts):
                if j == idx: continue
                if points[j][0] == new_x and points[j][1] == new_y:
                    valid = False
                    break
            
            if not valid:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
                continue
                
            old_x, old_y = points[idx]
            points[idx] = [new_x, new_y]
            new_score = count_incidences(points, lines)
            
            if new_score > current_score:
                current_score = new_score
                no_improve_count = 0
            else:
                points[idx] = [old_x, old_y]
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
        else:
            # Move a line to pass through a nearby point
            idx = random.randint(0, n_lines - 1)
            pt_idx = random.randint(0, n_pts - 1)
            
            A, B, C = lines[idx]
            px, py = points[pt_idx]
            
            # We want a new line (A', B', C') close to (A, B, C) passing through (px, py)
            # A'px + B'py + C' = 0
            # Let's perturb A, B and solve for C.
            # Or perturb slope and intercept.
            
            # Simple strategy: Pick a random point (qx, qy) close to (px, py) but not on the line?
            # No, we want the line to pass through px, py.
            # So C' = -A'px - B'py.
            
            # Let's try a few random (A', B') pairs near (A, B).
            found_new_line = False
            for _ in range(5):
                dA = random.randint(-1, 1)
                dB = random.randint(-1, 1)
                new_A = A + dA
                new_B = B + dB
                if new_A == 0 and new_B == 0:
                    continue
                new_C = -new_A * px - new_B * py
                nA, nB, nC = normalize_line(new_A, new_B, new_C)
                
                if nA == 0 and nB == 0:
                    continue
                
                # Check duplicate
                valid = True
                for j in range(n_lines):
                    if j == idx: continue
                    if lines[j][0] == nA and lines[j][1] == nB and lines[j][2] == nC:
                        valid = False
                        break
                
                if valid:
                    lines[idx] = [nA, nB, nC]
                    new_score = count_incidences(points, lines)
                    if new_score > current_score:
                        current_score = new_score
                        no_improve_count = 0
                        found_new_line = True
                        break
                    else:
                        lines[idx] = [A, B, C]
            
            if not found_new_line:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break

    return points, lines, current_score

def szemeredi_trotter_incidences():
    n_pts = 54
    n_lines = 27
    
    best_points = None
    best_lines = None
    best_score = 0
    
    start_time = time.time()
    total_time_limit = 600.0
    safety_margin = 30.0
    
    # 1. Baseline
    pts, lns = generate_grid_config()
    score = count_incidences(pts, lns)
    if score > best_score:
        best_score = score
        best_points = pts.copy()
        best_lines = lns.copy()
        
    # 2. Targeted Hill Climbing
    # Run for a long time on the best solution
    # We can also do random restarts, but targeted search on a good seed is powerful.
    
    # Let's do a few random restarts first to find better seeds
    for r in range(5):
        if time.time() - start_time > total_time_limit - safety_margin:
            break
        
        # Random config
        pts = np.random.randint(-10, 10, (n_pts, 2), dtype=np.int64)
        # Ensure unique points
        unique_pts = []
        seen = set()
        for p in pts:
            key = (p[0], p[1])
            if key not in seen:
                seen.add(key)
                unique_pts.append(p)
        while len(unique_pts) < n_pts:
            p = (np.random.randint(-10, 10), np.random.randint(-10, 10))
            if p not in seen:
                seen.add(p)
                unique_pts.append(p)
        pts = np.array(unique_pts, dtype=np.int64)
        
        lns = np.random.randint(-5, 5, (n_lines, 3), dtype=np.int64)
        # Normalize and ensure unique
        unique_lns = []
        seen_l = set()
        for l in lns:
            nA, nB, nC = normalize_line(int(l[0]), int(l[1]), int(l[2]))
            if nA == 0 and nB == 0:
                continue
            if (nA, nB, nC) not in seen_l:
                seen_l.add((nA, nB, nC))
                unique_lns.append((nA, nB, nC))
        while len(unique_lns) < n_lines:
            A, B, C = np.random.randint(-5, 5, 3)
            nA, nB, nC = normalize_line(int(A), int(B), int(C))
            if nA == 0 and nB == 0:
                continue
            if (nA, nB, nC) not in seen_l:
                seen_l.add((nA, nB, nC))
                unique_lns.append((nA, nB, nC))
        lns = np.array(unique_lns, dtype=np.int64)
        
        score = count_incidences(pts, lns)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            best_lines = lns.copy()
            
    # 3. Polish the best
    if time.time() - start_time < total_time_limit - safety_margin:
        remaining = total_time_limit - safety_margin - (time.time() - start_time)
        if remaining > 10.0:
            best_points, best_lines, best_score = hill_climb_targeted(best_points, best_lines, max_iters=1000000, time_limit=remaining)
            
    return best_points, best_lines
