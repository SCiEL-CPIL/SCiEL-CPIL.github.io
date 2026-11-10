# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random
from itertools import combinations

def normalize_line(A, B, C):
    """Normalize a line so first nonzero coefficient is positive and gcd=1."""
    g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
    if g > 0:
        A //= g
        B //= g
        C //= g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def count_incidences(points, lines):
    """Count total incidences exactly using int64 arithmetic."""
    total = 0
    for i in range(len(points)):
        x, y = points[i]
        for j in range(len(lines)):
            A, B, C = lines[j]
            if A * x + B * y + C == 0:
                total += 1
    return total

def build_3dir_grid(U, V, W):
    """Build points at triple intersections and lines for 3-direction grid."""
    points = []
    for u in U:
        for v in V:
            if u + v in W:
                points.append([u, v])
    
    lines = []
    for u in U:
        lines.append((1, 0, -u))
    for v in V:
        lines.append((0, 1, -v))
    for w in W:
        lines.append((1, 1, -w))
    
    return points, lines

def joint_local_search(points, lines, time_limit=60.0):
    """
    Joint local search: perturb both line offsets and point positions.
    Lines are represented as (A, B, C) where we can adjust C (offset).
    Points can move to nearby integer positions.
    """
    start_time = time.time()
    n_pts = len(points)
    n_lines = len(lines)
    
    current_score = count_incidences(points, lines)
    best_score = current_score
    best_points = [p[:] for p in points]
    best_lines = [l[:] for l in lines]
    
    # Convert to mutable lists
    pts = [p[:] for p in points]
    lns = [l[:] for l in lines]
    
    def fast_score(p, l):
        total = 0
        for i in range(n_pts):
            x, y = p[i]
            for j in range(n_lines):
                A, B, C = l[j]
                if A * x + B * y + C == 0:
                    total += 1
        return total
    
    def is_valid_config(p, l):
        # Check duplicate points
        seen = set()
        for pt in p:
            t = (pt[0], pt[1])
            if t in seen:
                return False
            seen.add(t)
        # Check duplicate lines
        norm_seen = set()
        for ln in l:
            A, B, C = int(ln[0]), int(ln[1]), int(ln[2])
            if A == 0 and B == 0:
                return False
            nl = normalize_line(A, B, C)
            if nl in norm_seen:
                return False
            norm_seen.add(nl)
        return True
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        
        # Try perturbing line offsets
        for j in range(n_lines):
            if time.time() - start_time > time_limit:
                break
            A, B, C = lns[j]
            for dC in range(-4, 5):
                if dC == 0:
                    continue
                old_C = lns[j][2]
                lns[j][2] = C + dC
                if not is_valid_config(pts, lns):
                    lns[j][2] = old_C
                    continue
                new_score = fast_score(pts, lns)
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                    if new_score > best_score:
                        best_score = new_score
                        best_points = [p[:] for p in pts]
                        best_lines = [l[:] for l in lns]
                else:
                    lns[j][2] = old_C
        
        # Try perturbing points
        for i in range(n_pts):
            if time.time() - start_time > time_limit:
                break
            x0, y0 = pts[i]
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x0 + dx, y0 + dy
                    # Check duplicate
                    dup = False
                    for k in range(n_pts):
                        if k != i and pts[k][0] == nx and pts[k][1] == ny:
                            dup = True
                            break
                    if dup:
                        continue
                    old_pt = pts[i][:]
                    pts[i] = [nx, ny]
                    new_score = fast_score(pts, lns)
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                        if new_score > best_score:
                            best_score = new_score
                            best_points = [p[:] for p in pts]
                            best_lines = [l[:] for l in lns]
                    else:
                        pts[i] = old_pt
    
    return best_points, best_lines, best_score

def joint_simulated_annealing(points, lines, time_limit=60.0):
    """
    Simulated annealing that perturbs both points and line offsets.
    """
    start_time = time.time()
    n_pts = len(points)
    n_lines = len(lines)
    
    pts = [p[:] for p in points]
    lns = [l[:] for l in lines]
    
    current_score = count_incidences(pts, lns)
    best_score = current_score
    best_points = [p[:] for p in pts]
    best_lines = [l[:] for l in lns]
    
    temp = 5.0
    cooling = 0.998
    
    while time.time() - start_time < time_limit and temp > 0.05:
        temp *= cooling
        
        # Randomly choose: perturb a point or a line offset
        if random.random() < 0.5:
            # Perturb a point
            i = random.randint(0, n_pts - 1)
            x0, y0 = pts[i]
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
            if dx == 0 and dy == 0:
                continue
            nx, ny = x0 + dx, y0 + dy
            dup = False
            for k in range(n_pts):
                if k != i and pts[k][0] == nx and pts[k][1] == ny:
                    dup = True
                    break
            if dup:
                continue
            old_pt = pts[i][:]
            pts[i] = [nx, ny]
            new_score = count_incidences(pts, lns)
            delta = new_score - current_score
            if delta > 0 or random.random() < math.exp(delta / max(temp, 1e-10)):
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = [p[:] for p in pts]
                    best_lines = [l[:] for l in lns]
            else:
                pts[i] = old_pt
        else:
            # Perturb a line offset
            j = random.randint(0, n_lines - 1)
            dC = random.randint(-5, 5)
            if dC == 0:
                continue
            old_C = lns[j][2]
            lns[j][2] += dC
            # Check for duplicate lines
            norm_seen = set()
            valid = True
            for ln in lns:
                A, B, C = int(ln[0]), int(ln[1]), int(ln[2])
                if A == 0 and B == 0:
                    valid = False
                    break
                nl = normalize_line(A, B, C)
                if nl in norm_seen:
                    valid = False
                    break
                norm_seen.add(nl)
            if not valid:
                lns[j][2] = old_C
                continue
            new_score = count_incidences(pts, lns)
            delta = new_score - current_score
            if delta > 0 or random.random() < math.exp(delta / max(temp, 1e-10)):
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = [p[:] for p in pts]
                    best_lines = [l[:] for l in lns]
            else:
                lns[j][2] = old_C
    
    return best_points, best_lines, best_score

def find_best_offsets(U, V, W_candidates):
    """
    For given U and V, find the best set of W values (diagonal offsets)
    that maximizes triple intersections.
    """
    best_W = None
    best_count = 0
    
    # Count how many times each sum appears
    sum_counts = {}
    for u in U:
        for v in V:
            s = u + v
            sum_counts[s] = sum_counts.get(s, 0) + 1
    
    # Sort by count descending, then by value
    sorted_sums = sorted(sum_counts.keys(), key=lambda s: (-sum_counts[s], s))
    
    # Try different subsets of size 9
    # Greedy: take top 9 by count
    W_greedy = sorted(sorted_sums[:9])
    count_greedy = 0
    W_set = set(W_greedy)
    for u in U:
        for v in V:
            if u + v in W_set:
                count_greedy += 1
    
    best_W = W_greedy
    best_count = count_greedy
    
    return best_W, best_count

def szemeredi_trotter_incidences():
    """
    Maximizes point-line incidences for n=54 points and m=27 lines.
    Joint optimization of points and lines using local search and simulated annealing.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    # Phase 1: Build initial 3-direction grid configuration
    U = list(range(1, 10))
    V = list(range(1, 10))
    
    W, _ = find_best_offsets(U, V, None)
    
    points, lines = build_3dir_grid(U, V, W)
    
    # Ensure we have exactly 54 points and 27 lines
    if len(points) > 54:
        points = points[:54]
    elif len(points) < 54:
        # Add extra points at other intersections
        added = set(tuple(p) for p in points)
        for u in range(1, 20):
            for v in range(1, 20):
                if len(points) >= 54:
                    break
                if (u, v) not in added:
                    points.append([u, v])
                    added.add((u, v))
            if len(points) >= 54:
                break
    
    points = points[:54]
    
    initial_score = count_incidences(points, lines)
    best_score = initial_score
    best_points = [p[:] for p in points]
    best_lines = [l[:] for l in lines]
    
    # Phase 2: Joint local search
    if time.time() < deadline - 120:
        ls_pts, ls_lines, ls_score = joint_local_search(points, lines, time_limit=50.0)
        if ls_score > best_score:
            best_score = ls_score
            best_points = ls_pts
            best_lines = ls_lines
    
    # Phase 3: Joint simulated annealing
    if time.time() < deadline - 50:
        sa_pts, sa_lines, sa_score = joint_simulated_annealing(best_points, best_lines, time_limit=60.0)
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_pts
            best_lines = sa_lines
    
    # Phase 4: Another joint local search pass
    if time.time() < deadline - 20:
        ls2_pts, ls2_lines, ls2_score = joint_local_search(best_points, best_lines, time_limit=30.0)
        if ls2_score > best_score:
            best_score = ls2_score
            best_points = ls2_pts
            best_lines = ls2_lines
    
    # Convert to numpy arrays
    points_arr = np.array(best_points, dtype=np.int64)
    lines_arr = np.array(best_lines, dtype=np.int64)
    
    # Validate
    assert points_arr.shape == (54, 2)
    assert lines_arr.shape == (27, 3)
    
    # Check for duplicate points
    unique_pts = set(map(tuple, points_arr.tolist()))
    if len(unique_pts) != 54:
        # Fallback: regenerate from grid
        U = list(range(1, 10))
        V = list(range(1, 10))
        W, _ = find_best_offsets(U, V, None)
        points, lines = build_3dir_grid(U, V, W)
        points_arr = np.array(points[:54], dtype=np.int64)
        lines_arr = np.array(lines, dtype=np.int64)
    
    # Check for duplicate lines
    norm_lines = set()
    for l in lines_arr:
        A, B, C = int(l[0]), int(l[1]), int(l[2])
        if A == 0 and B == 0:
            raise ValueError("Invalid line")
        nl = normalize_line(A, B, C)
        if nl in norm_lines:
            raise ValueError("Duplicate lines")
        norm_lines.add(nl)
    
    return points_arr, lines_arr

# EVOLVE-BLOCK-END
