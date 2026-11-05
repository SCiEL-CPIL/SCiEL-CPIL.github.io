import time
import random
import numpy as np
from collections import Counter
from itertools import combinations

def _make_grid(cols, rows, x0=0, y0=0):
    """Generate a cols x rows grid of points starting at (x0, y0)."""
    pts = []
    for r in range(rows):
        for c in range(cols):
            pts.append((x0 + c, y0 + r))
    return pts

def _make_triangular_lattice(n, x0=0, y0=0):
    """Generate n points on a triangular lattice (hexagonal grid subset)."""
    pts = set()
    # Triangular lattice basis: (1,0) and (0,1) for square, but for triangular use (1,0) and (1/2, sqrt(3)/2)
    # For integer coordinates, use a skewed grid: (i, j) -> (i, j) with offset
    # Actually for integer coords, use a "hexagonal" pattern:
    # Row r has points at x = r//2, r//2+1, ... 
    # Simpler: use a standard grid but with alternating row offsets (brick wall pattern)
    r = 0
    while len(pts) < n:
        offset = r % 2
        c = 0
        while len(pts) < n:
            x = x0 + c + offset
            y = y0 + r
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.add((x, y))
            c += 1
        r += 1
    return list(pts)[:n]

def _make_double_triangular(shift_x, shift_y, x0=-5, y0=-5):
    """Create 30 points as two 15-point triangular lattice clusters shifted by (shift_x, shift_y)."""
    pts = set()
    # First cluster
    r = 0
    count = 0
    while count < 15:
        offset = r % 2
        c = 0
        while count < 15:
            x = x0 + c + offset
            y = y0 + r
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.add((x, y))
                count += 1
            c += 1
        r += 1
    # Second cluster shifted
    r = 0
    count = 0
    while count < 15:
        offset = r % 2
        c = 0
        while count < 15:
            x = x0 + shift_x + c + offset
            y = y0 + shift_y + r
            if -30 <= x <= 30 and -30 <= y <= 30 and (x, y) not in pts:
                pts.add((x, y))
                count += 1
            c += 1
        r += 1
    result = list(pts)
    return result if len(result) == 30 else None

def _full_score_and_target(points):
    """Compute full histogram, return (max_count, target_area)."""
    n = len(points)
    counts = Counter()
    # Use numpy for faster determinant calculation if possible, but N=30 is small
    # Pure python with optimized loops is fine
    for i in range(n):
        xi, yi = points[i]
        for j in range(i+1, n):
            xj, yj = points[j]
            for k in range(j+1, n):
                xk, yk = points[k]
                d = abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
                if d > 0:
                    counts[d] += 1
    if not counts:
        return 0, 0
    target_area, max_count = counts.most_common(1)[0]
    return max_count, target_area

def _delta_count_for_target(points, idx, new_pt, target_area):
    """
    Calculate the change in the count of triangles with 'target_area'
    when points[idx] is replaced by new_pt.
    Inlined for speed.
    """
    old_x, old_y = points[idx]
    new_x, new_y = new_pt
    n = len(points)
    
    delta = 0
    
    # Iterate over all pairs of other points
    for j in range(n):
        if j == idx:
            continue
        xj, yj = points[j]
        for k in range(j+1, n):
            if k == idx:
                continue
            xk, yk = points[k]
            
            # Old area: det of (old_pt, j, k)
            d_old = abs((xj - old_x) * (yk - old_y) - (yj - old_y) * (xk - old_x))
            # New area: det of (new_pt, j, k)
            d_new = abs((xj - new_x) * (yk - new_y) - (yj - new_y) * (xk - new_x))
            
            if d_old == target_area:
                delta -= 1
            if d_new == target_area:
                delta += 1
                
    return delta

def _delta_count_for_swap(points, i, j, new_i, new_j, target_area):
    """
    Calculate the change in the count of triangles with 'target_area'
    when points[i] is replaced by new_i and points[j] is replaced by new_j.
    Inlined for speed.
    """
    old_ix, old_iy = points[i]
    old_jx, old_jy = points[j]
    new_ix, new_iy = new_i
    new_jx, new_jy = new_j
    n = len(points)
    
    delta = 0
    
    # Triangles involving both i and j: (i, j, k) for each k != i, j
    for k in range(n):
        if k == i or k == j:
            continue
        xk, yk = points[k]
        
        d_old = abs((old_jx - old_ix) * (yk - old_iy) - (old_jy - old_iy) * (xk - old_ix))
        d_new = abs((new_jx - new_ix) * (yk - new_iy) - (new_jy - new_iy) * (xk - new_ix))
        
        if d_old == target_area:
            delta -= 1
        if d_new == target_area:
            delta += 1
            
    # Triangles involving i but not j, and j but not i
    for k in range(n):
        if k == i or k == j:
            continue
        xk, yk = points[k]
        for l in range(k+1, n):
            if l == i or l == j:
                continue
            xl, yl = points[l]
            
            # Triangle (i, k, l)
            d_old_ikl = abs((xk - old_ix) * (yl - old_iy) - (yk - old_iy) * (xl - old_ix))
            d_new_ikl = abs((xk - new_ix) * (yl - new_iy) - (yk - new_iy) * (xl - new_ix))
            
            if d_old_ikl == target_area:
                delta -= 1
            if d_new_ikl == target_area:
                delta += 1
                
            # Triangle (j, k, l)
            d_old_jkl = abs((xk - old_jx) * (yl - old_jy) - (yk - old_jy) * (xl - old_jx))
            d_new_jkl = abs((xk - new_jx) * (yl - new_jy) - (yk - new_jy) * (xl - new_jx))
            
            if d_old_jkl == target_area:
                delta -= 1
            if d_new_jkl == target_area:
                delta += 1
                
    return delta

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin

    # Try several initial configurations
    best_pts = None
    best_count = 0
    target_area = 0
    
    configs = [
        (6, 5, -2, -2),
        (5, 6, -2, -2),
        (10, 3, -4, -1),
        (3, 10, -1, -4),
        (15, 2, -7, 0),
        (2, 15, 0, -7),
        (1, 30, -14, 0), # Degenerate, will be 0 area, but valid input
        (30, 1, 0, -14), # Degenerate
    ]
    
    for cols, rows, x0, y0 in configs:
        if cols * rows != 30:
            continue
        pts = _make_grid(cols, rows, x0, y0)
        if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts):
            c, t = _full_score_and_target(pts)
            if c > best_count:
                best_count = c
                target_area = t
                best_pts = pts[:]
    
    # Try triangular lattice initializations
    tri_pts = _make_triangular_lattice(30, -5, -5)
    if tri_pts and len(tri_pts) == 30:
        c, t = _full_score_and_target(tri_pts)
        if c > best_count:
            best_count = c
            target_area = t
            best_pts = tri_pts[:]
    
    # Try double triangular lattice with various shifts
    for sx in range(-10, 11, 3):
        for sy in range(-10, 11, 3):
            dbl_pts = _make_double_triangular(sx, sy, -5, -5)
            if dbl_pts:
                c, t = _full_score_and_target(dbl_pts)
                if c > best_count:
                    best_count = c
                    target_area = t
                    best_pts = dbl_pts[:]
                
    if best_pts is None:
        # Fallback random
        best_pts = []
        while len(best_pts) < 30:
            p = (random.randint(-30, 30), random.randint(-30, 30))
            if p not in best_pts:
                best_pts.append(p)
        best_count, target_area = _full_score_and_target(best_pts)

    random.seed(42)
    
    # Simulated Annealing Parameters
    T = 10.0
    cooling = 0.9995
    min_T = 0.01
    
    current_pts = best_pts[:]
    current_count = best_count
    current_target = target_area
    
    # We track the "score" as the count of the current target area.
    # We must be careful: if the global max area changes, our 'current_count' might not reflect the true score.
    # However, if we only accept moves that improve 'current_count' (or accept worse with prob),
    # we are effectively optimizing for that specific area.
    # To ensure we don't miss a better area, we re-evaluate periodically.
    
    last_reval_time = start_time
    steps = 0
    
    while time.time() < deadline:
        # Periodic re-evaluation to update target and check global best
        if time.time() - last_reval_time > 1.5:
            c, t = _full_score_and_target(current_pts)
            if c > best_count:
                best_count = c
                best_pts = current_pts[:]
            current_target = t
            current_count = c
            last_reval_time = time.time()
            T = max(T, 5.0) # Reheat slightly on re-eval to allow exploration
            
        # Propose a move
        # 40% chance small single move, 30% chance large single move, 30% chance swap
        r_move = random.random()
        if r_move < 0.4:
            # Small move
            idx = random.randint(0, 29)
            orig = current_pts[idx]
            radius = 3
            candidates = []
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = orig[0] + dx, orig[1] + dy
                    if -30 <= nx <= 30 and -30 <= ny <= 30:
                        if (nx, ny) not in current_pts:
                            candidates.append((nx, ny))
            if not candidates:
                continue
            new_pt = random.choice(candidates)
            delta = _delta_count_for_target(current_pts, idx, new_pt, current_target)
            
            # Acceptance
            if delta > 0 or random.random() < np.exp(delta / T):
                current_pts[idx] = new_pt
                current_count += delta
                if current_count > best_count:
                    best_count = current_count
                    best_pts = current_pts[:]
        elif r_move < 0.7:
            # Large move (random jump)
            idx = random.randint(0, 29)
            nx = random.randint(-30, 30)
            ny = random.randint(-30, 30)
            if (nx, ny) in current_pts:
                continue
            new_pt = (nx, ny)
            delta = _delta_count_for_target(current_pts, idx, new_pt, current_target)
            
            # Acceptance
            if delta > 0 or random.random() < np.exp(delta / T):
                current_pts[idx] = new_pt
                current_count += delta
                if current_count > best_count:
                    best_count = current_count
                    best_pts = current_pts[:]
        else:
            # Swap move: pick two indices, propose new positions for both
            i, j = random.sample(range(30), 2)
            orig_i = current_pts[i]
            orig_j = current_pts[j]
            
            # Propose small moves for both
            rad = 2
            cands_i = []
            cands_j = []
            for dx in range(-rad, rad+1):
                for dy in range(-rad, rad+1):
                    if dx == 0 and dy == 0: continue
                    nx, ny = orig_i[0]+dx, orig_i[1]+dy
                    if -30 <= nx <= 30 and -30 <= ny <= 30:
                        if (nx, ny) not in current_pts and (nx, ny) != orig_j:
                            cands_i.append((nx, ny))
            for dx in range(-rad, rad+1):
                for dy in range(-rad, rad+1):
                    if dx == 0 and dy == 0: continue
                    nx, ny = orig_j[0]+dx, orig_j[1]+dy
                    if -30 <= nx <= 30 and -30 <= ny <= 30:
                        if (nx, ny) not in current_pts and (nx, ny) != orig_i:
                            cands_j.append((nx, ny))
            
            if not cands_i or not cands_j:
                continue
                
            new_i = random.choice(cands_i)
            new_j = random.choice(cands_j)
            
            # Check if new_i and new_j collide with each other
            if new_i == new_j:
                continue
                
            delta = _delta_count_for_swap(current_pts, i, j, new_i, new_j, current_target)
            
            if delta > 0 or random.random() < np.exp(delta / T):
                current_pts[i] = new_i
                current_pts[j] = new_j
                current_count += delta
                if current_count > best_count:
                    best_count = current_count
                    best_pts = current_pts[:]
        
        T *= cooling
        if T < min_T:
            T = min_T
            # If T is too low, we are stuck in local optimum.
            # Try a random restart from best_pts with some perturbation?
            # Or just increase T slightly.
            T = 5.0
            
        steps += 1

    # Final check
    c, t = _full_score_and_target(best_pts)
    if c > best_count:
        best_count = c
        best_pts = [list(p) for p in best_pts] # Ensure list of lists

    return [list(p) for p in best_pts]
