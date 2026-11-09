"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def _count_ordinary_lines(pts):
    """Count lines containing exactly two of the points."""
    n = len(pts)
    if n < 3:
        return 0
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[j][0] - pts[i][0]
            dy = pts[j][1] - pts[i][1]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                key = (0, 0, pts[i][0])
            elif dy == 0:
                key = (0, 1, pts[i][1])
            else:
                if dy < 0:
                    dx, dy = -dx, -dy
                g = gcd(abs(dx), abs(dy))
                dx //= g
                dy //= g
                c = dy * pts[i][0] - dx * pts[i][1]
                key = (1, dx, dy, c)
            lines[key] = lines.get(key, 0) + 1
    return sum(1 for c in lines.values() if c == 1)

def _evaluate(points):
    """Score: number of ordinary lines (lower is better)."""
    return _count_ordinary_lines(points)

def _not_all_collinear(points):
    n = len(points)
    if n < 3:
        return True
    for j in range(1, n):
        for k in range(j + 1, n):
            x0, y0 = points[0]
            x1, y1 = points[j]
            x2, y2 = points[k]
            if (x1 - x0) * (y2 - y0) != (y1 - y0) * (x2 - x0):
                return True
    return False

def _get_line_key(x1, y1, x2, y2):
    """Get normalized line key from two points."""
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    if dx == 0:
        return (0, 0, x1)
    elif dy == 0:
        return (0, 1, y1)
    else:
        if dy < 0:
            dx, dy = -dx, -dy
        g = gcd(abs(dx), abs(dy))
        dx //= g
        dy //= g
        c = dy * x1 - dx * y1
        return (1, dx, dy, c)

def _compute_line_counts(points):
    """Compute count of points on each line through pairs."""
    n = len(points)
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _get_line_key(points[i][0], points[i][1], points[j][0], points[j][1])
            if key is not None:
                lines[key] = lines.get(key, 0) + 1
    return lines

def _count_ordinary_fast(points, line_counts):
    """Count ordinary lines using precomputed line counts."""
    return sum(1 for c in line_counts.values() if c == 1)

def _try_place_on_lines(points, exclude_idx, rng, max_trials=200):
    """Try to place point at exclude_idx on a line that already has 2+ other points."""
    n = len(points)
    existing = [(points[i][0], points[i][1]) for i in range(n) if i != exclude_idx]
    occupied = set(existing)
    
    # Collect all lines with 2+ existing points
    line_points = {}
    for i in range(len(existing)):
        for j in range(i + 1, len(existing)):
            key = _get_line_key(existing[i][0], existing[i][1], existing[j][0], existing[j][1])
            if key is not None:
                if key not in line_points:
                    line_points[key] = []
                line_points[key].append(existing[i])
                line_points[key].append(existing[j])
    
    # Filter to lines with 2+ points (already have at least 2 by construction)
    candidate_lines = [lp for lp in line_points.values() if len(lp) >= 2]
    
    if not candidate_lines:
        return None
    
    # Shuffle for diversity
    rng.shuffle(candidate_lines)
    
    for line_pts in candidate_lines[:50]:  # Try up to 50 lines
        if len(line_pts) < 2:
            continue
        
        # Get two points on this line to determine direction
        x1, y1 = line_pts[0]
        x2, y2 = line_pts[1]
        
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            continue
        
        # Normalize
        if dx < 0 or (dx == 0 and dy < 0):
            dx, dy = -dx, -dy
        
        g = gcd(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 1
        if g > 0:
            dx_n = dx // g
            dy_n = dy // g
        else:
            dx_n, dy_n = dx, dy
        
        # Try positions along this line
        # Parameterize: (x1 + t*dx_n, y1 + t*dy_n)
        # Existing points have specific t values; try other integer t
        existing_ts = set()
        for px, py in line_pts:
            # Find t such that px = x1 + t*dx_n, py = y1 + t*dy_n
            if dx_n != 0:
                t = (px - x1) // dx_n
                if x1 + t * dx_n == px and y1 + t * dy_n == py:
                    existing_ts.add(t)
            elif dy_n != 0:
                t = (py - y1) // dy_n
                if x1 + t * dx_n == px and y1 + t * dy_n == py:
                    existing_ts.add(t)
        
        # Try t values not in existing_ts
        for t in range(-5, 8):
            if t in existing_ts:
                continue
            new_x = x1 + t * dx_n
            new_y = y1 + t * dy_n
            
            if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                continue
            
            if (new_x, new_y) not in occupied:
                return [new_x, new_y]
    
    return None

def construct():
    start = time.time()
    deadline = start + 135  # safety margin before 150s kill
    
    n = 12
    rng = random.Random(42)
    
    # Start with a known good configuration: 4x3 grid
    best_points = [[x, y] for x in range(4) for y in range(3)]
    best_score = _evaluate(best_points)
    
    # Also try some other initial configurations
    initial_configs = [
        # 4x3 grid
        [[x, y] for x in range(4) for y in range(3)],
        # 3x4 grid
        [[x, y] for x in range(3) for y in range(4)],
        # Hexagonal-ish: points on two parallel lines with offsets
        [[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1],[1,2],[2,2],[0,2],[3,2]],
        # Points forming multiple 3-point lines
        [[0,0],[1,0],[2,0],[0,1],[1,1],[2,1],[0,2],[1,2],[2,2],[3,0],[3,1],[3,2]],
    ]
    
    for config in initial_configs:
        if len(config) != 12:
            continue
        if not _not_all_collinear(config):
            continue
        seen = set()
        valid = True
        for p in config:
            if (p[0], p[1]) in seen:
                valid = False
                break
            seen.add((p[0], p[1]))
        if not valid:
            continue
        score = _evaluate(config)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in config]
    
    # Local search with focus on placing points on existing lines
    for restart in range(30):
        if time.time() > deadline:
            break
        
        if restart == 0:
            current_points = [p[:] for p in best_points]
            current_score = best_score
        else:
            # Perturb: move 2-4 points randomly
            current_points = [p[:] for p in best_points]
            num_perturb = rng.randint(2, 4)
            for _ in range(num_perturb):
                idx = rng.randrange(n)
                current_points[idx] = [rng.randint(-10, 10), rng.randint(-10, 10)]
            
            # Ensure validity
            if not _not_all_collinear(current_points):
                continue
            
            # Check for duplicates
            seen = set()
            valid = True
            for p in current_points:
                if (p[0], p[1]) in seen:
                    valid = False
                    break
                seen.add((p[0], p[1]))
            if not valid:
                continue
            
            current_score = _evaluate(current_points)
            if current_score > best_score + 3:
                continue
        
        # Iterative improvement: try to improve each point one at a time
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            for idx in range(n):
                if time.time() > deadline:
                    break
                
                old_x, old_y = current_points[idx]
                
                # Try to place on an existing line
                new_pos = _try_place_on_lines(current_points, idx, rng)
                
                if new_pos is not None:
                    new_x, new_y = new_pos
                    if new_x != old_x or new_y != old_y:
                        # Check for duplicate
                        is_dup = False
                        for i, (px, py) in enumerate(current_points):
                            if i != idx and px == new_x and py == new_y:
                                is_dup = True
                                break
                        if not is_dup:
                            # Tentatively move
                            current_points[idx] = [new_x, new_y]
                            
                            if _not_all_collinear(current_points):
                                score = _evaluate(current_points)
                                if score < current_score:
                                    current_score = score
                                    improved = True
                                    break
                                else:
                                    current_points[idx] = [old_x, old_y]
                
                # Also try random nearby positions
                for _ in range(20):
                    new_x = old_x + rng.randint(-3, 3)
                    new_y = old_y + rng.randint(-3, 3)
                    
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                    
                    # Check for duplicate
                    is_dup = False
                    for i, (px, py) in enumerate(current_points):
                        if i != idx and px == new_x and py == new_y:
                            is_dup = True
                            break
                    if is_dup:
                        continue
                    
                    # Tentatively move
                    current_points[idx] = [new_x, new_y]
                    
                    if not _not_all_collinear(current_points):
                        current_points[idx] = [old_x, old_y]
                        continue
                    
                    score = _evaluate(current_points)
                    
                    if score < current_score:
                        current_score = score
                        improved = True
                        break
                    else:
                        current_points[idx] = [old_x, old_y]
                
                if improved:
                    break
            
            if current_score < best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
        
        if best_score <= 6:
            break
    
    # Final validation
    assert len(best_points) == 12
    assert all(isinstance(p, list) and len(p) == 2 for p in best_points)
    assert all(abs(p[0]) <= 10**9 and abs(p[1]) <= 10**9 for p in best_points)
    assert _not_all_collinear(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
