"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
from math import gcd

def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 3:
        return 0
    # Use a set of normalized line representations
    line_points = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            # Normalize direction
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            # Normalize so that dx > 0 or (dx == 0 and dy > 0)
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            # Line is determined by direction and a point on the line
            # Use the point with smaller x (or smaller y if x equal) as anchor
            if x1 < x2 or (x1 == x2 and y1 < y2):
                ax, ay = x1, y1
            else:
                ax, ay = x2, y2
            key = (dx, dy, ax, ay)
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    
    ordinary = 0
    for key, pts in line_points.items():
        if len(pts) == 2:
            ordinary += 1
    return ordinary

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _valid(points):
    """Check that points are valid: 12 distinct integer points, |x|,|y| <= 10^9, not all collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for p in points:
        if len(p) != 2:
            return False
        x, y = p
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    # Check not all collinear
    if len(points) >= 3:
        (x1, y1), (x2, y2), (x3, y3) = points[0], points[1], points[2]
        cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
        if cross == 0:
            # Check if all are collinear
            all_collinear = True
            for i in range(3, len(points)):
                x4, y4 = points[i]
                cross_i = (x2 - x1) * (y4 - y1) - (y2 - y1) * (x4 - x1)
                if cross_i != 0:
                    all_collinear = False
                    break
            if all_collinear:
                return False
    return True

def _line_key(x1, y1, x2, y2):
    """Compute normalized line key for two points."""
    dx = x2 - x1
    dy = y2 - y1
    g = gcd(abs(dx), abs(dy))
    if g == 0:
        return None
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    if x1 < x2 or (x1 == x2 and y1 < y2):
        ax, ay = x1, y1
    else:
        ax, ay = x2, y2
    return (dx, dy, ax, ay)

def _build_line_map(points):
    """Build a map from line key to set of point indices on that line."""
    n = len(points)
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i][0], points[i][1], points[j][0], points[j][1])
            if key is None:
                continue
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    return line_map

def _delta_count_ordinary(points, move_idx, new_pos, line_map):
    """
    Compute the change in ordinary line count when moving point move_idx to new_pos.
    Returns the new ordinary count, or None if the move is invalid.
    """
    x, y = new_pos
    old_x, old_y = points[move_idx]
    
    # Check duplicate
    for i, p in enumerate(points):
        if i != move_idx and p[0] == x and p[1] == y:
            return None
    
    if abs(x) > 10**9 or abs(y) > 10**9:
        return None
    
    # Count current ordinary lines
    current_ordinary = sum(1 for pts in line_map.values() if len(pts) == 2)
    
    # Find all lines that contain the old point
    lines_with_old = []
    for key, pts in line_map.items():
        if move_idx in pts:
            lines_with_old.append((key, pts))
    
    # Remove old point from all lines, update ordinary count
    delta = 0
    for key, pts in lines_with_old:
        if len(pts) == 2:
            delta -= 1  # was ordinary, now becomes a single point (not a line)
        # After removal, if len(pts) was 3, it becomes 2 (new ordinary)
        if len(pts) == 3:
            delta += 1
    
    # Now add new point to lines
    # Find all lines formed by new point with each other point
    new_lines = {}
    for j in range(12):
        if j == move_idx:
            continue
        key = _line_key(x, y, points[j][0], points[j][1])
        if key is None:
            continue
        if key not in new_lines:
            new_lines[key] = set()
        new_lines[key].add(j)
    
    # For each line that the new point lies on, check if it already existed
    for key, others in new_lines.items():
        # Full set of points on this line would be others + {move_idx}
        full_pts = set(others)
        full_pts.add(move_idx)
        
        if key in line_map:
            old_pts = line_map[key]
            # This line already existed. Check if old point was on it
            if move_idx in old_pts:
                # The line already had the old point, so we're just moving along the line
                # The set of other points stays the same
                pass
            else:
                # The line existed without the old point. Now we add move_idx
                # Old line had len(old_pts) points, now has len(old_pts)+1
                if len(old_pts) == 2:
                    delta -= 1  # was ordinary
                if len(old_pts) == 1:
                    # was a single point, now becomes 2 (ordinary)
                    delta += 1
                # if len was 3+, no change in ordinary status
        else:
            # New line created
            if len(full_pts) == 2:
                delta += 1
    
    new_ordinary = current_ordinary + delta
    return new_ordinary

def _init_3x4_grid():
    """Initialize with a 3x4 grid pattern."""
    points = []
    for i in range(3):
        for j in range(4):
            points.append([i * 3, j * 3])
    return points

def _generate_strategic_candidates(points, i, line_map):
    """
    Generate candidate positions for point i that are likely to reduce ordinary lines.
    Focus on positions that create or extend 3+ point collinearities.
    """
    candidates = set()
    x0, y0 = points[i]
    
    # For each pair of other points, consider positions that would create a 3-point line
    for j in range(12):
        if j == i:
            continue
        for k in range(j + 1, 12):
            if k == i:
                continue
            # Consider extending the line through points j and k
            xj, yj = points[j]
            xk, yk = points[k]
            dx = xk - xj
            dy = yk - yj
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dxn = dx // g
            dyn = dy // g
            
            # Try positions along this line near the existing points
            for t in range(-3, 4):
                nx = xj + t * dxn
                ny = yj + t * dyn
                if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                    # Check not duplicate
                    is_dup = False
                    for m, p in enumerate(points):
                        if m != i and p[0] == nx and p[1] == ny:
                            is_dup = True
                            break
                    if not is_dup:
                        candidates.add((nx, ny))
    
    # Also try small perturbations around current position
    for dx in range(-5, 6):
        for dy in range(-5, 6):
            nx = x0 + dx
            ny = y0 + dy
            if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                is_dup = False
                for m, p in enumerate(points):
                    if m != i and p[0] == nx and p[1] == ny:
                        is_dup = True
                        break
                if not is_dup:
                    candidates.add((nx, ny))
    
    return candidates

def construct():
    """
    Construct 12 integer points with a minimized number of ordinary lines.
    Strategy: Start from a 3x4 grid and use local search with strategic candidates
    and delta counting to minimize ordinary lines.
    """
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit
    
    # Initialize with 3x4 grid
    best_points = _init_3x4_grid()
    best_count = _count_ordinary(best_points)
    
    # Build line map
    line_map = _build_line_map(best_points)
    
    # Local search with strategic candidates
    rng = random.Random(42)
    no_improve = 0
    max_no_improve = 1000
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Pick a random point to move
        i = rng.randint(0, 11)
        
        # Generate strategic candidates
        candidates = _generate_strategic_candidates(best_points, i, line_map)
        
        if not candidates:
            no_improve += 1
            continue
        
        # Shuffle candidates for randomization
        cand_list = list(candidates)
        rng.shuffle(cand_list)
        
        # Try a subset of candidates (limit per iteration for speed)
        max_trials = min(len(cand_list), 50)
        
        found_improvement = False
        for trial in range(max_trials):
            if time.time() >= deadline:
                break
            
            x, y = cand_list[trial]
            
            # Use delta counting
            new_count = _delta_count_ordinary(best_points, i, (x, y), line_map)
            
            if new_count is None:
                continue
            
            if new_count < best_count:
                # Verify with full count
                new_points = [p[:] for p in best_points]
                new_points[i] = [x, y]
                
                if not _valid(new_points):
                    continue
                
                verified_count = _count_ordinary(new_points)
                
                if verified_count < best_count:
                    best_count = verified_count
                    best_points = new_points
                    line_map = _build_line_map(best_points)
                    no_improve = 0
                    found_improvement = True
                    break
        
        if not found_improvement:
            no_improve += 1
    
    # Final polish: try all single-point moves with wider range using delta counting
    while time.time() < deadline:
        improved = False
        for i in range(12):
            if time.time() >= deadline:
                break
            
            x0, y0 = best_points[i]
            # Try a range of offsets
            for dx in range(-20, 21):
                for dy in range(-20, 21):
                    if time.time() >= deadline:
                        break
                    
                    x = x0 + dx
                    y = y0 + dy
                    
                    if abs(x) > 10**9 or abs(y) > 10**9:
                        continue
                    
                    # Check duplicate
                    is_dup = False
                    for j, p in enumerate(best_points):
                        if j != i and p[0] == x and p[1] == y:
                            is_dup = True
                            break
                    if is_dup:
                        continue
                    
                    new_count = _delta_count_ordinary(best_points, i, (x, y), line_map)
                    if new_count is None:
                        continue
                    
                    if new_count < best_count:
                        new_points = [p[:] for p in best_points]
                        new_points[i] = [x, y]
                        
                        if not _valid(new_points):
                            continue
                        
                        verified_count = _count_ordinary(new_points)
                        if verified_count < best_count:
                            best_count = verified_count
                            best_points = new_points
                            line_map = _build_line_map(best_points)
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_points
# EVOLVE-BLOCK-END
