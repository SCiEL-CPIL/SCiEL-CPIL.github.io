"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

def _normalize_line(p1, p2):
    """Return a canonical representation of the line through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return (1, 0, x1)
    if dy == 0:
        return (0, 1, y1)
    g = math.gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    # Ensure consistent sign: first non-zero component positive
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    # Line: dx*y - dy*x = c
    c = dx * y1 - dy * x1
    return (dx, dy, c)

def _line_counts(points):
    """Calculate the number of points on every line defined by pairs of points."""
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            line = _normalize_line(points[i], points[j])
            line_counts[line] = line_counts.get(line, 0) + 1
    
    # Convert pair counts to point counts
    # C(k, 2) = pairs => k points
    point_counts = {}
    for line, pairs in line_counts.items():
        # Solve k(k-1)/2 = pairs
        # k^2 - k - 2*pairs = 0
        # k = (1 + sqrt(1 + 8*pairs)) / 2
        disc = 1 + 8 * pairs
        sqrt_disc = math.isqrt(disc)
        if sqrt_disc * sqrt_disc == disc and (1 + sqrt_disc) % 2 == 0:
            k = (1 + sqrt_disc) // 2
            point_counts[line] = k
    return point_counts

def _score_and_validity(points):
    """
    Returns (score, is_valid).
    Score: number of lines with exactly 4 points.
    is_valid: False if any line has >= 5 points.
    """
    if len(points) != 16:
        return -1, False
    
    # Check distinctness and bounds
    seen = set()
    for p in points:
        x, y = p
        if not isinstance(x, int) or not isinstance(y, int):
            return -1, False
        if abs(x) > 1000 or abs(y) > 1000:
            return -1, False
        if (x, y) in seen:
            return -1, False
        seen.add((x, y))
        
    pc = _line_counts(points)
    score = 0
    for line, count in pc.items():
        if count >= 5:
            return 0, False
        if count == 4:
            score += 1
    return score, True

def _generate_grid_config(w, h, offset_x=0, offset_y=0):
    """Generate a w x h grid configuration, padded with random points if needed."""
    points = []
    for i in range(w):
        for j in range(h):
            points.append([i + offset_x, j + offset_y])
    
    # If we have more than 16, we need to trim or we are done if exactly 16
    # If less, we add random points from a small neighborhood
    if len(points) > 16:
        # Randomly select 16
        random.shuffle(points)
        points = points[:16]
    elif len(points) < 16:
        # Add random points from a small grid to maintain density
        attempts = 0
        while len(points) < 16 and attempts < 1000:
            x = random.randint(-10, 10)
            y = random.randint(-10, 10)
            if [x, y] not in points:
                points.append([x, y])
            attempts += 1
    return points

def _local_search_optimized(points, max_iters=2000, time_limit=None):
    """
    Optimized local search.
    1. Fix invalid states (5+ collinear) by moving a point on the offending line.
    2. Improve score by trying to complete 3-point lines to 4-point lines.
    3. Random perturbations.
    """
    current = [list(p) for p in points]
    current_score, is_valid = _score_and_validity(current)
    if not is_valid:
        # Try to repair immediately
        current, current_score = _repair_invalid(current)
        if not is_valid:
             # Re-check after repair
             current_score, is_valid = _score_and_validity(current)
             if not is_valid:
                 return current, current_score

    best = [list(p) for p in current]
    best_score = current_score
    
    start_time = time.time()
    
    for it in range(max_iters):
        if time_limit and time.time() > time_limit:
            break
            
        # Determine move strategy
        # 1. If invalid, MUST fix
        if not is_valid:
            idx, new_point = _get_repair_move(current)
            if new_point is None:
                # Try random move if specific repair fails
                idx = random.randint(0, 15)
                new_point = [random.randint(-20, 20), random.randint(-20, 20)]
                # Ensure distinct
                while new_point in current:
                    new_point = [random.randint(-20, 20), random.randint(-20, 20)]
        else:
            # 2. Try to improve score
            move_made = False
            # Strategy A: Complete a 3-line to 4-line
            pc = _line_counts(current)
            lines_3 = [line for line, count in pc.items() if count == 3]
            if lines_3 and random.random() < 0.5:
                target_line = random.choice(lines_3)
                # Find a point NOT on this line that can be moved TO this line
                # We need to find a point on the line? No, we move a point FROM elsewhere TO the line.
                # Actually, moving a point to the line increases its count to 4.
                # But the point moves away from its old lines.
                
                # Identify points on the line
                points_on_line = []
                for i, p in enumerate(current):
                    if _point_on_line(p, target_line):
                        points_on_line.append(i)
                        
                # We need to pick a point NOT in points_on_line to move.
                # And the new position must be on the line.
                # Let's pick a random point not on the line.
                candidates = [i for i in range(16) if i not in points_on_line]
                if candidates:
                    idx = random.choice(candidates)
                    # Find a valid integer point on the line that is not currently occupied
                    # Line: a*x + b*y = c (canonical form from _normalize_line is dx, dy, c => dx*y - dy*x = c)
                    # Wait, _normalize_line returns (dx, dy, c) for dx*y - dy*x = c.
                    a, b, c = target_line
                    # Equation: a*y - b*x = c  => b*x - a*y = -c
                    # Let's just iterate through a small box around the current point or the line segment
                    # Simpler: Pick a random point on the line within bounds
                    # Parametric: x = x0 + t*b, y = y0 + t*a (since normal is (a, -b) ?? No)
                    # Line ax - by + c' = 0?
                    # My canonical form: (dx, dy, c) where dx*y - dy*x = c.
                    # So dy*x - dx*y + c = 0 is not right.
                    # dx*y - dy*x = c.
                    # Let's find one point on the line.
                    # If dx == 0, x = c/dy? No.
                    # If dx != 0, y = (dy*x + c) / dx.
                    
                    # Let's just try random points in a small range around the centroid of the 3 points on the line
                    xs = [current[i][0] for i in points_on_line]
                    ys = [current[i][1] for i in points_on_line]
                    cx = sum(xs) // 3
                    cy = sum(ys) // 3
                    
                    found = False
                    for _ in range(50):
                        # Try points near the centroid
                        nx = cx + random.randint(-10, 10)
                        ny = cy + random.randint(-10, 10)
                        if abs(nx) > 1000 or abs(ny) > 1000: continue
                        if [nx, ny] in current: continue
                        # Check if on line
                        if _point_on_line([nx, ny], target_line):
                            new_point = [nx, ny]
                            # Check if this move is valid (no 5-collinear)
                            current[idx] = new_point
                            s, v = _score_and_validity(current)
                            if v and s > current_score:
                                current_score = s
                                is_valid = v
                                if s > best_score:
                                    best_score = s
                                    best = [list(p) for p in current]
                                move_made = True
                                break
                            else:
                                # Revert
                                current[idx] = best[idx] if False else current[idx] # Need to store old
                                # Actually, I should store old point before modifying
                                pass
                    if move_made:
                        continue
            
            # Strategy B: Random move
            idx = random.randint(0, 15)
            old_point = list(current[idx])
            # Move to a nearby point
            new_point = [old_point[0] + random.randint(-5, 5), old_point[1] + random.randint(-5, 5)]
            # Ensure bounds
            new_point[0] = max(-1000, min(1000, new_point[0]))
            new_point[1] = max(-1000, min(1000, new_point[1]))
            
            if new_point == old_point:
                continue
            
            if new_point in current:
                continue
                
            current[idx] = new_point
            new_score, new_valid = _score_and_validity(current)
            
            if new_valid and new_score > current_score:
                current_score = new_score
                is_valid = new_valid
                if new_score > best_score:
                    best_score = new_score
                    best = [list(p) for p in current]
            else:
                # Revert
                current[idx] = old_point

    return best, best_score

def _repair_invalid(points):
    """Attempt to fix a configuration with >= 5 collinear points."""
    points = [list(p) for p in points]
    for _ in range(100):
        pc = _line_counts(points)
        bad_lines = [line for line, count in pc.items() if count >= 5]
        if not bad_lines:
            score, _ = _score_and_validity(points)
            return points, score
        
        bad_line = random.choice(bad_lines)
        # Pick a point on this line to move
        points_on_line = [i for i in range(16) if _point_on_line(points[i], bad_line)]
        if not points_on_line:
            continue
        idx = random.choice(points_on_line)
        old_point = list(points[idx])
        
        # Move to a random location
        for _ in range(20):
            new_point = [random.randint(-20, 20), random.randint(-20, 20)]
            if new_point in points:
                continue
            points[idx] = new_point
            _, is_valid = _score_and_validity(points)
            if is_valid:
                score, _ = _score_and_validity(points)
                return points, score
            points[idx] = old_point
        
        # If we couldn't fix it with small moves, try a larger jump
        points[idx] = [random.randint(-100, 100), random.randint(-100, 100)]
        _, is_valid = _score_and_validity(points)
        if is_valid:
            score, _ = _score_and_validity(points)
            return points, score
        points[idx] = old_point

    # Failed to repair, return original
    score, _ = _score_and_validity(points)
    return points, score

def _point_on_line(point, line):
    """Check if point is on line defined by canonical tuple (dx, dy, c)."""
    a, b, c = line
    x, y = point
    # Equation: a*y - b*x = c
    return a * y - b * x == c

def _get_repair_move(points):
    """Get a specific move to repair an invalid state."""
    pc = _line_counts(points)
    bad_lines = [line for line, count in pc.items() if count >= 5]
    if not bad_lines:
        return None, None
    bad_line = random.choice(bad_lines)
    points_on_line = [i for i in range(16) if _point_on_line(points[i], bad_line)]
    idx = random.choice(points_on_line)
    # Try to move to a point not on any bad line
    for _ in range(10):
        new_point = [random.randint(-10, 10), random.randint(-10, 10)]
        if new_point in points:
            continue
        # Check if new point creates new 5-collinear
        points[idx] = new_point
        _, is_valid = _score_and_validity(points)
        if is_valid:
            return idx, new_point
        points[idx] = points[idx] # revert
    return None, None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_points = None
    best_score = -1
    
    # Strategy: Try multiple diverse initializations
    # 1. 4x4 Grid
    # 2. 3x5 Grid
    # 3. 5x3 Grid
    # 4. Random points from small grid
    # 5. Perturbed grids
    
    initializers = []
    
    # 4x4
    initializers.append(_generate_grid_config(4, 4, -2, -2))
    # 3x5
    initializers.append(_generate_grid_config(3, 5, -1, -2))
    # 5x3
    initializers.append(_generate_grid_config(5, 3, -2, -1))
    # 2x8
    initializers.append(_generate_grid_config(2, 8, -1, -4))
    # 8x2
    initializers.append(_generate_grid_config(8, 2, -4, -1))
    
    # Random dense
    for _ in range(5):
        pts = []
        while len(pts) < 16:
            p = [random.randint(-10, 10), random.randint(-10, 10)]
            if p not in pts:
                pts.append(p)
        initializers.append(pts)

    for init_pts in initializers:
        if time.time() > deadline:
            break
        
        # Local search
        improved, score = _local_search_optimized(init_pts, max_iters=1000, time_limit=deadline)
        
        if score > best_score:
            best_score = score
            best_points = [list(p) for p in improved]
            
        # If we found a good score, do a deeper polish
        if best_score >= 12:
             polished, pol_score = _local_search_optimized(best_points, max_iters=2000, time_limit=deadline)
             if pol_score > best_score:
                 best_score = pol_score
                 best_points = [list(p) for p in polished]
                 
        # Check time
        if time.time() > deadline - 10:
            break

    # Final validation
    if best_points is None:
        # Fallback
        return _generate_grid_config(4, 4, -2, -2)
        
    final_score, is_valid = _score_and_validity(best_points)
    if not is_valid:
        # Try to repair
        repaired, rep_score = _repair_invalid(best_points)
        if _score_and_validity(repaired)[1]:
            return repaired
        return _generate_grid_config(4, 4, -2, -2)
        
    return best_points

# EVOLVE-BLOCK-END
