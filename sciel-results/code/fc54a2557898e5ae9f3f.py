"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import defaultdict
from itertools import combinations

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
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dx * y1 - dy * x1
    return (dx, dy, c)

def _build_line_structure(points):
    """
    Build line structure from 16 points.
    Returns:
        line_to_pts: dict mapping line -> frozenset of point indices on that line
        pt_to_lines: list of lists, pt_to_lines[i] = list of lines through point i
        line_counts: dict mapping line -> number of points on that line
    """
    n = len(points)
    line_to_pts = defaultdict(set)
    
    for i in range(n):
        for j in range(i + 1, n):
            line = _normalize_line(points[i], points[j])
            line_to_pts[line].add(i)
            line_to_pts[line].add(j)
    
    # Convert to frozensets and compute counts
    line_to_pts = {line: frozenset(pts) for line, pts in line_to_pts.items()}
    line_counts = {line: len(pts) for line, pts in line_to_pts.items()}
    
    # Build point-to-lines mapping
    pt_to_lines = [[] for _ in range(n)]
    for line, pts in line_to_pts.items():
        for p in pts:
            pt_to_lines[p].append(line)
    
    return line_to_pts, pt_to_lines, line_counts

def _score_from_line_counts(line_counts):
    """Count lines with exactly 4 points."""
    score = 0
    for count in line_counts.values():
        if count == 4:
            score += 1
    return score

def _is_valid_from_line_counts(line_counts):
    """Check no line has 5+ points."""
    for count in line_counts.values():
        if count >= 5:
            return False
    return True

def _validate_points(points):
    """Validate points meet all constraints."""
    if len(points) != 16:
        return False
    seen = set()
    for p in points:
        x, y = p
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > 1000 or abs(y) > 1000:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    return True

def _quick_score(points):
    """Fast score calculation."""
    line_counts = {}
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            line = _normalize_line(points[i], points[j])
            if line not in line_counts:
                line_counts[line] = set([i, j])
            else:
                line_counts[line].add(i)
                line_counts[line].add(j)
    
    score = 0
    valid = True
    for line, pts in line_counts.items():
        cnt = len(pts)
        if cnt >= 5:
            valid = False
            break
        if cnt == 4:
            score += 1
    return score, valid

def _delta_score(points, idx, new_pt, line_to_pts, pt_to_lines, line_counts):
    """
    Compute the change in score if we move point idx to new_pt.
    Returns (delta, new_valid) where delta = new_score - old_score.
    This is an approximation; for exactness we'd need full rebuild.
    For now, we'll just do a full rebuild if the delta looks promising.
    """
    # For simplicity and correctness, we'll just return 0 and let the caller
    # do a full evaluation if needed. This is a placeholder for a more
    # sophisticated delta computation.
    return 0, True

def _local_search(points, max_iters, time_limit, rng):
    """
    Local search: move one point at a time to a random nearby position.
    Accept if valid and score doesn't decrease.
    """
    current = [list(p) for p in points]
    current_score, current_valid = _quick_score(current)
    if not current_valid:
        return current, 0
    
    best = [list(p) for p in current]
    best_score = current_score
    
    line_to_pts, pt_to_lines, line_counts = _build_line_structure(current)
    
    start_time = time.time()
    n_no_improve = 0
    
    for it in range(max_iters):
        if time.time() > time_limit:
            break
        
        # Pick a random point to move
        idx = rng.randint(0, 15)
        old_pt = list(current[idx])
        
        # Try several candidate moves
        found_improvement = False
        for attempt in range(5):
            # Random perturbation
            radius = rng.choice([3, 5, 8, 12])
            new_pt = [old_pt[0] + rng.randint(-radius, radius),
                      old_pt[1] + rng.randint(-radius, radius)]
            new_pt[0] = max(-1000, min(1000, new_pt[0]))
            new_pt[1] = max(-1000, min(1000, new_pt[1]))
            
            if new_pt == old_pt:
                continue
            if any(new_pt == p for p in current):
                continue
            
            # Try the move
            current[idx] = new_pt
            new_score, new_valid = _quick_score(current)
            
            if new_valid and new_score > current_score:
                # Accept
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best = [list(p) for p in current]
                # Rebuild line structure
                line_to_pts, pt_to_lines, line_counts = _build_line_structure(current)
                n_no_improve = 0
                found_improvement = True
                break
            else:
                # Reject
                current[idx] = old_pt
        
        if not found_improvement:
            n_no_improve += 1
            if n_no_improve > 200:
                # Try a bigger perturbation
                idx = rng.randint(0, 15)
                old_pt = list(current[idx])
                radius = rng.randint(15, 30)
                new_pt = [old_pt[0] + rng.randint(-radius, radius),
                          old_pt[1] + rng.randint(-radius, radius)]
                new_pt[0] = max(-1000, min(1000, new_pt[0]))
                new_pt[1] = max(-1000, min(1000, new_pt[1]))
                
                if new_pt != old_pt and not any(new_pt == p for p in current):
                    current[idx] = new_pt
                    new_score, new_valid = _quick_score(current)
                    if new_valid:
                        if new_score >= current_score:
                            current_score = new_score
                            if new_score > best_score:
                                best_score = new_score
                                best = [list(p) for p in current]
                            line_to_pts, pt_to_lines, line_counts = _build_line_structure(current)
                            n_no_improve = 0
                    else:
                        current[idx] = old_pt
    
    return best, best_score

def _simulated_annealing(points, max_iters, time_limit, rng):
    """
    Simulated annealing: accept worse moves with probability exp(-delta/T).
    """
    current = [list(p) for p in points]
    current_score, current_valid = _quick_score(current)
    if not current_valid:
        return current, 0
    
    best = [list(p) for p in current]
    best_score = current_score
    
    T = 10.0
    T_min = 0.1
    cooling = 0.995
    
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() > time_limit:
            break
        
        idx = rng.randint(0, 15)
        old_pt = list(current[idx])
        
        radius = rng.choice([3, 5, 8, 15])
        new_pt = [old_pt[0] + rng.randint(-radius, radius),
                  old_pt[1] + rng.randint(-radius, radius)]
        new_pt[0] = max(-1000, min(1000, new_pt[0]))
        new_pt[1] = max(-1000, min(1000, new_pt[1]))
        
        if new_pt == old_pt or any(new_pt == p for p in current):
            continue
        
        current[idx] = new_pt
        new_score, new_valid = _quick_score(current)
        
        if not new_valid:
            current[idx] = old_pt
            continue
        
        delta = new_score - current_score
        if delta >= 0:
            # Accept
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best = [list(p) for p in current]
        else:
            # Maybe accept
            if T > T_min and rng.random() < math.exp(delta / T):
                current_score = new_score
            else:
                current[idx] = old_pt
        
        T *= cooling
    
    return best, best_score

def _generate_initial_points(rng):
    """Generate diverse initial configurations."""
    configs = []
    
    # 4x4 grid
    pts = []
    for i in range(4):
        for j in range(4):
            pts.append([i - 2, j - 2])
    configs.append(pts)
    
    # 4x4 grid with perturbation
    pts = []
    for i in range(4):
        for j in range(4):
            pts.append([i - 2 + rng.randint(-1, 1), j - 2 + rng.randint(-1, 1)])
    # Remove duplicates
    seen = set()
    unique = []
    for p in pts:
        if tuple(p) not in seen:
            seen.add(tuple(p))
            unique.append(p)
    while len(unique) < 16:
        p = [rng.randint(-4, 4), rng.randint(-4, 4)]
        if tuple(p) not in seen:
            seen.add(tuple(p))
            unique.append(p)
    configs.append(unique[:16])
    
    # Random points in a box
    pts = []
    seen = set()
    while len(pts) < 16:
        p = [rng.randint(-6, 6), rng.randint(-6, 6)]
        if tuple(p) not in seen:
            seen.add(tuple(p))
            pts.append(p)
    configs.append(pts)
    
    # Structured: two rows of 4 with offsets
    pts = []
    for j in range(4):
        pts.append([j - 2, 0])
        pts.append([j - 2 + 1, 3])
    configs.append(pts)
    
    # 3x3 grid plus 1
    pts = []
    for i in range(3):
        for j in range(3):
            pts.append([i - 1, j - 1])
    pts.append([4, 4])
    # Add 7 more random points
    seen = set(tuple(p) for p in pts)
    while len(pts) < 16:
        p = [rng.randint(-5, 5), rng.randint(-5, 5)]
        if tuple(p) not in seen:
            seen.add(tuple(p))
            pts.append(p)
    configs.append(pts)
    
    # Dense random
    pts = []
    seen = set()
    while len(pts) < 16:
        p = [rng.randint(-3, 3), rng.randint(-3, 3)]
        if tuple(p) not in seen:
            seen.add(tuple(p))
            pts.append(p)
    configs.append(pts)
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    rng = random.Random(42)
    
    best_points = None
    best_score = -1
    
    # Generate diverse initial configurations
    configs = _generate_initial_points(rng)
    
    # For each config, try local search and simulated annealing
    for config in configs:
        if time.time() > deadline - 5:
            break
        
        # Validate initial config
        if not _validate_points(config):
            continue
        
        # Local search
        ls_pts, ls_score = _local_search(config, max_iters=5000, time_limit=deadline - 10, rng=rng)
        if ls_score > best_score:
            best_score = ls_score
            best_points = [list(p) for p in ls_pts]
        
        # Simulated annealing from the best of local search
        sa_pts, sa_score = _simulated_annealing(ls_pts, max_iters=5000, time_limit=deadline - 10, rng=rng)
        if sa_score > best_score:
            best_score = sa_score
            best_points = [list(p) for p in sa_pts]
    
    # If we have a good solution, do more refinement
    if best_points is not None and best_score >= 5:
        # Do another round of local search with more iterations
        refined_pts, refined_score = _local_search(best_points, max_iters=10000, time_limit=deadline - 5, rng=rng)
        if refined_score > best_score:
            best_score = refined_score
            best_points = [list(p) for p in refined_pts]
        
        # And simulated annealing
        sa2_pts, sa2_score = _simulated_annealing(best_points, max_iters=10000, time_limit=deadline - 5, rng=rng)
        if sa2_score > best_score:
            best_score = sa2_score
            best_points = [list(p) for p in sa2_pts]
    
    # Final validation
    if best_points is None:
        # Fallback: simple 4x4 grid
        best_points = []
        for i in range(4):
            for j in range(4):
                best_points.append([i - 2, j - 2])
    
    final_score, is_valid = _quick_score(best_points)
    if not is_valid:
        # Fallback: simple 4x4 grid
        best_points = []
        for i in range(4):
            for j in range(4):
                best_points.append([i - 2, j - 2])
    
    return best_points

# EVOLVE-BLOCK-END
