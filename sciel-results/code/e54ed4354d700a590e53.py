"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math
import itertools
import time
from collections import defaultdict

def _line_key(p1, p2):
    """Compute normalized line key (a, b, c) for line through p1, p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    g = math.gcd(abs(dx), abs(dy))
    if g == 0:
        return None
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    a = dy
    b = -dx
    c = a * x1 + b * y1
    return (a, b, c)

def _score_config(points):
    """Count lines with exactly 4 points. Returns 0 if any line has 5+."""
    n = len(points)
    if n != 16:
        return 0
    line_to_points = defaultdict(set)
    for i, j in itertools.combinations(range(n), 2):
        key = _line_key(points[i], points[j])
        if key is None:
            continue
        line_to_points[key].add(i)
        line_to_points[key].add(j)
    max_collinear = 2
    count_4 = 0
    for key, pts in line_to_points.items():
        if len(pts) >= 5:
            return 0
        if len(pts) == 4:
            count_4 += 1
    return count_4

def _is_valid(points):
    """Check validity: 16 distinct points, bounds, no 5+ collinear."""
    n = len(points)
    if n != 16:
        return False
    seen = set()
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    line_to_points = defaultdict(set)
    for i, j in itertools.combinations(range(n), 2):
        key = _line_key(points[i], points[j])
        if key is None:
            continue
        line_to_points[key].add(i)
        line_to_points[key].add(j)
    for key, pts in line_to_points.items():
        if len(pts) >= 5:
            return False
    return True

def _backtrack_construct(deadline):
    """Backtracking search for 16 valid points maximizing 4-lines."""
    # Generate candidate pool: focus on structured regions
    # Use a mix of grid points and perturbed points
    candidates = []
    # 4x4 grid base
    for x in range(0, 8):
        for y in range(0, 8):
            candidates.append((x, y))
    # Add some spread
    for x in range(-4, 12):
        for y in range(-4, 12):
            if (x, y) not in candidates:
                candidates.append((x, y))
    
    random.shuffle(candidates)
    # Limit candidate pool for tractability
    candidates = candidates[:200]
    
    best_score = 0
    best_points = None
    line_counts = defaultdict(int)  # line_key -> count of points on that line
    line_points = defaultdict(set)  # line_key -> set of point indices
    
    def backtrack(idx, chosen, line_counts_local, line_points_local):
        nonlocal best_score, best_points
        if time.time() > deadline:
            return
        if idx == 16:
            # Count 4-lines
            count_4 = 0
            for key, pts in line_points_local.items():
                if len(pts) == 4:
                    count_4 += 1
            if count_4 > best_score:
                best_score = count_4
                best_points = [list(p) for p in chosen]
            return
        
        # Try candidates for position idx
        for cand in candidates:
            if cand in chosen:
                continue
            # Check if adding this point would create a 5+ collinear line
            valid = True
            for prev in chosen:
                key = _line_key(cand, prev)
                if key is None:
                    continue
                if line_counts_local[key] >= 4:
                    # Adding would make it 5
                    valid = False
                    break
            if not valid:
                continue
            
            # Also check: is cand already on a line with 4 existing points?
            # (covered above since we check all pairs)
            
            # Add cand
            chosen.append(cand)
            new_line_updates = []
            for prev in chosen[:-1]:
                key = _line_key(cand, prev)
                if key is None:
                    continue
                line_counts_local[key] += 1
                line_points_local[key].add(len(chosen) - 1)
                new_line_updates.append(key)
            
            backtrack(idx + 1, chosen, line_counts_local, line_points_local)
            
            # Undo
            for key in new_line_updates:
                line_counts_local[key] -= 1
                line_points_local[key].discard(len(chosen) - 1)
            chosen.pop()
            
            if time.time() > deadline:
                return
    
    backtrack(0, [], line_counts, line_points)
    return best_points, best_score

def _local_search(points, deadline):
    """Local search to improve a valid configuration."""
    best = [p[:] for p in points]
    best_score = _score_config(best)
    
    if best_score == 0:
        return best, 0
    
    for iteration in range(50000):
        if time.time() > deadline:
            break
        
        idx = random.randint(0, 15)
        new_pts = [p[:] for p in best]
        dx = random.randint(-5, 5)
        dy = random.randint(-5, 5)
        if dx == 0 and dy == 0:
            continue
        new_pts[idx][0] += dx
        new_pts[idx][1] += dy
        
        # Quick validity check
        if abs(new_pts[idx][0]) > 1000 or abs(new_pts[idx][1]) > 1000:
            continue
        # Check distinctness
        if any(new_pts[idx] == p for i, p in enumerate(new_pts) if i != idx):
            continue
        
        new_score = _score_config(new_pts)
        if new_score > best_score:
            best = new_pts
            best_score = new_score
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    random.seed(42)
    
    # Strategy 1: Backtracking construction
    bt_points, bt_score = _backtrack_construct(deadline - 60)
    
    # Strategy 2: Start from 4x4 grid and local search
    grid = [[x, y] for x in range(4) for y in range(4)]
    grid_score = _score_config(grid)  # Should be 10
    
    # Combine: use whichever is better as starting point
    if bt_score > grid_score and bt_points:
        start_points = bt_points
        start_score = bt_score
    else:
        start_points = grid
        start_score = grid_score
    
    # Local search polish
    deadline_ls = min(deadline, start_time + 120)
    final_points, final_score = _local_search(start_points, deadline_ls)
    
    # If backtracking found something better, also try local search from there
    if bt_points and bt_score > final_score:
        deadline_ls2 = min(deadline, start_time + 125)
        bt_final, bt_final_score = _local_search(bt_points, deadline_ls2)
        if bt_final_score > final_score:
            final_points = bt_final
            final_score = bt_final_score
    
    # Safety: ensure validity
    if not _is_valid(final_points):
        final_points = grid
        final_score = grid_score
    
    return final_points
# EVOLVE-BLOCK-END
