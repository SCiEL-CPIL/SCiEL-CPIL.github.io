import numpy as np
from math import gcd
import time
import itertools


def _gcd3(a, b, c):
    """GCD of three non-negative integers (with sign handling)."""
    a, b, c = abs(int(a)), abs(int(b)), abs(int(c))
    g = gcd(gcd(a, b), c)
    return g if g > 0 else 1


def _normalize_line(A, B, C):
    """Normalize a line so it's unique: divide by gcd, make first nonzero positive."""
    A, B, C = int(A), int(B), int(C)
    g = _gcd3(A, B, C)
    if g > 1:
        A //= g
        B //= g
        C //= g
    # Make first nonzero entry positive
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)


def _count_incidents(points, lines):
    """Count incidences exactly using int64 arithmetic."""
    n = len(points)
    m = len(lines)
    total = 0
    for i in range(m):
        A, B, C = lines[i]
        # A*x + B*y + C == 0
        vals = A * points[:, 0] + B * points[:, 1] + C
        total += int(np.sum(vals == 0))
    return total


def _validate(points, lines):
    """Check validity constraints. Returns True if valid."""
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    # Check no NaN/inf
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(lines)):
        return False
    # Check entries are near integers
    if not np.all(np.abs(points - np.round(points)) < 1e-6):
        return False
    if not np.all(np.abs(lines - np.round(lines)) < 1e-6):
        return False
    # Check no repeated points
    pts_set = set(map(tuple, points.tolist()))
    if len(pts_set) != 54:
        return False
    # Check no line with A=B=0
    for l in lines:
        if l[0] == 0 and l[1] == 0:
            return False
    # Check no duplicate lines (after normalization)
    norm_lines = set()
    for l in lines:
        nl = _normalize_line(l[0], l[1], l[2])
        if nl in norm_lines:
            return False
        norm_lines.add(nl)
    return True


def _line_through_two_points(p1, p2):
    """Get line (A,B,C) through two points, normalized. Returns None if same point."""
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])
    if x1 == x2 and y1 == y2:
        return None
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    return _normalize_line(A, B, C)


def _generate_candidate_lines(points, exclude_set, max_candidates=200):
    """Generate candidate lines through pairs of points, excluding those in exclude_set."""
    candidates = []
    seen = set()
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            nl = _line_through_two_points(points[i], points[j])
            if nl is None:
                continue
            if nl in exclude_set or nl in seen:
                continue
            seen.add(nl)
            candidates.append(nl)
            if len(candidates) >= max_candidates:
                return candidates
    return candidates


def _local_search(points, lines, deadline, max_iters=500):
    """Local search: try swapping lines and re-optimizing points."""
    best_points = points.copy()
    best_lines = lines.copy()
    best_score = _count_incidents(best_points, best_lines)
    
    start_time = time.time()
    
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        
        # Get current incidences per line
        line_incidents = []
        for i in range(len(best_lines)):
            A, B, C = best_lines[i]
            vals = A * best_points[:, 0] + B * best_points[:, 1] + C
            line_incidents.append(int(np.sum(vals == 0)))
        
        # Find the line with minimum incidence (target for replacement)
        min_line_idx = int(np.argmin(line_incidents))
        min_inc = line_incidents[min_line_idx]
        
        # Generate candidate replacement lines
        exclude_set = set()
        for i in range(len(best_lines)):
            if i != min_line_idx:
                exclude_set.add(_normalize_line(*best_lines[i]))
        
        candidates = _generate_candidate_lines(best_points, exclude_set, max_candidates=100)
        
        # Evaluate each candidate
        best_candidate = None
        best_candidate_score = best_score
        
        # Also try re-optimizing points for each candidate
        for cand in candidates:
            # Quick score: how many points does this line hit?
            A, B, C = cand
            vals = A * best_points[:, 0] + B * best_points[:, 1] + C
            cand_inc = int(np.sum(vals == 0))
            
            # Net gain = cand_inc - min_inc
            if cand_inc <= min_inc:
                continue
            
            # Try re-optimizing points
            new_lines = best_lines.copy()
            new_lines[min_line_idx] = np.array(cand, dtype=np.int64)
            
            # Re-optimize points: for each point, try moving it to maximize total
            new_points = _reoptimize_points(best_points, new_lines, deadline)
            if new_points is None:
                continue
            
            score = _count_incidents(new_points, new_lines)
            if score > best_candidate_score:
                best_candidate_score = score
                best_candidate = (new_points.copy(), new_lines.copy())
        
        if best_candidate is not None:
            best_points, best_lines = best_candidate
            best_score = best_candidate_score
        else:
            # Try point-only optimization
            new_points = _reoptimize_points(best_points, best_lines, deadline)
            if new_points is not None:
                score = _count_incidents(new_points, best_lines)
                if score > best_score:
                    best_points = new_points
                    best_score = score
            else:
                # No improvement, try a different strategy: random point perturbation
                if np.random.random() < 0.3:
                    # Move a random point slightly
                    idx = np.random.randint(len(best_points))
                    dx, dy = np.random.choice([-1, 0, 1], size=2)
                    if dx == 0 and dy == 0:
                        continue
                    new_pts = best_points.copy()
                    new_pts[idx] += [dx, dy]
                    # Check no duplicate
                    if len(set(map(tuple, new_pts.tolist()))) == 54:
                        score = _count_incidents(new_pts, best_lines)
                        if score > best_score:
                            best_points = new_pts
                            best_score = score
    
    return best_points, best_lines, best_score


def _reoptimize_points(points, lines, deadline, max_passes=3):
    """Re-optimize point positions to maximize incidences with given lines.
    For each point, try moving it to a nearby location that increases total incidence."""
    current_points = points.copy()
    current_score = _count_incidents(current_points, lines)
    
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        if time.time() > deadline:
            break
        
        for idx in range(len(current_points)):
            if time.time() > deadline:
                break
            
            orig_x, orig_y = int(current_points[idx, 0]), int(current_points[idx, 1])
            
            # Try moving this point to various nearby locations
            best_new_pos = None
            best_new_score = current_score
            
            # Try a range of offsets
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = orig_x + dx, orig_y + dy
                    new_pos = (nx, ny)
                    
                    # Check if this position is already occupied
                    occupied = False
                    for other_idx in range(len(current_points)):
                        if other_idx != idx:
                            if (int(current_points[other_idx, 0]), int(current_points[other_idx, 1])) == new_pos:
                                occupied = True
                                break
                    if occupied:
                        continue
                    
                    # Compute new score efficiently: only need to check which lines the old and new positions satisfy
                    # Old position contributions
                    old_on = 0
                    for l in lines:
                        A, B, C = l
                        if A * orig_x + B * orig_y + C == 0:
                            old_on += 1
                    
                    # New position contributions
                    new_on = 0
                    for l in lines:
                        A, B, C = l
                        if A * nx + B * ny + C == 0:
                            new_on += 1
                    
                    delta = new_on - old_on
                    if delta > 0:
                        candidate_score = current_score + delta
                        if candidate_score > best_new_score:
                            best_new_score = candidate_score
                            best_new_pos = (nx, ny)
            
            if best_new_pos is not None:
                current_points[idx] = best_new_pos
                current_score = best_new_score
                improved = True
    
    if current_score > _count_incidents(points, lines):
        return current_points
    return None


def _build_grid_and_lines():
    """Build the initial 6x9 grid with axis-aligned and diagonal lines."""
    a, b = 6, 9
    xs = np.arange(1, a + 1)
    ys = np.arange(1, b + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    lines = []
    
    for j in range(1, b + 1):
        lines.append((0, 1, -j))
    
    for i in range(1, a + 1):
        lines.append((1, 0, -i))
    
    diag_counts = {}
    for i in range(1, a + 1):
        for j in range(1, b + 1):
            c = j - i
            diag_counts[c] = diag_counts.get(c, 0) + 1
    
    sorted_diags = sorted(diag_counts.items(), key=lambda x: (-x[1], x[0]))
    selected_c = [c for c, cnt in sorted_diags[:12]]
    
    for c in selected_c:
        lines.append((1, -1, c))
    
    lines = np.array(lines, dtype=np.int64)
    return points, lines


def _try_configuration(points, lines, deadline):
    """Try a configuration: validate and run local search."""
    if not _validate(points, lines):
        return None, None, 0
    
    score = _count_incidents(points, lines)
    new_points, new_lines, new_score = _local_search(points, lines, deadline)
    if new_score > score:
        return new_points, new_lines, new_score
    return points, lines, score


def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences for 54 points and 27 lines on the integer lattice.
    Uses grid initialization + local search with line swaps and point re-optimization.
    """
    start_time = time.time()
    deadline = start_time + 570  # 30s safety margin before 600s limit
    
    # Initial grid configuration
    points, lines = _build_grid_and_lines()
    best_points, best_lines, best_score = _try_configuration(points, lines, deadline)
    
    if best_points is None:
        return points, lines
    
    # Try several different initial configurations
    configs = []
    
    # Config 1: 6x9 grid (already done)
    # Config 2: 3x18 grid
    a, b = 3, 18
    xs = np.arange(1, a + 1)
    ys = np.arange(1, b + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    pts2 = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    lines2 = []
    for j in range(1, b + 1):
        lines2.append((0, 1, -j))
    for i in range(1, a + 1):
        lines2.append((1, 0, -i))
    # That's only 21 lines, need 6 more diagonals
    diag_counts = {}
    for i in range(1, a + 1):
        for j in range(1, b + 1):
            c = j - i
            diag_counts[c] = diag_counts.get(c, 0) + 1
    sorted_diags = sorted(diag_counts.items(), key=lambda x: (-x[1], x[0]))
    for c, cnt in sorted_diags[:6]:
        lines2.append((1, -1, c))
    lines2 = np.array(lines2, dtype=np.int64)
    configs.append((pts2, lines2))
    
    # Config 3: 9x6 grid (transposed)
    a, b = 9, 6
    xs = np.arange(1, a + 1)
    ys = np.arange(1, b + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    pts3 = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    lines3 = []
    for j in range(1, b + 1):
        lines3.append((0, 1, -j))
    for i in range(1, a + 1):
        lines3.append((1, 0, -i))
    diag_counts = {}
    for i in range(1, a + 1):
        for j in range(1, b + 1):
            c = j - i
            diag_counts[c] = diag_counts.get(c, 0) + 1
    sorted_diags = sorted(diag_counts.items(), key=lambda x: (-x[1], x[0]))
    for c, cnt in sorted_diags[:12]:
        lines3.append((1, -1, c))
    lines3 = np.array(lines3, dtype=np.int64)
    configs.append((pts3, lines3))
    
    # Config 4: 54 points in a different arrangement - two overlapping grids
    # 18x3 grid
    a, b = 18, 3
    xs = np.arange(1, a + 1)
    ys = np.arange(1, b + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    pts4 = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    lines4 = []
    for j in range(1, b + 1):
        lines4.append((0, 1, -j))
    for i in range(1, a + 1):
        lines4.append((1, 0, -i))
    # That's 21 lines, need 6 more
    diag_counts = {}
    for i in range(1, a + 1):
        for j in range(1, b + 1):
            c = j - i
            diag_counts[c] = diag_counts.get(c, 0) + 1
    sorted_diags = sorted(diag_counts.items(), key=lambda x: (-x[1], x[0]))
    for c, cnt in sorted_diags[:6]:
        lines4.append((1, -1, c))
    lines4 = np.array(lines4, dtype=np.int64)
    configs.append((pts4, lines4))
    
    # Try each config
    for pts_c, lines_c in configs:
        if time.time() > deadline:
            break
        p, l, s = _try_configuration(pts_c, lines_c, deadline)
        if p is not None and s > best_score:
            best_points, best_lines, best_score = p, l, s
    
    # Final polish: target low-incidence lines for replacement
    if time.time() < deadline - 30:
        # Additional local search passes with more aggressive moves
        for _ in range(3):
            if time.time() > deadline:
                break
            new_p, new_l, new_s = _local_search(best_points, best_lines, deadline)
            if new_s > best_score:
                best_points, best_lines, best_score = new_p, new_l, new_s
            else:
                break
    
    # Final validation
    if not _validate(best_points, best_lines):
        # Fallback to original grid
        return _build_grid_and_lines()
    
    return best_points, best_lines
