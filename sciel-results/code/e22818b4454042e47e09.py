import numpy as np
import time
import random
from math import gcd

def _normalize_line(A, B, C):
    """Normalize line (A, B, C) so gcd(|A|,|B|,|C|) = 1 and first nonzero is positive."""
    # Handle case where A=B=C=0, though we prevent this
    if A == 0 and B == 0 and C == 0:
        return (0, 0, 0)
    
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 0:
        A //= g
        B //= g
        C //= g
    
    # Make first nonzero entry positive
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def _incidence_count(points, lines):
    """Count (p,l) pairs where A*x + B*y + C == 0."""
    A = lines[:, 0:1]
    B = lines[:, 1:2]
    C = lines[:, 2:3]
    x = points[:, 0:1].T  # (1, n)
    y = points[:, 1:2].T  # (1, n)
    val = A * x + B * y + C  # (m, n)
    return int(np.count_nonzero(val == 0))

def _build_initial_grid():
    """
    Build 54 points as a 9x6 grid and 27 lines.
    """
    cols = 9
    rows = 6
    xs = np.arange(cols)
    ys = np.arange(rows)
    PX, PY = np.meshgrid(xs, ys, indexing='ij')
    points = np.stack([PX.ravel(), PY.ravel()], axis=1).astype(np.int64)
    
    # Generate all lines passing through at least 2 points
    # There are 54 points. P(54, 2) = 2808 pairs.
    # We will collect unique normalized lines and count incidences.
    line_counts = {}
    
    # To speed up, we can just iterate through pairs
    n_pts = len(points)
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            
            if A == 0 and B == 0:
                continue
                
            norm = _normalize_line(A, B, C)
            if norm not in line_counts:
                line_counts[norm] = 0
                
    # Now count incidences for each unique line
    # This is O(L * N) where L is number of unique lines.
    # In a 9x6 grid, the number of unique lines is manageable (< 1000).
    lines_list = list(line_counts.keys())
    counts = []
    for A, B, C in lines_list:
        val = A * points[:, 0] + B * points[:, 1] + C
        cnt = int(np.count_nonzero(val == 0))
        counts.append(cnt)
        
    counts = np.array(counts)
    # Select top 27 lines
    top_idx = np.argsort(-counts)[:27]
    selected_lines = [lines_list[i] for i in top_idx]
    lines = np.array(selected_lines, dtype=np.int64)
    
    return points, lines

def _is_valid(points, lines):
    """Quick validity check."""
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    
    # Check for duplicate points
    if len(set(map(tuple, points.tolist()))) != 54:
        return False
        
    # Check for duplicate lines
    line_set = set()
    for i in range(27):
        A, B, C = lines[i]
        A, B, C = int(A), int(B), int(C)
        if A == 0 and B == 0:
            return False
        norm = _normalize_line(A, B, C)
        if norm in line_set:
            return False
        line_set.add(norm)
    return True

def _local_search(points, lines, deadline, seed=42):
    """
    Local search with incremental delta scoring and multiple move types.
    """
    points = points.copy()
    lines = lines.copy()
    
    # Precompute incidence matrix M (27, 54) boolean
    def compute_matrix(points, lines):
        A = lines[:, 0:1]
        B = lines[:, 1:2]
        C = lines[:, 2:3]
        x = points[:, 0:1].T
        y = points[:, 1:2].T
        val = A * x + B * y + C
        return (val == 0)
        
    M = compute_matrix(points, lines)
    current_score = int(np.sum(M))
    best_score = current_score
    best_points = points.copy()
    best_lines = lines.copy()
    
    # Set of points for fast duplicate checking
    point_set = set(map(tuple, points.tolist()))
    
    # Randomization for tie-breaking and exploration
    rng = np.random.default_rng(seed)
    
    iters = 0
    while time.time() < deadline:
        iters += 1
        if iters % 20000 == 0:
            # Periodic full re-evaluation to avoid drift
            current_score = _incidence_count(points, lines)
            M = compute_matrix(points, lines)
            point_set = set(map(tuple, points.tolist()))
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
                best_lines = lines.copy()
            
        move_type = rng.random()
        
        if move_type < 0.5:
            # 1. Try moving a single point to a better location
            p_idx = rng.integers(0, 54)
            orig_x, orig_y = points[p_idx]
            
            # Try several candidate moves
            candidates = []
            # Small random moves
            for _ in range(10):
                dx = rng.integers(-5, 6)
                dy = rng.integers(-5, 6)
                if dx == 0 and dy == 0:
                    continue
                candidates.append((orig_x + dx, orig_y + dy))
            # Also try moving to align with existing lines
            for l_idx in range(27):
                A, B, C = lines[l_idx]
                if A == 0 and B == 0:
                    continue
                # If we move point to lie on this line, pick a nearby integer point on the line
                # Parametric: if B != 0, y = -(A*x + C)/B; try nearby x values
                if B != 0:
                    for dx in range(-5, 6):
                        nx = orig_x + dx
                        num = -(A * nx + C)
                        if num % B == 0:
                            ny = num // B
                            candidates.append((nx, ny))
                elif A != 0:
                    for dy in range(-5, 6):
                        ny = orig_y + dy
                        num = -(B * ny + C)
                        if num % A == 0:
                            nx = num // A
                            candidates.append((nx, ny))
            
            found = False
            for new_x, new_y in candidates:
                if new_x == orig_x and new_y == orig_y:
                    continue
                # Check duplicate points
                if (new_x, new_y) in point_set:
                    continue
                
                # Calculate delta
                old_inc = int(np.sum(M[:, p_idx]))
                
                new_inc = 0
                for l_idx in range(27):
                    A, B, C = lines[l_idx]
                    if A * new_x + B * new_y + C == 0:
                        new_inc += 1
                        
                delta = new_inc - old_inc
                if delta > 0:
                    # Accept move
                    point_set.discard((orig_x, orig_y))
                    points[p_idx] = [new_x, new_y]
                    point_set.add((new_x, new_y))
                    for l_idx in range(27):
                        A, B, C = lines[l_idx]
                        M[l_idx, p_idx] = (A * new_x + B * new_y + C == 0)
                    
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_points = points.copy()
                        best_lines = lines.copy()
                    found = True
                    break
                    
        elif move_type < 0.8:
            # 2. Try replacing a line with the best line through any pair of points
            l_idx = rng.integers(0, 27)
            old_inc = int(np.sum(M[l_idx, :]))
            
            # Find best line through any pair of points (excluding those already used)
            best_new_inc = old_inc
            best_norm = None
            existing_lines = set()
            for i in range(27):
                if i != l_idx:
                    existing_lines.add(_normalize_line(int(lines[i][0]), int(lines[i][1]), int(lines[i][2])))
            
            # Sample pairs to find good candidates
            n_samples = 200
            for _ in range(n_samples):
                p1_idx = rng.integers(0, 54)
                p2_idx = rng.integers(0, 54)
                if p1_idx == p2_idx:
                    continue
                    
                x1, y1 = points[p1_idx]
                x2, y2 = points[p2_idx]
                
                A = y2 - y1
                B = x1 - x2
                C = x2 * y1 - x1 * y2
                
                if A == 0 and B == 0:
                    continue
                    
                norm = _normalize_line(A, B, C)
                if norm in existing_lines:
                    continue
                
                new_inc = 0
                for p_idx in range(54):
                    x, y = points[p_idx]
                    if norm[0] * x + norm[1] * y + norm[2] == 0:
                        new_inc += 1
                        
                if new_inc > best_new_inc:
                    best_new_inc = new_inc
                    best_norm = norm
            
            if best_norm is not None and best_new_inc > old_inc:
                delta = best_new_inc - old_inc
                lines[l_idx] = best_norm
                for p_idx in range(54):
                    x, y = points[p_idx]
                    M[l_idx, p_idx] = (best_norm[0] * x + best_norm[1] * y + best_norm[2] == 0)
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
                    best_lines = lines.copy()
                    
        else:
            # 3. Try moving two points simultaneously to create a new collinearity
            p1_idx = rng.integers(0, 54)
            p2_idx = rng.integers(0, 54)
            if p1_idx == p2_idx:
                continue
            
            orig_x1, orig_y1 = points[p1_idx]
            orig_x2, orig_y2 = points[p2_idx]
            
            # Try moving both points to new positions
            best_delta = 0
            best_move = None
            
            for _ in range(15):
                dx1 = rng.integers(-4, 5)
                dy1 = rng.integers(-4, 5)
                dx2 = rng.integers(-4, 5)
                dy2 = rng.integers(-4, 5)
                if dx1 == 0 and dy1 == 0 and dx2 == 0 and dy2 == 0:
                    continue
                
                nx1, ny1 = orig_x1 + dx1, orig_y1 + dy1
                nx2, ny2 = orig_x2 + dx2, orig_y2 + dy2
                
                # Check duplicates
                if (nx1, ny1) in point_set and (nx1, ny1) != (orig_x1, orig_y1):
                    continue
                if (nx2, ny2) in point_set and (nx2, ny2) != (orig_x2, orig_y2):
                    continue
                if (nx1, ny1) == (nx2, ny2):
                    continue
                
                old_inc1 = int(np.sum(M[:, p1_idx]))
                old_inc2 = int(np.sum(M[:, p2_idx]))
                
                new_inc1 = 0
                new_inc2 = 0
                for l_idx in range(27):
                    A, B, C = lines[l_idx]
                    if A * nx1 + B * ny1 + C == 0:
                        new_inc1 += 1
                    if A * nx2 + B * ny2 + C == 0:
                        new_inc2 += 1
                        
                delta = (new_inc1 + new_inc2) - (old_inc1 + old_inc2)
                if delta > best_delta:
                    best_delta = delta
                    best_move = (nx1, ny1, nx2, ny2)
            
            if best_move is not None and best_delta > 0:
                nx1, ny1, nx2, ny2 = best_move
                point_set.discard((orig_x1, orig_y1))
                point_set.discard((orig_x2, orig_y2))
                points[p1_idx] = [nx1, ny1]
                points[p2_idx] = [nx2, ny2]
                point_set.add((nx1, ny1))
                point_set.add((nx2, ny2))
                
                for l_idx in range(27):
                    A, B, C = lines[l_idx]
                    M[l_idx, p1_idx] = (A * nx1 + B * ny1 + C == 0)
                    M[l_idx, p2_idx] = (A * nx2 + B * ny2 + C == 0)
                    
                current_score += best_delta
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
                    best_lines = lines.copy()

    return best_points, best_lines, best_score

def _build_alternative_initial():
    """
    Build an alternative initial configuration using a different structure.
    Uses a 6x9 grid rotated or a different pattern.
    """
    # Try a 3x18 grid
    cols = 18
    rows = 3
    xs = np.arange(cols)
    ys = np.arange(rows)
    PX, PY = np.meshgrid(xs, ys, indexing='ij')
    points = np.stack([PX.ravel(), PY.ravel()], axis=1).astype(np.int64)
    
    # Generate all lines through pairs
    line_counts = {}
    n_pts = len(points)
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            norm = _normalize_line(A, B, C)
            if norm not in line_counts:
                line_counts[norm] = 0
    
    lines_list = list(line_counts.keys())
    counts = []
    for A, B, C in lines_list:
        val = A * points[:, 0] + B * points[:, 1] + C
        cnt = int(np.count_nonzero(val == 0))
        counts.append(cnt)
        
    counts = np.array(counts)
    top_idx = np.argsort(-counts)[:27]
    selected_lines = [lines_list[i] for i in top_idx]
    lines = np.array(selected_lines, dtype=np.int64)
    
    return points, lines

def _build_hexagonal():
    """
    Build points in a hexagonal-like pattern for better line incidences.
    """
    points_list = []
    # Create rows with offset for hexagonal packing
    for row in range(9):
        offset = row % 2
        for col in range(6):
            x = col + offset * 0.5
            y = row * 0.866
            # Scale to integers
            ix = int(round(x * 2))
            iy = int(round(y * 2))
            points_list.append((ix, iy))
    
    # Remove duplicates if any
    seen = set()
    unique_points = []
    for p in points_list:
        if p not in seen:
            seen.add(p)
            unique_points.append(p)
    
    # If we don't have exactly 54, pad or trim
    if len(unique_points) < 54:
        # Add more points
        for row in range(10, 20):
            offset = row % 2
            for col in range(6):
                x = col + offset * 0.5
                y = row * 0.866
                ix = int(round(x * 2))
                iy = int(round(y * 2))
                if (ix, iy) not in seen:
                    seen.add((ix, iy))
                    unique_points.append((ix, iy))
                    if len(unique_points) == 54:
                        break
            if len(unique_points) == 54:
                break
    elif len(unique_points) > 54:
        unique_points = unique_points[:54]
    
    points = np.array(unique_points, dtype=np.int64)
    
    # Generate lines
    line_counts = {}
    n_pts = len(points)
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            norm = _normalize_line(A, B, C)
            if norm not in line_counts:
                line_counts[norm] = 0
    
    lines_list = list(line_counts.keys())
    counts = []
    for A, B, C in lines_list:
        val = A * points[:, 0] + B * points[:, 1] + C
        cnt = int(np.count_nonzero(val == 0))
        counts.append(cnt)
        
    counts = np.array(counts)
    top_idx = np.argsort(-counts)[:27]
    selected_lines = [lines_list[i] for i in top_idx]
    lines = np.array(selected_lines, dtype=np.int64)
    
    return points, lines

def szemeredi_trotter_incidences():
    deadline = time.time() + 580  # Safety margin
    
    # 1. Try multiple initial configurations
    initial_configs = []
    
    # Grid 9x6
    p1, l1 = _build_initial_grid()
    if _is_valid(p1, l1):
        initial_configs.append((p1, l1, _incidence_count(p1, l1)))
    
    # Grid 3x18
    p2, l2 = _build_alternative_initial()
    if _is_valid(p2, l2):
        initial_configs.append((p2, l2, _incidence_count(p2, l2)))
    
    # Hexagonal
    p3, l3 = _build_hexagonal()
    if _is_valid(p3, l3):
        initial_configs.append((p3, l3, _incidence_count(p3, l3)))
    
    if not initial_configs:
        # Fallback
        p1, l1 = _build_initial_grid()
        return p1, l1
    
    # Sort by initial score
    initial_configs.sort(key=lambda x: -x[2])
    
    best_points, best_lines, best_score = initial_configs[0][0].copy(), initial_configs[0][1].copy(), initial_configs[0][2]
    
    # 2. Local search from each initial config with time allocation
    n_configs = len(initial_configs)
    time_per_config = (deadline - 30) / n_configs
    
    for i, (pts, lns, score) in enumerate(initial_configs):
        if time.time() >= deadline - 30:
            break
        
        config_deadline = min(deadline - 30, time.time() + time_per_config)
        
        # Run local search with different seeds
        pts2, lns2, score2 = _local_search(pts, lns, config_deadline, seed=42 + i)
        if score2 > best_score:
            best_score = score2
            best_points = pts2
            best_lines = lns2
        
        # If we have time, do another round with a different seed
        if time.time() < config_deadline - 5:
            pts3, lns3, score3 = _local_search(pts, lns, config_deadline, seed=100 + i)
            if score3 > best_score:
                best_score = score3
                best_points = pts3
                best_lines = lns3
    
    # Final check
    if not _is_valid(best_points, best_lines):
        # Return the best valid initial config
        for pts, lns, score in initial_configs:
            if _is_valid(pts, lns):
                return pts, lns
        return initial_configs[0][0], initial_configs[0][1]
        
    return best_points, best_lines
