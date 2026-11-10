import time
import random
import math
from collections import defaultdict

def _line_key(x1, y1, x2, y2):
    """Compute normalized line key through two points."""
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    
    g = math.gcd(abs(A), abs(B))
    if g == 0:
        g = 1
    A //= g
    B //= g
    C //= g
    
    if A < 0 or (A == 0 and B < 0):
        A = -A
        B = -B
        C = -C
    
    return (A, B, C)

def _build_line_map(points):
    """Build map from line key to count of points on that line."""
    lines = defaultdict(int)
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i][0], points[i][1], points[j][0], points[j][1])
            lines[key] += 1
    return lines

def _score_from_line_map(line_map):
    """Count lines with exactly 4 points."""
    count = 0
    for key, val in line_map.items():
        if val == 4:
            count += 1
    return count

def _is_valid(points):
    """Check no five collinear and all distinct."""
    n = len(points)
    if n != 16:
        return False
    
    # Check distinctness
    seen = set()
    for p in points:
        if p in seen:
            return False
        seen.add(p)
    
    # Check bounds
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return False
    
    # Check no five collinear
    line_map = _build_line_map(points)
    for key, val in line_map.items():
        if val >= 5:
            return False
    
    return True

def _count_4_lines(points):
    """Count lines with exactly 4 points."""
    line_map = _build_line_map(points)
    return _score_from_line_map(line_map)

def _generate_initial_configurations():
    """Generate diverse initial configurations with affine normalization."""
    configs = []
    
    # 4x4 grid
    grid_4x4 = [(x, y) for x in range(4) for y in range(4)]
    configs.append(grid_4x4)
    
    # Perturbed grids
    random.seed(42)
    for _ in range(10):
        pts = set()
        while len(pts) < 16:
            x = random.randint(-5, 5)
            y = random.randint(-5, 5)
            pts.add((x, y))
        configs.append(list(pts))
    
    # 5x5 grid subsets
    grid_5x5 = [(x, y) for x in range(5) for y in range(5)]
    for _ in range(10):
        removed = set(random.sample(range(25), 9))
        pts = [grid_5x5[i] for i in range(25) if i not in removed]
        configs.append(pts)
    
    return configs

def _normalize_affine(points):
    """
    Apply affine transformation to fix three non-collinear points at (0,0), (1,0), (0,1).
    Returns normalized points if possible, else None.
    """
    n = len(points)
    if n < 3:
        return None
    
    # Find three non-collinear points
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # Check if non-collinear using cross product
                v1 = (p2[0] - p1[0], p2[1] - p1[1])
                v2 = (p3[0] - p1[0], p3[1] - p1[1])
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                if cross != 0:
                    # Found non-collinear triple
                    # Affine transformation:
                    # new_p = M * (p - p1)
                    # where M maps (p2-p1) -> (1,0) and (p3-p1) -> (0,1)
                    # M = [1/(p2-p1) . 1/(p3-p1)] as columns inverse
                    # Let a = p2 - p1, b = p3 - p1
                    # We want M * a = (1,0), M * b = (0,1)
                    # So M = [a, b]^(-1)
                    ax, ay = p2[0] - p1[0], p2[1] - p1[1]
                    bx, by = p3[0] - p1[0], p3[1] - p1[1]
                    
                    # Inverse of [[ax, bx], [ay, by]]
                    det = ax * by - bx * ay
                    if det == 0:
                        continue
                    
                    # M = (1/det) * [[by, -bx], [-ay, ax]]
                    inv_det = 1.0 / det
                    
                    normalized = []
                    for px, py in points:
                        dx = px - p1[0]
                        dy = py - p1[1]
                        # new_x = (by * dx - bx * dy) / det
                        # new_y = (-ay * dx + ax * dy) / det
                        nx = (by * dx - bx * dy) / det
                        ny = (-ay * dx + ax * dy) / det
                        normalized.append((nx, ny))
                    
                    return normalized
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    random.seed(12345)
    
    # Generate initial configurations
    configs = _generate_initial_configurations()
    
    best_score = 0
    best_points = None
    
    for config in configs:
        if not _is_valid(config):
            continue
        score = _count_4_lines(config)
        if score > best_score:
            best_score = score
            best_points = list(config)
    
    if best_points is None:
        # Fallback to 4x4 grid
        return [[x, y] for x in range(4) for y in range(4)]
    
    # Work with normalized representation using affine invariance
    # Fix three points at (0,0), (1,0), (0,1) and search on remaining 13
    current_points = list(best_points)
    current_score = best_score
    
    # Try to normalize
    normalized = _normalize_affine(current_points)
    if normalized is not None:
        # Use normalized coordinates for search
        search_points = normalized
        search_score = current_score
    else:
        search_points = current_points
        search_score = current_score
    
    # Stochastic local search on the 13 free points (indices 3-15 in normalized frame)
    # In normalized frame, points[0]=(0,0), points[1]=(1,0), points[2]=(0,1) are fixed
    # We search on points[3:16]
    
    while time.time() < deadline:
        # Pick a random free point index (3 to 15)
        idx = random.randint(3, 15)
        old_x, old_y = search_points[idx]
        
        # Try moving it with various step sizes
        step_size = random.choice([1, 1, 1, 2, 2, 3, 4, 5, 8, 12])
        step_x = random.randint(-step_size, step_size)
        step_y = random.randint(-step_size, step_size)
        
        new_x = old_x + step_x
        new_y = old_y + step_y
        
        # Check distinctness against all other points
        new_point = (new_x, new_y)
        is_distinct = True
        for i, p in enumerate(search_points):
            if i != idx and p == new_point:
                is_distinct = False
                break
        
        if not is_distinct:
            continue
        
        # Create trial
        trial_points = search_points.copy()
        trial_points[idx] = new_point
        
        # Check validity (no five collinear)
        line_map = _build_line_map(trial_points)
        has_five = False
        for key, val in line_map.items():
            if val >= 5:
                has_five = True
                break
        
        if has_five:
            continue
        
        trial_score = _score_from_line_map(line_map)
        
        # Acceptance criterion: simulated annealing style
        if trial_score > search_score:
            search_score = trial_score
            search_points = trial_points
            
            # Update best
            # Convert back to original integer coordinates
            # We need to reconstruct original points from normalized
            # For now, just track in normalized space and convert at the end
            if search_score > best_score:
                best_score = search_score
                # Store normalized points for reconstruction
                best_normalized = list(search_points)
        elif trial_score == search_score and random.random() < 0.15:
            # Plateau move
            search_points = trial_points
    
    # Now we need to convert the best normalized configuration back to integer points
    # The normalized points are in a coordinate system where three points are at (0,0), (1,0), (0,1)
    # We need to find an integer affine transformation that maps these back to valid integer points
    
    # Strategy: The normalized points define the configuration up to affine transformation.
    # We can choose any affine map that sends (0,0)->P0, (1,0)->P1, (0,1)->P2 where P0,P1,P2 are non-collinear integer points.
    # Then all other points map to integer points if the matrix has integer entries.
    
    # Let's try to find a good affine transformation
    # M = [[a, b], [c, d]] where M*(0,0)=(0,0), M*(1,0)=(a,c), M*(0,1)=(b,d)
    # We need a, b, c, d to be integers such that all transformed points are integers and within bounds.
    
    # Try small integer matrices
    best_final_points = None
    best_final_score = 0
    
    # Use the best normalized configuration
    if 'best_normalized' in dir():
        norm_pts = best_normalized
    else:
        norm_pts = search_points
    
    # Try various affine transformations
    for a in range(-5, 6):
        for b in range(-5, 6):
            for c in range(-5, 6):
                for d in range(-5, 6):
                    # Check non-degenerate
                    det = a * d - b * c
                    if det == 0:
                        continue
                    
                    # Transform all points
                    transformed = []
                    valid = True
                    for nx, ny in norm_pts:
                        tx = a * nx + b * ny
                        ty = c * nx + d * ny
                        if abs(tx) > 1000 or abs(ty) > 1000:
                            valid = False
                            break
                        transformed.append((tx, ty))
                    
                    if not valid:
                        continue
                    
                    # Check distinctness
                    if len(set(transformed)) != 16:
                        continue
                    
                    # Check no five collinear
                    if not _is_valid(transformed):
                        continue
                    
                    score = _count_4_lines(transformed)
                    if score > best_final_score:
                        best_final_score = score
                        best_final_points = transformed
    
    # If no valid transformation found, use original best_points
    if best_final_points is not None and best_final_score >= best_score:
        return [[x, y] for x, y in best_final_points]
    else:
        return [[x, y] for x, y in best_points]
