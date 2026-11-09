"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

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

def _count_ordinary_lines(pts):
    """Count lines containing exactly two of the points."""
    n = len(pts)
    if n < 3:
        return 0
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _get_line_key(pts[i][0], pts[i][1], pts[j][0], pts[j][1])
            if key is not None:
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

def _is_valid(points):
    """Check if points are valid: 12 distinct, within bounds, not all collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for p in points:
        if len(p) != 2:
            return False
        if abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
            return False
        if (p[0], p[1]) in seen:
            return False
        seen.add((p[0], p[1]))
    return _not_all_collinear(points)

def _generate_hexagonal_config(scale=1):
    """Generate points from hexagonal lattice pattern.
    
    Uses the 6 vertices of a regular hexagon plus 6 more points
    to create many 3-point collinearities.
    """
    # Hexagon vertices (centered at origin)
    # For a regular hexagon with 'radius' r, vertices are at angles 0, 60, 120, 180, 240, 300
    # Using integer coordinates: (r, 0), (r/2, r*sqrt(3)/2), etc.
    # To keep integers, use a scaled hexagonal lattice
    
    # Hexagonal lattice basis: (1,0) and (1/2, sqrt(3)/2)
    # For integer coords, use basis vectors (2,0) and (1, sqrt(3))
    # Approximate with: (2,0) and (1,2) for a sheared hexagon
    
    # Better: use the standard hexagonal tiling with integer coordinates
    # Points: (0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2), (3,0), (3,1), (3,2)
    
    # Actually, let's use a configuration that maximizes 3-point lines
    # The best known for 12 points with fewest ordinary lines uses a 3x4 grid variant
    
    # Start with a 3x4 grid and modify
    base = [[0,0],[1,0],[2,0],[3,0],
            [0,1],[1,1],[2,1],[3,1],
            [0,2],[1,2],[2,2],[3,2]]
    
    # This is a 4x3 grid. Let's try to find better configurations
    
    # Configuration 1: 4x3 grid
    config1 = [[x, y] for x in range(4) for y in range(3)]
    
    # Configuration 2: Modified hexagonal
    # Place points to create many 3-point lines
    config2 = [
        [0,0], [1,0], [2,0], [3,0],
        [0,1], [1,1], [2,1], [3,1],
        [0,2], [1,2], [2,2], [3,2]
    ]
    
    # Configuration 3: Try to create more 3-point lines
    # Use a pattern where multiple lines have 3+ points
    config3 = [
        [0,0], [1,0], [2,0], [3,0],
        [0,1], [2,1], [4,1], [1,2],
        [0,2], [2,2], [4,2], [1,3]
    ]
    
    # Configuration 4: Symmetric pattern
    config4 = [
        [0,0], [2,0], [4,0], [6,0],
        [1,1], [3,1], [5,1], [0,2],
        [2,2], [4,2], [6,2], [1,3]
    ]
    
    configs = [config1, config2, config3, config4]
    
    best = None
    best_score = float('inf')
    for config in configs:
        if not _is_valid(config):
            continue
        score = _evaluate(config)
        if score < best_score:
            best_score = score
            best = [p[:] for p in config]
    
    return best if best else config1

def _generate_pencil_configs():
    """Generate configurations based on pencils of lines through a point."""
    configs = []
    
    # Pencil: multiple lines through one point
    # Place 12 points such that many lines pass through a common point
    center = [0, 0]
    
    # Lines through center with various slopes
    # For each line, place 2-3 points
    lines_data = [
        # (slope_num, slope_den, positions along line)
        (0, 1, [1, 2, 3]),      # horizontal right
        (0, -1, [-1, -2, -3]),  # horizontal left  
        (1, 1, [1, 2]),          # 45 deg
        (1, -1, [1, 2]),         # -45 deg
        (2, 1, [1]),             # steep
        (-2, 1, [1]),            # steep negative
    ]
    
    points = [center]
    for num, den, positions in lines_data:
        for pos in positions:
            x = pos * den
            y = pos * num
            points.append([x, y])
            if len(points) >= 12:
                break
        if len(points) >= 12:
            break
    
    points = points[:12]
    if _is_valid(points):
        configs.append(points)
    
    # Another pencil configuration
    center2 = [0, 0]
    points2 = [center2]
    # Use lines with more points
    for angle_idx in range(6):
        # 6 directions from center
        if angle_idx == 0:
            dx, dy = 1, 0
        elif angle_idx == 1:
            dx, dy = 0, 1
        elif angle_idx == 2:
            dx, dy = 1, 1
        elif angle_idx == 3:
            dx, dy = -1, 1
        elif angle_idx == 4:
            dx, dy = -1, 0
        else:
            dx, dy = 0, -1
        
        for dist in [1, 2]:
            points2.append([center2[0] + dx * dist, center2[1] + dy * dist])
            if len(points2) >= 12:
                break
        if len(points2) >= 12:
            break
    
    points2 = points2[:12]
    if _is_valid(points2):
        configs.append(points2)
    
    return configs

def _find_good_positions(points, idx, rng, max_candidates=100):
    """Find good positions for point at idx that create more 3-point collinearities."""
    n = len(points)
    old_x, old_y = points[idx]
    occupied = set()
    for i, p in enumerate(points):
        if i != idx:
            occupied.add((p[0], p[1]))
    
    existing = [(points[i][0], points[i][1]) for i in range(n) if i != idx]
    
    # Find all lines with 2+ existing points
    line_to_points = {}
    for i in range(len(existing)):
        for j in range(i + 1, len(existing)):
            key = _get_line_key(existing[i][0], existing[i][1], existing[j][0], existing[j][1])
            if key is not None:
                if key not in line_to_points:
                    line_to_points[key] = set()
                line_to_points[key].add(tuple(existing[i]))
                line_to_points[key].add(tuple(existing[j]))
    
    candidates = []
    
    # For each line with 2+ points, try to add a point on it
    for key, pts_on_line in line_to_points.items():
        if len(pts_on_line) < 2:
            continue
        
        # Get two points to determine line direction
        pts_list = list(pts_on_line)
        x1, y1 = pts_list[0]
        x2, y2 = pts_list[1]
        
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
        
        # Find existing t values
        existing_ts = set()
        for px, py in pts_on_line:
            if dx_n != 0:
                if (px - x1) % dx_n == 0:
                    t = (px - x1) // dx_n
                    if x1 + t * dx_n == px and y1 + t * dy_n == py:
                        existing_ts.add(t)
            elif dy_n != 0:
                if (py - y1) % dy_n == 0:
                    t = (py - y1) // dy_n
                    if x1 + t * dx_n == px and y1 + t * dy_n == py:
                        existing_ts.add(t)
        
        # Try nearby t values
        for t in range(-3, 6):
            if t in existing_ts:
                continue
            new_x = x1 + t * dx_n
            new_y = y1 + t * dy_n
            
            if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                continue
            
            if (new_x, new_y) not in occupied:
                # Estimate benefit: how many new 3-point lines does this create?
                benefit = 0
                for other_key, other_pts in line_to_points.items():
                    if other_key == key:
                        continue
                    # Check if (new_x, new_y) is on other_key's line
                    # This is expensive, so just count lines with 2+ points
                    benefit += 1  # Simplified: each existing line with 2+ points is a potential collinearity
                
                candidates.append((new_x, new_y, len(pts_on_line)))
    
    # Also try random positions near existing points
    for _ in range(30):
        base_pt = rng.choice(existing)
        new_x = base_pt[0] + rng.randint(-2, 2)
        new_y = base_pt[1] + rng.randint(-2, 2)
        
        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
            continue
        if (new_x, new_y) not in occupied:
            candidates.append((new_x, new_y, 0))
    
    # Sort by potential benefit (higher line count first)
    candidates.sort(key=lambda c: -c[2])
    
    return candidates[:max_candidates]

def _swap_improve(points, rng, deadline):
    """Try swapping pairs of points to improve the score."""
    n = len(points)
    current_score = _evaluate(points)
    improved = False
    
    for i in range(n):
        if time.time() > deadline:
            break
        for j in range(i + 1, n):
            if time.time() > deadline:
                break
            
            # Try swapping points i and j
            points[i], points[j] = points[j], points[i]
            
            if _is_valid(points):
                new_score = _evaluate(points)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    # Swap back
                    points[i], points[j] = points[j], points[i]
    
    return improved

def _local_search(points, rng, deadline, max_iter=50):
    """Local search: move one point at a time to improve score."""
    n = len(points)
    current_score = _evaluate(points)
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        
        for idx in range(n):
            if time.time() > deadline:
                break
            
            old_x, old_y = points[idx]
            
            # Find good positions
            candidates = _find_good_positions(points, idx, rng)
            
            best_candidate = None
            best_candidate_score = current_score
            
            for new_x, new_y, _ in candidates[:20]:
                if time.time() > deadline:
                    break
                
                # Tentatively move
                points[idx] = [new_x, new_y]
                
                if _is_valid(points):
                    score = _evaluate(points)
                    if score < best_candidate_score:
                        best_candidate_score = score
                        best_candidate = [new_x, new_y]
                
                # Restore
                points[idx] = [old_x, old_y]
            
            if best_candidate is not None:
                points[idx] = best_candidate
                current_score = best_candidate_score
                improved = True
                break
        
        if not improved:
            break
    
    return current_score

def construct():
    start = time.time()
    deadline = start + 135  # safety margin before 150s kill
    
    n = 12
    rng = random.Random(42)
    
    # Generate initial configurations
    best_points = None
    best_score = float('inf')
    
    # Try hexagonal and pencil configurations
    hex_config = _generate_hexagonal_config()
    if hex_config and _is_valid(hex_config):
        score = _evaluate(hex_config)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in hex_config]
    
    for config in _generate_pencil_configs():
        if _is_valid(config):
            score = _evaluate(config)
            if score < best_score:
                best_score = score
                best_points = [p[:] for p in config]
    
    # Also try the original 4x3 grid
    grid = [[x, y] for x in range(4) for y in range(3)]
    if _is_valid(grid):
        score = _evaluate(grid)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in grid]
    
    if best_points is None:
        best_points = grid
        best_score = _evaluate(best_points)
    
    # Local search from best configuration
    current_points = [p[:] for p in best_points]
    current_score = _local_search(current_points, rng, deadline, max_iter=30)
    
    if current_score < best_score:
        best_score = current_score
        best_points = [p[:] for p in current_points]
    
    # Try multiple restarts with perturbations
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Perturb current best
        current_points = [p[:] for p in best_points]
        num_perturb = rng.randint(2, 4)
        for _ in range(num_perturb):
            idx = rng.randrange(n)
            # Move to a nearby position
            current_points[idx] = [
                current_points[idx][0] + rng.randint(-3, 3),
                current_points[idx][1] + rng.randint(-3, 3)
            ]
        
        if not _is_valid(current_points):
            continue
        
        current_score = _local_search(current_points, rng, deadline, max_iter=20)
        
        if current_score < best_score:
            best_score = current_score
            best_points = [p[:] for p in current_points]
        
        if best_score <= 6:
            break
    
    # Final swap-based improvement
    if time.time() < deadline:
        _swap_improve(best_points, rng, deadline)
        final_score = _evaluate(best_points)
        if final_score < best_score:
            best_score = final_score
    
    # Final validation
    assert len(best_points) == 12
    assert all(isinstance(p, list) and len(p) == 2 for p in best_points)
    assert all(abs(p[0]) <= 10**9 and abs(p[1]) <= 10**9 for p in best_points)
    assert _not_all_collinear(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
