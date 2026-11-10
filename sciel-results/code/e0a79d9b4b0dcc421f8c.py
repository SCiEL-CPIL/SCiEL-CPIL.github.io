# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from collections import defaultdict

def compute_incidence(points, lines):
    """
    Compute the number of incidences between points and lines.
    points: (N, 2) array of integers
    lines: (M, 3) array of integers (A, B, C) such that A*x + B*y + C = 0
    Returns: int
    """
    if len(points) == 0 or len(lines) == 0:
        return 0
    
    # Vectorized check: (N, 1, 3) * (1, M, 3) -> sum over last axis -> (N, M)
    # To avoid memory blowup for large N, M, we can chunk it, but N=54, M=27 is small.
    # points[:, np.newaxis, :] * lines[np.newaxis, :, :]
    # Actually, simpler:
    # A*x + B*y + C = 0
    # Let P be points (N, 2), L be lines (M, 3)
    # We want count of P[i] on L[j]
    
    # Efficient way:
    # Expand points to (N, 1, 2) and lines to (1, M, 3)
    # But we only need x and y for the first two columns of lines.
    # L[:, 0] is A, L[:, 1] is B, L[:, 2] is C
    # P[:, 0] is x, P[:, 1] is y
    
    # Dot product trick:
    # [x, y, 1] dot [A, B, C] = 0
    ones = np.ones((len(points), 1))
    P_hom = np.hstack([points, ones]) # (N, 3)
    
    # (N, 3) @ (3, M) -> (N, M)
    # lines is (M, 3), so lines.T is (3, M)
    dot_prods = P_hom @ lines.T
    
    # Count zeros
    return np.sum(dot_prods == 0)

def get_neighbors(x, y, radius=2):
    """Generate nearby integer coordinates for local search."""
    neighbors = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            neighbors.append((x + dx, y + dy))
    return neighbors

def local_search(points, lines, max_iter=1000, time_limit=500):
    """
    Improve the configuration by moving points to better positions.
    points: (N, 2) array
    lines: (M, 3) array
    """
    points = points.copy().astype(np.int64)
    n = len(points)
    m = len(lines)
    
    # Precompute line coefficients for faster evaluation
    # We want to evaluate delta in incidences when moving point i to new_pos
    
    current_score = compute_incidence(points, lines)
    
    # Track used points to ensure uniqueness
    used_points = set(map(tuple, points))
    
    improved = True
    iter_count = 0
    start_time = time.time()
    
    while improved and iter_count < max_iter:
        if time.time() - start_time > time_limit:
            break
        improved = False
        iter_count += 1
        
        for i in range(n):
            old_x, old_y = points[i]
            best_new_pos = None
            best_new_score = current_score
            
            # Try moving to neighbors
            # Radius 1 is usually sufficient for local optima, but let's try 2 for robustness
            neighbors = get_neighbors(old_x, old_y, radius=2)
            
            for nx, ny in neighbors:
                if (nx, ny) in used_points:
                    continue
                
                # Temporarily move point i
                points[i] = [nx, ny]
                
                # Calculate new score
                # Optimization: Only recalculate incidences for point i
                # Old incidences for point i: count of lines where A*old_x + B*old_y + C == 0
                # New incidences for point i: count of lines where A*nx + B*ny + C == 0
                
                # Since N and M are small, full recompute is ~ 54*27 = 1458 ops.
                # 1000 iters * 54 points * ~20 neighbors * 1458 ops = ~1.5e8 ops.
                # This might be too slow in Python if not optimized.
                
                # Let's use a faster incremental check.
                # We need to know which lines passed through old point and which through new.
                
                # Precomputed line arrays
                A = lines[:, 0]
                B = lines[:, 1]
                C = lines[:, 2]
                
                # Incidences for old point
                old_eq = (A * old_x + B * old_y + C) == 0
                old_count = np.sum(old_eq)
                
                # Incidences for new point
                new_eq = (A * nx + B * ny + C) == 0
                new_count = np.sum(new_eq)
                
                new_score = current_score - old_count + new_count
                
                if new_score > best_new_score:
                    best_new_score = new_score
                    best_new_pos = (nx, ny)
                
                # Revert for next neighbor check
                points[i] = [old_x, old_y]
            
            if best_new_pos is not None:
                # Apply the best move
                used_points.discard((old_x, old_y))
                used_points.add(best_new_pos)
                points[i] = best_new_pos
                current_score = best_new_score
                improved = True

    return points, current_score

def generate_grid_config():
    k = 3
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    rows = []
    for a in range(1, k + 1):
        for b in range(1, k * k + 1):
            rows.append((a, -1, b))
    lines = np.array(rows, dtype=np.int64)
    return points, lines

def generate_l_shape_config():
    # L-shape: Points along x-axis and y-axis
    # 27 points on x-axis (1..27, 0), 27 points on y-axis (0, 1..27)
    # Total 54 points.
    # Lines: x=0, y=0, and 25 other lines?
    # This is a heuristic.
    pts = []
    for i in range(1, 28):
        pts.append((i, 0))
    for j in range(1, 28):
        pts.append((0, j))
    points = np.array(pts, dtype=np.int64)
    
    # Lines: 
    # 1. x=0 (0,1,0)
    # 2. y=0 (1,0,0)
    # 3-27. Lines passing through (i, j) pairs?
    # Let's just use random lines for now to get a valid config, then polish.
    # Or better, lines that cover many points.
    # Line x=0 covers 27 points.
    # Line y=0 covers 27 points.
    # Remaining 25 lines.
    # We can pick lines that pass through (i, 0) and (0, j) for various i,j.
    # Line through (i, 0) and (0, j): x/i + y/j = 1 -> j*x + i*y - i*j = 0.
    # Let's pick 25 such lines.
    lines = [[0, 1, 0], [1, 0, 0]]
    for i in range(1, 26):
        for j in range(1, 26):
            if len(lines) >= 27: break
            # Ensure unique
            A, B, C = j, i, -i*j
            # Normalize
            g = np.gcd(np.gcd(abs(A), abs(B)), abs(C))
            if g > 0:
                A, B, C = A//g, B//g, C//g
            if A < 0 or (A == 0 and B < 0):
                A, B, C = -A, -B, -C
            l = [A, B, C]
            if l not in lines:
                lines.append(l)
            if len(lines) >= 27: break
        if len(lines) >= 27: break
        
    lines = np.array(lines[:27], dtype=np.int64)
    return points, lines

def generate_random_config():
    # Random points in a grid, random lines
    np.random.seed(42)
    points = []
    used = set()
    while len(points) < 54:
        x = np.random.randint(-50, 50)
        y = np.random.randint(-50, 50)
        if (x, y) not in used:
            points.append((x, y))
            used.add((x, y))
    points = np.array(points, dtype=np.int64)
    
    lines = []
    used_lines = set()
    while len(lines) < 27:
        A = np.random.randint(-10, 10)
        B = np.random.randint(-10, 10)
        C = np.random.randint(-100, 100)
        if A == 0 and B == 0:
            continue
        g = np.gcd(np.gcd(abs(A), abs(B)), abs(C))
        if g > 0:
            A, B, C = A//g, B//g, C//g
        if A < 0 or (A == 0 and B < 0):
            A, B, C = -A, -B, -C
        l = (A, B, C)
        if l not in used_lines:
            lines.append(l)
            used_lines.add(l)
    lines = np.array(lines, dtype=np.int64)
    return points, lines

def generate_diamond_config():
    # Points on a diamond shape
    pts = []
    for i in range(-10, 11):
        for j in range(-10, 11):
            if abs(i) + abs(j) <= 10:
                pts.append((i, j))
    # Take first 54
    points = np.array(pts[:54], dtype=np.int64)
    
    # Lines: x=0, y=0, x+y=0, x-y=0, etc.
    lines = [[1, 0, 0], [0, 1, 0], [1, 1, 0], [1, -1, 0]]
    # Add random lines to fill up to 27
    np.random.seed(43)
    used_lines = set(map(tuple, lines))
    while len(lines) < 27:
        A = np.random.randint(-5, 5)
        B = np.random.randint(-5, 5)
        C = np.random.randint(-20, 20)
        if A == 0 and B == 0: continue
        g = np.gcd(np.gcd(abs(A), abs(B)), abs(C))
        if g > 0: A, B, C = A//g, B//g, C//g
        if A < 0 or (A == 0 and B < 0): A, B, C = -A, -B, -C
        l = (A, B, C)
        if l not in used_lines:
            lines.append(list(l))
            used_lines.add(l)
    lines = np.array(lines, dtype=np.int64)
    return points, lines

def validate(points, lines):
    """Check validity of the solution."""
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    if not np.issubdtype(points.dtype, np.integer) or not np.issubdtype(lines.dtype, np.integer):
        return False
    
    # Check duplicates
    if len(set(map(tuple, points))) != 54:
        return False
    
    # Check lines
    for i in range(27):
        A, B, C = lines[i]
        if A == 0 and B == 0:
            return False
        # Check uniqueness
        g = np.gcd(np.gcd(abs(A), abs(B)), abs(C))
        if g > 0:
            An, Bn, Cn = A//g, B//g, C//g
        else:
            An, Bn, Cn = A, B, C
        if An < 0 or (An == 0 and Bn < 0):
            An, Bn, Cn = -An, -Bn, -Cn
        for j in range(i+1, 27):
            A2, B2, C2 = lines[j]
            g2 = np.gcd(np.gcd(abs(A2), abs(B2)), abs(C2))
            if g2 > 0:
                A2n, B2n, C2n = A2//g2, B2//g2, C2//g2
            else:
                A2n, B2n, C2n = A2, B2, C2
            if A2n < 0 or (A2n == 0 and B2n < 0):
                A2n, B2n, C2n = -A2n, -B2n, -C2n
            if (An, Bn, Cn) == (A2n, B2n, C2n):
                return False
    return True

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 580  # Leave 20s margin
    
    configs = [
        generate_grid_config,
        generate_l_shape_config,
        generate_random_config,
        generate_diamond_config
    ]
    
    best_points = None
    best_lines = None
    best_score = -1
    
    for gen_func in configs:
        if time.time() > deadline:
            break
        try:
            points, lines = gen_func()
            if not validate(points, lines):
                continue
            
            score = compute_incidence(points, lines)
            if score > best_score:
                best_score = score
                best_points = points.copy()
                best_lines = lines.copy()
                
            # Polish this config
            polished_points, polished_score = local_search(points, lines, max_iter=50, time_limit=10)
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished_points.copy()
                best_lines = lines.copy()
        except Exception as e:
            print(f"Error in config generation: {e}")
            continue
            
    # If we still have time, do a final polish on the best config
    if best_points is not None and time.time() < deadline - 10:
        # Use remaining time for polish
        remaining_time = deadline - time.time() - 5
        if remaining_time > 1:
            final_points, final_score = local_search(best_points, best_lines, max_iter=1000, time_limit=remaining_time)
            if final_score > best_score:
                best_points = final_points
                best_score = final_score
                
    # Ensure return types are correct
    if best_points is None:
        # Fallback to grid
        best_points, best_lines = generate_grid_config()
        
    return best_points, best_lines

# EVOLVE-BLOCK-END
