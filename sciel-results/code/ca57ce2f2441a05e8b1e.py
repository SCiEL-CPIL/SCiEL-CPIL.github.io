# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from math import gcd
from itertools import combinations

def _normalize_line(A, B, C):
    """Normalize line coefficients (A, B, C) to a canonical form."""
    if A == 0 and B == 0:
        return (0, 0, 0)
    
    g = gcd(abs(A), abs(B))
    g = gcd(g, abs(C))
    
    if g > 1:
        A //= g
        B //= g
        C //= g
        
    if A < 0:
        A = -A
        B = -B
        C = -C
    elif A == 0 and B < 0:
        B = -B
        C = -C
        
    return (A, B, C)

def _count_incidences(points, lines):
    """
    Count total incidences between points and lines.
    """
    if len(points) == 0 or len(lines) == 0:
        return 0
    total = 0
    for A, B, C in lines:
        vals = A * points[:, 0] + B * points[:, 1] + C
        total += np.sum(vals == 0)
    return total

def _generate_line_pool_from_grid(grid_size=15, max_coord=10):
    """
    Generate a pool of lines from a dense grid.
    We consider all pairs of points in a grid of size grid_size x grid_size
    centered at origin, with coordinates in [-max_coord, max_coord].
    This creates a rich pool of lines with small integer coefficients.
    """
    pool = set()
    # Create grid points
    coords = list(range(-max_coord, max_coord + 1))
    # To keep pool size manageable, we don't use the full grid_size^2 pairs if grid_size is large
    # But grid_size=15, max_coord=10 gives 21x21=441 points. Pairs ~ 100k. That's fine.
    
    # Actually, let's just iterate over all pairs in the grid defined by coords
    # But 21*21 = 441 points. Pairs = 441*440/2 = ~97k.
    # For each pair, compute line.
    
    for i, (x1, y1) in enumerate([(x, y) for x in coords for y in coords]):
        for j, (x2, y2) in enumerate([(x, y) for x in coords for y in coords]):
            if i >= j:
                continue
            # Line through (x1, y1) and (x2, y2)
            # A = y2 - y1
            # B = x1 - x2
            # C = x2*y1 - x1*y2
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            key = _normalize_line(A, B, C)
            if key != (0, 0, 0):
                pool.add(key)
                
    # Add some random lines to diversify
    for _ in range(1000):
        A = random.randint(-5, 5)
        B = random.randint(-5, 5)
        C = random.randint(-30, 30)
        if A != 0 or B != 0:
            pool.add(_normalize_line(A, B, C))
            
    return list(pool)

def _greedy_line_selection(points, pool, k=27):
    """
    Greedy set cover: Select k lines from pool to maximize number of unique points covered.
    """
    n_points = len(points)
    # Precompute which points are on which line in the pool?
    # That's expensive: |pool| * n_points.
    # Let's just calculate on the fly or precompute incidence matrix if pool is small enough.
    # Pool size ~ 100k. n_points = 54. 5.4M ops. Fast enough in numpy?
    # Let's convert pool to numpy array for vectorization.
    
    pool_arr = np.array(pool, dtype=np.int64)
    # Shape (M, 3)
    
    # Incidence matrix: (M, N)
    # M lines, N points.
    # vals = A*x + B*y + C
    # points: (N, 2)
    # pool_arr: (M, 3)
    
    # We can do:
    # x = points[:, 0] (N,)
    # y = points[:, 1] (N,)
    # A = pool_arr[:, 0] (M,)
    # B = pool_arr[:, 1] (M,)
    # C = pool_arr[:, 2] (M,)
    
    # vals = A[:, None] * x[None, :] + B[:, None] * y[None, :] + C[:, None]
    # This creates an (M, N) array.
    # M ~ 100k, N = 54. 5.4M floats. ~40MB. OK.
    
    x = points[:, 0]
    y = points[:, 1]
    A = pool_arr[:, 0]
    B = pool_arr[:, 1]
    C = pool_arr[:, 2]
    
    # To save memory, we can do it in chunks or just let numpy handle it.
    # Let's try direct.
    vals = A[:, None] * x[None, :] + B[:, None] * y[None, :] + C[:, None]
    incidence = (vals == 0).astype(np.int64) # (M, N)
    
    # Now greedy selection
    selected_lines = []
    covered_points = np.zeros(n_points, dtype=bool)
    
    # We want to maximize total incidences, not just unique points?
    # The task says "I is the number of (point, line) pairs...".
    # So we want to maximize sum of incidences.
    # Greedy: pick line with max (sum of incidences - 0) ?
    # Actually, if we just pick the top K lines by incidence count, that's what the current program does.
    # Why is that not optimal?
    # Because it doesn't account for the fact that we can MOVE points.
    # If we fix lines, we can move points to lie on them.
    # So the "value" of a line depends on how many points can be placed on it.
    # But points are shared.
    
    # Let's stick to the current strategy of fixing lines and optimizing points,
    # but use a better initial line selection.
    # The current program uses "top 27 lines by incidence in initial grid".
    # Maybe we should select lines that are "diverse" to allow more points to be placed?
    # Or maybe we should just use the pool of lines defined by the points themselves?
    
    # Let's try a different approach:
    # 1. Start with a random set of 27 lines from the pool.
    # 2. Optimize points to lie on these lines (each point moves to the nearest intersection of two selected lines, or just projects onto the line).
    # 3. Re-evaluate lines: For each point, find the line in the selected set it's on.
    # 4. Swap lines: Replace a line with a new line from the pool that covers more points (or allows more points to be placed).
    
    # This is essentially what the current program does, but the line pool is static.
    # Let's make the line pool dynamic: generate lines from the current points.
    
    return pool_arr, incidence

def _optimize_points_to_lines(points, lines):
    """
    Given fixed lines, move each point to the nearest intersection of two lines,
    or to a point on a line that maximizes incidence.
    Actually, just moving to the nearest point on the line is not enough,
    because we want integer coordinates.
    
    Better: For each point, find the best integer coordinate nearby that lies on one of the lines.
    """
    n_points = len(points)
    n_lines = len(lines)
    
    # Precompute line coeffs
    A = lines[:, 0]
    B = lines[:, 1]
    C = lines[:, 2]
    
    def score_point(pt):
        x, y = pt
        vals = A * x + B * y + C
        return np.sum(vals == 0)
    
    # Local search
    improved = True
    while improved:
        improved = False
        for i in range(n_points):
            old_pt = points[i]
            old_score = score_point(old_pt)
            
            best_cand = None
            best_cand_score = old_score
            
            # Search nearby
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    cand = (old_pt[0] + dx, old_pt[1] + dy)
                    # Check if used
                    is_used = False
                    for j, p in enumerate(points):
                        if j != i and p[0] == cand[0] and p[1] == cand[1]:
                            is_used = True
                            break
                    if is_used:
                        continue
                    
                    new_score = score_point(cand)
                    if new_score > best_cand_score:
                        best_cand_score = new_score
                        best_cand = cand
            
            if best_cand is not None:
                points[i] = best_cand
                improved = True
                
    return points

def _select_best_lines(points, pool_arr, k=27):
    """
    Select k lines from pool_arr to maximize total incidences with points.
    """
    x = points[:, 0]
    y = points[:, 1]
    A = pool_arr[:, 0]
    B = pool_arr[:, 1]
    C = pool_arr[:, 2]
    
    vals = A[:, None] * x[None, :] + B[:, None] * y[None, :] + C[:, None]
    incidence = (vals == 0).astype(np.int64)
    counts = np.sum(incidence, axis=1)
    
    # Top k
    top_k_indices = np.argsort(counts)[::-1][:k]
    return pool_arr[top_k_indices]

def szemeredi_trotter_incidences():
    start_time = time.time()
    timeout = 590
    
    # 1. Generate a large pool of lines
    # Use a grid of size 15x15, coords -10..10
    pool_lines = _generate_line_pool_from_grid(max_coord=10)
    pool_arr = np.array(pool_lines, dtype=np.int64)
    
    # 2. Initial points: 9x6 grid
    xs = np.arange(1, 10)
    ys = np.arange(1, 7)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # 3. Select initial lines
    lines = _select_best_lines(points, pool_arr, 27)
    
    best_score = _count_incidences(points, lines)
    best_points = points.copy()
    best_lines = lines.copy()
    
    # 4. Iterative optimization
    while time.time() < timeout - 10:
        # Optimize points for current lines
        points = _optimize_points_to_lines(points, lines)
        
        # Select new lines for current points
        # Re-generate pool? Or use existing pool?
        # Using existing pool is faster.
        lines = _select_best_lines(points, pool_arr, 27)
        
        # Calculate score
        current_score = _count_incidences(points, lines)
        
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
            best_lines = lines.copy()
            
        # If no improvement, try a random restart of points?
        # Or just break?
        
    return best_points, best_lines

# EVOLVE-BLOCK-END
