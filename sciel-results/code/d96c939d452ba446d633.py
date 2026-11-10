import numpy as np
import time
import random
import math
from itertools import product

def _canon_line(A, B, C):
    """Canonicalize a line (A, B, C) -> (A, B, C) such that gcd(|A|,|B|,|C|)=1 and first non-zero is positive."""
    if A == 0 and B == 0:
        return None
    g = math.gcd(abs(A), math.gcd(abs(B), abs(C)))
    if g == 0:
        return None
    A //= g
    B //= g
    C //= g
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidences(points, lines):
    """Count incidences exactly."""
    I = 0
    for x, y in points:
        for A, B, C in lines:
            if A * x + B * y + C == 0:
                I += 1
    return I

def _generate_initial_points():
    """Generate 54 distinct integer points in a reasonable range."""
    points = set()
    while len(points) < 54:
        x = random.randint(-10, 10)
        y = random.randint(-10, 10)
        points.add((x, y))
    return list(points)

def _generate_initial_lines():
    """Generate 27 distinct lines."""
    lines = set()
    while len(lines) < 27:
        # Random slope and intercept
        # Use small integers to keep coordinates small
        A = random.randint(-5, 5)
        B = random.randint(-5, 5)
        if A == 0 and B == 0:
            continue
        C = random.randint(-20, 20)
        cl = _canon_line(A, B, C)
        if cl:
            lines.add(cl)
    return list(lines)

def _local_search(points, lines, max_iters=10000, temp_start=1.0, temp_end=0.01):
    """Simulated annealing local search."""
    points = list(points)
    lines = list(lines)
    current_I = _count_incidences(points, lines)
    best_I = current_I
    best_points = [tuple(p) for p in points]
    best_lines = [tuple(l) for l in lines]
    
    # Precompute incidence matrix for speed?
    # M[i, j] = 1 if point i is on line j
    # Updating M when a point moves: recalculate row i.
    # Updating M when a line moves: recalculate column j.
    
    M = np.zeros((54, 27), dtype=np.int8)
    for i in range(54):
        x, y = points[i]
        for j in range(27):
            A, B, C = lines[j]
            if A * x + B * y + C == 0:
                M[i, j] = 1
    
    current_I = np.sum(M)
    
    for it in range(max_iters):
        temp = temp_start * (temp_end / temp_start) ** (it / max_iters)
        
        if random.random() < 0.5:
            # Move a point
            i = random.randint(0, 53)
            old_x, old_y = points[i]
            # Try nearby points
            dx = random.randint(-2, 2)
            dy = random.randint(-2, 2)
            if dx == 0 and dy == 0:
                continue
            new_x, new_y = old_x + dx, old_y + dy
            new_pt = (new_x, new_y)
            
            # Check duplicate
            if new_pt in points:
                continue
            
            # Calculate delta I
            delta = 0
            for j in range(27):
                A, B, C = lines[j]
                old_inc = M[i, j]
                new_inc = 1 if (A * new_x + B * new_y + C == 0) else 0
                delta += new_inc - old_inc
                
            if delta > 0 or random.random() < math.exp(delta / temp):
                # Accept
                points[i] = new_pt
                for j in range(27):
                    A, B, C = lines[j]
                    M[i, j] = 1 if (A * new_x + B * new_y + C == 0) else 0
                current_I += delta
                if current_I > best_I:
                    best_I = current_I
                    best_points = [tuple(p) for p in points]
                    best_lines = [tuple(l) for l in lines]
        else:
            # Change a line
            j = random.randint(0, 26)
            old_A, old_B, old_C = lines[j]
            
            # Propose new line
            # Perturb A, B, C
            new_A = old_A + random.randint(-1, 1)
            new_B = old_B + random.randint(-1, 1)
            new_C = old_C + random.randint(-2, 2)
            
            if new_A == 0 and new_B == 0:
                continue
                
            cl = _canon_line(new_A, new_B, new_C)
            if not cl:
                continue
            
            # Check duplicate line
            if cl in lines:
                continue
            
            new_A, new_B, new_C = cl
            
            # Calculate delta I
            delta = 0
            for i in range(54):
                x, y = points[i]
                old_inc = M[i, j]
                new_inc = 1 if (new_A * x + new_B * y + new_C == 0) else 0
                delta += new_inc - old_inc
                
            if delta > 0 or random.random() < math.exp(delta / temp):
                # Accept
                lines[j] = cl
                for i in range(54):
                    x, y = points[i]
                    M[i, j] = 1 if (new_A * x + new_B * y + new_C == 0) else 0
                current_I += delta
                if current_I > best_I:
                    best_I = current_I
                    best_points = [tuple(p) for p in points]
                    best_lines = [tuple(l) for l in lines]
                    
    return best_points, best_lines, best_I

def _seed_grid():
    """The baseline Szemeredi-Trotter grid."""
    k = 3
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    points = [tuple(p) for p in points]
    
    lines = []
    for a in range(1, k + 1):
        for b in range(1, k * k + 1):
            lines.append(_canon_line(a, -1, b))
    return points, lines

def _seed_random():
    """Random seed."""
    points = _generate_initial_points()
    lines = _generate_initial_lines()
    return points, lines

def _seed_hybrid():
    """Start with grid, then randomize some points."""
    points, lines = _seed_grid()
    # Randomize 10 points
    for _ in range(10):
        idx = random.randint(0, 53)
        x, y = points[idx]
        new_x = x + random.randint(-5, 5)
        new_y = y + random.randint(-5, 5)
        # Ensure distinct
        while (new_x, new_y) in points:
            new_x = x + random.randint(-5, 5)
            new_y = y + random.randint(-5, 5)
        points[idx] = (new_x, new_y)
    return points, lines

def szemeredi_trotter_incidences():
    """
    Maximizes point-line incidences for 54 points and 27 lines.
    """
    start_time = time.time()
    deadline = start_time + 580  # 20s safety margin
    
    best_I = 0
    best_points = None
    best_lines = None
    
    # Seeds
    seeds = [
        _seed_grid(),
        _seed_random(),
        _seed_hybrid(),
        _seed_random(),
        _seed_hybrid()
    ]
    
    # Run local search on each seed
    for i, (pts, lns) in enumerate(seeds):
        if time.time() > deadline:
            break
            
        # Estimate time per seed
        # Let's run 5000 iterations per seed
        max_iters = 5000
        # Adjust based on time
        if i == 0:
            # First seed gets more time to establish baseline
            max_iters = 10000
            
        res_pts, res_lns, res_I = _local_search(pts, lns, max_iters=max_iters)
        
        if res_I > best_I:
            best_I = res_I
            best_points = res_pts
            best_lines = res_lns
            
        # If we found a very high score, we can stop early?
        # Theoretical max is ~400. If we get > 150, it's good.
        # But let's use the time.
        
    # If time remains, run more iterations on the best solution found
    if best_points is not None and time.time() < deadline - 5:
        remaining_time = deadline - 5 - time.time()
        # Estimate iterations per second
        # From previous run, assume 1000 iters/sec is safe for Python
        extra_iters = int(remaining_time * 500) 
        if extra_iters > 0:
             res_pts, res_lns, res_I = _local_search(best_points, best_lines, max_iters=extra_iters, temp_start=0.1, temp_end=0.001)
             if res_I > best_I:
                 best_I = res_I
                 best_points = res_pts
                 best_lines = res_lns

    # Convert to numpy arrays
    points_arr = np.array(best_points, dtype=np.int64)
    lines_arr = np.array(best_lines, dtype=np.int64)
    
    return points_arr, lines_arr

# EVOLVE-BLOCK-START
# The code above is the complete program.
# EVOLVE-BLOCK-END
