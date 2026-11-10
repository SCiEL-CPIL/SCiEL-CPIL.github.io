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
    
    # Find GCD of A, B, C
    g = gcd(abs(A), abs(B))
    g = gcd(g, abs(C))
    
    if g > 1:
        A //= g
        B //= g
        C //= g
        
    # Ensure first non-zero entry is positive
    if A < 0:
        A = -A
        B = -B
        C = -C
    elif A == 0 and B < 0:
        B = -B
        C = -C
        
    return (A, B, C)

def _generate_line_pool(size=2000, max_coeff=10):
    """Generate a diverse pool of lines with small integer coefficients."""
    pool = set()
    
    # 1. Grid lines: x=c, y=c
    for c in range(-max_coeff, max_coeff + 1):
        pool.add(_normalize_line(1, 0, -c))
        pool.add(_normalize_line(0, 1, -c))
        
    # 2. Diagonals: x+y=c, x-y=c
    for c in range(-2*max_coeff, 2*max_coeff + 1):
        pool.add(_normalize_line(1, 1, -c))
        pool.add(_normalize_line(1, -1, c))
        
    # 3. Other simple slopes: y = mx + c
    for m in range(-max_coeff, max_coeff + 1):
        if m == 0: continue # already covered
        for c in range(-max_coeff, max_coeff + 1):
            # y = mx + c => -mx + y - c = 0 => A=-m, B=1, C=-c
            # Or mx - y + c = 0 => A=m, B=-1, C=c
            pool.add(_normalize_line(m, -1, c))
            
    # 4. Random lines with small coefficients
    while len(pool) < size:
        A = random.randint(-max_coeff, max_coeff)
        B = random.randint(-max_coeff, max_coeff)
        C = random.randint(-max_coeff*2, max_coeff*2)
        if A == 0 and B == 0:
            continue
        norm = _normalize_line(A, B, C)
        if norm != (0,0,0):
            pool.add(norm)
            
    return list(pool)

def _get_intersections(lines):
    """
    Get all integer intersection points of the given lines.
    Returns a list of (point, incidence_count) where incidence_count is the number of lines passing through that point.
    """
    points_dict = {} # point -> set of line indices
    
    n_lines = len(lines)
    for i in range(n_lines):
        A1, B1, C1 = lines[i]
        for j in range(i + 1, n_lines):
            A2, B2, C2 = lines[j]
            
            # A1*x + B1*y = -C1
            # A2*x + B2*y = -C2
            
            det = A1 * B2 - A2 * B1
            if det == 0:
                continue # Parallel or coincident
            
            # Cramer's rule
            x_num = (-C1) * B2 - B1 * (-C2)
            y_num = A1 * (-C2) - (-C1) * A2
            
            if x_num % det != 0 or y_num % det != 0:
                continue # Not integer intersection
            
            x = x_num // det
            y = y_num // det
            
            pt = (x, y)
            if pt not in points_dict:
                points_dict[pt] = set()
            points_dict[pt].add(i)
            points_dict[pt].add(j)
            
    # We need to count how many lines pass through each point.
    # The set above only contains the lines that *formed* the intersection.
    # A point might lie on a 3rd line.
    # Efficient way: For each point, check against all lines?
    # Or, since we only have ~351 points and 27 lines, it's cheap.
    
    final_points = []
    for pt, line_indices in points_dict.items():
        x, y = pt
        count = 0
        for k in range(n_lines):
            A, B, C = lines[k]
            if A * x + B * y + C == 0:
                count += 1
        final_points.append((pt, count))
        
    return final_points

def _optimize_points_for_lines(lines, num_points=54):
    """
    Given a set of lines, find the best num_points integer coordinates
    to maximize total incidences.
    """
    intersections = _get_intersections(lines)
    
    # Sort by incidence count descending
    intersections.sort(key=lambda x: x[1], reverse=True)
    
    selected_points = []
    selected_set = set()
    
    for pt, count in intersections:
        if pt not in selected_set:
            selected_points.append(pt)
            selected_set.add(pt)
            if len(selected_points) == num_points:
                break
                
    # If we don't have enough integer intersections, fill with random points?
    # Or points on the lines themselves?
    if len(selected_points) < num_points:
        # Fallback: pick points on the lines
        # Or just random points in a box
        # Better: pick points that are on at least 1 line
        # We can iterate over lines and pick integer points on them
        # But we need to avoid duplicates.
        
        # Let's just fill with random points in a reasonable range
        # This case is rare if the line pool is good.
        existing = selected_set
        while len(selected_points) < num_points:
            x = random.randint(-20, 20)
            y = random.randint(-20, 20)
            if (x, y) not in existing:
                selected_points.append((x, y))
                existing.add((x, y))
                
    return np.array(selected_points, dtype=np.int64)

def _count_incidences(points, lines):
    """Count total incidences between points and lines."""
    total = 0
    for line in lines:
        A, B, C = line
        vals = A * points[:, 0] + B * points[:, 1] + C
        total += np.sum(vals == 0)
    return total

def _find_best_lines_and_points(line_pool, num_lines=27, num_points=54, max_time=580):
    """
    Structural Search:
    1. Start with a random set of lines from pool.
    2. Optimize points for these lines.
    3. Calculate score.
    4. Propose a new line (swap one line in the set).
    5. Re-optimize points.
    6. Accept based on Simulated Annealing.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    # Initialize lines
    # Pick lines that are likely to have many intersections
    # Start with a mix of grid and diagonal
    initial_lines = []
    # Add some vertical/horizontal
    for c in range(0, 10):
        initial_lines.append(_normalize_line(1, 0, -c))
        initial_lines.append(_normalize_line(0, 1, -c))
    # Add some diagonals
    for c in range(0, 10):
        initial_lines.append(_normalize_line(1, 1, -c))
        
    # Randomly select num_lines from the initial set or pool
    # Ensure no duplicates
    current_lines = []
    used = set()
    for l in line_pool:
        if l not in used:
            current_lines.append(l)
            used.add(l)
            if len(current_lines) == num_lines:
                break
    if len(current_lines) < num_lines:
        # Pad with random from pool
        for l in line_pool:
            if l not in used:
                current_lines.append(l)
                used.add(l)
                if len(current_lines) == num_lines:
                    break
                    
    # Optimize points
    current_points = _optimize_points_for_lines(current_lines, num_points)
    current_score = _count_incidences(current_points, current_lines)
    
    best_score = current_score
    best_points = current_points.copy()
    best_lines = current_lines.copy()
    
    T = 10.0 # Temperature
    T_min = 0.1
    cooling = 0.995
    
    while time.time() < deadline - 10:
        if T < T_min:
            break
            
        # Propose a move: Swap one line
        idx = random.randint(0, num_lines - 1)
        old_line = current_lines[idx]
        
        # Pick a new line from the pool
        # Prefer lines that are "close" or just random
        # Random from pool is good for diversity
        new_line = random.choice(line_pool)
        
        # If same line, skip
        if new_line == old_line:
            continue
            
        # Update current lines
        current_lines[idx] = new_line
        
        # Re-optimize points
        new_points = _optimize_points_for_lines(current_lines, num_points)
        new_score = _count_incidences(new_points, current_lines)
        
        # Simulated Annealing acceptance
        delta = new_score - current_score
        if delta > 0 or random.random() < np.exp(delta / T):
            # Accept
            current_points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
                best_lines = current_lines.copy()
        else:
            # Reject, revert line
            current_lines[idx] = old_line
            
        T *= cooling
        
    return best_points, best_lines, best_score

def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization.
    Uses a line-centric structural search with point optimization.
    """
    start_time = time.time()
    timeout = 580  # Safety margin
    
    # Generate a pool of candidate lines
    # Larger pool gives more diversity
    line_pool = _generate_line_pool(size=3000, max_coeff=8)
    
    best_points, best_lines, best_score = _find_best_lines_and_points(line_pool, max_time=timeout)
    
    # Final validation and formatting
    points_out = best_points.astype(np.int64)
    lines_out = np.array(best_lines, dtype=np.int64)
    
    # Ensure shapes
    if points_out.shape != (54, 2):
        # Fallback to grid
        xs = np.arange(1, 10)
        ys = np.arange(1, 7)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points_out = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        # Get top 27 lines for this grid
        pair_counts = {}
        pts = points_out
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                l = _normalize_line(pts[i][1]-pts[j][1], -(pts[i][0]-pts[j][0]), (pts[j][0]-pts[i][0])*pts[i][1] - (pts[j][1]-pts[i][1])*pts[i][0])
                if l != (0,0,0):
                    pair_counts[l] = pair_counts.get(l, 0) + 1
        sorted_lines = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
        lines_out = np.array([l for l, _ in sorted_lines[:27]], dtype=np.int64)
        
    if lines_out.shape != (27, 3):
        lines_out = np.array([_normalize_line(1,0,0)]*27, dtype=np.int64) # Invalid fallback, but shape correct
        
    # Check for duplicates in points
    pts_set = set([tuple(p) for p in points_out])
    if len(pts_set) < 54:
        # Fallback
        xs = np.arange(1, 10)
        ys = np.arange(1, 7)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points_out = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        pair_counts = {}
        pts = points_out
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                l = _normalize_line(pts[i][1]-pts[j][1], -(pts[i][0]-pts[j][0]), (pts[j][0]-pts[i][0])*pts[i][1] - (pts[j][1]-pts[i][1])*pts[i][0])
                if l != (0,0,0):
                    pair_counts[l] = pair_counts.get(l, 0) + 1
        sorted_lines = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
        lines_out = np.array([l for l, _ in sorted_lines[:27]], dtype=np.int64)
        
    # Check for invalid lines
    for l in lines_out:
        if l[0] == 0 and l[1] == 0:
            # Fallback
            xs = np.arange(1, 10)
            ys = np.arange(1, 7)
            px, py = np.meshgrid(xs, ys, indexing="ij")
            points_out = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
            pair_counts = {}
            pts = points_out
            for i in range(len(pts)):
                for j in range(i+1, len(pts)):
                    l = _normalize_line(pts[i][1]-pts[j][1], -(pts[i][0]-pts[j][0]), (pts[j][0]-pts[i][0])*pts[i][1] - (pts[j][1]-pts[i][1])*pts[i][0])
                    if l != (0,0,0):
                        pair_counts[l] = pair_counts.get(l, 0) + 1
            sorted_lines = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
            lines_out = np.array([l for l, _ in sorted_lines[:27]], dtype=np.int64)
            break
            
    # Check for duplicate lines
    lines_set = set([tuple(l) for l in lines_out])
    if len(lines_set) < 27:
        # Fallback
        xs = np.arange(1, 10)
        ys = np.arange(1, 7)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points_out = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        pair_counts = {}
        pts = points_out
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                l = _normalize_line(pts[i][1]-pts[j][1], -(pts[i][0]-pts[j][0]), (pts[j][0]-pts[i][0])*pts[i][1] - (pts[j][1]-pts[i][1])*pts[i][0])
                if l != (0,0,0):
                    pair_counts[l] = pair_counts.get(l, 0) + 1
        sorted_lines = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
        lines_out = np.array([l for l, _ in sorted_lines[:27]], dtype=np.int64)

    return points_out, lines_out

# EVOLVE-BLOCK-END
