import itertools
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _line_key(p1, p2):
    """
    Returns a normalized key for the line passing through p1 and p2.
    p1, p2 are (x, y) tuples or numpy arrays.
    Line equation: Ax + By + C = 0
    """
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    
    if A == 0 and B == 0:
        return None
        
    # Find GCD of A, B, C
    def gcd(a, b):
        a, b = abs(a), abs(b)
        while b:
            a, b = b, a % b
        return a
        
    g = gcd(gcd(A, B), C)
    if g != 0:
        A //= g
        B //= g
        C //= g
        
    # Normalize sign: first non-zero of A, B, C should be positive
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
        
    return (A, B, C)

def _count_ordinary_lines(points):
    """
    Counts the number of lines containing exactly two of the given points.
    points: list of (x, y) tuples
    """
    n = len(points)
    if n < 2:
        return 0
        
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            line_counts[key] = line_counts.get(key, 0) + 1
            
    ordinary_count = 0
    for count in line_counts.values():
        if count == 1:
            ordinary_count += 1
    return ordinary_count

def _generate_hesse_config():
    """
    Generates a Hesse-like configuration or a known high-symmetry set.
    The Hesse configuration has 9 points and 12 lines. 
    We need 12 points. 
    A known configuration for 12 points with few ordinary lines is derived from 
    the affine plane of order 3 (9 points) plus 3 more, or specific algebraic curves.
    
    Let's try a configuration based on the 3x3 grid (affine plane AG(2,3)) which has 0 ordinary lines?
    No, AG(2,3) has 9 points. Lines have 3 points. 
    If we take the 3x3 grid:
    (0,0) (0,1) (0,2)
    (1,0) (1,1) (1,2)
    (2,0) (2,1) (2,2)
    
    Lines in 3x3 grid:
    3 rows (3 pts each)
    3 cols (3 pts each)
    2 main diagonals (3 pts each)
    4 "broken" diagonals (2 pts each) -> These are ordinary lines in the 3x3 grid!
    Wait, in a 3x3 grid, the lines with slope 1/2 or 2/1 passing through corners?
    Let's check 3x3 grid ordinary lines.
    Pairs:
    (0,0)-(2,1): key? 
    In a 3x3 grid, are there lines with exactly 2 points?
    Total lines determined by pairs: C(9,2)=36.
    Lines with 3 points: 3+3+2 = 8 lines. Each contributes C(3,2)=3 pairs. 8*3=24 pairs.
    Remaining pairs: 36-24=12. These form 12 lines with exactly 2 points.
    So 3x3 grid has 12 ordinary lines.
    
    We need 12 points. 
    If we take a 4x3 grid?
    Or maybe a configuration from a cubic curve?
    
    Let's use a known result: 
    For n=12, the minimum number of ordinary lines is 6 (over reals).
    Can we achieve 6 with integers?
    
    Strategy: 
    1. Start with a high-symmetry seed (e.g., 3x3 grid + 3 points added to break symmetry but maintain collinearity where possible).
    2. Use local search to minimize ordinary lines.
    """
    # Seed 1: 3x3 grid plus 3 points at infinity? No, must be finite integers.
    # Let's try a 4x3 grid.
    # Or better, start with the 3x3 grid and add 3 points far away to see if we can reduce ordinary lines?
    # Actually, adding points usually increases ordinary lines unless they lie on existing lines.
    
    # Let's try a configuration based on the complete quadrilateral or similar.
    
    # A good seed is often a grid. 
    # Let's try 4x3 grid (12 points).
    pts = []
    for x in range(4):
        for y in range(3):
            pts.append((x, y))
    return pts

def _generate_random_seed(scale=10):
    """
    Generates a random set of 12 points.
    """
    pts = []
    for _ in range(12):
        x = np.random.randint(-scale, scale + 1)
        y = np.random.randint(-scale, scale + 1)
        pts.append((x, y))
    return pts

def _local_search(points, max_iters=500, time_limit=100.0):
    """
    Hill climbing local search.
    Tries to move one point at a time to a new integer coordinate 
    to reduce the number of ordinary lines.
    """
    current_points = [list(p) for p in points]
    current_score = _count_ordinary_lines(current_points)
    
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 50
    
    # Pre-calculate line keys for current points to speed up evaluation?
    # Since we only move one point, we can re-calculate efficiently.
    
    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        if no_improve_count > max_no_improve:
            break
            
        improved = False
        
        # Try moving each point
        for i in range(12):
            orig_x, orig_y = current_points[i]
            
            # Try a few random moves around the current position
            # Also try moving to specific "interesting" locations like collinear with other pairs
            moves = []
            
            # Random small moves
            for _ in range(10):
                dx = np.random.randint(-5, 6)
                dy = np.random.randint(-5, 6)
                if dx == 0 and dy == 0:
                    continue
                new_x = orig_x + dx
                new_y = orig_y + dy
                if abs(new_x) <= 10**9 and abs(new_y) <= 10**9:
                    moves.append((new_x, new_y))
            
            # Try to align with other points
            # Pick two other points j, k and try to place i on the line jk
            # Or place i such that it creates a new line with 3 points.
            for j in range(12):
                if j == i: continue
                for k in range(j+1, 12):
                    if k == i: continue
                    # Line through j and k
                    xj, yj = current_points[j]
                    xk, yk = current_points[k]
                    # Direction vector
                    dx = xk - xj
                    dy = yk - yj
                    # Extend line
                    # Try points at distance 1, 2, 3 from j in direction of k
                    for dist in [1, 2, -1, -2]:
                        new_x = xj + dist * dx
                        new_y = yj + dist * dy
                        if abs(new_x) <= 10**9 and abs(new_y) <= 10**9:
                            # Check if this point is already in the set (to avoid duplicates)
                            # We will check duplicates in the evaluation
                            moves.append((new_x, new_y))
            
            # Evaluate moves
            best_move_score = current_score
            best_move_pos = None
            
            # Limit moves to avoid excessive computation
            if len(moves) > 50:
                moves = moves[:50]
                
            for new_x, new_y in moves:
                # Create candidate points
                candidate = [p[:] for p in current_points]
                candidate[i] = [new_x, new_y]
                
                # Check for duplicates
                is_valid = True
                seen = set()
                for p in candidate:
                    if tuple(p) in seen:
                        is_valid = False
                        break
                    seen.add(tuple(p))
                
                if not is_valid:
                    continue
                
                # Check if all collinear (invalid for task)
                # Quick check: are all points on the same line?
                # If yes, score is infinity (or very high), we want to avoid this.
                # Actually, task says "not all collinear".
                # If all collinear, it's invalid.
                
                score = _count_ordinary_lines(candidate)
                
                # Penalize if all collinear
                # Check collinearity
                if len(set([_line_key(candidate[0], candidate[k]) for k in range(1, 12)])) == 1:
                    score = 1000 # High penalty
                
                if score < best_move_score:
                    best_move_score = score
                    best_move_pos = (new_x, new_y)
                    
            if best_move_pos is not None:
                current_points[i] = list(best_move_pos)
                current_score = best_move_score
                improved = True
                
                if current_score < best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
                if best_score <= 6:
                    return best_points, best_score
                    
        if not improved:
            # Restart with a random perturbation if stuck
            # Or just break if stuck for a while
            pass
            
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    best_points = None
    best_score = float('inf')
    
    # 1. Try known configurations
    seeds = []
    
    # 4x3 Grid
    grid_4x3 = [(x, y) for x in range(4) for y in range(3)]
    seeds.append(grid_4x3)
    
    # 3x4 Grid (same essentially)
    
    # 3x3 Grid + 3 points
    # Try adding points to 3x3 grid to reduce ordinary lines
    base_3x3 = [(x, y) for x in range(3) for y in range(3)]
    # Add points at (3,0), (0,3), (3,3)?
    seeds.append(base_3x3 + [(3,0), (0,3), (3,3)])
    seeds.append(base_3x3 + [(3,1), (1,3), (3,3)])
    
    # Random seeds
    for _ in range(5):
        seeds.append(_generate_random_seed(20))
    
    # Evaluate seeds
    for seed in seeds:
        if time.time() > deadline:
            break
        score = _count_ordinary_lines(seed)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in seed]
            
    # 2. Local search from best seed
    if best_points is not None:
        improved_points, improved_score = _local_search(best_points, max_iters=2000, time_limit=130.0)
        if improved_score < best_score:
            best_score = improved_score
            best_points = improved_points
            
    # 3. If we still have time, try local search from other seeds
    if time.time() < deadline - 10:
        for seed in seeds:
            if time.time() > deadline:
                break
            if seed == best_points:
                continue
            # Only search from seeds that are reasonably good
            seed_score = _count_ordinary_lines(seed)
            if seed_score > best_score + 5:
                continue
                
            improved_points, improved_score = _local_search(seed, max_iters=1000, time_limit=120.0)
            if improved_score < best_score:
                best_score = improved_score
                best_points = improved_points
                if best_score <= 6:
                    break
                    
    # Final check and formatting
    if best_points is None:
        # Fallback: 4x3 grid
        return [[x, y] for x in range(4) for y in range(3)]
        
    # Ensure distinct and within bounds
    final_points = []
    for p in best_points:
        x, y = p
        if abs(x) <= 10**9 and abs(y) <= 10**9:
            final_points.append([int(x), int(y)])
            
    # Check distinct
    if len(set(map(tuple, final_points))) != 12:
        # If duplicates crept in, fallback
        return [[x, y] for x in range(4) for y in range(3)]
        
    # Check not all collinear
    if len(set([_line_key(final_points[0], final_points[i]) for i in range(1, 12)])) == 1:
        return [[x, y] for x in range(4) for y in range(3)]
        
    return final_points

# EVOLVE-BLOCK-END
