import time
import random
import math
from collections import defaultdict

def _count_4_lines(points):
    """
    Counts the number of lines containing exactly 4 of the given points.
    Also returns True if any line contains 5 or more points (invalid).
    """
    n = len(points)
    if n < 4:
        return 0, False
    
    line_counts = defaultdict(int)
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            
            if A == 0 and B == 0:
                continue
                
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g > 0:
                A //= g
                B //= g
                C //= g
            
            if A < 0:
                A = -A
                B = -B
                C = -C
            elif A == 0:
                if B < 0:
                    B = -B
                    C = -C
                elif B == 0:
                    if C < 0:
                        C = -C
            
            line_counts[(A, B, C)] += 1

    count_4 = 0
    has_5 = False
    for count in line_counts.values():
        if count == 4:
            count_4 += 1
        elif count >= 5:
            has_5 = True
            
    return count_4, has_5

def _try_complete_grid(grid_points, max_iter=1000):
    """
    Attempts to improve a grid-like configuration by moving points 
    to create more 4-lines.
    """
    current_points = [p[:] for p in grid_points]
    current_score, is_invalid = _count_4_lines(current_points)
    if is_invalid:
        return current_points, current_score
    
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    # Try moving each point to nearby integer coordinates
    # This is a targeted local search, but with a specific bias towards
    # maintaining grid-like structures or creating new 4-point lines.
    
    for _ in range(max_iter):
        improved = False
        for idx in range(16):
            ox, oy = current_points[idx]
            # Try a set of candidate moves
            candidates = []
            # Nearby moves
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    nx = ox + dx
                    ny = oy + dy
                    if abs(nx) <= 1000 and abs(ny) <= 1000:
                        # Check distinctness
                        is_dup = False
                        for k in range(16):
                            if k != idx and current_points[k] == [nx, ny]:
                                is_dup = True
                                break
                        if not is_dup:
                            candidates.append([nx, ny])
            
            # Shuffle candidates for diversity
            random.shuffle(candidates)
            
            best_candidate = None
            best_candidate_score = current_score
            
            for cand in candidates:
                current_points[idx] = cand
                new_score, is_invalid = _count_4_lines(current_points)
                if not is_invalid:
                    if new_score > best_candidate_score:
                        best_candidate_score = new_score
                        best_candidate = cand
                    elif new_score == best_candidate_score and random.random() < 0.1:
                        # Accept lateral moves with small probability to escape plateaus
                        best_candidate = cand
                        best_candidate_score = new_score
                
            if best_candidate is not None:
                current_points[idx] = best_candidate
                if best_candidate_score > best_score:
                    best_score = best_candidate_score
                    best_points = [p[:] for p in current_points]
                    improved = True
        
        if not improved:
            # Try some random jumps if stuck
            for _ in range(5):
                idx = random.randint(0, 15)
                ox, oy = current_points[idx]
                nx = ox + random.randint(-20, 20)
                ny = oy + random.randint(-20, 20)
                if abs(nx) <= 1000 and abs(ny) <= 1000:
                    is_dup = False
                    for k in range(16):
                        if k != idx and current_points[k] == [nx, ny]:
                            is_dup = True
                            break
                    if not is_dup:
                        current_points[idx] = [nx, ny]
                        new_score, is_invalid = _count_4_lines(current_points)
                        if not is_invalid and new_score > best_score:
                            best_score = new_score
                            best_points = [p[:] for p in current_points]
                            improved = True
            if not improved:
                break
                
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 150s limit, keep 5s safety margin
    
    random.seed(42)
    
    best_points_global = None
    best_score_global = 0
    
    # Strategy: Generate multiple diverse initial configurations 
    # based on known structures and refine them.
    
    # 1. Standard 4x4 Grid (known to have 10 4-lines: 4 rows, 4 cols, 2 main diags? 
    #    Actually 4x4 grid has: 4 rows, 4 cols. Diagonals of length 4 are only 2.
    #    Other 4-point lines? (0,0)-(1,1)-(2,2)-(3,3) is one. (0,3)-(1,2)-(2,1)-(3,0) is another.
    #    Slope 1/2? (0,0)-(2,1)... no.
    #    Let's just let the counter handle it.
    grid_4x4 = [[x, y] for x in range(4) for y in range(4)]
    
    # 2. Perturbed Grids
    # 3. Other structures: e.g., points on a parabola? y = x^2? 
    #    (0,0), (1,1), (2,4), (3,9)... 4 points on parabola? No, 3 points determine a conic.
    #    4 points on a line? No, max 4.
    
    # Let's generate a set of diverse starting points
    starts = []
    
    # Start 1: 4x4 Grid
    starts.append([p[:] for p in grid_4x4])
    
    # Start 2: 4x4 Grid with one corner moved
    g2 = [p[:] for p in grid_4x4]
    g2[0] = [0, 4] # Move (0,0) to (0,4)? No, (0,4) is distinct.
    # Actually, let's just use random perturbations of the grid
    for i in range(50):
        g = [p[:] for p in grid_4x4]
        # Move 1-3 points randomly
        for _ in range(random.randint(1, 3)):
            idx = random.randint(0, 15)
            g[idx] = [random.randint(-5, 5), random.randint(-5, 5)]
        # Ensure distinctness
        unique = []
        for p in g:
            if p not in unique:
                unique.append(p)
        if len(unique) == 16:
            starts.append(unique)
            
    # Start 3: Random points in a small box
    for i in range(20):
        pts = []
        while len(pts) < 16:
            p = [random.randint(-5, 5), random.randint(-5, 5)]
            if p not in pts:
                pts.append(p)
        starts.append(pts)
        
    # Evaluate and refine each start
    for start_pts in starts:
        if time.time() > deadline:
            break
            
        # Quick check if valid
        score, invalid = _count_4_lines(start_pts)
        if invalid:
            continue
            
        if score > best_score_global:
            best_score_global = score
            best_points_global = [p[:] for p in start_pts]
            
        # Refine
        refined_pts, refined_score = _try_complete_grid(start_pts, max_iter=50)
        if refined_score > best_score_global:
            best_score_global = refined_score
            best_points_global = refined_pts
            
        # If we found a very good score, we might want to do a deeper search
        # from this specific point
        if best_score_global >= 12:
             # Do a more intensive local search from the best found so far
             # to see if we can push higher
             deep_pts, deep_score = _try_complete_grid(best_points_global, max_iter=200)
             if deep_score > best_score_global:
                 best_score_global = deep_score
                 best_points_global = deep_pts

    # If we haven't found anything better than the grid (10), 
    # let's try a dedicated search for "near-grid" configurations
    # that might have more 4-lines by breaking the grid symmetry slightly.
    
    # The 4x4 grid has 10 lines. Can we get 11, 12, 13...?
    # The theoretical max is 15.
    
    # Let's run a final global random search with a bias towards 
    # "structured" moves if we are still at or below 10.
    if best_score_global <= 10:
        # Try to find a configuration with > 10
        # Use a "simulated annealing" but with a very specific move set:
        # Move a point to a position that completes a line of 3.
        
        current = [p[:] for p in best_points_global] if best_points_global else [p[:] for p in grid_4x4]
        current_score = best_score_global if best_points_global else 10
        
        # Precompute lines of 3 for the current config? 
        # Too slow to do dynamically in Python for 115s.
        # Instead, just do a smarter random search.
        
        for _ in range(10000): # Fixed number of steps, fast
            if time.time() > deadline:
                break
            idx = random.randint(0, 15)
            ox, oy = current[idx]
            
            # Pick a random target point in a larger box
            nx = ox + random.randint(-10, 10)
            ny = oy + random.randint(-10, 10)
            
            if abs(nx) > 1000 or abs(ny) > 1000:
                continue
                
            is_dup = False
            for k in range(16):
                if k != idx and current[k] == [nx, ny]:
                    is_dup = True
                    break
            if is_dup:
                continue
                
            current[idx] = [nx, ny]
            new_score, is_invalid = _count_4_lines(current)
            
            if not is_invalid:
                if new_score > current_score:
                    current_score = new_score
                    if new_score > best_score_global:
                        best_score_global = new_score
                        best_points_global = [p[:] for p in current]
                elif new_score == current_score and random.random() < 0.05:
                    pass # Accept lateral
                else:
                    # Reject
                    current[idx] = [ox, oy]
            else:
                current[idx] = [ox, oy]

    if best_points_global is None:
        # Fallback to grid
        best_points_global = grid_4x4
        
    return best_points_global
