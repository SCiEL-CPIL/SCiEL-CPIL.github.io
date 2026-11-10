import time
import random
import sys
from itertools import combinations

def _check_no_four_collinear(points):
    """Check if no four points are collinear. Returns True if valid."""
    n = len(points)
    # Map lines to counts
    lines = {}
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            if dx == 0 and dy == 0:
                continue
            
            # Normalize line equation to canonical form
            # Ax + By + C = 0
            A = dy
            B = -dx
            C = dx * y1 - dy * x1
            
            def gcd(a, b):
                while b:
                    a, b = b, a % b
                return abs(a)
            
            g = gcd(gcd(A, B), C)
            if g != 0:
                A //= g
                B //= g
                C //= g
            
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            if key not in lines:
                lines[key] = 2
            else:
                lines[key] += 1
                
            if lines[key] >= 4:
                return False
    return True

def _max_independent_set(points):
    """
    Exact branch and bound for the largest subset with no three collinear.
    Uses bitmasks for speed.
    """
    n = len(points)
    if n == 0:
        return 0
    
    # Precompute conflict masks
    # collinear_with[i][j] is a bitmask of k such that i, j, k are collinear.
    collinear_with = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            mask = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    mask |= (1 << k)
            collinear_with[i][j] = mask
            collinear_with[j][i] = mask

    best = 0
    
    def backtrack(current_mask, candidates_mask):
        nonlocal best
        
        cand_count = candidates_mask.bit_count()
        cur_count = current_mask.bit_count()
        
        if cur_count + cand_count <= best:
            return
            
        if cand_count == 0:
            if cur_count > best:
                best = cur_count
            return

        p_idx = (candidates_mask & -candidates_mask).bit_length() - 1
        rest_mask = candidates_mask & ~(1 << p_idx)
        
        # Branch 1: Include p_idx
        blocked_mask = 0
        cm = current_mask
        while cm:
            r_idx = (cm & -cm).bit_length() - 1
            blocked_mask |= collinear_with[p_idx][r_idx]
            cm &= cm - 1
            
        new_candidates = rest_mask & ~blocked_mask
        backtrack(current_mask | (1 << p_idx), new_candidates)
        
        # Branch 2: Exclude p_idx
        backtrack(current_mask, rest_mask)

    backtrack(0, (1 << n) - 1)
    return best

def _count_triples_fast(points):
    """Count total number of collinear triples."""
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            for k in range(j+1, n):
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    count += 1
    return count

def _greedy_construct(time_limit):
    """
    Greedily build a set of 20 points by adding one point at a time.
    At each step, choose the candidate point that maximizes the number of
    new collinear triples formed, subject to no 4-collinear constraint.
    """
    start_time = time.time()
    points = []
    
    # Initial 2 points: choose far apart to allow space
    points.append([-50, -50])
    points.append([50, 50])
    
    # Candidate pool: We will search in a grid around the origin
    # To ensure we find good triple-forming points, we should look at
    # intersections of lines defined by existing points.
    
    for step in range(2, 20):
        if time.time() - start_time > time_limit:
            break
            
        best_candidate = None
        best_gain = -1
        best_triples = 0
        
        # Generate candidates
        # 1. Points on lines defined by pairs of existing points
        # 2. Random points in the bounding box
        
        candidates = set()
        
        # Add points on lines
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                # Extend line in both directions
                # Step size 1 is too fine, try steps that keep us in bounds
                # We want integer points.
                # Line equation: x = x1 + t*dx, y = y1 + t*dy
                # We need to normalize dx, dy to coprime to step by 1 unit of the line vector
                def gcd(a, b):
                    while b:
                        a, b = b, a % b
                    return abs(a)
                
                g = gcd(dx, dy)
                if g == 0: continue
                step_x = dx // g
                step_y = dy // g
                
                # Search t from -20 to 20
                for t in range(-20, 21):
                    if t == 0 or t == 1: # Skip existing points i and j roughly
                        # Actually t=0 is p1, t=1 is p2.
                        # We want other points.
                        pass
                    nx = x1 + t * step_x
                    ny = y1 + t * step_y
                    if -100 <= nx <= 100 and -100 <= ny <= 100:
                        candidates.add((nx, ny))
        
        # Add some random candidates to escape local minima
        for _ in range(50):
            nx = random.randint(-100, 100)
            ny = random.randint(-100, 100)
            candidates.add((nx, ny))
            
        # Evaluate candidates
        for cand in candidates:
            if cand in points:
                continue
                
            new_points = points + [list(cand)]
            
            # Check 4-collinear
            if not _check_no_four_collinear(new_points):
                continue
                
            # Count triples
            triples = _count_triples_fast(new_points)
            
            if triples > best_gain:
                best_gain = triples
                best_candidate = cand
                
        if best_candidate is not None:
            points.append(list(best_candidate))
        else:
            # Fallback: add random valid point
            while True:
                nx = random.randint(-100, 100)
                ny = random.randint(-100, 100)
                cand = [nx, ny]
                if cand in points:
                    continue
                new_points = points + [cand]
                if _check_no_four_collinear(new_points):
                    points.append(cand)
                    break
            
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_points = None
    best_score = float('inf')
    
    # Run multiple greedy constructions with different random seeds
    for seed in range(10):
        if time.time() > deadline - 10:
            break
            
        random.seed(seed)
        time_limit = 10.0
        pts = _greedy_construct(time_limit)
        
        if len(pts) != 20:
            continue
            
        if not _check_no_four_collinear(pts):
            continue
            
        score = _max_independent_set(pts)
        if score < best_score:
            best_score = score
            best_points = pts
            print(f"Seed {seed}: Score {score}, Triples {_count_triples_fast(pts)}", file=sys.stderr)
            
    # If greedy failed, fallback to random valid
    if best_points is None:
        random.seed(123)
        pts = []
        while len(pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            p = [x, y]
            if p not in pts:
                temp_pts = pts + [p]
                if _check_no_four_collinear(temp_pts):
                    pts.append(p)
        best_points = pts
        best_score = _max_independent_set(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_max_independent_set(pts)}")
    print(f"Valid: {_check_no_four_collinear(pts)}")
    print(f"Triples: {_count_triples_fast(pts)}")
