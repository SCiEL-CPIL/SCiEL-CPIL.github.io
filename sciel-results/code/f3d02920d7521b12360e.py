"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def _line_key(i, pts):
    """Compute line keys for point i against all other points."""
    x1, y1 = pts[i]
    keys = []
    for j in range(len(pts)):
        if j == i:
            continue
        x2, y2 = pts[j]
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        g = gcd(gcd(abs(a), abs(b)), abs(c))
        if g > 0:
            a //= g
            b //= g
            c //= g
        if a < 0 or (a == 0 and b < 0):
            a, b, c = -a, -b, -c
        keys.append((j, (a, b, c)))
    return keys

def _count_ordinary(pts):
    """Count lines containing exactly 2 of the given points."""
    n = len(pts)
    if n < 2:
        return 0
    line_map = {}
    for i in range(n):
        for j, key in _line_key(i, pts):
            if j > i:
                if key not in line_map:
                    line_map[key] = set()
                line_map[key].add(i)
                line_map[key].add(j)
    return sum(1 for v in line_map.values() if len(v) == 2)

def _is_valid(pts):
    """Check 12 distinct integer points, |x|,|y| <= 1e9, not all collinear."""
    if len(pts) != 12:
        return False
    seen = set()
    for p in pts:
        if not isinstance(p[0], int) or not isinstance(p[1], int):
            return False
        if abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
            return False
        key = (p[0], p[1])
        if key in seen:
            return False
        seen.add(key)
    if len(pts) < 3:
        return True
    x0, y0 = pts[0]
    x1, y1 = pts[1]
    for i in range(2, len(pts)):
        xi, yi = pts[i]
        if (xi - x0) * (y1 - y0) != (yi - y0) * (x1 - x0):
            return True
    return False

def _generate_hesse_plus_3():
    """
    Generate the Hesse configuration (9 points, 0 ordinary lines) 
    and add 3 points on a line far away.
    
    The Hesse configuration is a 3x3 grid where each row, column, and 
    'diagonal' (in the affine plane sense, specific lines) contains 3 points.
    Actually, the standard 3x3 grid has 12 lines of 3 points.
    Number of ordinary lines in a 3x3 grid:
    Total pairs = C(9,2) = 36.
    Lines with 3 points: 3 rows + 3 cols + 3 "diagonals" (if we define them correctly) = 9 lines.
    Each such line accounts for 3 pairs.
    9 * 3 = 27 pairs are covered by non-ordinary lines.
    Remaining pairs = 36 - 27 = 9.
    Each ordinary line has 1 pair.
    So a generic 3x3 grid has 9 ordinary lines?
    
    Wait, the Hesse configuration is NOT a simple 3x3 grid of integers like (0,0)..(2,2).
    The Hesse configuration is defined over a field where 1+1+1=0 is not true, or specifically uses cube roots of unity.
    Over reals/integers, we cannot form a perfect Hesse configuration (which requires complex numbers or finite fields).
    However, we can approximate it or use a known integer configuration with few ordinary lines.
    
    Known result: For n=9, the minimum ordinary lines is 0? No, Sylvester-Gallai theorem says if not all collinear, there is at least one ordinary line.
    So min ordinary lines for 9 points is 1? No, the theorem says at least one.
    Can we have 1 ordinary line for 9 points?
    The 3x3 grid has 9 ordinary lines.
    
    Let's look at the task again: "Over the reals the minimum for 12 points is 6".
    This implies there exists a configuration of 12 points with 6 ordinary lines.
    One such configuration is the "Hesse configuration" plus 3 points?
    Or perhaps a configuration based on the roots of a cubic.
    
    A known construction for few ordinary lines uses the roots of a cubic polynomial.
    Let the points be (x, x^3) for x in S.
    If we choose S carefully, we can create many collinearities.
    
    However, a simpler structural change is to use the specific integer configuration that is known to be optimal or near-optimal.
    Since I don't have the exact coordinates memorized for the 12-point 6-ordinary-line configuration, I will implement a search that is biased towards algebraic structures.
    
    BUT, the directive says "Make a STRUCTURAL change".
    The current program does random search.
    I will implement a **Simulated Annealing** with a specific cooling schedule and a move set that favors creating collinearities (e.g., moving a point to lie on a line defined by two others).
    
    Actually, let's try a different structural approach: **Constraint Satisfaction / Backtracking**.
    We want to minimize ordinary lines.
    An ordinary line is a line with exactly 2 points.
    We want to maximize the number of "rich" lines (lines with >= 3 points).
    
    Let's try a specific known configuration for 12 points.
    The configuration of 12 points with 6 ordinary lines is often related to the affine plane of order 3 (9 points) plus 3 more?
    Or the complete graph K4?
    
    Let's use a heuristic search that explicitly tries to form lines of 3.
    Strategy:
    1. Start with a 3x3 grid (9 points).
    2. Add 3 points.
    3. Use local search to minimize ordinary lines.
    
    To make it structurally different from the seed's "random restart local search", I will use **Differential Evolution** or **Simulated Annealing** with a specific neighborhood function that prioritizes collinearity.
    
    Let's implement Simulated Annealing.
    """
    pass

def construct():
    deadline = time.time() + 145
    
    # Candidate 1: 3x3 Grid + 3 points on a line at infinity (approximated by large coordinates)
    # The 3x3 grid {(i,j) for i in 0..2, j in 0..2} has 9 ordinary lines.
    # If we add 3 points on a line L, and L doesn't intersect the grid in a way that creates more ordinary lines...
    
    # Let's try a specific algebraic construction:
    # Points on the curve y = x^3.
    # If we take roots of a polynomial, we get collinearities.
    
    # However, the most reliable "structural" change is to use a **Global Optimization** algorithm 
    # that is not local search. Let's use **Randomized Hill Climbing with Large Jumps** 
    # but initialized with **Algebraic Seeds**.
    
    # Seed 1: 3x3 Grid
    pts1 = [[i, j] for i in range(3) for j in range(3)]
    # Add 3 points far away on a line
    # Line y = 1000. Points: (0, 1000), (1, 1000), (2, 1000)
    pts1 += [[0, 1000], [1, 1000], [2, 1000]]
    
    # Seed 2: 2x6 Grid
    pts2 = [[i, j] for i in range(2) for j in range(6)]
    
    # Seed 3: Triangle with midpoints and centroid?
    # (0,0), (6,0), (0,6), (3,0), (0,3), (3,3), (6,3), (3,6) ...
    
    best_pts = None
    best_score = float('inf')
    
    seeds = [pts1, pts2]
    
    for seed in seeds:
        if _is_valid(seed):
            score = _count_ordinary(seed)
            if score < best_score:
                best_score = score
                best_pts = [list(p) for p in seed]
                
    if best_score <= 6:
        return [list(p) for p in best_pts]
        
    # Simulated Annealing
    import random
    random.seed(42)
    
    current_pts = [list(p) for p in best_pts]
    current_score = best_score
    
    T = 100.0
    cooling = 0.99
    
    while time.time() < deadline:
        if T < 1.0:
            break
            
        # Pick a random point to move
        idx = random.randint(0, 11)
        orig_x, orig_y = current_pts[idx]
        
        # Move strategy: 
        # 1. Small random jump
        # 2. Jump to a line defined by two other points (to create a 3-point line)
        # 3. Jump to a line defined by two other points and check if it breaks other lines
        
        move_type = random.choice(['small', 'collinear', 'large'])
        
        new_x, new_y = orig_x, orig_y
        
        if move_type == 'small':
            new_x = orig_x + random.randint(-5, 5)
            new_y = orig_y + random.randint(-5, 5)
        elif move_type == 'large':
            new_x = random.randint(-100, 100)
            new_y = random.randint(-100, 100)
        elif move_type == 'collinear':
            # Pick two other points i, j
            others = [k for k in range(12) if k != idx]
            i, j = random.sample(others, 2)
            x1, y1 = current_pts[i]
            x2, y2 = current_pts[j]
            
            # Line through (x1,y1) and (x2,y2)
            # Parametric form: P = P1 + t * (P2 - P1)
            # We want an integer point.
            # If we pick t as an integer, we get an integer point.
            # But we might land on an existing point.
            # Let's pick t from a small set of integers not 0, 1.
            t_candidates = list(range(-10, 11))
            random.shuffle(t_candidates)
            
            found_move = False
            for t in t_candidates:
                if t == 0 or t == 1:
                    continue
                cand_x = x1 + t * (x2 - x1)
                cand_y = y1 + t * (y2 - y1)
                if abs(cand_x) > 10**9 or abs(cand_y) > 10**9:
                    continue
                # Check if it's distinct from all points
                is_distinct = True
                for p in current_pts:
                    if p[0] == cand_x and p[1] == cand_y:
                        is_distinct = False
                        break
                if is_distinct:
                    new_x, new_y = cand_x, cand_y
                    found_move = True
                    break
            
            if not found_move:
                # Fallback to small move
                new_x = orig_x + random.randint(-5, 5)
                new_y = orig_y + random.randint(-5, 5)
                
        new_pts = [list(p) for p in current_pts]
        new_pts[idx] = [new_x, new_y]
        
        if not _is_valid(new_pts):
            continue
            
        new_score = _count_ordinary(new_pts)
        
        # Acceptance criterion
        if new_score < current_score:
            current_pts = new_pts
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_pts = [list(p) for p in current_pts]
                if best_score <= 6:
                    return [list(p) for p in best_pts]
        else:
            # Metropolis criterion
            delta = new_score - current_score
            if delta == 0:
                # Accept equal with some probability to escape plateaus
                if random.random() < 0.5:
                    current_pts = new_pts
            else:
                prob = 1.0 / (1.0 + delta) # Simplified acceptance
                # Standard SA: exp(-delta/T)
                import math
                try:
                    prob = math.exp(-delta / T)
                except:
                    prob = 0.0
                if random.random() < prob:
                    current_pts = new_pts
                    current_score = new_score
                    
        T *= cooling
        
    return [list(p) for p in best_pts]
# EVOLVE-BLOCK-END
