# EVOLVE-BLOCK-START
import itertools
import math
import random

def is_perfect_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

def dist_sq(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy

def are_collinear(p1, p2, p3):
    dx1 = p2[0] - p1[0]
    dy1 = p2[1] - p1[1]
    dx2 = p3[0] - p1[0]
    dy2 = p3[1] - p1[1]
    return dx1 * dy2 - dy1 * dx2 == 0

def are_concyclic(p1, p2, p3, p4):
    def row(p):
        return [p[0]*p[0] + p[1]*p[1], p[0], p[1], 1]
    
    r1 = row(p1)
    r2 = row(p2)
    r3 = row(p3)
    r4 = row(p4)
    
    def det4(m):
        sign = 1
        total = 0
        for col in range(4):
            minor = []
            for i in range(1, 4):
                minor.append([m[i][j] for j in range(4) if j != col])
            def det3(mat):
                return (mat[0][0]*(mat[1][1]*mat[2][2]-mat[1][2]*mat[2][1])
                      - mat[0][1]*(mat[1][0]*mat[2][2]-mat[1][2]*mat[2][0])
                      + mat[0][2]*(mat[1][0]*mat[2][1]-mat[1][1]*mat[2][0]))
            total += sign * m[0][col] * det3(minor)
            sign = -sign
        return total
    
    return det4([r1, r2, r3, r4]) == 0

def construct():
    # Strategy: 
    # 1. Generate candidates in a larger grid.
    # 2. Filter candidates to those with many integer-distance neighbors (Core).
    # 3. Use Greedy with Dynamic Degree on the filtered set.
    
    bound = 300
    candidates = []
    for x in range(-bound, bound + 1):
        for y in range(-bound, bound + 1):
            candidates.append((x, y))
    
    n = len(candidates)
    
    # Pre-compute neighbor counts to filter
    # This is O(N^2) which is ~3.6e9 for 600x600. Too slow.
    # We need a faster filter or smaller grid.
    # Let's try a smaller grid but more attempts.
    # Or, use a random subset of candidates for the "score" estimation in Greedy.
    
    random.seed(42)
    best_set = []
    
    # Try to find 7 points. If not, keep 6.
    target_size = 7
    
    for attempt in range(100):
        # Random subset of candidates to keep search space manageable?
        # No, we need the full space to find the specific set.
        # Let's use a smaller grid: 200
        # 400x400 = 160,000 points.
        
        # Greedy with Dynamic Degree
        selected = []
        available = candidates[:]
        random.shuffle(available)
        
        # To speed up, we'll only consider a subset of 'available' for the dynamic score
        # Or we'll just use a standard greedy but with a larger grid and hope.
        
        # Actually, let's just use the current greedy but with a larger grid and more attempts.
        # The current code uses 500 attempts on 150 bound.
        # Let's try 100 attempts on 300 bound.
        
        # Optimization: Pre-check for integer distance is fast.
        
        for p in available:
            if len(selected) >= target_size:
                break
            
            compatible = True
            for q in selected:
                if not is_perfect_square(dist_sq(p, q)):
                    compatible = False
                    break
            
            if compatible:
                # Check collinear
                collinear_ok = True
                for i in range(len(selected) - 1):
                    for j in range(i + 1, len(selected)):
                        if are_collinear(selected[i], selected[j], p):
                            collinear_ok = False
                            break
                    if not collinear_ok:
                        break
                if not collinear_ok:
                    continue
                
                # Check concyclic
                concyclic_ok = True
                if len(selected) >= 3:
                    for i in range(len(selected) - 2):
                        for j in range(i + 1, len(selected) - 1):
                            for k in range(j + 1, len(selected)):
                                if are_concyclic(selected[i], selected[j], selected[k], p):
                                    concyclic_ok = False
                                    break
                            if not concyclic_ok:
                                break
                        if not concyclic_ok:
                            break
                if concyclic_ok:
                    selected.append(p)
        
        if len(selected) > len(best_set):
            best_set = selected
            if len(best_set) == target_size:
                break

    # Fallback to 6
    if len(best_set) < 6:
        points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
        # Add a 6th point if possible? The current code finds 6.
        # Let's return the best found.
        if not best_set:
             best_set = points
        else:
             # If we found 5, we might want to return 5? No, maximize.
             # But 6 is better.
             pass

    # Ensure we return at least 6 if we have a known 6-set
    if len(best_set) < 6:
        # Use the hardcoded 6-set
        # We need a valid 6-set. The one in the code is:
        # (0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63) is 5 points.
        # The code says "Fallback to 5-point configuration" but returns 5?
        # Wait, the current code returns 5 if it fails to find 6.
        # But the score is 1.0, which implies it found 6?
        # "Maximize the number of points."
        # If the seed score is 1.0, and it finds 6, then seed=6.
        
        # Let's just return the best_set found.
        pass

    points = best_set
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in points]}
# EVOLVE-BLOCK-END
