"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np

def count_ordinary_lines(points):
    """
    Count the number of lines containing exactly 2 of the points.
    points: list of (x, y) tuples.
    """
    n = len(points)
    if n < 3:
        return 0
    
    # Store lines as sets of indices
    # To avoid duplicates, we can store the line by its canonical form (slope, intercept) or just use sets of indices.
    # A better way: iterate over all pairs, determine the line, and count how many points are on it.
    # Then count lines with exactly 2 points.
    
    # Since n=12 is small, we can be simple.
    # For each pair, find the line.
    # Use a dictionary: key = canonical line representation, value = count of points.
    # Canonical representation:
    # For a line ax + by + c = 0, we can normalize (a,b,c) such that gcd(a,b,c)=1 and a>0 or (a==0 and b>0).
    # If a=0, b>0. If a!=0, a>0.
    
    lines = {}
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            # Line equation: (y2-y1)x - (x2-x1)y + (x2-y2-x1) = 0 ? No.
            # A = y1 - y2
            # B = x2 - x1
            # C = x1*y2 - x2*y1
            
            A = y1 - y2
            B = x2 - x1
            C = x1*y2 - x2*y1
            
            # Normalize
            # Find GCD of A, B, C
            import math
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g != 0:
                A //= g
                B //= g
                C //= g
            
            # Ensure canonical sign
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            
            if key not in lines:
                lines[key] = set([i, j])
            else:
                lines[key].add(i)
                lines[key].add(j)
                
    # Now count lines with exactly 2 points
    ordinary_count = 0
    for key, pts in lines.items():
        if len(pts) == 2:
            ordinary_count += 1
            
    return ordinary_count

def construct():
    # Try to find a configuration with < 11 ordinary lines using a small search space.
    # We search in the box [-5, 5] x [-5, 5].
    # Number of points in box: 11*11 = 121.
    # We need to choose 12.
    # This is too many to brute force all combinations.
    # Instead, we can try a few specific structured configurations.
    
    # Config 1: 11+1
    pts_11_1 = [(i, 0) for i in range(11)] + [(0, 1)]
    score_11_1 = count_ordinary_lines(pts_11_1)
    
    # Config 2: 3x4 Grid
    pts_grid = [(x, y) for x in range(4) for y in range(3)]
    score_grid = count_ordinary_lines(pts_grid)
    
    # Config 3: "Star" / Symmetric
    # Center (0,0), 4 on axes, 4 on diagonals?
    # (0,0)
    # (1,0), (-1,0), (2,0), (-2,0)
    # (0,1), (0,-1), (0,2), (0,-2)
    # (1,1), (-1,-1), (1,-1), (-1,1)
    pts_star = [(0,0), (1,0), (-1,0), (2,0), (-2,0), 
                (0,1), (0,-1), (0,2), (0,-2),
                (1,1), (-1,-1), (1,-1), (-1,1)]
    score_star = count_ordinary_lines(pts_star)
    
    # Config 4: Random search for better config?
    # Given time constraints, we just return the best of the known ones.
    # If we want to be more aggressive, we can do a local search starting from the best known.
    
    best_score = min(score_11_1, score_grid, score_star)
    best_pts = pts_11_1
    if score_11_1 == best_score:
        best_pts = pts_11_1
    elif score_grid == best_score:
        best_pts = pts_grid
    else:
        best_pts = pts_star
        
    # Convert to list of lists
    return [list(p) for p in best_pts]
# EVOLVE-BLOCK-END
