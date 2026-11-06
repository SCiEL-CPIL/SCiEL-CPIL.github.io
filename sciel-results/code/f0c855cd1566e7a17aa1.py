"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import Counter
import time

def _max_mult(pts):
    """Compute max multiplicity of positive doubled area for 30 points."""
    n = len(pts)
    pts = np.array(pts, dtype=np.int64)
    areas = []
    for i in range(n-2):
        pi = pts[i]
        for j in range(i+1, n-1):
            d1 = pts[j] - pi
            for k in range(j+1, n):
                d2 = pts[k] - pi
                a = abs(d1[0]*d2[1] - d1[1]*d2[0])
                if a > 0:
                    areas.append(a)
    if not areas:
        return 0
    c = Counter(areas)
    return max(c.values())

def construct():
    t0 = time.time()
    deadline = t0 + 75
    
    # Strong initial: 10x9 grid centered
    best_pts = [(x,y) for x in range(-5,5) for y in range(-4,5)]
    best_score = _max_mult(best_pts)
    
    # Try various grid sizes
    grids = [
        (10,9), (9,10), (12,8), (8,12), (15,7), (7,15),
        (18,6), (6,18), (20,5), (5,20), (11,8), (8,11),
        (14,7), (7,14), (16,6), (6,16), (25,4), (4,25),
        (30,3), (3,30), (24,4), (4,24), (22,4), (4,22),
        (21,4), (4,21), (20,4), (4,20), (19,4), (4,19),
        (18,4), (4,18), (17,4), (4,17), (16,4), (4,16),
    ]
    
    for (r,c) in grids:
        if time.time() > deadline:
            break
        # Center the grid
        x0 = -(c-1)//2
        y0 = -(r-1)//2
        pts = [(x0+i, y0+j) for i in range(c) for j in range(r)]
        if len(pts) == 30 and all(-30<=x<=30 and -30<=y<=30 for x,y in pts):
            s = _max_mult(pts)
            if s > best_score:
                best_score = s
                best_pts = pts
    
    # Local search: perturb points
    pts = list(best_pts)
    score = best_score
    rng = np.random.RandomState(42)
    
    for it in range(5000):
        if time.time() > deadline:
            break
        i = rng.randint(0, 30)
        x, y = pts[i]
        nx = x + rng.randint(-4, 5)
        ny = y + rng.randint(-4, 5)
        if -30 <= nx <= 30 and -30 <= ny <= 30 and (nx,ny) not in pts:
            new_pts = pts[:]
            new_pts[i] = (nx,ny)
            if len(set(new_pts)) == 30:
                ns = _max_mult(new_pts)
                if ns > score:
                    score = ns
                    pts = new_pts
    
    return pts

# EVOLVE-BLOCK-END
