import numpy as np
import time
from math import gcd
import random

def normalize_line(A, B, C):
    """Normalize line coefficients to ensure uniqueness and integer validity."""
    if A == 0 and B == 0:
        return None
    
    # Ensure integers
    A = int(A)
    B = int(B)
    C = int(C)
    
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        g = 1
    A //= g
    B //= g
    C //= g
    
    # Make first non-zero entry positive
    if A < 0 or (A == 0 and B < 0):
        A = -A
        B = -B
        C = -C
        
    return (A, B, C)

def count_incidences(points, lines):
    """
    points: (n, 2) array
    lines: (m, 3) array (A, B, C)
    Returns number of (p, l) such that A*x + B*y + C == 0
    """
    if len(points) == 0 or len(lines) == 0:
        return 0
    P = points.astype(np.int64)
    L = lines.astype(np.int64)
    
    px = P[:, 0].reshape(-1, 1)
    py = P[:, 1].reshape(-1, 1)
    
    A = L[:, 0].reshape(1, -1)
    B = L[:, 1].reshape(1, -1)
    C = L[:, 2].reshape(1, -1)
    
    vals = A * px + B * py + C
    count = np.sum(vals == 0)
    return int(count)

def generate_triangular_grid_lines(n_lines=27, k=3):
    """
    Generate lines in 3 families (directions) to create a triangular lattice structure.
    We aim for roughly 9 lines in each direction.
    The lines are of the form:
    Family 1: y = m1*x + c1  -> -m1*x + y - c1 = 0  (A=-m1, B=1, C=-c1)
    Family 2: y = m2*x + c2  -> -m2*x + y - c2 = 0  (A=-m2, B=1, C=-c2)
    Family 3: x = c3         -> x - c3 = 0          (A=1, B=0, C=-c3)
    
    To maximize triple intersections, we choose slopes and intercepts such that
    intersections of F1 and F2 often lie on F3 lines.
    
    Let's use slopes m1=1, m2=-1, and vertical lines.
    Intersection of y=x+c1 and y=-x+c2 is x=(c2-c1)/2, y=(c1+c2)/2.
    For this to be on a vertical line x=c3, we need c3 = (c2-c1)/2.
    
    Let's try a specific construction:
    Lines:
    L1_i: y = x + i       (i in range)
    L2_j: y = -x + j      (j in range)
    L3_k: x = k           (k in range)
    
    Intersection of L1_i and L2_j:
    x + i = -x + j => 2x = j - i => x = (j-i)/2.
    y = x + i = (j-i)/2 + i = (j+i)/2.
    
    For the intersection to be an integer point, j-i must be even.
    If j-i is even, x is integer, y is integer.
    This point lies on vertical line x=k if k = (j-i)/2.
    
    So if we pick i, j such that j-i = 2k, the intersection is (k, (i+j)/2).
    
    Let's pick ranges for i, j, k to get 27 lines total.
    9 lines in each family.
    
    Let i in [0, 8]
    Let j in [0, 8]
    Let k in [0, 8]
    
    Total lines: 27.
    
    Let's check the number of triple intersections.
    A triple intersection occurs when L1_i, L2_j, L3_k meet at one point.
    This requires k = (j-i)/2.
    So j - i = 2k.
    Given i, k, j is determined: j = i + 2k.
    For j to be in [0, 8], we need 0 <= i + 2k <= 8.
    
    Let's count how many (i, j, k) tuples satisfy this.
    i in 0..8, k in 0..8.
    j = i + 2k.
    Conditions: 0 <= i, 0 <= k, 0 <= j <= 8.
    
    Sum over k:
    k=0: j=i. i in 0..8. (9 solutions)
    k=1: j=i+2. i in 0..6. (7 solutions)
    k=2: j=i+4. i in 0..4. (5 solutions)
    k=3: j=i+6. i in 0..2. (3 solutions)
    k=4: j=i+8. i in 0..0. (1 solution)
    k>=5: no solutions.
    
    Total triple intersections = 9+7+5+3+1 = 25.
    Each triple intersection is a point with 3 incidences.
    There are 25 such points.
    
    What about other points?
    Intersections of L1_i and L2_j where j-i is odd are not integer points, so they don't count for integer grid points.
    However, we are free to choose ANY 54 integer points. They don't have to be on the lines?
    No, the points must be integers. The lines are defined by integer coefficients.
    The incidence is A*x+B*y+C=0.
    
    If we pick the 25 triple points, we have 25 points with 3 incidences each = 75 incidences.
    We need 54 points. We have 29 more points to pick.
    These points should lie on as many lines as possible.
    The remaining points in the grid formed by these lines might have 2 incidences.
    
    Let's generate the lines and find the best 54 points from a large grid.
    """
    lines = []
    
    # Family 1: y = x + i  =>  -x + y - i = 0
    for i in range(9):
        lines.append((-1, 1, -i))
        
    # Family 2: y = -x + j  =>  x + y - j = 0
    for j in range(9):
        lines.append((1, 1, -j))
        
    # Family 3: x = k  =>  x - k = 0
    for k in range(9):
        lines.append((1, 0, -k))
        
    # Normalize and deduplicate
    unique_lines = []
    for l in lines:
        n_l = normalize_line(l[0], l[1], l[2])
        if n_l and n_l not in unique_lines:
            unique_lines.append(n_l)
            
    # We expect 27 unique lines
    return np.array(unique_lines, dtype=np.int64)

def find_best_points(lines, num_points=54, box_range=50):
    """
    Find the num_points distinct integer points in [-box_range, box_range]^2
    with the highest number of incidences with the given lines.
    """
    xs = np.arange(-box_range, box_range + 1)
    ys = np.arange(-box_range, box_range + 1)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    candidate_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    L = lines.astype(np.int64)
    px_c = candidate_points[:, 0].reshape(-1, 1)
    py_c = candidate_points[:, 1].reshape(-1, 1)
    
    A = L[:, 0].reshape(1, -1)
    B = L[:, 1].reshape(1, -1)
    C = L[:, 2].reshape(1, -1)
    
    vals = A * px_c + B * py_c + C
    counts = np.sum(vals == 0, axis=1)
    
    # Sort by count descending, then by coordinate for determinism
    indices = np.argsort(-counts, kind='stable')
    best_indices = indices[:num_points]
    
    return candidate_points[best_indices]

def szemeredi_trotter_incidences():
    start_time = time.time()
    
    # 1. Generate the specific line configuration
    lines = generate_triangular_grid_lines()
    
    # 2. Find the best points
    points = find_best_points(lines, num_points=54, box_range=50)
    
    # 3. Verify and return
    I = count_incidences(points, lines)
    
    # Fallback if something went wrong (e.g. not enough unique points found in box)
    if len(points) < 54:
        # Expand box
        points = find_best_points(lines, num_points=54, box_range=100)
        
    # Ensure points are unique
    unique_points, indices = np.unique(points, axis=0, return_index=True)
    if len(unique_points) < 54:
        # This shouldn't happen with box_range=100 and 27 lines
        # But just in case, fill with random
        while len(unique_points) < 54:
            p = np.array([random.randint(-100, 100), random.randint(-100, 100)], dtype=np.int64)
            if not np.any(np.all(unique_points == p, axis=1)):
                unique_points = np.vstack([unique_points, p])
        unique_points = unique_points[:54]
    else:
        unique_points = unique_points[indices]
        
    return unique_points, lines

# EVOLVE-BLOCK-END
