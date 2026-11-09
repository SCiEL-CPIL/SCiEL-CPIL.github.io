# EVOLVE-BLOCK-START
import numpy as np
import time


def _gcd(a, b, c):
    from math import gcd
    g = gcd(abs(a), abs(b))
    g = gcd(g, abs(c))
    return g if g != 0 else 1


def _normalize_line(A, B, C):
    g = _gcd(A, B, C)
    A //= g
    B //= g
    C //= g
    # make first nonzero entry positive
    for v in (A, B, C):
        if v != 0:
            if v < 0:
                A, B, C = -A, -B, -C
            break
    return (A, B, C)


def _count_incidents(points, lines):
    # points: (n,2) int64, lines: (m,3) int64
    # Compute A*x + B*y + C for all pairs
    n = points.shape[0]
    m = lines.shape[0]
    # Use broadcasting
    px = points[:, 0:1]  # (n,1)
    py = points[:, 1:2]  # (n,1)
    A = lines[0:1, 0]  # (1,1)
    B = lines[0:1, 1]  # (1,1)
    C = lines[0:1, 2]  # (1,1)
    # We need to compute for each (i,j): lines[j,0]*points[i,0] + lines[j,1]*points[i,1] + lines[j,2]
    # Vectorized: (n,1) * (1,m) 
    vals = lines[:, 0][None, :] * points[:, 0:1] + lines[:, 1][None, :] * points[:, 1:2] + lines[:, 2][None, :]
    return int(np.sum(vals == 0))


def _construct_bundle_hybrid():
    """
    Construct a hybrid configuration that aims for high incidences.
    
    Strategy: Use a "near-bundle" structure.
    We have 54 points and 27 lines.
    
    Consider a set of lines passing through a common point P0.
    Let's say we have `L` lines through P0.
    Each of these lines can hold many points.
    If we place `k` points on each of `L` lines, plus P0, total points = L*k + 1.
    Incidences: P0 is on all L lines (L incidences).
    The other L*k points: each is on exactly 1 of the L lines (unless it's an intersection, which we avoid by choosing distinct lines and placing points in disjoint segments).
    So incidences from these = L*k.
    Total = L*k + L = L*(k+1).
    Constraints: L*k + 1 <= 54, L <= 27.
    We can add more lines.
    
    Example:
    L = 27 lines through (0,0).
    Points: (0,0) plus points on these lines.
    If we put k=1 point on each line (other than origin), total points = 28.
    Incidences = 27 (origin) + 27 (others) = 54.
    Remaining points: 54 - 28 = 26.
    Remaining lines: 27 - 27 = 0.
    This gives 54 incidences.
    
    Can we do better?
    Grid gave 81.
    
    Let's try a different construction.
    Maybe two clusters?
    
    Actually, the grid construction is known to be optimal for asymptotic bounds.
    However, for small n,m, specific configurations might beat it.
    
    Let's look at the "dual" problem or other known extremal configurations.
    The Szemeredi-Trotter bound is I <= O(n m^{2/3} + n^{2/3} m + n + m).
    For n=54, m=27:
    54 * 27^{2/3} + 54^{2/3} * 27 + 54 + 27
    27^{2/3} = 9.
    54^{2/3} = (54)^{0.666} ~ 14.4.
    54 * 9 = 486.
    14.4 * 27 = 388.8.
    Sum ~ 486 + 389 + 81 = 956.
    So the bound is loose for small numbers. 81 is much lower than the upper bound.
    
    Can we get significantly more than 81?
    
    Consider a "fan" of lines with many points on each.
    Let's try to maximize incidences by having points lie on multiple lines.
    
    Construction:
    Let's use a grid but denser?
    Current: 3x18 grid. Lines y = ax + b.
    
    What if we use a smaller grid but with more lines?
    No, number of points is fixed at 54.
    
    Let's try a "lattice" of points and lines that are not just y=ax+b.
    
    Alternative:
    Take a 6x9 grid of points (54 points).
    Points: (i, j) for i in 0..5, j in 0..8.
    Lines:
    1. Vertical lines: x = i (6 lines). Each has 9 points. Incidences: 6*9 = 54.
    2. Horizontal lines: y = j (9 lines). Each has 6 points. Incidences: 9*6 = 54.
    Total lines so far: 15. Total incidences: 108.
    We have 27 - 15 = 12 lines left.
    We can add diagonal lines.
    Diagonal lines x - y = c.
    Range of c: x-y from -8 to 5.
    If we add diagonals, each diagonal passes through several points.
    For a 6x9 grid, a diagonal x-y=c passes through min(6, 9, ...) points.
    Specifically, length of diagonal c is:
    If c >= 0: x ranges from c to 5, y = x-c. Need y in 0..8.
    x-c >= 0 => x >= c.
    x-c <= 8 => x <= 8+c.
    x in 0..5.
    So x from c to 5. Number of points = 5 - c + 1 = 6 - c.
    If c < 0: let c = -d. y - x = d.
    y = x + d.
    x in 0..5. y in 0..8.
    x+d <= 8 => x <= 8-d.
    x+d >= 0 => x >= -d (always true if x>=0, d>0).
    So x from 0 to min(5, 8-d).
    If d <= 3: x from 0 to 5, 6 points.
    If d = 4: x from 0 to 4, 5 points.
    If d = 5: x from 0 to 3, 4 points.
    If d = 6: x from 0 to 2, 3 points.
    If d = 7: x from 0 to 1, 2 points.
    If d = 8: x from 0 to 0, 1 point.
    
    We have 12 lines to add. We should pick the diagonals with the most points.
    c = 0: 6 pts.
    c = 1: 5 pts.
    c = 2: 4 pts.
    c = 3: 3 pts.
    c = 4: 2 pts.
    c = 5: 1 pt.
    c = -1 (d=1): 6 pts.
    c = -2 (d=2): 6 pts.
    c = -3 (d=3): 6 pts.
    c = -4 (d=4): 5 pts.
    c = -5 (d=5): 4 pts.
    c = -6 (d=6): 3 pts.
    c = -7 (d=7): 2 pts.
    c = -8 (d=8): 1 pt.
    
    Sorted by length:
    6: c=0, -1, -2, -3 (4 lines)
    5: c=1, -4 (2 lines)
    4: c=2, -5 (2 lines)
    3: c=3, -6 (2 lines)
    2: c=4, -7 (2 lines)
    1: c=5, -8 (2 lines)
    
    We need 12 lines.
    Take all 4 of length 6: 24 pts.
    Take all 2 of length 5: 10 pts.
    Take all 2 of length 4: 8 pts.
    Take all 2 of length 3: 6 pts.
    Total added: 4+2+2+2 = 10 lines.
    Incidences added: 24+10+8+6 = 48.
    We have 2 lines left.
    Take 2 of length 2: 2 lines, 4 pts.
    Total added lines: 12.
    Total added incidences: 48 + 4 = 52.
    Total incidences: 108 (grid) + 52 (diagonals) = 160.
    
    This is much higher than 81!
    
    Let's verify the points are unique.
    Grid points: (i, j) for i in 0..5, j in 0..8. All unique.
    Lines:
    Verticals: x=0, x=1, ..., x=5. (6 lines)
    Horizontals: y=0, y=1, ..., y=8. (9 lines)
    Diagonals: x-y=c for selected c.
    
    Check for duplicate lines:
    Verticals are distinct.
    Horizontals are distinct.
    Diagonals are distinct (different c).
    No vertical is a horizontal or diagonal.
    No horizontal is a diagonal.
    
    Check for integer coordinates: Yes.
    
    Let's implement this.
    
    Points:
    np.meshgrid(np.arange(6), np.arange(9), indexing='ij')
    
    Lines:
    Verticals: (1, 0, -i) for i in 0..5.
    Horizontals: (0, 1, -j) for j in 0..8.
    Diagonals: (1, -1, c) for selected c.
    
    Selected c:
    c in [0, -1, -2, -3, 1, -4, 2, -5, 3, -6, 4, -7]
    
    Let's double check the count.
    6 + 9 + 12 = 27 lines.
    54 points.
    
    Incidences:
    Verticals: 6 * 9 = 54.
    Horizontals: 9 * 6 = 54.
    Diagonals:
    c=0: (0,0),(1,1)...(5,5) -> 6 pts.
    c=-1: (0,1),(1,2)...(5,6) -> 6 pts.
    c=-2: (0,2)...(5,7) -> 6 pts.
    c=-3: (0,3)...(5,8) -> 6 pts.
    c=1: (1,0),(2,1)...(5,4) -> 5 pts.
    c=-4: (0,4)...(4,8) -> 5 pts.
    c=2: (2,0)...(5,3) -> 4 pts.
    c=-5: (0,5)...(3,8) -> 4 pts.
    c=3: (3,0)...(5,2) -> 3 pts.
    c=-6: (0,6)...(2,8) -> 3 pts.
    c=4: (4,0),(5,1) -> 2 pts.
    c=-7: (0,7),(1,8) -> 2 pts.
    
    Sum: 6+6+6+6+5+5+4+4+3+3+2+2 = 52.
    Total = 54 + 54 + 52 = 160.
    
    This is a solid improvement over 81.
    Can we do even better?
    
    What if we add more diagonals?
    We have exactly 27 lines.
    Could we use different lines?
    
    What about the other diagonal direction? x+y=c.
    Let's see if mixing directions helps.
    
    But 160 is already > 2x the baseline.
    
    Let's check if we can swap some vertical/horizontal for more diagonals.
    Verticals give 9 incidences each.
    Horizontals give 6 incidences each.
    Best diagonals give 6 incidences each.
    So Verticals are very efficient.
    
    If we remove a horizontal (6 inc), we free 1 line slot.
    We can add a diagonal (max 6 inc).
    Net change: 0.
    So swapping Horizontal for Diagonal is neutral if diagonal is length 6.
    
    If we remove a vertical (9 inc), we can add a diagonal (max 6 inc).
    Net change: -3. Bad.
    
    So we should keep all verticals.
    We should keep as many horizontals as possible, but we are limited by total lines.
    6 verticals + 9 horizontals + 12 diagonals = 27 lines.
    
    Is there a configuration with more incidences?
    
    What if we use a non-rectangular grid?
    Or a different set of lines?
    
    Let's stick with this 160 incidence construction.
    
    One potential issue: "two rows describing the same line".
    Our lines are:
    (1, 0, -i)
    (0, 1, -j)
    (1, -1, c)
    
    Normalized:
    (1, 0, -i) -> gcd 1, first nonzero 1 > 0.
    (0, 1, -j) -> gcd 1, first nonzero 1 > 0.
    (1, -1, c) -> gcd 1, first nonzero 1 > 0.
    
    All distinct.
    
    Let's write the code.
    """
    points_x = np.arange(6)
    points_y = np.arange(9)
    px, py = np.meshgrid(points_x, points_y, indexing='ij')
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    lines = []
    # Verticals
    for i in range(6):
        lines.append((1, 0, -i))
    # Horizontals
    for j in range(9):
        lines.append((0, 1, -j))
    # Diagonals x - y = c
    # Selected c values
    c_vals = [0, -1, -2, -3, 1, -4, 2, -5, 3, -6, 4, -7]
    for c in c_vals:
        lines.append((1, -1, c))
        
    lines = np.array(lines, dtype=np.int64)
    
    return points, lines


def szemeredi_trotter_incidences():
    start_time = time.time()
    
    # Try the grid+diagonal construction
    points, lines = _construct_bundle_hybrid()
    
    # Validate
    if points.shape != (54, 2) or lines.shape != (27, 3):
        # Fallback to simple grid
        k = 3
        xs = np.arange(1, k + 1)
        ys = np.arange(1, 2 * k * k + 1)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        rows = []
        for a in range(1, k + 1):
            for b in range(1, k * k + 1):
                rows.append((a, -1, b))
        lines = np.array(rows, dtype=np.int64)
        
    # Check for NaN/Inf
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(lines)):
        # Fallback
        k = 3
        xs = np.arange(1, k + 1)
        ys = np.arange(1, 2 * k * k + 1)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        rows = []
        for a in range(1, k + 1):
            for b in range(1, k * k + 1):
                rows.append((a, -1, b))
        lines = np.array(rows, dtype=np.int64)
        
    # Check for repeated points
    # Use a set of tuples
    pt_set = set()
    for p in points:
        t = (int(p[0]), int(p[1]))
        if t in pt_set:
            # Fallback
            k = 3
            xs = np.arange(1, k + 1)
            ys = np.arange(1, 2 * k * k + 1)
            px, py = np.meshgrid(xs, ys, indexing="ij")
            points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
            rows = []
            for a in range(1, k + 1):
                for b in range(1, k * k + 1):
                    rows.append((a, -1, b))
            lines = np.array(rows, dtype=np.int64)
            break
        pt_set.add(t)
        
    # Check for invalid lines (A=B=0)
    for l in lines:
        if l[0] == 0 and l[1] == 0:
            # Fallback
            k = 3
            xs = np.arange(1, k + 1)
            ys = np.arange(1, 2 * k * k + 1)
            px, py = np.meshgrid(xs, ys, indexing="ij")
            points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
            rows = []
            for a in range(1, k + 1):
                for b in range(1, k * k + 1):
                    rows.append((a, -1, b))
            lines = np.array(rows, dtype=np.int64)
            break
            
    # Check for duplicate lines
    line_set = set()
    for l in lines:
        nl = _normalize_line(int(l[0]), int(l[1]), int(l[2]))
        if nl in line_set:
            # Fallback
            k = 3
            xs = np.arange(1, k + 1)
            ys = np.arange(1, 2 * k * k + 1)
            px, py = np.meshgrid(xs, ys, indexing="ij")
            points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
            rows = []
            for a in range(1, k + 1):
                for b in range(1, k * k + 1):
                    rows.append((a, -1, b))
            lines = np.array(rows, dtype=np.int64)
            break
        line_set.add(nl)
        
    return points, lines


# EVOLVE-BLOCK-END
