"""
Strategy: Model the problem as a Constraint Satisfaction Problem (CSP).
We hypothesize that the optimal configuration (6 ordinary lines) for 12 points
can be constructed by partitioning the 12 points into 4 disjoint collinear triples (pencils).
This creates 4 "main" lines with 3 points each.
We use a targeted backtracking search with exact rational arithmetic (fractions.Fraction)
to determine the unique intersection points of these lines.

The structure is:
- 4 lines L1, L2, L3, L4.
- Each Li contains 3 points.
- The points are the 12 distinct intersection points of pairs of lines, plus potentially
  other points if the intersection graph doesn't cover all 12.
  
Actually, a simpler and more robust structural approach for 12 points with 4 triples:
Consider the 4 lines. If they are in general position (no two parallel, no three concurrent),
they form 6 intersection points. But we need 12 points.
So we must place the 3 points on each line such that they are distinct and we minimize
ordinary lines (lines with exactly 2 points).

Let's try a specific known configuration or search for one.
The "Hesse configuration" has 9 points.
For 12 points, a common structure is a 3x4 grid (12 points).
In a 3x4 grid:
Points: (0,0)..(3,0), (0,1)..(3,1), (0,2)..(3,2).
Horizontal lines: 3 lines with 4 points. (0 ordinary)
Vertical lines: 4 lines with 3 points. (0 ordinary)
Diagonals:
Slope 1: (0,0)-(1,1)-(2,2) [3 pts], (1,0)-(2,1)-(3,2) [3 pts].
Slope -1: (0,2)-(1,1)-(2,0) [3 pts], (1,2)-(2,1)-(3,0) [3 pts].
Other slopes:
(0,0)-(3,1) slope 1/3. (1,0)-(4,1) no.
(0,0)-(2,1) slope 1/2. (1,0)-(3,1) [2 pts? No, (1,0) and (3,1) are on line y=0.5x-0.5. 
Check (0,0) and (2,1): y=0.5x. (1,0) not on it. (3,1) not on it.
So (0,0)-(2,1) is an ordinary line? 
Let's count ordinary lines for 3x4 grid.
Total pairs: C(12,2) = 66.
Lines with >=3 points:
3 horizontal (4 pts each): 3 lines.
4 vertical (3 pts each): 4 lines.
2 diagonal slope 1 (3 pts each): 2 lines.
2 diagonal slope -1 (3 pts each): 2 lines.
Any others?
Slope 1/2: (0,0)-(2,1) [2], (1,0)-(3,1) [2]. 
Wait, (0,0) and (2,1) define a line. Are there other grid points on it?
x=0,y=0. x=2,y=1. x=4,y=2 (out of range x<=3).
So (0,0)-(2,1) is a line with 2 points. Ordinary.
Similarly (1,0)-(3,1) is ordinary.
Slope 2: (0,0)-(1,2) [2], (1,0)-(2,2) [2].
Slope 1/3: (0,0)-(3,1) [2], (0,1)-(3,2) [2].
Slope -1/2: (0,2)-(2,1) [2], (1,2)-(3,1) [2].
Slope -2: (0,2)-(1,0) [2], (1,2)-(2,0) [2].
Slope -1/3: (0,1)-(3,0) [2], (0,2)-(3,1) [2].

Let's list all lines with exactly 2 points in 3x4 grid.
We can write a quick check in the code to verify the count for the 3x4 grid.
If the count is 6, we are done. If not, we search for a better configuration.

The prompt says: "Over the reals the minimum for 12 points is 6; whether integer (rational) coordinates reach 6 is not documented".
This suggests 6 is the target.
The 3x4 grid is a strong candidate. Let's calculate its ordinary lines count in the code.
If it's not 6, we will use a local search starting from the 3x4 grid to minimize the count.
"""

import time
import math
import random
from fractions import Fraction
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140  # Safety margin


def _line_key(p1, p2):
    """Canonical key for line through p1=(x1,y1) and p2=(x2,y2) using integers."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return (0, 0, x1)
    g = math.gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0:
        dx = -dx
        dy = -dy
    c = dy * x1 - dx * y1
    return (dx, dy, c)


def count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            line_counts[key] = line_counts.get(key, 0) + 1
    return sum(1 for c in line_counts.values() if c == 1)


def _is_valid(points):
    """Check that points are 12 distinct integer points with |x|,|y| <= 1e9, not all collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for p in points:
        if len(p) != 2:
            return False
        x, y = p
        if not (isinstance(x, int) and isinstance(y, int)):
            return False
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    if len(seen) < 3:
        return False
    p0, p1, p2 = points[0], points[1], points[2]
    if (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]) != 0:
        return True
    for i in range(3, 12):
        pi = points[i]
        if (p1[0] - p0[0]) * (pi[1] - p0[1]) - (p1[1] - p0[1]) * (pi[0] - p0[0]) != 0:
            return True
    return False


def _grid_3x4():
    """Generate the 3x4 grid configuration."""
    pts = []
    for x in range(4):
        for y in range(3):
            pts.append([x, y])
    return pts


def _local_search(points, max_iters=50000):
    """
    Perform a local search to minimize ordinary lines.
    Strategy: Randomly pick a point and move it to a position that lies on a line
    defined by two other points (to increase collinearity) or a random position.
    """
    current = [p[:] for p in points]
    current_count = count_ordinary_lines(current)
    best = [p[:] for p in current]
    best_count = current_count
    
    # Pre-calculate line keys for all pairs to speed up evaluation?
    # No, count_ordinary_lines is O(N^2) = 144 ops. 50k iters = 7M ops. Fast enough.
    
    for _ in range(max_iters):
        if time.time() > _DEADLINE - 5:
            break
            
        idx = random.randint(0, 11)
        old_pt = current[idx][:]
        
        # Strategy 1: Move to a point on a line defined by two other points
        # Pick two other points i, j
        i, j = random.sample([k for k in range(12) if k != idx], 2)
        p1 = current[i]
        p2 = current[j]
        
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        g = math.gcd(abs(dx), abs(dy))
        if g == 0:
            continue
        dx //= g
        dy //= g
        
        # Parametric point: p1 + t * (dx, dy)
        # Choose t such that the point is new and within bounds
        # Try small t values
        t_candidates = random.sample(range(-20, 21), 10)
        moved = False
        for t in t_candidates:
            if t == 0:
                continue
            nx = p1[0] + t * dx
            ny = p1[1] + t * dy
            if abs(nx) > 10**9 or abs(ny) > 10**9:
                continue
            new_pt = [nx, ny]
            # Check if new_pt is already in current (excluding idx)
            # Since we are replacing current[idx], we check against others
            is_dup = False
            for k in range(12):
                if k == idx:
                    continue
                if current[k] == new_pt:
                    is_dup = True
                    break
            if is_dup:
                continue
            
            current[idx] = new_pt
            new_count = count_ordinary_lines(current)
            
            if new_count < current_count:
                current_count = new_count
                if new_count < best_count:
                    best_count = new_count
                    best = [p[:] for p in current]
                moved = True
                break
            else:
                # Revert if not improved (or accept with prob for simulated annealing? 
                # Let's stick to strict improvement for simplicity and speed)
                current[idx] = old_pt
                moved = True
                break
        
        if not moved:
            # Strategy 2: Random small perturbation
            nx = old_pt[0] + random.randint(-5, 5)
            ny = old_pt[1] + random.randint(-5, 5)
            if abs(nx) > 10**9 or abs(ny) > 10**9:
                continue
            new_pt = [nx, ny]
            is_dup = False
            for k in range(12):
                if k == idx:
                    continue
                if current[k] == new_pt:
                    is_dup = True
                    break
            if is_dup:
                continue
            
            current[idx] = new_pt
            new_count = count_ordinary_lines(current)
            if new_count < current_count:
                current_count = new_count
                if new_count < best_count:
                    best_count = new_count
                    best = [p[:] for p in current]
            else:
                current[idx] = old_pt

    return best, best_count


def construct():
    """
    Find 12 integer points minimizing ordinary lines.
    1. Start with 3x4 grid.
    2. Evaluate ordinary lines.
    3. If not 6, perform local search.
    """
    # Initial candidate: 3x4 grid
    pts = _grid_3x4()
    count = count_ordinary_lines(pts)
    
    # If we already hit the theoretical minimum (6), return immediately
    if count == 6:
        return pts
    
    # Otherwise, try to improve via local search
    # We might also try other structured seeds if local search fails to improve
    # But 3x4 is a very dense, high-symmetry seed.
    
    best_pts, best_count = _local_search(pts, max_iters=100000)
    
    # Verify validity
    if not _is_valid(best_pts):
        # Fallback to a known valid configuration
        # e.g. 3x3 grid + 3 points far away
        pts = _grid_3x4()
        best_pts = pts
        best_count = count_ordinary_lines(pts)

    return best_pts

# EVOLVE-BLOCK-END
