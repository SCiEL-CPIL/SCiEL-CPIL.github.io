"""Program using MODEL-AND-OPTIMIZE: local search + geometric verification for integral point sets.

Model:
- Parameterize as (D, list of points with rational x and y*sqrt(D)).
- Objective: maximize number of distinct points n (3..12) such that:
  * all pairwise squared distances are rational (equivalently integral after scaling for a fixed D)
  * no three points are collinear (exact determinant 0 check in affine 2-space)
  * no four points are concyclic (determinant-based cyclic condition using squared distances)
  * |numerators|, |denominators| <= 1e40, D squarefree <= 1e12
- For D=1, reduce to integer lattice with integer squared distances.

Optimization:
- Start from a known 5-point integral set.
- Perform a time-budgeted local search over a bounded integer grid to add a 6th point
  that (i) has integer squared distances to the existing 5 and (ii) preserves all
  collinearity/concyclicity constraints.
- If time remains, do a broader search up to n=8 on a slightly larger grid but keep
  constraints tight to stay within the wall-clock limit.
- Always return a valid configuration of maximum found n (starting from n=5).

Deadline:
- Use time.perf_counter(); stop loops before ~110s wall-clock, then finalize.
"""
import time
from math import isqrt
from itertools import combinations

TIME_LIMIT = 150.0
SAFETY = 30.0
DEADLINE = TIME_LIMIT - SAFETY

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _gcd(a, b):
    a, b = abs(int(a)), abs(int(b))
    while b:
        a, b = b, a % b
    return a

def _sqdist(p, q):
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return dx * dx + dy * dy

def _collinear3(A, B, C):
    # (B.x - A.x)*(C.y - A.y) - (B.y - A.y)*(C.x - A.x)
    return (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])

def _conic_coeff(A, B, C, D):
    # determinant
    # | x1 y1 x1^2+y1^2 1 |
    # | x2 y2 x2^2+y2^2 1 |
    # | x3 y3 x3^2+y3^2 1 |
    # | x4 y4 x4^2+y4^2 1 |
    (x1, y1) = A
    (x2, y2) = B
    (x3, y3) = C
    (x4, y4) = D
    s1 = x1*x1 + y1*y1
    s2 = x2*x2 + y2*y2
    s3 = x3*x3 + y3*y3
    s4 = x4*x4 + y4*y4
    # Expand determinant with column 4 elimination (use row1 as pivot)
    # D = x1*y2*(s3 - s4) - x1*y3*(s2 - s4) + x1*y4*(s2 - s3)
    #   - y1*x2*(s3 - s4) + y1*x3*(s2 - s4) - y1*x4*(s2 - s3)
    #   + (s2*y3*x4 - s2*y4*x3) - (s3*y2*x4 - s3*y4*x2) + (s4*y2*x3 - s4*y3*x2)
    d = (
        x1 * y2 * (s3 - s4) - x1 * y3 * (s2 - s4) + x1 * y4 * (s2 - s3)
        - y1 * x2 * (s3 - s4) + y1 * x3 * (s2 - s4) - y1 * x4 * (s2 - s3)
        + s2 * (y3 * x4 - y4 * x3)
        - s3 * (y2 * x4 - y4 * x2)
        + s4 * (y2 * x3 - y3 * x2)
    )
    return d

def _valid_set(pts):
    # Check distinct
    seen = set(pts)
    if len(seen) != len(pts):
        return False
    # Check all squared distances are perfect squares (D=1)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if not _is_square(_sqdist(pts[i], pts[j])):
                return False
    # No 3 collinear
    for a, b, c in combinations(pts, 3):
        if _collinear3(a, b, c) == 0:
            return False
    # No 4 concyclic
    for a, b, c, d in combinations(pts, 4):
        if _conic_coeff(a, b, c, d) == 0:
            return False
    return True

def _format_points(pts):
    # D=1 so y is rational y with den=1
    return {"D": 1, "points": [[[x, 1], [y, 1]] for (x, y) in pts]}

def _try_extend(current, deadline):
    # current: list of (x, y) integer points
    # Use circle-based candidate generation: for each candidate squared distance r2 to P0,
    # and using the linear constraint from P1, solve for possible integer (x,y).
    n = len(current)
    if n < 2:
        return current
    
    P0 = current[0]
    P1 = current[1]
    x0, y0 = P0
    x1, y1 = P1
    
    # For a new point P=(x,y) with integer distances to all current points:
    # (x-x0)^2 + (y-y0)^2 = r0^2  (perfect square)
    # (x-x1)^2 + (y-y1)^2 = r1^2  (perfect square)
    # Subtracting: 2*(x1-x0)*x + 2*(y1-y0)*y = (x0^2+y0^2 - x1^2-y1^2) + r1^2 - r0^2
    # This is a linear equation in x,y.
    
    dx01 = x1 - x0
    dy01 = y1 - y0
    # Linear constraint: 2*dx01*x + 2*dy01*y = C(r0^2, r1^2)
    # where C = (x0^2+y0^2 - x1^2-y1^2) + r1^2 - r0^2
    
    # Also need integer distances to P2..P_{n-1}
    
    # Strategy: enumerate r0^2 (squared distance to P0) as a perfect square,
    # then for each r1^2, check if the linear equation has integer solutions
    # that also satisfy the other distance constraints.
    
    # Bound the search: points should be within a reasonable range
    xs = [p[0] for p in current]
    ys = [p[1] for p in current]
    xmin, xmax = min(xs) - 200, max(xs) + 200
    ymin, ymax = min(ys) - 200, max(ys) + 200
    
    existing = set(current)
    max_r2 = (xmax - xmin) ** 2 + (ymax - ymin) ** 2  # max possible squared distance
    
    t0 = time.perf_counter()
    cand = None
    
    # Enumerate r0 as integer (so r0^2 is perfect square)
    # r0^2 ranges from 0 to max_r2, so r0 from 0 to isqrt(max_r2)
    max_r0 = isqrt(max_r2)
    
    # For efficiency, we'll iterate over possible x values and compute required y
    # from the linear constraint, then check if y is integer and satisfies all constraints
    
    # The linear constraint: 2*dx01*x + 2*dy01*y = (x0^2+y0^2 - x1^2-y1^2) + r1^2 - r0^2
    # Let LHS_base = 2*dx01*x
    # Then 2*dy01*y = (x0^2+y0^2 - x1^2-y1^2) + r1^2 - r0^2 - LHS_base
    # For this to have integer y, RHS must be divisible by 2*dy01 (if dy01 != 0)
    
    # Alternative: iterate over x in [xmin, xmax], and for each x, 
    # r0^2 = (x-x0)^2 + (y-y0)^2 must be a perfect square for some integer y
    # This means (y-y0)^2 = r0^2 - (x-x0)^2 must be a perfect square
    # So we can enumerate x, then enumerate r0^2 >= (x-x0)^2 that are perfect squares,
    # compute (y-y0)^2, check if perfect square, get y candidates
    
    # But this is still expensive. Let's use the linear constraint to reduce work.
    
    # If dx01 and dy01 are both nonzero, we can parameterize:
    # From linear eq: y = [(x0^2+y0^2 - x1^2-y1^2) + r1^2 - r0^2 - 2*dx01*x] / (2*dy01)
    # For y to be integer, numerator must be divisible by 2*dy01.
    
    # Let's try a grid-based approach but with better pruning:
    # For each x in range, compute which y values could work based on P0 distance
    # being a perfect square.
    
    for x in range(xmin, xmax + 1):
        if time.perf_counter() > deadline:
            break
        
        dx0 = x - x0
        dx0_sq = dx0 * dx0
        
        # (y-y0)^2 = r0^2 - dx0_sq must be a perfect square
        # So r0^2 - dx0_sq = dy0^2 for some integer dy0
        # r0^2 = dx0_sq + dy0^2
        # We need r0^2 to be a perfect square AND dy0^2 to be a perfect square
        # i.e., dx0^2 + dy0^2 is a perfect square (Pythagorean)
        
        # For given dx0, find all dy0 such that dx0^2 + dy0^2 is a perfect square
        # and y = y0 + dy0 is in [ymin, ymax]
        
        # Bound: dy0^2 <= max_r2 - dx0_sq, so |dy0| <= isqrt(max_r2 - dx0_sq)
        max_dy0 = isqrt(max_r2 - dx0_sq) if max_r2 >= dx0_sq else 0
        
        for dy0 in range(-max_dy0, max_dy0 + 1):
            y = y0 + dy0
            if y < ymin or y > ymax:
                continue
            if (x, y) in existing:
                continue
            
            # Check distance to P0 is perfect square
            r0_sq = dx0_sq + dy0 * dy0
            if not _is_square(r0_sq):
                continue
            
            # Check distance to P1
            dx1 = x - x1
            dy1 = y - y1
            r1_sq = dx1 * dx1 + dy1 * dy1
            if not _is_square(r1_sq):
                continue
            
            # Check distances to P2..P_{n-1}
            ok = True
            for i in range(2, n):
                dx = x - current[i][0]
                dy = y - current[i][1]
                if not _is_square(dx * dx + dy * dy):
                    ok = False
                    break
            if not ok:
                continue
            
            # Quick collinearity test vs pairs
            bad = False
            for a in range(n):
                for b in range(a + 1, n):
                    if _collinear3(current[a], current[b], (x, y)) == 0:
                        bad = True
                        break
                if bad:
                    break
            if bad:
                continue
            
            # Quick concyclicity with 3 of the existing (any 3)
            for a, b, c in combinations(range(n), 3):
                if _conic_coeff(current[a], current[b], current[c], (x, y)) == 0:
                    bad = True
                    break
            if bad:
                continue
            
            # If passed local tests, candidate found; stop to do full validation
            cand = (x, y)
            break
        
        if cand is not None:
            break
    
    if cand is None:
        return current
    
    new_set = current + [cand]
    # full verification for the new set
    if _valid_set(new_set):
        return new_set
    
    # fallback: try a few nearby alternative candidates if time remains
    # Continue scanning further to look for another candidate
    t0b = time.perf_counter()
    # We need to continue from where we left off. Let's restart the search but skip already-checked points
    checked = set()
    checked.add(cand)
    
    for x in range(xmin, xmax + 1):
        if time.perf_counter() > deadline:
            break
        
        dx0 = x - x0
        dx0_sq = dx0 * dx0
        
        max_dy0 = isqrt(max_r2 - dx0_sq) if max_r2 >= dx0_sq else 0
        
        for dy0 in range(-max_dy0, max_dy0 + 1):
            y = y0 + dy0
            if y < ymin or y > ymax:
                continue
            if (x, y) in existing or (x, y) in checked:
                continue
            checked.add((x, y))
            
            r0_sq = dx0_sq + dy0 * dy0
            if not _is_square(r0_sq):
                continue
            
            dx1 = x - x1
            dy1 = y - y1
            r1_sq = dx1 * dx1 + dy1 * dy1
            if not _is_square(r1_sq):
                continue
            
            ok = True
            for i in range(2, n):
                dx = x - current[i][0]
                dy = y - current[i][1]
                if not _is_square(dx * dx + dy * dy):
                    ok = False
                    break
            if not ok:
                continue
            
            bad = False
            for a in range(n):
                for b in range(a + 1, n):
                    if _collinear3(current[a], current[b], (x, y)) == 0:
                        bad = True
                        break
                if bad:
                    break
            if bad:
                continue
            
            for a, b, c in combinations(range(n), 3):
                if _conic_coeff(current[a], current[b], current[c], (x, y)) == 0:
                    bad = True
                    break
            if bad:
                continue
            
            new_set2 = current + [(x, y)]
            if _valid_set(new_set2):
                return new_set2
            break
        else:
            continue
        break
    
    return current

def construct():
    # Seed 5-point set
    base = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]

    # Validate base (should be valid)
    if not _valid_set(base):
        # Fallback to smaller valid known set: use the first 3 points
        base = base[:3]

    t0 = time.perf_counter()
    deadline = t0 + DEADLINE

    best = base[:]

    # Try iterative extension with time guard
    while len(best) < 12 and (time.perf_counter() < deadline):
        extended = _try_extend(best, deadline)
        if len(extended) <= len(best):
            break
        best = extended

    # Final safeguard
    if not _valid_set(best):
        # Ensure we return at least a minimal valid set
        best = base[:3]

    return _format_points(best)
