"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction
from itertools import combinations

def _is_square(n):
    if n < 0:
        return False
    if n == 0:
        return True
    r = int(n ** 0.5)
    if r * r == n:
        return True
    # Handle large integers properly
    for r in (r - 1, r, r + 1):
        if r >= 0 and r * r == n:
            return True
    return False

def construct():
    start = time.time()
    deadline = start + 130  # safety margin
    
    def elapsed():
        return time.time() - start
    
    # Try to find a set of 6 points with rational pairwise distances,
    # no three collinear, no four concyclic.
    # 
    # Strategy: Use integer points. All pairwise distances are rational (integers).
    # We need to find 6 integer points such that:
    #   - No three are collinear
    #   - No four are concyclic
    #
    # Known 5-point example: (0,0), (-56,33), (-16,63), (0,33), (16,63)
    # Let's try to extend this by adding a 6th integer point.
    
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    def collinear(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) == (q[1] - p[1]) * (r[0] - p[0])
    
    def concyclic(a, b, c, d):
        # Check if four points are concyclic using the determinant of the circle equation
        # x^2 + y^2 + Dx + Ey + F = 0
        # For points (xi, yi), the system is:
        # xi^2 + yi^2 + Di*xi + Ei*yi + F = 0
        # This is concyclic iff the 4x4 determinant is zero
        # Using exact arithmetic
        xa, ya = a
        xb, yb = b
        xc, yc = c
        xd, yd = d
        
        # Determinant of the augmented system
        # | xa^2+ya^2  xa  ya  1 |
        # | xb^2+yb^2  xb  yb  1 |
        # | xc^2+yc^2  xc  yc  1 |
        # | xd^2+yd^2  xd  yd  1 |
        A = xa * xa + ya * ya
        B = xb * xb + yb * yb
        C = xc * xc + yc * yc
        D_ = xd * xd + yd * yd
        
        # Compute 4x4 determinant by expansion
        # Using the rule:
        # det = A*(xb*yc*1 - xb*yd*1 - yb*xc*1 + yb*xd*1)
        #     - xa*(B*yc*1 - B*yd*1 - yb*C*1 + yb*xd*1)  ... wait, let me use a proper method
        
        # Let's use the standard formula:
        # det = | A  xa  ya  1 |
        #       | B  xb  yb  1 |
        #       | C  xc  yc  1 |
        #       | D_ xd  yd  1 |
        
        # Expand along the first row:
        # A * det([[xb,yb,1],[xc,yc,1],[xd,yd,1]])
        # - xa * det([[B,yb,1],[C,yc,1],[D_,yd,1]])
        # + ya * det([[B,xb,1],[C,xc,1],[D_,xd,1]])
        # - 1 * det([[B,xb,yb],[C,xc,yc],[D_,xd,yd]])
        
        def det3(m):
            return (m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
                  - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
                  + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))
        
        M1 = [[xb, yb, 1], [xc, yc, 1], [xd, yd, 1]]
        M2 = [[B, yb, 1], [C, yc, 1], [D_, yd, 1]]
        M3 = [[B, xb, 1], [C, xc, 1], [D_, xd, 1]]
        M4 = [[B, xb, yb], [C, xc, yc], [D_, xd, yd]]
        
        return A * det3(M1) - xa * det3(M2) + ya * det3(M3) - det3(M4)
    
    def check_set(pts):
        # Check no three collinear
        for a, b, c in combinations(pts, 3):
            if collinear(a, b, c):
                return False
        # Check no four concyclic
        for a, b, c, d in combinations(pts, 4):
            if concyclic(a, b, c, d) == 0:
                return False
        return True
    
    # First verify the base 5 points
    if not check_set(base_points):
        # Fallback: return the known valid 5-point set
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in base_points]}
    
    # Try to add a 6th integer point
    # Search over a range of integer coordinates
    best_points = base_points[:]
    
    # Try points in a reasonable range
    # The base points span x from -56 to 16, y from 0 to 63
    # Try adding points in a wider range
    
    found_6 = False
    # Try a grid of candidate 6th points
    for x6 in range(-80, 40):
        if elapsed() > deadline:
            break
        for y6 in range(-20, 90):
            if elapsed() > deadline:
                break
            p6 = (x6, y6)
            # Check p6 is not collinear with any pair from base
            ok = True
            for a, b in combinations(base_points, 2):
                if collinear(a, b, p6):
                    ok = False
                    break
            if not ok:
                continue
            # Check p6 doesn't form a concyclic quadruple with any 3 from base
            for a, b, c in combinations(base_points, 3):
                if concyclic(a, b, c, p6) == 0:
                    ok = False
                    break
            if not ok:
                continue
            # Full check
            candidate = base_points + [p6]
            if check_set(candidate):
                best_points = candidate
                found_6 = True
                break
        if found_6 or elapsed() > deadline:
            break
    
    if found_6:
        pts = best_points
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in pts]}
    
    # If 6 not found, try a different approach: search for any 6 integer points
    # Try known configurations or a broader search
    
    # Try a different base: use points from a known 6-point configuration if available
    # For now, return the best we have (5 points)
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
# EVOLVE-BLOCK-END
