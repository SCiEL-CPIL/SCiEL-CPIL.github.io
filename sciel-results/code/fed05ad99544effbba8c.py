"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from fractions import Fraction
import math

# EVOLVE-BLOCK-START
import math
from itertools import combinations

def _is_perfect_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _check_pairwise_rational_distances(points, D):
    """Check all pairwise squared distances are perfect squares (for D=1 case)."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x1 - x2
            dy = y1 - y2
            sq_dist = dx * dx + D * dy * dy
            if not _is_perfect_square(sq_dist):
                return False
    return True

def _check_no_three_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                # Cross product for collinearity
                cross = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
                if cross == 0:
                    return False
    return True

def _check_no_four_concyclic(points):
    """Check no four points are concyclic using exact arithmetic."""
    n = len(points)
    for quad in combinations(range(n), 4):
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = [points[i] for i in quad]
        # Four points are concyclic if the determinant of the circle equation is zero.
        # Circle: x^2 + y^2 + Dx + Ey + F = 0
        # For each point: xi^2 + yi^2 + D*xi + E*yi + F = 0
        # We solve for D, E, F from first 3 points and check if 4th satisfies.
        a1, b1, c1 = x1*x1 + y1*y1, x1, y1
        a2, b2, c2 = x2*x2 + y2*y2, x2, y2
        a3, b3, c3 = x3*x3 + y3*y3, x3, y3
        a4, b4, c4 = x4*x4 + y4*y4, x4, y4
        # Det of matrix [[a,b,c,1],[a2,b2,c2,1],[a3,b3,c3,1],[a4,b4,c4,1]]
        # Actually, we need: x^2+y^2+Dx+Ey+F=0
        # So: D*xi + E*yi + F = -(xi^2+yi^2)
        # System for D,E,F from points 1,2,3:
        # D*x1 + E*y1 + F = -(x1^2+y1^2)
        # D*x2 + E*y2 + F = -(x2^2+y2^2)
        # D*x3 + E*y3 + F = -(x3^2+y3^2)
        # Check if point 4 satisfies: D*x4 + E*y4 + F = -(x4^2+y4^2)
        # Using Cramer's rule or elimination.
        # Let's use the determinant method.
        # The four points are concyclic iff:
        # |x1^2+y1^2, x1, y1, 1|
        # |x2^2+y2^2, x2, y2, 1| = 0
        # |x3^2+y3^2, x3, y3, 1|
        # |x4^2+y4^2, x4, y4, 1|
        # Compute this 4x4 determinant.
        m = [[a1, b1, c1, 1],
             [a2, b2, c2, 1],
             [a3, b3, c3, 1],
             [a4, b4, c4, 1]]
        # Compute determinant
        det = (m[0][0]*(m[1][1]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) -
                        m[1][2]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) +
                        m[1][3]*(m[2][1]*m[3][2]-m[2][2]*m[3][1])) -
               m[0][1]*(m[1][0]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) -
                        m[1][2]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) +
                        m[1][3]*(m[2][0]*m[3][2]-m[2][2]*m[3][0])) +
               m[0][2]*(m[1][0]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) -
                        m[1][1]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) +
                        m[1][3]*(m[2][0]*m[3][1]-m[2][1]*m[3][0])) -
               m[0][3]*(m[1][0]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]) -
                        m[1][1]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]) +
                        m[1][2]*(m[2][0]*m[3][1]-m[2][1]*m[3][0])))
        if det == 0:
            return False
    return True

def _search_maximal_set(limit=30):
    """Search for a maximal set of points with integer coords in [-limit, limit]^2."""
    # Generate candidate points
    candidates = [(x, y) for x in range(-limit, limit + 1) for y in range(-limit, limit + 1)]
    
    # First, filter to points that have integer distance to (0,0)
    # Actually, let's just do a greedy/backtracking search.
    # For efficiency, let's start with (0,0) and try to build up.
    
    best_set = [(0, 0)]
    
    # Try adding points greedily
    for pt in candidates:
        if pt == (0, 0):
            continue
        trial = best_set + [pt]
        if (_check_pairwise_rational_distances(trial, 1) and
            _check_no_three_collinear(trial)):
            best_set = trial
            if len(best_set) >= 7:
                break
    
    return best_set

def construct():
    # Search for a large integral point set
    # Start with a known good 5-point set and try to extend
    # Known 5-point set: (0,0), (-56,33), (-16,63), (0,33), (16,63)
    # But these have large coords. Let me search with smaller coords.
    
    # Try to find a set of size >= 6 with small coordinates
    best = _search_maximal_set(limit=50)
    
    if len(best) < 3:
        # Fallback to known 5-point set
        best = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Verify the found set
    if not (_check_pairwise_rational_distances(best, 1) and
            _check_no_three_collinear(best) and
            _check_no_four_concyclic(best)):
        # Fallback
        best = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best]}
# EVOLVE-BLOCK-END
