"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import math
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def _is_square(n):
    """Check if n is a perfect square (n is a non-negative int)."""
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _dist_sq(p1, p2, D):
    """Squared distance between two points (x1,y1),(x2,y2) with y*sqrt(D)."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x1 - x2
    dy = y1 - y2
    # squared distance = dx^2 + D * dy^2
    return dx * dx + D * dy * dy

def _is_collinear(p1, p2, p3, D):
    """Check if three points are collinear using determinant."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    # det = (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
    det = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    return det == 0

def _is_concyclic(p1, p2, p3, p4, D):
    """Check if four points are concyclic.
    
    Points (xi, yi*sqrt(D)) are concyclic iff there exist a,b,c such that:
    xi^2 + D*yi^2 + a*xi + b*(yi*sqrt(D)) + c = 0 for all i
    
    This is equivalent to: xi^2 + D*yi^2 + a*xi + b*sqrt(D)*yi + c = 0
    
    For this to hold with rational a,b,c, we need the 4x4 determinant to be zero:
    | xi^2 + D*yi^2, xi, yi, 1 |
    
    Wait, let me reconsider. The circle equation in the plane where point i is at (xi, yi*sqrt(D)):
    (xi - h)^2 + (yi*sqrt(D) - k)^2 = r^2
    xi^2 - 2h*xi + h^2 + D*yi^2 - 2k*yi*sqrt(D) + k^2 = r^2
    xi^2 + D*yi^2 - 2h*xi - 2k*sqrt(D)*yi + (h^2 + k^2 - r^2) = 0
    
    Let A = -2h, B = -2k*sqrt(D), C = h^2 + k^2 - r^2
    Then: xi^2 + D*yi^2 + A*xi + B*yi + C = 0
    
    For four points to be concyclic, the 4x4 system must be singular:
    | xi^2 + D*yi^2, xi, yi, 1 | = 0 determinant
    """
    vals = []
    for p in [p1, p2, p3, p4]:
        x, y = p
        vals.append([x * x + D * y * y, x, y, 1])
    # Compute determinant of 4x4 matrix
    # Using cofactor expansion
    m = vals
    def det4(m):
        return (m[0][0]*(m[1][1]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) + m[1][3]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]))
            - m[0][1]*(m[1][0]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]))
            + m[0][2]*(m[1][0]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) - m[1][1]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][1]-m[2][1]*m[3][0]))
            - m[0][3]*(m[1][0]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]) - m[1][1]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]) + m[1][2]*(m[2][0]*m[3][1]-m[2][1]*m[3][0])))
    return det4(m) == 0

def _check_valid(points, D):
    """Check if a set of points is valid."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            ds = _dist_sq(points[i], points[j], D)
            if not _is_square(ds):
                return False
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _is_collinear(points[i], points[j], points[k], D):
                    return False
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    if _is_concyclic(points[i], points[j], points[k], points[l], D):
                        return False
    return True

def _try_add_point(points, D, candidate):
    """Try to add a candidate point to the set."""
    new_points = points + [candidate]
    return _check_valid(new_points, D)

def _search_integer_points(max_coord, deadline, initial_points=None):
    """Search for integer points with integer pairwise distances."""
    D = 1
    if initial_points is None:
        points = [(0, 0)]
    else:
        points = list(initial_points)
    
    best = list(points)
    
    # Generate candidate points
    candidates = []
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            if (x, y) not in points:
                candidates.append((x, y))
    
    random.shuffle(candidates)
    
    start_time = time.time()
    
    # Try to add points greedily
    for candidate in candidates:
        if time.time() > deadline:
            break
        if _try_add_point(points, D, candidate):
            points.append(candidate)
            best = list(points)
            if len(points) >= 12:
                break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Start with the known 5-point set
    initial = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    best_points = list(initial)
    
    # Try to extend with local search
    random.seed(42)
    
    # Try multiple restarts with different random seeds
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Randomly perturb the initial set
        if attempt == 0:
            points = list(initial)
        else:
            # Start fresh with a random small set
            points = [(0, 0)]
            # Add a few random points
            for _ in range(min(attempt % 5, 5)):
                if time.time() > deadline:
                    break
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                if (x, y) not in points and _try_add_point(points, 1, (x, y)):
                    points.append((x, y))
        
        # Now try to extend
        extended = _search_integer_points(150, deadline, points)
        
        if len(extended) > len(best_points):
            best_points = extended
    
    # Convert to required format
    D = 1
    result_points = []
    for x, y in best_points:
        result_points.append([[x, 1], [y, 1]])
    
    return {'D': D, 'points': result_points}
# EVOLVE-BLOCK-END
