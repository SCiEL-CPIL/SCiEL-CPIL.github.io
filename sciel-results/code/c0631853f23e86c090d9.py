"""Circle packing in unit square using Apollonian-gasket-inspired construction."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from math import sqrt, atan2, sin, cos, pi

_DEADLINE = 140.0  # safety margin before 150s kill


def _dist(a, b):
    """Euclidean distance between two 2D points."""
    return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def _circle_in_gap(p1, r1, p2, r2, p3, r3):
    """
    Find the largest circle that fits in the gap between three mutually
    non-overlapping circles (or circle-wall-wall). Uses Descartes' theorem
    generalized for arbitrary curvatures.
    
    Returns (center, radius) or None if no valid circle fits.
    """
    # Convert to curvatures (k = 1/r, with sign for orientation)
    # For three circles, use the formula for the fourth tangent circle
    k1, k2, k3 = 1.0/r1 if r1 > 0 else 0.0, 1.0/r2 if r2 > 0 else 0.0, 1.0/r3 if r3 > 0 else 0.0
    
    # Descartes' theorem: (k1+k2+k3+k4)^2 = 2*(k1^2+k2^2+k3^2+k4^2)
    # => k4 = k1+k2+k3 ± 2*sqrt(k1*k2 + k2*k3 + k3*k1)
    s = k1*k2 + k2*k3 + k3*k1
    if s < 0:
        return None
    k4_plus = k1 + k2 + k3 + 2.0*sqrt(s)
    k4_minus = k1 + k2 + k3 - 2.0*sqrt(s)
    
    # We want the smaller circle (larger curvature) that fits in the gap
    # k4_plus gives the small circle in the interstice
    if k4_plus <= 0:
        return None
    r4 = 1.0 / k4_plus
    
    # Find center using complex Descartes or geometric construction
    # Use the fact that the new circle is tangent to all three
    # Solve for center (cx, cy) such that:
    #   dist((cx,cy), pi) = r4 + ri  for i=1,2,3
    # This is a system of 3 equations. Use least squares approach.
    
    # Actually, use the complex Descartes theorem for center
    # z4 = (z1/k1 + z2/k2 + z3/k3 + 2*sqrt(z1*z2/k1/k2 + z2*z3/k2/k3 + z3*z1/k3/k1)) / k4
    # where zi = xi + yi*i (complex), but this is tricky with signs
    
    # Simpler: solve geometrically. The center must satisfy:
    # (cx - x1)^2 + (cy - y1)^2 = (r4 + r1)^2
    # (cx - x2)^2 + (cy - y2)^2 = (r4 + r2)^2
    # (cx - x3)^2 + (cy - y3)^2 = (r4 + r3)^2
    
    # Subtract first from second:
    # 2*cx*(x2-x1) + 2*cy*(y2-y1) = (r4+r2)^2 - (r4+r1)^2 + x2^2-x1^2 + y2^2-y1^2
    # = 2*r4*(r2-r1) + r2^2 - r1^2 + x2^2-x1^2 + y2^2-y1^2
    
    def _solve_center(c1, r1, c2, r2, c3, r3, r4):
        # Linear system from subtracting circle equations
        A = np.array([
            [2*(c2[0]-c1[0]), 2*(c2[1]-c1[1])],
            [2*(c3[0]-c1[0]), 2*(c3[1]-c1[1])]
        ])
        b = np.array([
            (r4+r2)**2 - (r4+r1)**2 + c2[0]**2 - c1[0]**2 + c2[1]**2 - c1[1]**2,
            (r4+r3)**2 - (r4+r1)**2 + c3[0]**2 - c1[0]**2 + c3[1]**2 - c1[1]**2
        ])
        try:
            center = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None
        # Verify it's valid (not outside unit square, positive radius)
        if center[0] < -r4 - 0.01 or center[0] > 1 + r4 + 0.01:
            return None
        if center[1] < -r4 - 0.01 or center[1] > 1 + r4 + 0.01:
            return None
        return (float(center[0]), float(center[1]))
    
    center = _solve_center(p1, r1, p2, r2, p3, r3, r4)
    if center is None:
        return None
    
    # Verify no overlap with existing circles will be checked by caller
    return (center, r4)


def _circle_in_wall_gap(p1, r1, p2, r2, wall1, wall2):
    """
    Find largest circle tangent to two circles and two walls.
    wall1, wall2 are 'x=0', 'x=1', 'y=0', 'y=1'
    """
    # This is more complex; use iterative approach
    # For now, return None to skip these cases
    return None


def _max_radius_at(pos, circles):
    """
    Given a position, find the maximum radius of a circle centered there
    that doesn't overlap any existing circle and stays in [0,1]^2.
    """
    x, y = pos
    # Distance to walls
    r_max = min(x, 1-x, y, 1-y)
    # Distance to existing circles
    for (cx, cy, cr) in circles:
        d = sqrt((x-cx)**2 + (y-cy)**2)
        r_max = min(r_max, d - cr)
    return max(0, r_max)


def _find_best_position(circles, n_samples=500):
    """
    Find the position that maximizes the largest circle that can be placed there.
    Uses a combination of strategic sampling.
    """
    best_radius = 0
    best_pos = (0.5, 0.5)
    
    # Sample positions: corners, edges, and random interior
    candidates = []
    
    # Corners
    candidates += [(0.05, 0.05), (0.95, 0.05), (0.05, 0.95), (0.95, 0.95)]
    # Edge midpoints
    candidates += [(0.5, 0.05), (0.5, 0.95), (0.05, 0.5), (0.95, 0.5)]
    # Center
    candidates.append((0.5, 0.5))
    
    # Random sampling
    rng = np.random.default_rng(42)
    for _ in range(n_samples):
        candidates.append((rng.random(), rng.random()))
    
    # Also sample near existing circles (interstices)
    if len(circles) >= 2:
        for i in range(min(len(circles), 10)):
            ci = circles[i]
            for j in range(i+1, min(len(circles), 10)):
                cj = circles[j]
                mid = ((ci[0]+cj[0])/2, (ci[1]+cj[1])/2)
                candidates.append(mid)
                # Points offset from the line between centers
                dx = cj[0] - ci[0]
                dy = cj[1] - ci[1]
                d = sqrt(dx*dx + dy*dy)
                if d > 0:
                    nx, ny = -dy/d, dx/d
                    for t in [0.2, 0.5, 0.8]:
                        px = mid[0] + t*nx*0.1
                        py = mid[1] + t*ny*0.1
                        if 0 <= px <= 1 and 0 <= py <= 1:
                            candidates.append((px, py))
                        px2 = mid[0] - t*nx*0.1
                        py2 = mid[1] - t*ny*0.1
                        if 0 <= px2 <= 1 and 0 <= py2 <= 1:
                            candidates.append((px2, py2))
    
    for pos in candidates:
        if not (0 <= pos[0] <= 1 and 0 <= pos[1] <= 1):
            continue
        r = _max_radius_at(pos, circles)
        if r > best_radius:
            best_radius = r
            best_pos = pos
    
    return best_pos, best_radius


def _try_apollonian_pair(circles):
    """
    For each pair of circles, try to find a circle tangent to both and
    one or two walls. Return the best candidate.
    """
    best = None
    best_r = 0
    
    walls = ['x0', 'x1', 'y0', 'y1']
    
    for i in range(len(circles)):
        for j in range(i+1, len(circles)):
            c1 = circles[i]
            c2 = circles[j]
            # Try each pair of walls
            for w1 in walls:
                for w2 in walls:
                    if w1 == w2:
                        continue
                    # Check if the gap between c1, c2, w1, w2 can fit a circle
                    # Simplified: sample the region
                    # This is complex; skip for now
                    pass
    
    return best


def _construct_apollonian():
    """
    Apollonian-gasket-inspired constructive algorithm:
    1. Start with a few large circles in corners
    2. Iteratively place the next largest circle in the best interstice
    """
    circles = []
    
    # Initial placement: 4 corner circles
    r_corner = 0.25
    corners = [(r_corner, r_corner), (1-r_corner, r_corner), 
               (r_corner, 1-r_corner), (1-r_corner, 1-r_corner)]
    for (x, y) in corners:
        circles.append((x, y, r_corner))
    
    # Place center circle
    r_center = 0.25
    circles.append((0.5, 0.5, r_center))
    
    # Now iteratively place remaining circles
    rng = np.random.default_rng(123)
    
    while len(circles) < N:
        if time.time() > _DEADLINE - 5:
            break
        
        # Find the best position for the next circle
        best_pos, best_r = _find_best_position(circles, n_samples=300)
        
        if best_r < 1e-6:
            # Try random positions
            found = False
            for _ in range(100):
                pos = (rng.random(), rng.random())
                r = _max_radius_at(pos, circles)
                if r > best_r:
                    best_r = r
                    best_pos = pos
                    found = True
            if not found or best_r < 1e-6:
                break
        
        # Add the circle with a small safety margin
        r_safe = best_r * 0.999
        circles.append((best_pos[0], best_pos[1], r_safe))
    
    return circles


def _construct_grid_fallback():
    """Fallback grid construction."""
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
             for i in range(side) for j in range(side)]
    return cells[:N]


def _validate(circles):
    """Check that circles are valid (in bounds, non-overlapping)."""
    for (x, y, r) in circles:
        if x - r < -1e-9 or x + r > 1 + 1e-9:
            return False
        if y - r < -1e-9 or y + r > 1 + 1e-9:
            return False
        if r <= 0:
            return False
    
    for i in range(len(circles)):
        for j in range(i+1, len(circles)):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            d = sqrt((x1-x2)**2 + (y1-y2)**2)
            if d < r1 + r2 - 1e-9:
                return False
    return True


def _objective(circles):
    """Sum of radii."""
    return sum(r for (_, _, r) in circles)


def construct():
    start_time = time.time()
    
    # Try Apollonian construction
    try:
        circles = _construct_apollonian()
        
        if len(circles) < N:
            # Pad with very small circles if needed
            rng = np.random.default_rng(99)
            while len(circles) < N:
                pos = (rng.random(), rng.random())
                r = _max_radius_at(pos, circles)
                if r > 1e-5:
                    circles.append((pos[0], pos[1], r * 0.99))
                else:
                    circles.append((pos[0], pos[1], 1e-5))
        
        # Validate
        if not _validate(circles):
            # Try to fix by reducing radii slightly
            circles_fixed = [(x, y, max(1e-5, r * 0.95)) for (x, y, r) in circles]
            if not _validate(circles_fixed):
                circles = _construct_grid_fallback()
            else:
                circles = circles_fixed
    except Exception:
        circles = _construct_grid_fallback()
    
    # Ensure exactly N circles
    if len(circles) > N:
        circles = circles[:N]
    elif len(circles) < N:
        # This shouldn't happen, but pad with tiny circles
        while len(circles) < N:
            circles.append((0.5, 0.5, 1e-5))
    
    result = np.array(circles, dtype=np.float64)
    
    # Final validation
    if not _validate(result):
        result = np.array(_construct_grid_fallback(), dtype=np.float64)
    
    return result
# EVOLVE-BLOCK-END
