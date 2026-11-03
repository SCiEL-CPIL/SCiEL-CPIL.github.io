"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the minimum triangle area."""
import numpy as np
import time
from scipy.optimize import minimize, root
from itertools import combinations
import sys

N = 11
SQRT3 = 3 ** 0.5
# Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
# Area of this triangle = sqrt(3)/4
TRI_AREA = SQRT3 / 4.0

def _cross2d(a, b):
    """2D cross product of vectors a and b (scalar)."""
    return a[0] * b[1] - a[1] * b[0]

def _triangle_area(p1, p2, p3):
    """Area of triangle formed by three points."""
    return 0.5 * abs(_cross2d(p2 - p1, p3 - p1))

def _all_triangle_areas(points):
    """Compute areas of all triangles formed by triples of points."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                areas.append(_triangle_area(points[i], points[j], points[k]))
    return np.array(areas)

def _min_area(points):
    """Minimum triangle area among all triples."""
    return np.min(_all_triangle_areas(points))

def _in_triangle(p):
    """Check if point p is inside or on the boundary of the triangle."""
    # Barycentric coordinates for triangle (0,0), (1,0), (0.5, sqrt(3)/2)
    x, y = p
    # Edge 1: from (0,0) to (1,0): y >= 0
    if y < -1e-10:
        return False
    # Edge 2: from (1,0) to (0.5, sqrt(3)/2): line is y = -sqrt(3)*(x-1) => sqrt(3)*x + y <= sqrt(3)
    if SQRT3 * x + y > SQRT3 + 1e-10:
        return False
    # Edge 3: from (0.5, sqrt(3)/2) to (0,0): line is y = sqrt(3)*x => y <= sqrt(3)*x
    if y > SQRT3 * x + 1e-10:
        return False
    return True

def _project_to_triangle(p):
    """Project point p to the closest point in the triangle."""
    if _in_triangle(p):
        return np.array(p, dtype=float)
    
    # Vertices
    v0 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, SQRT3 / 2.0])
    
    edges = [(v0, v1), (v1, v2), (v2, v0)]
    
    best = None
    best_dist = float('inf')
    
    for a, b in edges:
        # Project p onto line segment ab
        ab = b - a
        ap = p - a
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = np.clip(t, 0.0, 1.0)
        proj = a + t * ab
        d = np.linalg.norm(p - proj)
        if d < best_dist:
            best_dist = d
            best = proj
    
    return best

def _is_valid(points):
    """Check all points are inside the triangle."""
    return all(_in_triangle(p) for p in points)

def _objective(x, t):
    """
    Objective for optimization: maximize t (the minimum area).
    We minimize -t subject to all triangle areas >= t.
    Using a smooth penalty: sum of soft-min violations.
    """
    points = x.reshape(-1, 2)
    
    # Compute all triangle areas
    areas = _all_triangle_areas(points)
    
    # Penalty for areas below t
    # Use a smooth approximation: -log(sum(exp(-areas/t_scale))) 
    # Or simply: t - max(0, t - min_area) but that's not smooth
    # Let's use: minimize (t - min_area)^2 + small regularization
    # Actually, let's use a barrier approach
    
    # Soft minimum: t_soft = -logsumexp(-areas / beta) * beta
    beta = 0.01
    logsumexp = np.log(np.sum(np.exp(-areas / beta)))
    t_soft = -beta * logsumexp
    
    # We want to maximize t_soft, so minimize -t_soft
    # But we also need to handle the constraint that points stay in triangle
    # Add penalty for points outside triangle
    penalty = 0.0
    for i in range(len(points)):
        if not _in_triangle(points[i]):
            # Distance to triangle
            proj = _project_to_triangle(points[i])
            penalty += 100.0 * np.linalg.norm(points[i] - proj) ** 2
    
    return -t_soft + penalty

def _gradient(x, t):
    """Gradient of the objective with respect to x."""
    points = x.reshape(-1, 2)
    n = len(points)
    beta = 0.01
    
    # Compute all triangle areas and their gradients
    grad = np.zeros_like(x)
    
    # For each triangle (i,j,k), area = 0.5 * |cross(pj-pi, pk-pi)|
    # The sign matters for differentiability; use signed area
    # Let's use signed area for smoothness
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                pi = points[i]
                pj = points[j]
                pk = points[k]
                
                # Signed area
                signed_area = 0.5 * _cross2d(pj - pi, pk - pi)
                
                # For the soft-min, we need d(area)/d(point)
                # area = 0.5 * ((pj_x - pi_x)(pk_y - pi_y) - (pk_x - pi_x)(pj_y - pi_y))
                # d(area)/d(pi_x) = 0.5 * (-(pk_y - pi_y) - (-(pj_y - pi_y))) = 0.5 * (pj_y - pk_y)
                # d(area)/d(pi_y) = 0.5 * ((pk_x - pi_x) - (pj_x - pi_x)) = 0.5 * (pk_x - pj_x)
                # d(area)/d(pj_x) = 0.5 * (pk_y - pi_y)
                # d(area)/d(pj_y) = 0.5 * (-(pk_x - pi_x))
                # d(area)/d(pk_x) = 0.5 * (-(pj_y - pi_y))
                # d(area)/d(pk_y) = 0.5 * (pj_x - pi_x)
                
                d_area_d_pi = np.array([0.5 * (pj[1] - pk[1]), 0.5 * (pk[0] - pj[0])])
                d_area_d_pj = np.array([0.5 * (pk[1] - pi[1]), -0.5 * (pk[0] - pi[0])])
                d_area_d_pk = np.array([-0.5 * (pj[1] - pi[1]), 0.5 * (pj[0] - pi[0])])
                
                # Weight for soft-min: w = exp(-area/beta) / sum(exp(-area/beta))
                # For now, approximate with equal weights if we don't track all areas
                # This is getting complex; let's use numerical gradient or a simpler approach
                pass
    
    # For simplicity, let's use a different approach: direct optimization with constraints
    return grad

def _optimize_from_start(start_points, max_time=100.0):
    """Optimize from a starting configuration."""
    x0 = np.array(start_points, dtype=float).flatten()
    
    # Use L-BFGS-B with bounds for each coordinate
    # Bounds: x in [0, 1], y in [0, sqrt(3)/2]
    # But we need to respect the triangle constraint, so use projection
    
    best_x = x0.copy()
    best_score = _min_area(start_points)
    
    # Use minimize with a smooth objective
    # Let's define a simpler objective: minimize negative of a soft-min of areas
    def obj(x):
        pts = x.reshape(-1, 2)
        # Project to triangle
        for i in range(len(pts)):
            if not _in_triangle(pts[i]):
                pts[i] = _project_to_triangle(pts[i])
        areas = _all_triangle_areas(pts)
        beta = 0.05
        # Soft min
        logsumexp = np.log(np.sum(np.exp(-areas / beta)))
        return -(-beta * logsumexp)  # = beta * logsumexp, minimize this
    
    # Actually, let's just use a direct approach: maximize min area via iterative local search
    # Or use scipy's minimize with a penalty method
    
    # Let's use a simpler strategy: coordinate ascent with random restarts
    # For each point, try to move it to increase the minimum area
    
    deadline = time.time() + max_time
    pts = np.array(start_points, dtype=float)
    
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        # Find the minimum area triangle
        min_area = float('inf')
        min_tri = None
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    a = _triangle_area(pts[i], pts[j], pts[k])
                    if a < min_area:
                        min_area = a
                        min_tri = (i, j, k)
        
        if min_area > best_score:
            best_score = min_area
            best_x = pts.copy()
        
        # Try to improve by moving one of the points in the minimum triangle
        i, j, k = min_tri
        for idx in [i, j, k]:
            # Try small random perturbations
            for _ in range(20):
                new_pts = pts.copy()
                dx = np.random.randn(2) * 0.05
                new_pts[idx] += dx
                new_pts[idx] = _project_to_triangle(new_pts[idx])
                
                new_min = _min_area(new_pts)
                if new_min > min_area:
                    pts = new_pts
                    break
    
    return best_x.reshape(-1, 2), best_score

def _random_start(rng):
    """Generate a random starting configuration."""
    points = []
    for _ in range(N):
        while True:
            u, v = rng.random(), rng.random()
            if u + v < 1.0:
                p = np.array([u + 0.5 * v, SQRT3 / 2.0 * v])
                break
        points.append(p)
    return np.array(points)

def _greedy_construction():
    """Construct points using a greedy approach."""
    points = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.5, SQRT3 / 2.0])]
    
    # Add remaining points to maximize minimum area
    for _ in range(N - 3):
        best_point = None
        best_min_area = -1
        for _ in range(200):
            u, v = np.random.random(), np.random.random()
            if u + v < 1.0:
                p = np.array([u + 0.5 * v, SQRT3 / 2.0 * v])
                new_points = points + [p]
                min_a = _min_area(new_points)
                if min_a > best_min_area:
                    best_min_area = min_a
                    best_point = p
        if best_point is not None:
            points.append(best_point)
    
    return np.array(points)

def construct():
    """Main entry point: returns array of shape (N, 2)."""
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    rng = np.random.RandomState(42)
    
    # Start with a few different initializations
    best_points = None
    best_score = -1
    
    # Greedy construction
    pts_greedy = _greedy_construction()
    score_greedy = _min_area(pts_greedy)
    if score_greedy > best_score:
        best_score = score_greedy
        best_points = pts_greedy
    
    # Random starts with local optimization
    num_starts = 0
    while time.time() < deadline:
        start_pts = _random_start(rng)
        pts, score = _optimize_from_start(start_pts, max_time=10.0)
        if score > best_score:
            best_score = score
            best_points = pts
        num_starts += 1
        if time.time() > deadline - 20:
            break
    
    # Final polish: try to improve best_points further
    if best_points is not None and time.time() < deadline - 5:
        pts, score = _optimize_from_start(best_points, max_time=15.0)
        if score > best_score:
            best_score = score
            best_points = pts
    
    if best_points is None:
        # Fallback: simple random
        best_points = _random_start(rng)
    
    # Ensure all points are in the triangle
    for i in range(len(best_points)):
        best_points[i] = _project_to_triangle(best_points[i])
    
    return best_points
