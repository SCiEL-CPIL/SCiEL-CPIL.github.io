"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools

_DEADLINE = 140.0  # safety margin before 150s kill
_start_time = None


def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def _covering_radius(points):
    """Compute covering radius in degrees using convex hull circumradii."""
    try:
        hull = ConvexHull(points)
        max_cr = 0.0
        for simplex in hull.simplices:
            pts = points[simplex]
            # Compute circumradius of the simplex
            # For a triangle: R = abc / (4*area)
            # For a tetrahedron: use formula
            if len(pts) == 3:
                a = np.linalg.norm(pts[1] - pts[0])
                b = np.linalg.norm(pts[2] - pts[1])
                c = np.linalg.norm(pts[0] - pts[2])
                # Area of triangle
                v1 = pts[1] - pts[0]
                v2 = pts[2] - pts[0]
                cross = np.cross(v1, v2)
                area = 0.5 * np.linalg.norm(cross)
                if area < 1e-12:
                    continue
                R = a * b * c / (4.0 * area)
            else:
                # Tetrahedron circumradius
                # Use the formula: R = |a||b||c||d| / (8*V) where V is volume
                # More robust: solve for center
                p0, p1, p2, p3 = pts
                v1 = p1 - p0
                v2 = p2 - p0
                v3 = p3 - p0
                # Volume
                vol = np.abs(np.dot(v1, np.cross(v2, v3))) / 6.0
                if vol < 1e-12:
                    continue
                # Circumradius formula for tetrahedron
                # R = sqrt(|a|^2 * |b|^2 * |c|^2 / (72 * V^2)) ... no, use proper formula
                # Actually: R = |a x b| * |c| / (4 * 3V) is for triangle...
                # For tetrahedron: R = (a*b*c) / (12*V) where a,b,c are edge lengths from one vertex? No.
                # Let's use: circumradius = |a|^2 * |b|^2 * |c|^2 / (72 * V^2) ... not quite
                # Better: R = sqrt( (|a|^2 * |b|^2 * |c|^2) / (72 * V^2) ) is wrong
                # Correct: for tetrahedron with vertices, R can be found by solving
                # |C - p_i|^2 = R^2 for all i
                # Let's just compute it numerically
                # Center is equidistant from all 4 vertices
                # Solve linear system
                A = np.array([
                    [2*(p1[0]-p0[0]), 2*(p1[1]-p0[1]), 2*(p1[2]-p0[2])],
                    [2*(p2[0]-p0[0]), 2*(p2[1]-p0[1]), 2*(p2[2]-p0[2])],
                    [2*(p3[0]-p0[0]), 2*(p3[1]-p0[1]), 2*(p3[2]-p0[2])],
                ])
                b = np.array([
                    np.dot(p1, p1) - np.dot(p0, p0),
                    np.dot(p2, p2) - np.dot(p0, p0),
                    np.dot(p3, p3) - np.dot(p0, p0),
                ])
                try:
                    center = np.linalg.solve(A, b)
                    R = np.linalg.norm(center - p0)
                except:
                    continue
            max_cr = max(max_cr, R)
        return np.degrees(np.arcsin(min(1.0, max_cr)))
    except:
        return 90.0


def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _refine(points, time_budget):
    """Local optimization: minimize max circumradius via gradient-free methods."""
    global _start_time
    best = points.copy()
    best_score = _covering_radius(best)
    
    # Use Nelder-Mead on flattened coordinates
    def objective(x):
        if time.time() - _start_time > _DEADLINE:
            return best_score
        pts = x.reshape(-1, 3)
        # Normalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        pts = pts / norms
        # Check no coincident points
        for i in range(N):
            for j in range(i+1, N):
                if np.linalg.norm(pts[i] - pts[j]) < 1e-8:
                    return 90.0
        return _covering_radius(pts)
    
    x0 = best.flatten()
    
    # Multiple restarts with small perturbations
    n_restarts = 5
    rng = np.random.RandomState(42)
    
    for restart in range(n_restarts):
        if time.time() - _start_time > _DEADLINE:
            break
        
        if restart == 0:
            x_init = x0.copy()
        else:
            x_init = x0 + rng.normal(0, 0.1, size=x0.shape)
            # Normalize
            for i in range(N):
                x_init[3*i:3*i+3] = _normalize(x_init[3*i:3*i+3])
        
        try:
            result = minimize(
                objective, x_init, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            if result.fun < best_score:
                best = result.x.reshape(-1, 3)
                norms = np.linalg.norm(best, axis=1, keepdims=True)
                norms = np.where(norms < 1e-12, 1.0, norms)
                best = best / norms
                best_score = result.fun
        except:
            pass
    
    return best, best_score


def construct():
    global _start_time
    _start_time = time.time()
    
    # Start with Fibonacci spiral
    points = _fibonacci_spiral(N)
    
    # Refine
    points, score = _refine(points, _DEADLINE - (time.time() - _start_time))
    
    # Final normalization
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    points = points / norms
    
    return points
# EVOLVE-BLOCK-END
