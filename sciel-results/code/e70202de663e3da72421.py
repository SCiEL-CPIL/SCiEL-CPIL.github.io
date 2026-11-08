"""Heilbronn triangles: maximize min triangle area for 11 points in unit-area equilateral triangle."""
import numpy as np
import random
import time
from itertools import combinations

N = 11
AREA = np.sqrt(3) / 4.0  # area of equilateral triangle with side 1

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])


def _min_triangle_area(pts):
    """Compute minimum area of any triangle formed by 3 of the n points. Vectorized."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            # Vectorize over k > j
            v1 = pts[j] - pts[i]
            v2 = pts[j + 1:] - pts[i]
            areas = 0.5 * np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area


def _is_inside_triangle(p):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    x, y = p
    if y < -1e-12 or y > np.sqrt(3) / 2 + 1e-12:
        return False
    if x < -1e-12 or x > 1 + 1e-12:
        return False
    # Check left edge: line from (0,0) to (0.5, sqrt(3)/2): y = sqrt(3)*x
    if y > np.sqrt(3) * x + 1e-12:
        return False
    # Check right edge: line from (1,0) to (0.5, sqrt(3)/2): y = -sqrt(3)*(x-1)
    if y > -np.sqrt(3) * (x - 1) + 1e-12:
        return False
    return True


def _project_to_triangle(p):
    """Project point p onto the equilateral triangle (closest point in triangle)."""
    x, y = p
    # Simple approach: if outside, move toward centroid
    cx, cy = 0.5, np.sqrt(3) / 6.0
    if _is_inside_triangle(p):
        return np.array([x, y])
    # Move toward centroid
    t = 0.5
    for _ in range(20):
        q = np.array([x + t * (cx - x), y + t * (cy - y)])
        if _is_inside_triangle(q):
            return q
        t *= 0.5
    return np.array([cx, cy])


def _local_optimize(pts, max_iter=200, step_init=0.05):
    """Coordinate descent: move each point to improve min triangle area."""
    pts = pts.copy()
    n = len(pts)
    current_min = _min_triangle_area(pts)
    step = step_init
    no_improve = 0
    
    for it in range(max_iter):
        improved = False
        for i in range(n):
            best_pt = pts[i].copy()
            best_min = current_min
            # Try moving point i in various directions
            for dx in np.linspace(-step, step, 9):
                for dy in np.linspace(-step, step, 9):
                    new_pt = pts[i] + np.array([dx, dy])
                    if not _is_inside_triangle(new_pt):
                        new_pt = _project_to_triangle(new_pt)
                    if np.allclose(new_pt, pts[i]):
                        continue
                    pts[i] = new_pt
                    new_min = _min_triangle_area(pts)
                    if new_min > best_min + 1e-12:
                        best_min = new_min
                        best_pt = new_pt.copy()
                    else:
                        pts[i] = best_pt.copy()
            if np.any(best_pt != pts[i]):
                pts[i] = best_pt
                current_min = best_min
                improved = True
        
        if not improved:
            step *= 0.7
            no_improve += 1
            if no_improve > 10 or step < 1e-6:
                break
        else:
            no_improve = 0
    
    return pts, current_min


def _generate_structured_init(rng):
    """Generate a structured initial configuration using barycentric coordinates."""
    # Use a grid-like arrangement in barycentric coordinates
    # For 11 points, try different patterns
    pattern = rng.randint(0, 3)
    
    if pattern == 0:
        # 3-4-4 grid in barycentric
        pts = []
        # Row 0: 3 points
        for i in range(3):
            a, b, c = 0.0, i / 2.0, 1.0 - i / 2.0
            p = a * V0 + b * V1 + c * V2
            pts.append(p + rng.uniform(-0.02, 0.02, 2))
        # Row 1: 4 points
        for i in range(4):
            a, b, c = 1/6.0, i / 3.0, 1.0 - 1/6.0 - i / 3.0
            p = a * V0 + b * V1 + c * V2
            pts.append(p + rng.uniform(-0.02, 0.02, 2))
        # Row 2: 4 points
        for i in range(4):
            a, b, c = 2/6.0, i / 3.0, 1.0 - 2/6.0 - i / 3.0
            p = a * V0 + b * V1 + c * V2
            pts.append(p + rng.uniform(-0.02, 0.02, 2))
    elif pattern == 1:
        # Hexagonal-ish arrangement
        pts = []
        # Center
        pts.append(np.array([0.5, np.sqrt(3) / 6.0]) + rng.uniform(-0.03, 0.03, 2))
        # 6 around center
        for ang in range(6):
            r = 0.25
            theta = ang * np.pi / 3.0 + rng.uniform(-0.1, 0.1)
            p = np.array([0.5 + r * np.cos(theta), np.sqrt(3) / 6.0 + r * np.sin(theta)])
            pts.append(p)
        # 4 more near edges
        for i in range(4):
            t = rng.uniform(0.1, 0.9)
            if i < 2:
                p = t * V1 + (1 - t) * V0 + rng.uniform(-0.02, 0.02, 2)
            else:
                p = t * V2 + (1 - t) * V1 + rng.uniform(-0.02, 0.02, 2)
            pts.append(p)
    elif pattern == 2:
        # Random barycentric with some structure
        pts = []
        # Corners with jitter
        for v in [V0, V1, V2]:
            pts.append(v + rng.uniform(0.05, 0.15, 2) * np.array([1, 0.577]))
            pts[-1] = _project_to_triangle(pts[-1])
        # Edge midpoints with jitter
        for v1, v2 in [(V0, V1), (V1, V2), (V2, V0)]:
            mid = 0.5 * (v1 + v2)
            pts.append(mid + rng.uniform(-0.05, 0.05, 2))
            pts[-1] = _project_to_triangle(pts[-1])
        # Interior points
        for _ in range(2):
            a, b, c = rng.random(), rng.random(), rng.random()
            s = a + b + c
            a, b, c = a / s, b / s, c / s
            p = a * V0 + b * V1 + c * V2
            pts.append(p)
    else:
        # Pure random
        pts = []
        for _ in range(N):
            a, b, c = rng.random(), rng.random(), rng.random()
            s = a + b + c
            a, b, c = a / s, b / s, c / s
            p = a * V0 + b * V1 + c * V2
            pts.append(p)
    
    # Project all to triangle
    pts = [_project_to_triangle(p) for p in pts]
    return np.array(pts)


def _generate_random_init(rng):
    """Generate random points inside the triangle."""
    pts = []
    for _ in range(N):
        a, b, c = rng.random(), rng.random(), rng.random()
        s = a + b + c
        a, b, c = a / s, b / s, c / s
        p = a * V0 + b * V1 + c * V2
        # Add small jitter
        p = p + rng.uniform(-0.01, 0.01, 2)
        pts.append(_project_to_triangle(p))
    return np.array(pts)


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    rng = random.Random(42)
    best_pts = None
    best_min = 0.0
    
    # Try multiple restarts with different strategies
    n_restarts = 0
    
    while time.time() < deadline:
        n_restarts += 1
        # Alternate between structured and random inits
        if n_restarts % 3 == 0:
            pts = _generate_random_init(rng)
        else:
            pts = _generate_structured_init(rng)
        
        # Local optimization
        pts_opt, min_area = _local_optimize(pts, max_iter=150, step_init=0.05)
        
        if min_area > best_min:
            best_min = min_area
            best_pts = pts_opt.copy()
        
        # If we've found a good solution, do finer local search
        if best_min > 0.01 and time.time() < deadline - 30:
            # Try smaller steps
            pts_fine, min_fine = _local_optimize(best_pts, max_iter=100, step_init=0.01)
            if min_fine > best_min:
                best_min = min_fine
                best_pts = pts_fine.copy()
        
        # Check if we should stop early if convergence seems good
        if best_min > 0.03 and n_restarts > 10:
            break
    
    # Final fine-tuning if time permits
    if best_pts is not None and time.time() < deadline - 10:
        best_pts, best_min = _local_optimize(best_pts, max_iter=50, step_init=0.005)
    
    if best_pts is None:
        # Fallback: return random points
        pts = _generate_random_init(rng)
        return pts
    
    return best_pts
