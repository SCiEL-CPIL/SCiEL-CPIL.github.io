"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # area of the equilateral triangle with side 1

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

DEADLINE = 140  # safety margin from 150s limit


def _barycentric_check(pts):
    """Check if all points are inside or on the triangle."""
    for p in pts:
        x, y = p
        # Barycentric coordinates
        den = S3H
        w0 = (S3H - y) / den  # weight for V0
        w1 = (y) / den * (1 - x - (y - S3H * x) / S3H)  # simplified
        # Use standard barycentric
        w0 = (S3H * (1 - x) - S3H * x + y) / S3H  # this is getting messy, let me redo
        pass
    # Better: use the standard formula
    # For triangle with vertices A, B, C, point P is inside if all barycentric coords are in [0,1]
    # A=(0,0), B=(1,0), C=(0.5, S3H)
    # Barycentric: P = a*A + b*B + c*C, a+b+c=1
    # x = b + 0.5*c, y = S3H*c
    # So c = y/S3H, b = x - 0.5*c, a = 1 - b - c
    for p in pts:
        x, y = p
        c = y / S3H
        b = x - 0.5 * c
        a = 1.0 - b - c
        if a < -1e-9 or b < -1e-9 or c < -1e-9:
            return False
        if a > 1 + 1e-9 or b > 1 + 1e-9 or c > 1 + 1e-9:
            return False
    return True


def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    n = len(pts)
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross product|
                ax, ay = pts[i]
                bx, by = pts[j]
                cx, cy = pts[k]
                cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
                area = abs(cross) * 0.5
                if area < min_a:
                    min_a = area
    return min_a


def _objective_flat(flat):
    """Objective function for optimization: negative of minimum triangle area.
    Returns large value if points are outside the triangle."""
    pts = flat.reshape(-1, 2)
    # Check if all points are inside the triangle
    for p in pts:
        x, y = p
        c = y / S3H
        b = x - 0.5 * c
        a = 1.0 - b - c
        if a < -1e-7 or b < -1e-7 or c < -1e-7 or a > 1 + 1e-7 or b > 1 + 1e-7 or c > 1 + 1e-7:
            return 1e6
    return -_min_triangle_area(pts)


def _random_point_in_triangle(rng):
    """Generate a random point inside the triangle using barycentric coordinates."""
    u = rng.random()
    v = rng.random()
    if u + v > 1:
        u = 1 - u
        v = 1 - v
    # Barycentric: a=1-u-v, b=u, c=v
    a = 1 - u - v
    b = u
    c = v
    x = a * 0 + b * 1 + c * 0.5
    y = a * 0 + b * 0 + c * S3H
    return np.array([x, y])


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_area = 0.0
    
    # Initial random configuration
    pts = np.array([_random_point_in_triangle(rng) for _ in range(N)])
    area = _min_triangle_area(pts)
    best_pts = pts.copy()
    best_area = area
    
    # Local optimization from multiple random starts
    n_starts = 50
    for s in range(n_starts):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Generate random starting configuration
        pts = np.array([_random_point_in_triangle(rng) for _ in range(N)])
        
        # Optimize
        try:
            result = minimize(
                _objective_flat,
                pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10}
            )
        except Exception:
            continue
        
        opt_pts = result.x.reshape(-1, 2)
        
        # Check validity
        valid = True
        for p in opt_pts:
            x, y = p
            c = y / S3H
            b = x - 0.5 * c
            a = 1.0 - b - c
            if a < -1e-7 or b < -1e-7 or c < -1e-7 or a > 1 + 1e-7 or b > 1 + 1e-7 or c > 1 + 1e-7:
                valid = False
                break
        
        if not valid:
            continue
        
        area = _min_triangle_area(opt_pts)
        if area > best_area:
            best_area = area
            best_pts = opt_pts.copy()
    
    # If we have time, try more starts with the best configuration perturbed
    if best_pts is not None and time.time() - start_time < DEADLINE - 5:
        for s in range(20):
            if time.time() - start_time > DEADLINE - 5:
                break
            # Perturb the best configuration
            perturbed = best_pts + rng.normal(0, 0.05, size=(N, 2))
            # Clip to triangle (approximate)
            valid = True
            for p in perturbed:
                x, y = p
                c = y / S3H
                b = x - 0.5 * c
                a = 1.0 - b - c
                if a < -1e-7 or b < -1e-7 or c < -1e-7 or a > 1 + 1e-7 or b > 1 + 1e-7 or c > 1 + 1e-7:
                    valid = False
                    break
            if not valid:
                continue
            
            try:
                result = minimize(
                    _objective_flat,
                    perturbed.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-10}
                )
            except Exception:
                continue
            
            opt_pts = result.x.reshape(-1, 2)
            valid = True
            for p in opt_pts:
                x, y = p
                c = y / S3H
                b = x - 0.5 * c
                a = 1.0 - b - c
                if a < -1e-7 or b < -1e-7 or c < -1e-7 or a > 1 + 1e-7 or b > 1 + 1e-7 or c > 1 + 1e-7:
                    valid = False
                    break
            
            if not valid:
                continue
            
            area = _min_triangle_area(opt_pts)
            if area > best_area:
                best_area = area
                best_pts = opt_pts.copy()
    
    if best_pts is None:
        # Fallback: return random points
        return np.array([_random_point_in_triangle(rng) for _ in range(N)])
    
    return best_pts
# EVOLVE-BLOCK-END
