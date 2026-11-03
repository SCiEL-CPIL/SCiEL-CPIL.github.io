"""Optimized sphere covering with 130 points via Convex Hull Facet Optimization."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130
DEADLINE = time.time() + 140  # 10s safety margin for the 150s limit

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _covering_radius(pts):
    """
    Compute the exact covering radius in radians using ConvexHull.
    This is the ground truth metric for evaluation.
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi / 2  # Fallback if hull fails (e.g., points in hemisphere)
    
    max_r = 0.0
    for simplex in hull.simplices:
        p = pts[simplex]
        a, b, c = p[0], p[1], p[2]
        v1 = b - a
        v2 = c - a
        cross = np.cross(v1, v2)
        cnorm = np.linalg.norm(cross)
        if cnorm < 1e-12:
            continue
        x = cross / cnorm
        centroid = (a + b + c) / 3.0
        # Ensure normal points outward
        if np.dot(x, centroid) < 0:
            x = -x
        dot_val = np.dot(x, a)
        dot_val = np.clip(dot_val, -1.0, 1.0)
        r = np.arccos(dot_val)
        if r > max_r:
            max_r = r
    return max_r

def _fibonacci_spiral(n):
    j = np.arange(n) + 0.5
    z = 1.0 - 2.0 * j / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * j
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    best_pts = _fibonacci_spiral(N)
    best_cr = _covering_radius(best_pts)
    
    # Strategy: Multi-restart with Facet-Based Gradient Descent
    # Instead of Lloyd's (which moves to Voronoi centroids), we directly minimize
    # the max circumradius of convex hull facets.
    # Gradient for a facet (a,b,c):
    # The circumcenter O is the normalized vector perpendicular to the plane abc.
    # To reduce the circumradius, we should move a, b, c towards O.
    # But we also need to consider the effect on other facets.
    # A simple heuristic: for the worst facet, move its vertices slightly towards its circumcenter.
    
    max_restarts = 10
    iters_per_restart = 100
    
    for restart in range(max_restarts):
        if time.time() > DEADLINE - 10:
            break
            
        # Initialize
        if restart == 0:
            pts = _fibonacci_spiral(N)
            pts += rng.randn(N, 3) * 0.1
        else:
            pts = rng.randn(N, 3)
        
        pts = _normalize(pts)
        current_cr = _covering_radius(pts)
        
        if current_cr < best_cr:
            best_cr = current_cr
            best_pts = pts.copy()
        
        # Facet-based gradient descent
        for it in range(iters_per_restart):
            if time.time() > DEADLINE - 5:
                break
            
            # Compute hull and identify worst facet
            try:
                hull = ConvexHull(pts)
            except Exception:
                break
            
            # Find the facet with maximum circumradius
            max_r = 0.0
            worst_simplex = None
            worst_normal = None
            
            for simplex in hull.simplices:
                p = pts[simplex]
                a, b, c = p[0], p[1], p[2]
                v1 = b - a
                v2 = c - a
                cross = np.cross(v1, v2)
                cnorm = np.linalg.norm(cross)
                if cnorm < 1e-12:
                    continue
                x = cross / cnorm
                centroid = (a + b + c) / 3.0
                if np.dot(x, centroid) < 0:
                    x = -x
                dot_val = np.dot(x, a)
                dot_val = np.clip(dot_val, -1.0, 1.0)
                r = np.arccos(dot_val)
                if r > max_r:
                    max_r = r
                    worst_simplex = simplex
                    worst_normal = x
            
            if worst_simplex is None:
                break
            
            # Move the vertices of the worst facet towards its circumcenter (normal)
            # Step size: proportional to the circumradius
            step_size = 0.01 * max_r
            a_idx, b_idx, c_idx = worst_simplex
            pts[a_idx] += step_size * worst_normal
            pts[b_idx] += step_size * worst_normal
            pts[c_idx] += step_size * worst_normal
            pts = _normalize(pts)
            
            # Evaluate
            current_cr = _covering_radius(pts)
            if current_cr < best_cr:
                best_cr = current_cr
                best_pts = pts.copy()
            
            # Early stopping if no improvement for a while
            if current_cr > best_cr * 1.01:
                break
    
    # Final polish: run a few more iterations on the best candidate
    if time.time() < DEADLINE - 2:
        pts = best_pts.copy()
        for _ in range(50):
            if time.time() > DEADLINE - 1:
                break
            try:
                hull = ConvexHull(pts)
            except Exception:
                break
            
            max_r = 0.0
            worst_simplex = None
            worst_normal = None
            
            for simplex in hull.simplices:
                p = pts[simplex]
                a, b, c = p[0], p[1], p[2]
                v1 = b - a
                v2 = c - a
                cross = np.cross(v1, v2)
                cnorm = np.linalg.norm(cross)
                if cnorm < 1e-12:
                    continue
                x = cross / cnorm
                centroid = (a + b + c) / 3.0
                if np.dot(x, centroid) < 0:
                    x = -x
                dot_val = np.dot(x, a)
                dot_val = np.clip(dot_val, -1.0, 1.0)
                r = np.arccos(dot_val)
                if r > max_r:
                    max_r = r
                    worst_simplex = simplex
                    worst_normal = x
            
            if worst_simplex is None:
                break
            
            step_size = 0.005 * max_r
            a_idx, b_idx, c_idx = worst_simplex
            pts[a_idx] += step_size * worst_normal
            pts[b_idx] += step_size * worst_normal
            pts[c_idx] += step_size * worst_normal
            pts = _normalize(pts)
            
            current_cr = _covering_radius(pts)
            if current_cr < best_cr:
                best_cr = current_cr
                best_pts = pts.copy()
    
    return _normalize(best_pts)

# Test
if __name__ == "__main__":
    pts = construct()
    print("Shape:", pts.shape)
    print("Covering radius (deg):", np.degrees(_covering_radius(pts)))
