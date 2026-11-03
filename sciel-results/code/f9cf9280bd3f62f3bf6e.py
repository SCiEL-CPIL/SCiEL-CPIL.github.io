"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130
_DEADLINE = 140.0  # safety margin before 150s kill

def _covering_radius_deg(pts):
    """Compute covering radius in degrees via convex hull Delaunay triangulation."""
    pts = np.asarray(pts, dtype=np.float64)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    if np.any(norms < 1e-15):
        return 180.0
    pts = pts / norms
    
    # Check for coincident points
    dots = np.clip(pts @ pts.T, -1.0, 1.0)
    off_diag = dots - np.eye(N)
    if np.min(off_diag) > 1 - 1e-10:
        return 180.0
    
    # Check if origin is inside convex hull (not in a hemisphere)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0
    
    # Compute covering radius as max circumradius over Delaunay triangles
    max_cr = 0.0
    for simplex in hull.simplices:
        if len(simplex) != 3:
            continue
        v0, v1, v2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        
        # Spherical circumcenter: point equidistant from v0, v1, v2 on the sphere
        u = v1 - v0
        v = v2 - v0
        w = np.cross(u, v)
        wn = np.linalg.norm(w)
        if wn < 1e-15:
            # Degenerate, use midpoint of longest edge
            d01 = np.linalg.norm(v0 - v1)
            d02 = np.linalg.norm(v0 - v2)
            d12 = np.linalg.norm(v1 - v2)
            if d01 >= d02 and d01 >= d12:
                c = (v0 + v1) / 2.0
            elif d02 >= d01 and d02 >= d12:
                c = (v0 + v2) / 2.0
            else:
                c = (v1 + v2) / 2.0
            cn = np.linalg.norm(c)
            if cn < 1e-15:
                continue
            c = c / cn
        else:
            c = w / wn
            # Choose the hemisphere containing the face
            if np.dot(c, v0) < 0:
                c = -c
        
        # Angular distance from c to v0 (should be same for v1, v2)
        cos_d = np.clip(np.dot(c, v0), -1.0, 1.0)
        ang = np.arccos(cos_d)
        if ang > max_cr:
            max_cr = ang
    
    return np.degrees(max_cr)


def _fibonacci_sphere(seed=0):
    """Generate n points on the sphere using the Fibonacci spiral with random rotation."""
    rng = np.random.RandomState(seed)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Apply random rotation
    q = rng.randn(4)
    q = q / np.linalg.norm(q)
    w, x, y, zq = q
    R = np.array([
        [1 - 2*(y*y + zq*zq), 2*(x*y - w*zq), 2*(x*zq + w*y)],
        [2*(x*y + w*zq), 1 - 2*(x*x + zq*zq), 2*(y*zq - w*x)],
        [2*(x*zq - w*y), 2*(y*zq + w*x), 1 - 2*(x*x + y*y)]
    ])
    pts = pts @ R.T
    return pts


def _random_sphere(seed=0):
    """Generate n random points on the sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(N, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _lloyd_iterate(pts, max_iter=50, deadline=140.0):
    """
    Lloyd's algorithm for sphere covering.
    1. Compute Voronoi cells (approximated by assigning a grid of points to nearest neighbor).
    2. Move each point to the centroid of its cell.
    """
    pts = np.array(pts, dtype=np.float64)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    
    # Create a fine grid of points on the sphere to approximate the Voronoi cells
    # Using a Fibonacci spiral for the grid as well, with much higher density
    n_grid = 5000
    i_grid = np.arange(n_grid) + 0.5
    z_grid = 1.0 - 2.0 * i_grid / n_grid
    phi_grid = np.pi * (1.0 + np.sqrt(5.0)) * i_grid
    r_grid = np.sqrt(np.maximum(0, 1.0 - z_grid * z_grid))
    grid_pts = np.column_stack([r_grid * np.cos(phi_grid), r_grid * np.sin(phi_grid), z_grid])
    
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        # Assign each grid point to the nearest configuration point
        # Compute distances (using dot product for speed on sphere)
        dots = grid_pts @ pts.T  # (n_grid, N)
        nearest = np.argmax(dots, axis=1)  # (n_grid,)
        
        new_pts = np.zeros_like(pts)
        counts = np.zeros(N)
        
        # Accumulate centroids
        # We need the vector sum of grid points assigned to each site
        # Using np.add.at for efficiency
        np.add.at(new_pts, nearest, grid_pts)
        np.add.at(counts, nearest, 1)
        
        # Handle empty cells (should not happen with enough grid points, but just in case)
        empty = counts == 0
        if np.any(empty):
            # Keep old position for empty cells
            new_pts[empty] = pts[empty]
            counts[empty] = 1  # Avoid division by zero
        
        # Normalize to move to centroid on the sphere
        new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
        
        # Check for convergence (max movement)
        move = np.linalg.norm(new_pts - pts, axis=1)
        if np.max(move) < 1e-6:
            break
            
        pts = new_pts
        
    return pts


def construct():
    start_time = time.time()
    
    # Generate multiple initializations
    inits = []
    for seed in range(10):
        inits.append(_fibonacci_sphere(seed))
    for seed in range(5):
        inits.append(_random_sphere(seed))
    
    # Evaluate all initializations and store
    init_scores = []
    for pts in inits:
        if time.time() > _DEADLINE - 30:
            break
        cr = _covering_radius_deg(pts)
        init_scores.append((cr, pts))
        
    # Sort by covering radius
    init_scores.sort(key=lambda x: x[0])
    
    # Ensure we have at least one valid result
    if not init_scores:
        # Fallback: return a Fibonacci sphere
        return _fibonacci_sphere(0)
        
    best_cr, best_pts = init_scores[0]
    
    # Run Lloyd's algorithm on the top few initializations
    for cr, pts in init_scores[:5]:
        if time.time() > _DEADLINE - 10:
            break
        try:
            opt_pts = _lloyd_iterate(pts, max_iter=30, deadline=_DEADLINE)
            cr_opt = _covering_radius_deg(opt_pts)
            if cr_opt < best_cr:
                best_cr = cr_opt
                best_pts = opt_pts.copy()
        except Exception:
            continue
    
    # Final normalization and validation
    best_pts = np.asarray(best_pts, dtype=np.float64)
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    if np.any(norms < 1e-15):
        return _fibonacci_sphere(0)
    best_pts = best_pts / norms
    
    return best_pts
