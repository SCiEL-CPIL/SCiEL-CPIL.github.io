import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

# Precompute triangle indices once
_IDX = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_I, _J, _K = _IDX[:, 0], _IDX[:, 1], _IDX[:, 2]
_NTRIS = len(_IDX)

def compute_score(points):
    """Compute min triangle area / convex hull area, vectorized."""
    n = len(points)
    if n < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0

    p_i = points[_I]
    p_j = points[_J]
    p_k = points[_K]

    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)

    min_area = np.min(areas)
    if min_area < 1e-12:
        return 0.0
    return min_area / hull_area

def neg_score(x):
    points = x.reshape((N, 2))
    return -compute_score(points)

def _build_convex_polygon(n_hull, angles, radii):
    """Build convex polygon vertices from angles and radii."""
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack((x, y))

def _interior_point_at_centroid(tri_verts):
    """Return centroid of a triangle given 3 vertices."""
    return np.mean(tri_verts, axis=0)

def _place_interior_points(hull_pts, n_interior, mode='centroid_fan'):
    """
    Place interior points using geometric constructions.
    mode: 'centroid_fan' - use centroids of triangles formed by center and consecutive hull edges
          'grid' - use a grid of points inside the hull
          'rings' - use concentric ring positions
    """
    n_hull = len(hull_pts)
    center = np.mean(hull_pts, axis=0)
    
    if mode == 'centroid_fan':
        # Create triangles from center to each edge, place points at centroids
        interior_pts = []
        for i in range(n_hull):
            j = (i + 1) % n_hull
            tri = np.array([center, hull_pts[i], hull_pts[j]])
            interior_pts.append(_interior_point_at_centroid(tri))
        # If we need fewer, subsample; if more, add more rings
        if len(interior_pts) >= n_interior:
            # Subsample evenly
            indices = np.linspace(0, len(interior_pts) - 1, n_interior, dtype=int)
            return np.array([interior_pts[i] for i in indices])
        else:
            # Add more by scaling
            result = list(interior_pts)
            while len(result) < n_interior:
                for i in range(n_hull):
                    if len(result) >= n_interior:
                        break
                    j = (i + 1) % n_hull
                    tri = np.array([center, hull_pts[i], hull_pts[j]])
                    cent = _interior_point_at_centroid(tri)
                    # Scale toward center
                    scaled = center + 0.5 * (cent - center)
                    result.append(scaled)
            return np.array(result[:n_interior])
    
    elif mode == 'rings':
        # Place interior points on concentric rings
        interior_pts = []
        n_rings = min(3, n_interior)
        per_ring = [n_interior // n_rings + (1 if i < n_interior % n_rings else 0) for i in range(n_rings)]
        for r_idx in range(n_rings):
            radius = 0.7 * (1 - r_idx / n_rings)
            n_pts = per_ring[r_idx]
            if n_pts == 0:
                continue
            angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False) + r_idx * np.pi / n_rings
            for a in angles:
                interior_pts.append(center + radius * np.array([np.cos(a), np.sin(a)]))
        return np.array(interior_pts[:n_interior])
    
    else:  # grid
        # Simple grid inside bounding box, filter to inside hull
        interior_pts = []
        n_per_dim = int(np.ceil(np.sqrt(n_interior)))
        xs = np.linspace(-0.5, 0.5, n_per_dim)
        ys = np.linspace(-0.5, 0.5, n_per_dim)
        for x in xs:
            for y in ys:
                pt = np.array([x, y])
                interior_pts.append(pt)
        # Keep only first n_interior
        return np.array(interior_pts[:n_interior])

def _score_from_params(n_hull, angles, radii, n_interior, mode='centroid_fan'):
    """Score a configuration given hull params and interior placement mode."""
    hull_pts = _build_convex_polygon(n_hull, angles, radii)
    if n_interior > 0:
        interior_pts = _place_interior_points(hull_pts, n_interior, mode)
        all_pts = np.vstack((hull_pts, interior_pts))
    else:
        all_pts = hull_pts
    
    if len(all_pts) != N:
        # Pad or trim
        if len(all_pts) < N:
            center = np.mean(all_pts, axis=0)
            pad = np.tile(center, (N - len(all_pts), 1)) + 0.01 * np.random.randn(N - len(all_pts), 2)
            all_pts = np.vstack((all_pts, pad))
        else:
            all_pts = all_pts[:N]
    
    return compute_score(all_pts)

def _optimize_hull_only(n_hull, n_interior, mode='centroid_fan', x0=None, maxiter=300):
    """Optimize hull angles and radii only, with interior points placed geometrically."""
    if x0 is None:
        # Regular polygon initialization
        angles = np.linspace(0, 2 * np.pi, n_hull, endpoint=False)
        radii = np.ones(n_hull)
        x0 = np.concatenate([angles, radii])
    
    def objective(x):
        angles = x[:n_hull]
        radii = x[n_hull:]
        # Ensure radii are positive
        radii = np.maximum(radii, 0.1)
        return -_score_from_params(n_hull, angles, radii, n_interior, mode)
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-7, 'ftol': 1e-9})
        return res.x
    except Exception:
        return x0

def _optimize_full(points, maxiter=500):
    """Full 2D optimization of all 14 points."""
    x0 = points.flatten()
    try:
        res = minimize(neg_score, x0, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-8, 'ftol': 1e-10})
        return res.x.reshape((N, 2))
    except Exception:
        return points

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    best_points = None
    best_score = -float('inf')
    rng = np.random.default_rng(42)

    def try_update(points):
        nonlocal best_points, best_score
        s = compute_score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
        return s

    def remaining():
        return deadline - (time.time() - start_time)

    # --- Phase 1: Convex polygon parameterization with geometric interior placement ---
    # Try different splits: n_hull + n_interior = 14
    splits = [(14, 0), (12, 2), (11, 3), (10, 4), (9, 5), (8, 6), (7, 7)]
    modes = ['centroid_fan', 'rings']
    
    for n_hull, n_interior in splits:
        if remaining() < 15:
            break
        for mode in modes:
            if remaining() < 10:
                break
            # Regular polygon init
            angles = np.linspace(0, 2 * np.pi, n_hull, endpoint=False)
            radii = np.ones(n_hull)
            x0 = np.concatenate([angles, radii])
            
            x_opt = _optimize_hull_only(n_hull, n_interior, mode, x0, maxiter=200)
            angles_opt = x_opt[:n_hull]
            radii_opt = np.maximum(x_opt[n_hull:], 0.1)
            
            hull_pts = _build_convex_polygon(n_hull, angles_opt, radii_opt)
            if n_interior > 0:
                interior_pts = _place_interior_points(hull_pts, n_interior, mode)
                all_pts = np.vstack((hull_pts, interior_pts))
            else:
                all_pts = hull_pts
            
            if len(all_pts) == N:
                try_update(all_pts)
    
    # --- Phase 2: Diverse initial points for full 2D optimization ---
    inits = []
    
    # Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    p_reg = np.column_stack((np.cos(angles), np.sin(angles)))
    inits.append(p_reg)
    
    # Two concentric rings
    for n_out in [6, 7, 8, 9]:
        n_in = N - n_out
        a_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
        for ratio in [0.4, 0.5, 0.6, 0.7]:
            for offset in [0, np.pi / n_out]:
                a_in = np.linspace(0, 2 * np.pi, n_in, endpoint=False) + offset
                p_out = np.column_stack((np.cos(a_out), np.sin(a_out)))
                p_in = np.column_stack((ratio * np.cos(a_in), ratio * np.sin(a_in)))
                inits.append(np.vstack((p_out, p_in)))
    
    # Jittered regular polygon
    for _ in range(5):
        inits.append(p_reg + rng.normal(0, 0.1, (N, 2)))
    
    # Random in disk
    for _ in range(10):
        r = np.sqrt(rng.random(N))
        th = rng.random(N) * 2 * np.pi
        inits.append(np.column_stack((r * np.cos(th), r * np.sin(th))))
    
    # Evaluate all inits
    for p in inits:
        try_update(p)
    
    # Sort by score
    scored = [(compute_score(p), p) for p in inits]
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # --- Phase 3: Full 2D Nelder-Mead from top inits ---
    for init_score, init_points in scored[:6]:
        if remaining() < 15:
            break
        new_p = _optimize_full(init_points, maxiter=400)
        try_update(new_p)
    
    # --- Phase 4: Powell refinement from best ---
    if best_points is not None and remaining() > 20:
        x0 = best_points.flatten()
        try:
            res = minimize(neg_score, x0, method='Powell',
                           options={'maxiter': 1000, 'xtol': 1e-9, 'ftol': 1e-11})
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass
    
    # --- Phase 5: Basin hopping ---
    if best_points is not None and remaining() > 25:
        for step in [0.3, 0.5, 0.7]:
            if remaining() < 15:
                break
            x0 = best_points.flatten()
            res = minimize(neg_score, x0, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-8, 'ftol': 1e-10})
            for _ in range(5):
                if remaining() < 10:
                    break
                perturbation = rng.normal(0, step, (N, 2))
                x_pert = res.x + perturbation.flatten()
                res2 = minimize(neg_score, x_pert, method='Nelder-Mead',
                                options={'maxiter': 150, 'xatol': 1e-8, 'ftol': 1e-10})
                if res2.fun < res.fun:
                    res = res2
            try_update(res.x.reshape((N, 2)))
    
    # --- Phase 6: Tight polish ---
    if best_points is not None and remaining() > 12:
        x0 = best_points.flatten()
        try:
            res = minimize(neg_score, x0, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-10, 'ftol': 1e-12})
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass
    
    # --- Phase 7: Final Powell polish ---
    if best_points is not None and remaining() > 8:
        x0 = best_points.flatten()
        try:
            res = minimize(neg_score, x0, method='Powell',
                           options={'maxiter': 2000, 'xtol': 1e-10, 'ftol': 1e-12})
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass
    
    if best_points is None:
        return p_reg
    return best_points
