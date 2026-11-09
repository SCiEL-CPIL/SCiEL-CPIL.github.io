# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize

def _triangle_areas(pts, tri_idx):
    """Compute all triangle areas vectorized."""
    a = pts[tri_idx[:, 0]]
    b = pts[tri_idx[:, 1]]
    c = pts[tri_idx[:, 2]]
    area = 0.5 * np.abs(
        a[:, 0] * (b[:, 1] - c[:, 1]) +
        b[:, 0] * (c[:, 1] - a[:, 1]) +
        c[:, 0] * (a[:, 1] - b[:, 1])
    )
    return area

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _get_tri_indices(n):
    idx = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx.append((i, j, k))
    return np.array(idx, dtype=int)

def _compute_score(pts, tri_idx):
    """Compute score given points and precomputed triangle indices."""
    if not np.all(np.isfinite(pts)):
        return 0.0
    areas = _triangle_areas(pts, tri_idx)
    min_area = areas.min()
    hull_area = _hull_area(pts)
    if hull_area <= 1e-9:
        return 0.0
    return min_area / hull_area

def _objective(pts_flat, tri_idx):
    """Objective function for optimization (minimize negative score)."""
    pts = pts_flat.reshape(-1, 2)
    score = _compute_score(pts, tri_idx)
    return -score

def _generate_lattice_config(n_points, offset_idx, scale=1.0, rot=0.0):
    """
    Generate a configuration based on a triangular lattice.
    We select points from a larger lattice and center them.
    """
    # Triangular lattice basis vectors
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3)/2])
    
    # Generate a grid of lattice points
    # We need enough points to select n_points from
    R = int(np.ceil(np.sqrt(n_points))) + 2
    pts = []
    for i in range(-R, R+1):
        for j in range(-R, R+1):
            p = i * a1 + j * a2
            pts.append(p)
    pts = np.array(pts)
    
    # Center the lattice points
    pts = pts - pts.mean(axis=0)
    
    # Sort by distance from center to pick the most compact cluster
    dists = np.linalg.norm(pts, axis=1)
    sorted_idx = np.argsort(dists)
    
    # Select n_points closest to center
    selected = pts[sorted_idx[:n_points]]
    
    # Apply scale and rotation
    cos_r, sin_r = np.cos(rot), np.sin(rot)
    rot_mat = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    selected = (selected * scale) @ rot_mat.T
    
    return selected

def heilbronn_convex13() -> np.ndarray:
    n = 13
    tri_idx = _get_tri_indices(n)
    
    start_time = time.time()
    deadline = start_time + 580  # Safety margin for 600s limit
    
    # 1. Generate initial candidates from lattice structures
    candidates = []
    
    # Candidate 0: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    candidates.append(reg_pts)
    
    # Candidate 1-3: Lattice subsets with slight rotations
    for rot in [0.0, np.pi/13, np.pi/7]:
        lat_pts = _generate_lattice_config(13, 0, scale=1.0, rot=rot)
        candidates.append(lat_pts)
        
    # Candidate 4: Slightly perturbed lattice
    lat_pts = _generate_lattice_config(13, 0, scale=1.0, rot=0.0)
    lat_pts += np.random.normal(0, 0.05, lat_pts.shape)
    candidates.append(lat_pts)
    
    # Evaluate initial candidates
    best_score = -1.0
    best_pts = None
    for c in candidates:
        s = _compute_score(c, tri_idx)
        if s > best_score:
            best_score = s
            best_pts = c.copy()
            
    # 2. Global Optimization using Differential Evolution
    # Bounds: The hull area should be kept reasonable. 
    # Since the score is scale-invariant, we can fix the scale by normalizing,
    # but for DE, we just bound the coordinates to a box.
    # A box of [-2, 2] is sufficient for a normalized hull area.
    bounds = [(-2.0, 2.0)] * (n * 2)
    
    # Run DE if we have time
    if time.time() < deadline - 60:
        # DE can be slow, so we limit maxiter and popsize
        # popsize=20, maxiter=50 is a reasonable start for 26 dims
        # We use the best initial guess as the starting point for DE? No, DE is population based.
        # We can seed the population with our best guess and its perturbations.
        
        # Define a custom seed population to help DE
        pop_size = 20
        init_pop = np.zeros((pop_size, n * 2))
        # Fill first few rows with best guess and perturbations
        init_pop[0] = best_pts.flatten()
        for i in range(1, pop_size):
            init_pop[i] = best_pts.flatten() + np.random.normal(0, 0.2, n * 2)
        
        try:
            # time_limit is not directly supported in all versions, so we rely on maxiter
            # 26 dimensions, popsize 20, maxiter 50 -> 1000 function evals.
            # Each eval is ~286 triangle areas + hull. ~1ms? 1s total.
            # We can afford more. Let's go for maxiter=100, popsize=30.
            # 3000 evals ~ 3-5 seconds.
            # We have ~500 seconds left. We can do multiple restarts or a larger run.
            
            result = differential_evolution(
                _objective, 
                bounds, 
                args=(tri_idx,),
                seed=42,
                popsize=30,
                maxiter=100,
                tol=1e-5,
                init=init_pop,
                polish=False # We will polish manually later
            )
            
            if result.success or result.fun < -best_score:
                de_pts = result.x.reshape(-1, 2)
                de_score = _compute_score(de_pts, tri_idx)
                if de_score > best_score:
                    best_score = de_score
                    best_pts = de_pts.copy()
                    
        except Exception as e:
            pass # Ignore errors and continue with current best
            
    # 3. Local Refinement using L-BFGS-B
    # Run multiple local optimizations from the best point and its perturbations
    local_starts = [best_pts]
    # Add some random perturbations around the best point
    for _ in range(5):
        perturbed = best_pts + np.random.normal(0, 0.05, best_pts.shape)
        local_starts.append(perturbed)
        
    for start_pts in local_starts:
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(
                _objective, 
                start_pts.flatten(), 
                args=(tri_idx,),
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.success:
                res_pts = res.x.reshape(-1, 2)
                res_score = _compute_score(res_pts, tri_idx)
                if res_score > best_score:
                    best_score = res_score
                    best_pts = res_pts.copy()
        except Exception:
            pass

    # 4. Final Nelder-Mead polish for precision
    if time.time() < deadline - 5:
        try:
            res_nm = minimize(
                _objective, 
                best_pts.flatten(), 
                args=(tri_idx,),
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            res_pts = res_nm.x.reshape(-1, 2)
            res_score = _compute_score(res_pts, tri_idx)
            if res_score > best_score:
                best_score = res_score
                best_pts = res_pts.copy()
        except Exception:
            pass

    # Normalize to unit hull area for consistency (optional but good practice)
    hull_area = _hull_area(best_pts)
    if hull_area > 1e-9:
        best_pts = best_pts / np.sqrt(hull_area)
        
    # Ensure finite
    if not np.all(np.isfinite(best_pts)):
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_pts

# EVOLVE-BLOCK-END
