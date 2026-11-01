"""Spherical 130-point covering minimisation via Lloyd relaxation + local search."""
import numpy as np
from scipy.spatial import SphericalVoronoi
import time

N = 130
SEED_TIME_LIMIT = 150.0  # evaluator kills at 150 s wall-clock; keep safety margin


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _covering_radius(pts):
    """Compute covering radius (in radians) via spherical Voronoi circumradii."""
    pts = _normalize(pts)
    try:
        sv = SphericalVoronoi(pts, radius=1.0)
    except Exception:
        return np.pi
    # Facets
    facets = sv.facets
    radii = []
    for facet in facets:
        if len(facet) < 3:
            continue
        # Get the vertices of this facet
        vidx = [int(f) for f in facet]
        # If facet has only 2 points, it's an edge - skip (shouldn't happen for full Voronoi)
        if len(vidx) < 3:
            continue
        # Compute circumradius of the spherical polygon
        # For a spherical polygon, the circumradius is the max angular distance from center to vertices
        # But we need the circumradius of the facet as a cap
        # Actually, the covering radius is the max over facets of the circumradius
        # where circumradius = max angular distance from facet center to any vertex
        # But more precisely, for a spherical polygon, the covering radius contribution
        # is the radius of the smallest spherical cap containing the polygon
        # which equals the max angular distance from the polygon's centroid (on sphere) to vertices
        # Actually no - the covering radius is the max over all points on sphere of distance to nearest site
        # This equals the max over all Voronoi facets of the circumradius of that facet
        # The circumradius of a spherical polygon is the radius of the smallest spherical cap
        # containing all vertices, which is the max over vertices of the angular distance
        # from the spherical centroid... no, it's the circumradius in the sense of the
        # smallest enclosing circle on the sphere.
        # 
        # For a spherical polygon, the covering radius contribution is:
        # max_{y in polygon} min_{i in facet} angle(y, v_i)
        # This equals the circumradius of the polygon on the sphere.
        # 
        # A practical approximation: the circumradius is the max angular distance from
        # the spherical mean (normalized sum) to the vertices. But this is only exact
        # if the spherical mean is the circumcenter. For convex spherical polygons with
        # small angular size, this is a good approximation.
        # 
        # More rigorous: the circumradius of a set of points on the sphere is the radius
        # of the smallest spherical cap containing them. This can be found by checking
        # all pairs/triples, but for a Voronoi facet with ~6 vertices, we can use the
        # fact that the circumcenter is the intersection of perpendicular bisectors.
        # 
        # Simple approach: compute the spherical centroid (normalized sum of vertices),
        # then take max angular distance to vertices. This underestimates the true
        # circumradius but is a good proxy.
        # 
        # Better approach for accuracy: the covering radius is the max over all points
        # on the sphere. For each Voronoi cell, the farthest point from the cell's sites
        # is at one of the cell's vertices (Voronoi vertices, i.e., intersection points
        # of three or more sites' perpendicular bisectors on the sphere).
        # 
        # Actually, the covering radius equals the maximum circumradius over all
        # "empty caps" defined by triples of sites that bound a Voronoi vertex.
        # Each Voronoi vertex is equidistant to 3 (or more) sites, and the angular
        # distance to those sites is the circumradius for that vertex.
        # The covering radius is the max of these circumradii.
        
        # Get the Voronoi vertices (points on sphere equidistant to 3+ sites)
        pass
    # Use sv's vertices directly
    vertices = sv.vertices  # These are points on the unit sphere
    # For each vertex, compute distance to nearest site
    if len(vertices) == 0:
        return np.pi
    # Distance from each vertex to all sites
    # vertices: (M, 3), pts: (N, 3)
    # cos_angles = vertices @ pts.T  (M, N)
    cos_angles = vertices @ pts.T
    # Clip for numerical safety
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    min_cos = np.min(cos_angles, axis=1)  # (M,)
    max_min_angle = np.arccos(min_cos)  # (M,)
    return float(np.max(max_min_angle))


def _lloyd_iteration(pts, num_iters=50):
    """Lloyd relaxation on the sphere."""
    pts = _normalize(pts.copy())
    for it in range(num_iters):
        try:
            sv = SphericalVoronoi(pts, radius=1.0)
        except Exception:
            break
        # Compute the centroid of each Voronoi cell
        # sv.points gives the original points
        # We need to compute the spherical centroid of each cell
        # SphericalVoronoi provides region info via .regions and .vertices
        # For each region (Voronoi cell), the vertices are the points on the sphere
        # bounding the cell. The centroid is the normalized sum of the vertices
        # weighted by area, but a simple normalized sum works as an approximation.
        
        # Actually, SphericalVoronoi has a method to get the centroid?
        # Let's compute it manually: for each point i, find its Voronoi cell vertices
        # and take the normalized mean.
        
        # sv.regions: list of lists of vertex indices
        # sv.vertices: (M, 3) array of Voronoi vertices on the sphere
        
        new_pts = np.zeros_like(pts)
        for i in range(len(pts)):
            region = sv.regions[i]
            if len(region) == 0:
                new_pts[i] = pts[i]
                continue
            # Get the vertices of this region
            region_verts = sv.vertices[region]
            # Spherical centroid: normalized sum
            centroid = np.sum(region_verts, axis=0)
            norm = np.linalg.norm(centroid)
            if norm < 1e-10:
                new_pts[i] = pts[i]
            else:
                new_pts[i] = centroid / norm
        pts = _normalize(new_pts)
    return pts


def _random_perturbation(pts, sigma):
    """Add random Gaussian noise and renormalize."""
    noise = np.random.randn(*pts.shape) * sigma
    return _normalize(pts + noise)


def _local_search(pts, max_time=None, rng=None):
    """Local search with decreasing perturbation sizes."""
    if rng is None:
        rng = np.random.default_rng(42)
    
    best_pts = _normalize(pts.copy())
    best_radius = _covering_radius(best_pts)
    start_time = time.time()
    
    # Schedule of perturbation sizes
    sigma_schedule = [0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001]
    
    for sigma in sigma_schedule:
        if max_time is not None and time.time() - start_time > max_time:
            break
        
        n_attempts = 100
        for attempt in range(n_attempts):
            if max_time is not None and time.time() - start_time > max_time:
                break
            
            # Perturb
            cand = _random_perturbation(best_pts, sigma)
            
            # Lloyd relaxation
            cand = _lloyd_iteration(cand, num_iters=20)
            
            # Evaluate
            radius = _covering_radius(cand)
            
            if radius < best_radius - 1e-10:
                best_radius = radius
                best_pts = cand
                # Restart with smaller sigma immediately
                break
        
        # Also do a pure Lloyd refinement on the best found
        best_pts = _lloyd_iteration(best_pts, num_iters=50)
        best_radius = _covering_radius(best_pts)
    
    return best_pts, best_radius


def construct():
    start_time = time.time()
    deadline = start_time + SEED_TIME_LIMIT - 10.0  # 10s safety margin
    
    rng = np.random.default_rng(12345)
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    best_pts = pts.copy()
    best_radius = _covering_radius(pts)
    
    # Phase 1: Initial Lloyd relaxation
    elapsed = time.time() - start_time
    if elapsed < deadline - 30.0:
        pts_lloyd = _lloyd_iteration(pts, num_iters=100)
        r_lloyd = _covering_radius(pts_lloyd)
        if r_lloyd < best_radius:
            best_pts = pts_lloyd
            best_radius = r_lloyd
    
    # Phase 2: Multiple random restarts with local search
    n_restarts = 8
    for restart in range(n_restarts):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 15.0:
            break
        
        # Create a random starting point
        # Mix of random and perturbed Fibonacci
        if restart % 2 == 0:
            # Random points
            cand = rng.standard_normal((N, 3))
            cand = _normalize(cand)
        else:
            # Perturbed Fibonacci
            cand = _random_perturbation(pts, sigma=0.5)
        
        # Time budget for this restart
        time_per_restart = min(remaining / (n_restarts - restart), 20.0)
        
        cand, rad = _local_search(cand, max_time=time_per_restart, rng=rng)
        
        if rad < best_radius:
            best_pts = cand
            best_radius = rad
    
    # Phase 3: Final high-precision polish
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    if remaining > 5.0:
        best_pts = _lloyd_iteration(best_pts, num_iters=200)
        best_radius = _covering_radius(best_pts)
    
    # Ensure output is valid
    best_pts = _normalize(best_pts)
    
    # Final safety check
    if not np.all(np.isfinite(best_pts)):
        # Fall back to Fibonacci
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        best_pts = _normalize(best_pts)
    
    return best_pts
