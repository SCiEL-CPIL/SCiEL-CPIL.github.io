import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.optimize import minimize
import math

N = 130

def _normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def _spherical_circumradius(p1, p2, p3):
    """
    Calculate the circumradius of a spherical triangle defined by 3 unit vectors.
    The circumradius is the angular distance from the circumcenter to any vertex.
    """
    # Calculate side lengths (angular distances)
    a = np.arccos(np.clip(np.dot(p2, p3), -1.0, 1.0))
    b = np.arccos(np.clip(np.dot(p1, p3), -1.0, 1.0))
    c = np.arccos(np.clip(np.dot(p1, p2), -1.0, 1.0))
    
    # Spherical law of cosines to find angles? No, we need circumradius R.
    # Formula for spherical circumradius R:
    # sin(R) = sin(a)sin(b)sin(c) / (2 * sin(S)) where S is spherical area?
    # Or use the formula:
    # 1/sin^2(R) = 1/sin^2(a) + 1/sin^2(b) + 1/sin^2(c) - (cos^2(a)+cos^2(b)+cos^2(c)+2cos(a)cos(b)cos(c))/sin^2(S) ?
    
    # Simpler: Find the circumcenter O.
    # O is equidistant from p1, p2, p3.
    # O is in the plane defined by p1, p2, p3.
    # Let O = x*p1 + y*p2 + z*p3.
    # O dot p1 = O dot p2 = O dot p3 = cos(R)
    
    # Construct matrix M where rows are p1, p2, p3.
    M = np.array([p1, p2, p3])
    # We want O such that M @ coeffs = cos(R) * ones(3) ? No.
    # O dot p_i = cos(R).
    # Let O = sum c_i p_i.
    # Then (sum c_j p_j) dot p_i = cos(R).
    # sum c_j (p_j dot p_i) = cos(R).
    # Let G be the Gram matrix G_ij = p_i dot p_j.
    # G @ c = cos(R) * np.ones(3).
    
    G = np.array([
        [np.dot(p1, p1), np.dot(p1, p2), np.dot(p1, p3)],
        [np.dot(p2, p1), np.dot(p2, p2), np.dot(p2, p3)],
        [np.dot(p3, p1), np.dot(p3, p2), np.dot(p3, p3)]
    ])
    
    # G is 3x3. c is 3x1.
    # We also know that O is a unit vector? No, O is the circumcenter on the sphere, so |O|=1.
    # So we have 4 equations (3 dot products + norm) and 3 unknowns (coeffs) + 1 unknown (cos R).
    # Actually, we can just solve for c in terms of cos R, then use norm constraint.
    
    # G is positive definite if points are not collinear.
    try:
        # c = G^-1 @ (cos R * ones)
        # Let v = G^-1 @ ones. Then c = cos R * v.
        v = np.linalg.solve(G, np.ones(3))
    except np.linalg.LinAlgError:
        return 1.0 # Degenerate, return max possible
    
    # O = c1 p1 + c2 p2 + c3 p3 = cos R * (v1 p1 + v2 p2 + v3 p3)
    # Let w = v1 p1 + v2 p2 + v3 p3.
    w = np.dot(v, M)
    
    # O = cos R * w.
    # |O|^2 = 1 => cos^2 R * |w|^2 = 1 => cos R = 1/sqrt(|w|^2) (assuming cos R > 0, i.e., R < 90 deg)
    w_norm = np.linalg.norm(w)
    if w_norm < 1e-9:
        return 1.0
        
    cos_R = 1.0 / w_norm
    # Check if cos_R is valid (<= 1)
    if cos_R > 1.0:
        cos_R = 1.0
    R = np.arccos(cos_R)
    return R

def _covering_radius(points):
    """
    Calculate the covering radius of the configuration.
    """
    try:
        # Delaunay triangulation in 3D
        tri = Delaunay(points)
        # Get facets of the convex hull
        # tri.simplices contains all tetrahedra. We need the surface facets.
        # A facet is a triangle that belongs to only one tetrahedron.
        # Alternatively, use ConvexHull.
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        facets = hull.simplices
        
        max_r = 0.0
        worst_facet_idx = -1
        for i in range(len(facets)):
            f = facets[i]
            p1 = points[f[0]]
            p2 = points[f[1]]
            p3 = points[f[2]]
            r = _spherical_circumradius(p1, p2, p3)
            if r > max_r:
                max_r = r
                worst_facet_idx = i
        
        return max_r, worst_facet_idx, facets[worst_facet_idx] if worst_facet_idx >= 0 else None
    except Exception:
        return 1.0, -1, None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Step 1: Spread points using electrostatic repulsion (Thomson problem approximation)
    # Minimize sum of 1/d^2
    def repulsion_energy(pts):
        # Calculate pairwise distances
        # Use broadcasting
        dots = np.dot(pts, pts.T)
        # Distance squared = 2 - 2*dots
        d2 = 2.0 - 2.0 * dots
        np.fill_diagonal(d2, 1e-9) # Avoid div by 0
        inv_d2 = 1.0 / d2
        np.fill_diagonal(inv_d2, 0.0)
        return np.sum(inv_d2)
        
    # Run a few iterations of gradient descent or L-BFGS for repulsion
    # To save time, we will do a simple local search on repulsion first.
    # Actually, L-BFGS on 390 variables is fast.
    try:
        res = minimize(repulsion_energy, points, method='L-BFGS-B', options={'maxiter': 50, 'gtol': 1e-4})
        if res.success:
            points = res.x
            # Normalize
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
    except Exception:
        pass
    
    # Step 2: Minimize Covering Radius
    current_r, worst_idx, worst_facet = _covering_radius(points)
    
    best_r = current_r
    best_points = points.copy()
    
    stagnation_count = 0
    
    while time.time() < deadline:
        if worst_facet is None:
            break
            
        p1_idx, p2_idx, p3_idx = worst_facet
        p1 = points[p1_idx]
        p2 = points[p2_idx]
        p3 = points[p3_idx]
        
        # Try to move p1, p2, p3 to reduce the circumradius of this facet
        # Heuristic: Move towards the centroid
        centroid = (p1 + p2 + p3) / 3.0
        centroid = _normalize(centroid)
        
        improved = False
        # Try several step sizes
        for step in [0.1, 0.05, 0.01, 0.001]:
            new_p1 = _normalize(p1 + step * (centroid - p1))
            new_p2 = _normalize(p2 + step * (centroid - p2))
            new_p3 = _normalize(p3 + step * (centroid - p3))
            
            new_points = points.copy()
            new_points[p1_idx] = new_p1
            new_points[p2_idx] = new_p2
            new_points[p3_idx] = new_p3
            
            new_r, new_worst_idx, new_worst_facet = _covering_radius(new_points)
            
            if new_r < current_r:
                points = new_points
                current_r = new_r
                worst_idx = new_worst_idx
                worst_facet = new_worst_facet
                improved = True
                stagnation_count = 0
                break
        
        if not improved:
            # Try random perturbations of the worst facet vertices
            improved = False
            for _ in range(10):
                new_points = points.copy()
                # Perturb p1, p2, p3 randomly
                for idx in [p1_idx, p2_idx, p3_idx]:
                    p = points[idx]
                    # Random direction on sphere
                    rand_vec = np.random.randn(3)
                    # Make it orthogonal to p
                    rand_vec = rand_vec - np.dot(rand_vec, p) * p
                    rand_vec = _normalize(rand_vec)
                    new_p = _normalize(p + 0.05 * rand_vec)
                    new_points[idx] = new_p
                
                new_r, new_worst_idx, new_worst_facet = _covering_radius(new_points)
                if new_r < current_r:
                    points = new_points
                    current_r = new_r
                    worst_idx = new_worst_idx
                    worst_facet = new_worst_facet
                    improved = True
                    break
            
            if not improved:
                stagnation_count += 1
                # If stuck for too long, try a larger perturbation or restart local search
                if stagnation_count > 50:
                    # Perturb all points slightly
                    points = points + np.random.randn(N, 3) * 0.01
                    points = points / np.linalg.norm(points, axis=1, keepdims=True)
                    current_r, worst_idx, worst_facet = _covering_radius(points)
                    stagnation_count = 0
                    if current_r < best_r:
                        best_r = current_r
                        best_points = points.copy()
        
        if current_r < best_r:
            best_r = current_r
            best_points = points.copy()
            
    return best_points
