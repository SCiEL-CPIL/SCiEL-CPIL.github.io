"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill
    
    # The goal is to minimize the covering radius.
    # Covering radius = max over all y on sphere of min_i angular_dist(y, p_i)
    # This is equivalent to the circumradius of the Delaunay triangulation (spherical).
    # We can approximate this by minimizing a soft-max of the angular distances to the nearest neighbor.
    # However, a better approach for "covering" is to consider the convex hull of the points.
    # The covering radius is the maximum circumradius of the spherical triangles (facets) of the convex hull.
    
    # Strategy:
    # 1. Use a multi-start local search.
    # 2. Objective function: A smooth approximation of the max angular distance to the nearest point.
    #    For a given point y, d(y) = min_i arccos(y . p_i).
    #    We want to minimize max_y d(y).
    #    This is hard to optimize directly over y.
    #    Alternative: Minimize the sum of the squared circumradii of the Delaunay triangles, 
    #    or simply use a "covering energy" that penalizes large empty caps.
    
    # A standard "covering" objective is the sum of 1/d_min(y)^k for a grid of points y, 
    # but that's expensive.
    
    # Let's try a different approach: 
    # The covering radius is determined by the farthest point in the Voronoi cells.
    # For a set of points on a sphere, the farthest point in a Voronoi cell is the circumcenter of the 
    # Delaunay triangle containing it (if the circumcenter lies inside the triangle).
    # So, we can compute the Delaunay triangulation, find the circumcenter of each triangle,
    # and the covering radius is the max angular distance from these circumcenters to the nearest point.
    
    # Since we can't differentiate the Delaunay triangulation easily, we can use a heuristic:
    # Minimize the maximum circumradius of the spherical triangles formed by the points.
    # We can approximate this by minimizing the sum of the squared circumradii, or using a soft-max.
    
    # Let's define the objective as the soft-max of the circumradii of all triples of points.
    # But there are O(N^3) triples. N=50 -> 125000 triples. This is feasible.
    
    def covering_energy(x):
        pts = x.reshape(-1, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        pts = pts / norms
        
        # Compute all pairwise dot products
        dots = pts @ pts.T
        # We need circumradius for each triangle (i, j, k).
        # Circumradius R of a spherical triangle with side lengths a, b, c (angular distances)
        # can be computed. But it's easier to compute the distance from the circumcenter to the vertices.
        
        # A simpler proxy: The covering radius is bounded by the max circumradius of the Delaunay triangles.
        # We can approximate the covering radius by the max over all triples (i,j,k) of the 
        # distance from the centroid (normalized) to the nearest point? No, that's not right.
        
        # Let's use a different energy: The "covering" energy is often approximated by
        # E = sum_i (1 / d_i^2) where d_i is the distance to the nearest neighbor? No, that's Tammes.
        
        # For covering, we want to minimize the largest empty cap.
        # Let's try minimizing the max over a set of sample points on the sphere of the min distance to the config.
        # We can use a fixed set of sample points (e.g., a Fibonacci sphere with 10000 points).
        
        return 0.0 # Placeholder
    
    # Actually, let's use the Delaunay approach directly in the evaluation, but for optimization,
    # we can use a gradient-free method like Nelder-Mead or CMA-ES if available, or just L-BFGS-B with a different objective.
    
    # Let's stick to the Tammes energy but refine it with a local search that specifically targets the covering radius.
    # Or, we can use the fact that the covering radius is the max circumradius of the Delaunay triangles.
    # We can compute this for a given configuration.
    
    def compute_covering_radius(pts):
        # Compute Delaunay triangulation
        try:
            hull = ConvexHull(pts)
            # The faces of the convex hull are the Delaunay triangles.
            # For each face, compute the circumcenter and the distance to the vertices.
            max_r = 0.0
            for face in hull.simplices:
                p0, p1, p2 = pts[face[0]], pts[face[1]], pts[face[2]]
                # Compute circumcenter of the spherical triangle
                # The circumcenter is the point equidistant from p0, p1, p2 on the sphere.
                # It lies in the plane defined by p0, p1, p2.
                # We can find it by solving the linear system.
                # Let c = a*p0 + b*p1 + d*p2.
                # c.p0 = c.p1 = c.p2 = R^2.
                # This is a bit complex. Let's use a simpler approximation:
                # The circumcenter of a spherical triangle is the intersection of the perpendicular bisectors.
                # For a small triangle, it's close to the centroid.
                # Let's just use the centroid as an approximation for the "worst point" in the cell.
                centroid = (p0 + p1 + p2) / 3.0
                centroid /= np.linalg.norm(centroid)
                # Distance from centroid to nearest point
                dists = np.arccos(np.clip(pts @ centroid, -1.0, 1.0))
                min_dist = np.min(dists)
                if min_dist > max_r:
                    max_r = min_dist
            return np.degrees(max_r)
        except Exception:
            return 180.0 # Invalid
    
    # We will use a multi-start approach with L-BFGS-B on the Tammes energy, 
    # but then we will evaluate the covering radius for the best candidates.
    
    def energy(x):
        pts = x.reshape(-1, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        pts = pts / norms
        dots = pts @ pts.T
        np.fill_diagonal(dots, 2.0)
        d2 = 2.0 - 2.0 * dots
        d2 = np.clip(d2, 1e-12, None)
        mask = ~np.eye(N, dtype=bool)
        return np.sum(1.0 / d2[mask])
    
    best_pts = None
    best_cr = 180.0
    rng = np.random.default_rng(42)
    
    max_restarts = 20
    
    for k in range(max_restarts):
        if time.time() > deadline:
            break
        
        init_pts = rng.normal(size=(N, 3))
        init_pts /= np.linalg.norm(init_pts, axis=1, keepdims=True)
        x0 = init_pts.flatten()
        
        res = minimize(energy, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.success or res.fun < np.inf:
            pts = res.x.reshape(-1, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            
            cr = compute_covering_radius(pts)
            if cr < best_cr:
                best_cr = cr
                best_pts = pts
    
    if best_pts is None:
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
    return best_pts
# EVOLVE-BLOCK-END
