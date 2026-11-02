import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

N = 14

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit
    
    # Pre-compute triangle indices for fast area calculation
    tri_indices = np.array(list(itertools.combinations(range(N), 3)))
    
    def compute_score_from_pts(pts):
        """
        Compute min triangle area / convex hull area for a set of 2D points.
        Returns 0 if the points do not form a valid convex polygon.
        """
        # Check for degeneracy
        if np.allclose(pts, pts[0]):
            return 0.0
        
        # Check convexity
        # A simple check: the cross product of consecutive edges must have the same sign
        # and the hull must contain all points (or be the points themselves if in convex position)
        try:
            # Compute convex hull to ensure we are dealing with a convex region
            # If the points are not in convex position, the hull area will be larger 
            # than the area of the polygon formed by the points in given order, 
            # but the task implies the points define the convex region.
            # Usually, for maximizing min triangle area, points will be on the hull.
            hull = ConvexHull(pts)
            hull_area = hull.volume
            if hull_area < 1e-10:
                return 0.0
            
            # Verify that all points are on the hull (or very close to it)
            # If a point is inside the hull, it's not part of the boundary of the convex region
            # defined by the points. The problem states "Place 14 points in a planar convex region".
            # It doesn't strictly say the points *are* the vertices of the convex hull, 
            # but to maximize min triangle area, interior points are suboptimal 
            # because they reduce the area of triangles involving them compared to boundary points.
            # So we assume optimal points are on the hull.
            
            # Calculate min triangle area
            pi = pts[tri_indices[:, 0]]
            pj = pts[tri_indices[:, 1]]
            pk = pts[tri_indices[:, 2]]
            
            v1 = pj - pi
            v2 = pk - pi
            cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
            areas = 0.5 * np.abs(cross)
            
            min_area = np.min(areas)
            
            return min_area / hull_area
        except Exception:
            return 0.0

    def fourier_to_pts(coeffs):
        """
        Convert Fourier coefficients to 14 points.
        coeffs: array of length 2*K.
        r(theta) = 1 + sum_{k=1}^{K} (a_k cos(k theta) + b_k sin(k theta))
        """
        K = len(coeffs) // 2
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        radii = np.ones(N)
        
        for k in range(1, K+1):
            a_k = coeffs[2*(k-1)]
            b_k = coeffs[2*(k-1)+1]
            radii += a_k * np.cos(k * angles) + b_k * np.sin(k * angles)
            
        # Ensure radii are positive
        radii = np.maximum(radii, 0.1)
        
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        return pts

    def neg_score_fourier(coeffs):
        pts = fourier_to_pts(coeffs)
        score = compute_score_from_pts(pts)
        # Penalize if the polygon formed by pts in order is not convex
        # Check cross products of consecutive edges
        edges = np.roll(pts, -1, axis=0) - pts
        cross_prods = edges[:, 0] * np.roll(pts, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(pts, -1, axis=0)[:, 0]
        # Wait, simpler convexity check for polygon in order:
        # Cross product of (P_{i+1}-P_i) and (P_{i+2}-P_{i+1}) should have same sign
        v1 = np.roll(pts, -1, axis=0) - pts
        v2 = np.roll(pts, -2, axis=0) - np.roll(pts, -1, axis=0)
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        if np.any(cross < -1e-6): # Allow small numerical noise
            return -score - 100.0 # Heavy penalty for non-convex ordering
        return -score

    best_score_val = 0
    best_coeffs = None
    
    # Try different numbers of Fourier terms
    # K=1: 2 params (ellipse)
    # K=2: 4 params
    # K=3: 6 params
    # K=4: 8 params
    # K=5: 10 params
    K_values = [2, 3, 4, 5]
    
    for K in K_values:
        if time.time() > deadline - 20:
            break
            
        dim = 2 * K
        # Initial guess: small random perturbations
        # We can also try specific symmetries, e.g. only cos terms
        initial_guesses = [
            np.zeros(dim),
            np.random.normal(0, 0.05, dim),
            np.random.normal(0, 0.1, dim)
        ]
        
        for x0 in initial_guesses:
            if time.time() > deadline - 10:
                break
                
            try:
                # Use Nelder-Mead for small dimensions
                res = minimize(neg_score_fourier, x0, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'disp': False})
                
                score_val = -res.fun
                if score_val > best_score_val:
                    best_score_val = score_val
                    best_coeffs = res.x.copy()
                    
            except Exception:
                pass
    
    # If we found a good solution in Fourier space, refine it in Cartesian space
    # This allows breaking the Fourier symmetry if needed
    if best_coeffs is not None:
        initial_pts = fourier_to_pts(best_coeffs)
        
        # Flatten for optimization
        x0_cart = initial_pts.flatten()
        
        def neg_score_cart(flat_pts):
            pts = flat_pts.reshape(N, 2)
            score = compute_score_from_pts(pts)
            # Check convexity of the polygon formed by points in order
            v1 = np.roll(pts, -1, axis=0) - pts
            v2 = np.roll(pts, -2, axis=0) - np.roll(pts, -1, axis=0)
            cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
            if np.any(cross < -1e-6):
                return -score - 100.0
            return -score

        try:
            # Use Powell for Cartesian refinement
            res_cart = minimize(neg_score_cart, x0_cart, method='Powell',
                                options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False})
            
            score_val = -res_cart.fun
            if score_val > best_score_val:
                best_score_val = score_val
                # We need to return the points, not coeffs.
                # But the function signature returns points.
                # We can just store the best points directly.
                best_pts = res_cart.x.reshape(N, 2)
            else:
                best_pts = initial_pts
        except Exception:
            best_pts = initial_pts
            
        return best_pts

    # Fallback: Return regular 14-gon if optimization failed
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])
