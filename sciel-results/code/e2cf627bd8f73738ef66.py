import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit

    # Precompute triangle indices
    tri_indices = list(combinations(range(N), 3))
    tri_i = np.array([t[0] for t in tri_indices], dtype=np.int32)
    tri_j = np.array([t[1] for t in tri_indices], dtype=np.int32)
    tri_k = np.array([t[2] for t in tri_indices], dtype=np.int32)
    n_tris = len(tri_i)

    def compute_score(points):
        """Compute min triangle area / convex hull area."""
        try:
            pts = points - points.mean(axis=0)
            hull = ConvexHull(pts)
            area_hull = hull.volume
        except Exception:
            return 0.0
        if area_hull < 1e-9:
            return 0.0
        
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        min_area = np.min(areas)
        return min_area / area_hull

    def _normalize_points(points):
        """Center and scale points to unit convex hull area."""
        pts = points - points.mean(axis=0)
        try:
            hull = ConvexHull(pts)
            area = hull.volume
            if area < 1e-9:
                return pts
            pts = pts * np.sqrt(1.0 / area)
        except Exception:
            pass
        return pts

    # Fourier Parameterization
    # Points are defined as deformations of a regular 13-gon
    # r_k = 1 + sum_{m=1}^{M} (a_m cos(m * theta_k) + b_m sin(m * theta_k))
    # theta_k = 2 * pi * k / N
    
    M = 4  # Number of harmonics. 2*M + 1 (scale) = 9 variables? 
    # Actually, we can just use 2*M variables for radius deformation, 
    # and maybe one for global scale? 
    # Since we normalize to unit area, global scale is redundant.
    # But the shape deformation changes the area.
    # Let's use 2*M variables.
    n_params = 2 * M
    
    def _fourier_to_points(params):
        """Convert Fourier coefficients to 2D points."""
        k = np.arange(N)
        theta = 2 * np.pi * k / N
        
        r = np.ones(N)
        for m in range(1, M + 1):
            a = params[2 * (m - 1)]
            b = params[2 * (m - 1) + 1]
            r += a * np.cos(m * theta) + b * np.sin(m * theta)
            
        # Ensure r > 0 to avoid weird inversions, though normalization handles scale
        # If r becomes negative, it flips the point to the other side, breaking convexity/structure
        # We should probably constrain r > 0.5 or similar, or just let it be and rely on optimization
        # To be safe, let's clip r to a minimum positive value? 
        # No, better to let the optimizer handle it, but if r <= 0, the point is at origin or flipped.
        # Let's just compute x, y.
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack([x, y])

    def _soft_min(areas, beta):
        """Compute soft minimum of areas using log-sum-exp trick."""
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 0.0
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        return -log_sum_exp / beta

    def _fourier_objective(params):
        """Objective function for Fourier parameterization."""
        try:
            pts = _fourier_to_points(params)
            pts = _normalize_points(pts)
            
            x = pts[:, 0]
            y = pts[:, 1]
            v1x = x[tri_j] - x[tri_i]
            v1y = y[tri_j] - y[tri_i]
            v2x = x[tri_k] - x[tri_i]
            v2y = y[tri_k] - y[tri_i]
            cross = np.abs(v1x * v2y - v1y * v2x)
            areas = 0.5 * cross
            
            # Use soft min to make it differentiable for L-BFGS
            # Beta needs to be high enough to approximate min, low enough for gradients
            beta = 50.0
            soft_min_val = _soft_min(areas, beta)
            
            # Also penalize if convex hull area is small (points collapsing)
            # Normalization should prevent this, but if points are collinear, area ~ 0
            # compute_score handles the division. Here we return negative soft_min.
            # We want to maximize min area. So minimize -soft_min.
            
            # Check if points are valid (non-degenerate)
            if np.min(areas) < 1e-6:
                return 1e6 # Large penalty for near-degenerate triangles
                
            return -soft_min_val
        except Exception:
            return 1e6

    # Strategy:
    # 1. Differential Evolution in Fourier space (8D)
    # 2. L-BFGS-B refinement in Fourier space
    # 3. Final L-BFGS-B refinement in full 2D space (26D) starting from the Fourier result
    
    best_score = -float('inf')
    best_points = None

    # 1. Differential Evolution
    # Bounds for Fourier coefficients:
    # If r = 1 + sum(...), we want r to stay positive.
    # Max possible deviation is sum of abs coeffs.
    # Let's bound coeffs to [-0.4, 0.4] to ensure r >= 1 - 4*0.4 = -0.6? No.
    # If all 4 harmonics align, max deviation is 4*0.4 = 1.6. r could be 1-1.6 = -0.6.
    # Let's use tighter bounds: [-0.2, 0.2]. Max deviation 0.8. r >= 0.2.
    bounds = [(-0.2, 0.2)] * n_params
    
    rng = np.random.RandomState(42)
    
    # DE can be slow, so we limit iterations and popsize
    try:
        de_res = opt.differential_evolution(
            _fourier_objective, 
            bounds, 
            maxiter=50, 
            popsize=15, 
            tol=1e-6, 
            seed=42,
            polish=False # We will polish manually
        )
        
        if de_res.success or de_res.fun < 1e5:
            best_params = de_res.x
            pts = _fourier_to_points(best_params)
            pts = _normalize_points(pts)
            score = compute_score(pts)
            
            if score > best_score:
                best_score = score
                best_points = pts.copy()
                
    except Exception:
        pass

    # If DE failed or found nothing, try random initialization in Fourier space
    if best_points is None:
        for _ in range(10):
            params = rng.uniform(-0.2, 0.2, n_params)
            pts = _fourier_to_points(params)
            pts = _normalize_points(pts)
            score = compute_score(pts)
            if score > best_score:
                best_score = score
                best_points = pts.copy()
                best_params = params

    # 2. L-BFGS-B in Fourier space
    if best_points is not None:
        # We need to recover best_params from best_points? 
        # No, we kept best_params from DE or random init.
        # If we came from random init, best_params is defined.
        # If we came from DE, best_params is defined.
        
        try:
            res_lbfgs_fourier = opt.minimize(
                _fourier_objective, 
                best_params, 
                method='L-BFGS-B', 
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res_lbfgs_fourier.fun < 1e5:
                new_params = res_lbfgs_fourier.x
                pts = _fourier_to_points(new_params)
                pts = _normalize_points(pts)
                score = compute_score(pts)
                
                if score > best_score:
                    best_score = score
                    best_points = pts.copy()
                    
        except Exception:
            pass

    # 3. Full space refinement (26D)
    # The Fourier space restricts the geometry to "star-shaped with respect to origin" and 
    # angularly regular. Full space allows breaking angular regularity.
    # We use the best points found so far as the start point.
    
    if best_points is not None:
        def _full_objective(points_flat):
            pts = points_flat.reshape(-1, 2)
            pts = _normalize_points(pts)
            
            x = pts[:, 0]
            y = pts[:, 1]
            v1x = x[tri_j] - x[tri_i]
            v1y = y[tri_j] - y[tri_i]
            v2x = x[tri_k] - x[tri_i]
            v2y = y[tri_k] - y[tri_i]
            cross = np.abs(v1x * v2y - v1y * v2x)
            areas = 0.5 * cross
            
            beta = 50.0
            soft_min_val = _soft_min(areas, beta)
            
            if np.min(areas) < 1e-6:
                return 1e6
                
            return -soft_min_val

        try:
            # L-BFGS-B in full space
            res_full = opt.minimize(
                _full_objective, 
                best_points.flatten(), 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res_full.fun < 1e5:
                pts = res_full.x.reshape(-1, 2)
                pts = _normalize_points(pts)
                score = compute_score(pts)
                
                if score > best_score:
                    best_score = score
                    best_points = pts.copy()
                    
        except Exception:
            pass

        # Try Powell as a fallback or secondary refiner if L-BFGS got stuck?
        # Powell is more robust to non-smoothness (min function is non-smooth).
        # L-BFGS uses soft-min, which is smooth. 
        # Let's try Powell on the hard objective (min area) for a final kick.
        
        def _hard_objective(points_flat):
            pts = points_flat.reshape(-1, 2)
            pts = _normalize_points(pts)
            score = compute_score(pts)
            if score == 0.0:
                return 1e6
            return -score

        try:
            res_powell = opt.minimize(
                _hard_objective, 
                best_points.flatten(), 
                method='Powell', 
                options={'maxiter': 50, 'xtol': 1e-10, 'ftol': 1e-10}
            )
            
            pts = res_powell.x.reshape(-1, 2)
            pts = _normalize_points(pts)
            score = compute_score(pts)
            
            if score > best_score:
                best_score = score
                best_points = pts.copy()
                
        except Exception:
            pass

    # Fallback if everything failed
    if best_points is None:
        # Regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_points = _normalize_points(best_points)
        best_score = compute_score(best_points)

    return best_points
