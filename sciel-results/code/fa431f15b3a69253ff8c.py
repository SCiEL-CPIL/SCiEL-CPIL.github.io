import numpy as np
import time
import itertools
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a Model-and-Optimize approach:
    1. Initializes with a hexagonal lattice pattern (strong geometric prior).
    2. Optimizes using D2 symmetry (8 parameters) to quickly find a good basin.
    3. Refines with full 32-parameter optimization.
    4. Uses multiple restarts and methods to avoid local minima.
    5. Explores hexagonal cluster variants (19 sites, 3 removed) with aspect ratio optimization.
    """
    n = 16
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    def score(points):
        """Compute R = (d_min / d_max)^2"""
        if points is None or len(points) != n:
            return 0.0
        if not np.all(np.isfinite(points)):
            return 0.0
        dists = pdist(points, 'euclidean')
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 0.0
        return (d_min / d_max) ** 2

    def objective(params):
        """
        Objective function to minimize: -d_min / d_max
        """
        pts = params.reshape((n, 2))
        dists = pdist(pts, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 0.0
        ratio = d_min / d_max
        return -ratio

    def smooth_objective(params, beta=100.0):
        """
        Smooth surrogate using soft-min and soft-max via log-sum-exp.
        Helps L-BFGS-B navigate non-smooth landscape.
        """
        pts = params.reshape((n, 2))
        dists = pdist(pts, 'euclidean')
        
        # Soft min
        neg_dists = -beta * dists
        max_neg_dists = np.max(neg_dists)
        log_sum_exp_neg = max_neg_dists + np.log(np.sum(np.exp(neg_dists - max_neg_dists)))
        d_min_soft = -log_sum_exp_neg / beta
        
        # Soft max
        pos_dists = beta * dists
        max_pos_dists = np.max(pos_dists)
        log_sum_exp_pos = max_pos_dists + np.log(np.sum(np.exp(pos_dists - max_pos_dists)))
        d_max_soft = log_sum_exp_pos / beta
        
        if d_max_soft < 1e-10:
            return 0.0
            
        ratio = d_min_soft / d_max_soft
        return -ratio

    best_points = None
    best_score = -1.0

    def update_best(pts):
        nonlocal best_points, best_score
        if pts is None:
            return
        if not np.all(np.isfinite(pts)):
            return
        s = score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()

    def generate_hexagonal():
        """Generate 16 points in a hexagonal pattern."""
        # 4 rows, 4 columns
        # Row 0: y=0, x=0, 1, 2, 3
        # Row 1: y=sqrt(3)/2, x=0.5, 1.5, 2.5, 3.5
        # Row 2: y=sqrt(3), x=0, 1, 2, 3
        # Row 3: y=1.5*sqrt(3), x=0.5, 1.5, 2.5, 3.5
        h = np.sqrt(3) / 2
        pts = np.zeros((16, 2))
        for i in range(4):
            y = i * 2 * h
            for j in range(4):
                idx = i * 4 + j
                if i % 2 == 0:
                    x = j
                else:
                    x = j + 0.5
                pts[idx] = [x, y]
        return pts

    def generate_hex_cluster_16(removed_sites, aspect_ratio=1.0):
        """
        Generate 16 points from a 19-site hexagonal cluster by removing 3 sites.
        The 19 sites form a hexagon with center at origin.
        Sites are indexed by (row, col) in axial coordinates.
        removed_sites: tuple of 3 indices (0-18) to remove.
        aspect_ratio: scales y-axis to allow for non-regular hexagons.
        """
        # Define the 19 sites of a hexagonal cluster (radius 2)
        # Axial coordinates (q, r) where distance from center is max(|q|, |r|, |q+r|)
        # For radius 2, we have all (q, r) such that max(|q|, |r|, |q+r|) <= 2
        
        # Let's just hardcode the 19 points for a regular hexagonal lattice
        # Center at (0,0)
        # The 19 points are:
        # Row -2: (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0) -> No, this is linear.
        
        # Let's use a simpler generation:
        # The 19 points in a hexagonal grid of radius 2 (including center)
        # Coordinates in Cartesian:
        # (0,0)
        # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
        # (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
        
        # Let's generate them programmatically to be safe
        sites = []
        for q in range(-2, 3):
            for r in range(-2, 3):
                if max(abs(q), abs(r), abs(q+r)) <= 2:
                    sites.append((q, r))
        
        # There should be 19 sites
        if len(sites) != 19:
            # Fallback to hardcoded if generation fails
            sites = [
                (0,0),
                (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
                (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
            ]
        
        # Convert to Cartesian
        # x = q + r/2
        # y = r * sqrt(3)/2
        pts_all = []
        for q, r in sites:
            x = q + r / 2.0
            y = r * np.sqrt(3) / 2.0 * aspect_ratio
            pts_all.append([x, y])
        
        pts_all = np.array(pts_all)
        
        # Remove the specified sites
        keep_indices = [i for i in range(19) if i not in removed_sites]
        pts = pts_all[keep_indices]
        
        return pts

    def generate_d2_symmetric(params):
        """
        Map 8 parameters (x1,y1, x2,y2, x3,y3, x4,y4) to 16 points with D2 symmetry.
        Points are (±xi, ±yi) for i=1..4.
        """
        pts = np.zeros((16, 2))
        for i in range(4):
            x = params[i*2]
            y = params[i*2+1]
            # Ensure positive to avoid duplicate orbits if not careful, 
            # but symmetry generates all quadrants.
            # To ensure distinct points, we assume x,y > 0.
            # If x or y is 0, we might get duplicates. 
            # We will optimize in unconstrained space and let symmetry handle it,
            # but duplicates reduce d_min to 0. So we should avoid x=0 or y=0.
            
            # Quadrant 1: (x, y)
            pts[i*4] = [x, y]
            # Quadrant 2: (-x, y)
            pts[i*4+1] = [-x, y]
            # Quadrant 3: (-x, -y)
            pts[i*4+2] = [-x, -y]
            # Quadrant 4: (x, -y)
            pts[i*4+3] = [x, -y]
        return pts

    def objective_d2(params):
        """Objective for D2 symmetric parameters."""
        pts = generate_d2_symmetric(params)
        return objective(pts.flatten())

    def smooth_objective_d2(params, beta=100.0):
        """Smooth objective for D2 symmetric parameters."""
        pts = generate_d2_symmetric(params)
        return smooth_objective(pts.flatten(), beta)

    def refine(pts, label=""):
        """
        Multi-stage refinement.
        """
        if time.time() > deadline - 10:
            return

        current = pts.copy()
        
        # Stage 1: Smooth L-BFGS-B on full 32 dims
        try:
            res1 = minimize(
                smooth_objective, 
                current.flatten(), 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res1.x is not None and np.all(np.isfinite(res1.x)):
                current = res1.x.reshape((n, 2))
        except Exception:
            pass

        if time.time() > deadline - 10:
            update_best(current)
            return

        # Stage 2: Raw L-BFGS-B on full 32 dims
        try:
            res2 = minimize(
                objective, 
                current.flatten(), 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res2.x is not None and np.all(np.isfinite(res2.x)):
                current = res2.x.reshape((n, 2))
        except Exception:
            pass

        if time.time() > deadline - 10:
            update_best(current)
            return

        # Stage 3: Powell (Derivative-free)
        try:
            res3 = minimize(
                objective, 
                current.flatten(), 
                method='Powell', 
                options={'maxiter': 300, 'ftol': 1e-12, 'xtol': 1e-12}
            )
            if res3.x is not None and np.all(np.isfinite(res3.x)):
                current = res3.x.reshape((n, 2))
        except Exception:
            pass

        update_best(current)

    # 1. Hexagonal Initialization
    hex_pts = generate_hexagonal()
    refine(hex_pts, "hex")

    # 1b. Hexagonal Cluster Variants (19 sites, 3 removed)
    # Try different removal patterns and aspect ratios
    # The 19 sites are indexed 0-18. We want to remove 3.
    # Symmetric removals are likely best.
    # Center is index 0 (if we ordered it that way). Let's check our generation.
    # My generation loop: q from -2 to 2, r from -2 to 2.
    # (0,0) is the 5th element in the loop?
    # q=-2: r=-2..2 (5 pts)
    # q=-1: r=-2..2 (5 pts)
    # q=0: r=-2..2 (5 pts) -> (0,0) is index 10
    # q=1: r=-2..2 (5 pts)
    # q=2: r=-2..2 (5 pts)
    # Total 25? No, the condition max(abs(q), abs(r), abs(q+r)) <= 2 filters them.
    
    # Let's just try a few specific removal patterns.
    # Pattern 1: Remove 3 corners (e.g., (2,0), (0,2), (-2,2) in axial -> indices?)
    # It's easier to just try random subsets of 3 indices and optimize aspect ratio.
    
    num_cluster_restarts = 5
    np.random.seed(42)
    for i in range(num_cluster_restarts):
        if time.time() > deadline - 60:
            break
        
        # Randomly select 3 sites to remove
        removed = np.random.choice(19, size=3, replace=False)
        
        # Optimize aspect ratio
        def obj_aspect(params):
            ar = params[0]
            # Constrain aspect ratio to [0.5, 2.0]
            ar = np.clip(ar, 0.5, 2.0)
            pts = generate_hex_cluster_16(tuple(removed), ar)
            return objective(pts.flatten())
        
        # Initial guess for aspect ratio
        x0 = np.array([1.0])
        
        try:
            res_ar = minimize(
                obj_aspect, 
                x0, 
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-6}
            )
            if res_ar.x is not None and np.all(np.isfinite(res_ar.x)):
                ar_best = np.clip(res_ar.x[0], 0.5, 2.0)
                pts_cluster = generate_hex_cluster_16(tuple(removed), ar_best)
                refine(pts_cluster, f"cluster_{i}")
        except Exception:
            pass

    # 2. D2 Symmetric Initialization
    # Initial guess for 4 points in first quadrant
    # Spread them out
    np.random.seed(42)
    
    # Try a few symmetric configurations
    for _ in range(5):
        if time.time() > deadline - 30:
            break
        # Random points in first quadrant
        params_init = np.random.rand(8) * 5 + 0.5
        try:
            res_sym = minimize(
                smooth_objective_d2, 
                params_init, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res_sym.x is not None and np.all(np.isfinite(res_sym.x)):
                pts_sym = generate_d2_symmetric(res_sym.x)
                refine(pts_sym, "d2_sym")
        except Exception:
            pass

    # 3. Random Restarts (Full 32D)
    # Use best found so far to guide random restarts? Or just random?
    # Random is safer to find different basins.
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > deadline - 30:
            break
        # Random points in a square, scaled to roughly match hexagonal extent
        rand_pts = np.random.rand(n, 2) * 6 - 3
        refine(rand_pts, f"rand_{i}")
        
        # Also try perturbations of the best found point
        if best_points is not None:
            if time.time() > deadline - 30:
                break
            # Small random perturbation
            perturbed = best_points + np.random.normal(0, 0.1, (n, 2))
            refine(perturbed, f"perturb_{i}")

    # 4. Final Polish on best found
    if best_points is not None and time.time() < deadline - 5:
        try:
            res_final = minimize(
                objective, 
                best_points.flatten(), 
                method='L-BFGS-B', 
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res_final.x is not None and np.all(np.isfinite(res_final.x)):
                cand = res_final.x.reshape((n, 2))
                update_best(cand)
        except Exception:
            pass
        
        # Additional high-precision polish with Powell if time remains
        if best_points is not None and time.time() < deadline - 2:
            try:
                res_polish = minimize(
                    objective, 
                    best_points.flatten(), 
                    method='Powell', 
                    options={'maxiter': 500, 'ftol': 1e-14, 'xtol': 1e-14}
                )
                if res_polish.x is not None and np.all(np.isfinite(res_polish.x)):
                    cand = res_polish.x.reshape((n, 2))
                    update_best(cand)
            except Exception:
                pass

    if best_points is None:
        return hex_pts

    return best_points
