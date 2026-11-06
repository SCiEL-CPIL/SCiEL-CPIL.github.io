"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, basinhopping
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    n = 8
    num_pairs = n * (n - 1) // 2  # 28

    # Strategy: Two-concentric-squares topology
    # 4 points on inner square (radius r), 4 on outer square (radius R), rotated by theta
    # Diameter = 2*R*sqrt(2) for outer square diagonal, so R = sqrt(2) for diameter 2
    # We optimize r and theta (and possibly center offset)
    
    def two_squares_config(params):
        """Generate 8 points from two concentric squares.
        params = [r, theta] where r is inner radius, theta is rotation of inner square
        Outer square has fixed radius R=sqrt(2) for diameter 2, centered at origin.
        """
        r, theta = params[0], params[1]
        R = np.sqrt(2.0)  # Fixed for diameter 2
        
        # Outer square: vertices at angles 0, pi/2, pi, 3pi/2 with radius R
        outer_angles = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        outer_pts = np.column_stack((R * np.cos(outer_angles), R * np.sin(outer_angles)))
        
        # Inner square: vertices at angles theta, theta+pi/2, theta+pi, theta+3pi/2 with radius r
        inner_angles = theta + np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        inner_pts = np.column_stack((r * np.cos(inner_angles), r * np.sin(inner_angles)))
        
        # Interleave: outer[0], inner[0], outer[1], inner[1], ...
        pts = np.zeros((8, 2))
        pts[0] = outer_pts[0]
        pts[1] = inner_pts[0]
        pts[2] = outer_pts[1]
        pts[3] = inner_pts[1]
        pts[4] = outer_pts[2]
        pts[5] = inner_pts[2]
        pts[6] = outer_pts[3]
        pts[7] = inner_pts[3]
        
        return pts

    def log_product_distance(pts):
        """Compute log of product of all pairwise distances."""
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diffs**2, axis=-1)
        iu = np.triu_indices(n, k=1)
        d = np.sqrt(dists_sq[iu])
        d = np.maximum(d, 1e-15)
        return np.sum(np.log(d))

    def objective_two_squares(params):
        """Minimize negative log product of distances for two-squares config."""
        r, theta = params
        if r < 0 or r > 2:
            return 1e10
        pts = two_squares_config(params)
        # Verify diameter is 2 (should be by construction since outer square diagonal = 2R = 2*sqrt(2)*sqrt(2) = 4? No!)
        # Wait: outer square with radius R has vertices at distance R from center.
        # Diagonal of square = 2*R*sqrt(2)? No. For a square inscribed in circle of radius R,
        # the side length is R*sqrt(2), and the diagonal is 2*R.
        # So diameter = 2*R. For diameter 2, R = 1.
        # Let me fix: R should be 1 for diameter 2.
        return -log_product_distance(pts)

    # Fix: For a square inscribed in circle of radius R, the diagonal is 2R.
    # So for diameter 2, we need R = 1.
    
    def two_squares_config_v2(params):
        """Generate 8 points from two concentric squares with diameter 2.
        Outer square radius = 1 (so diagonal = 2 = diameter).
        params = [r, theta]
        """
        r, theta = params[0], params[1]
        R = 1.0  # Outer square radius, diagonal = 2R = 2
        
        outer_angles = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        outer_pts = np.column_stack((R * np.cos(outer_angles), R * np.sin(outer_angles)))
        
        inner_angles = theta + np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        inner_pts = np.column_stack((r * np.cos(inner_angles), r * np.sin(inner_angles)))
        
        pts = np.zeros((8, 2))
        pts[0] = outer_pts[0]
        pts[1] = inner_pts[0]
        pts[2] = outer_pts[1]
        pts[3] = inner_pts[1]
        pts[4] = outer_pts[2]
        pts[5] = inner_pts[2]
        pts[6] = outer_pts[3]
        pts[7] = inner_pts[3]
        
        return pts

    def objective_v2(params):
        r, theta = params
        if r < -0.5 or r > 1.5:
            return 1e10
        pts = two_squares_config_v2(params)
        return -log_product_distance(pts)

    # Optimize r and theta
    best_score = -np.inf
    best_pts = None
    
    # Grid search over theta to handle periodicity
    for theta0 in np.linspace(0, np.pi/2, 13):
        for r0 in np.linspace(0.1, 0.9, 17):
            x0 = np.array([r0, theta0])
            try:
                res = minimize(objective_v2, x0, method='Nelder-Mead',
                               options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'disp': False})
                if res.fun < best_score:
                    best_score = res.fun
                    best_pts = two_squares_config_v2(res.x)
            except Exception:
                pass
    
    # Polish with high precision
    if best_pts is not None and time.time() < deadline - 20:
        # Find best (r, theta) from grid
        best_params = None
        best_val = np.inf
        for theta0 in np.linspace(0, np.pi/2, 25):
            for r0 in np.linspace(0.05, 0.95, 37):
                x0 = np.array([r0, theta0])
                val = objective_v2(x0)
                if val < best_val:
                    best_val = val
                    best_params = x0
        
        if best_params is not None:
            for method in ['Nelder-Mead', 'Powell']:
                try:
                    res = minimize(objective_v2, best_params, method=method,
                                   options={'maxiter': 5000, 'xatol': 1e-15, 'fatol': 1e-16, 'disp': False})
                    if res.fun < best_val:
                        best_val = res.fun
                        best_params = res.x
                except Exception:
                    pass
            best_pts = two_squares_config_v2(best_params)
    
    # Also try the general optimization as fallback/alternative
    fixed_idx = [0, 4]
    free_idx = [i for i in range(n) if i not in fixed_idx]
    n_free = len(free_idx)
    n_dims = n_free * 2

    def unpack(x):
        p = np.zeros((n, 2))
        p[0] = [0.0, 0.0]
        p[4] = [2.0, 0.0]
        for i, idx in enumerate(free_idx):
            p[idx] = [x[2 * i], x[2 * i + 1]]
        return p

    def objective_general(x, beta=100.0):
        p = unpack(x)
        diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists_sq = np.sum(diffs**2, axis=-1)
        iu = np.triu_indices(n, k=1)
        d = np.sqrt(dists_sq[iu])
        d = np.maximum(d, 1e-15)
        sum_log_d = np.sum(np.log(d))
        max_d = np.max(d)
        scaled = beta * (d - max_d)
        scaled = np.clip(scaled, -500, 500)
        log_sum = np.log(np.sum(np.exp(scaled)))
        soft_max = max_d + log_sum / beta
        excess = np.maximum(soft_max - 2.0, 0.0)
        penalty = 100.0 * excess * excess
        val = -sum_log_d + penalty
        return val

    # Regular octagon init
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts_unit = np.column_stack((np.cos(angles), np.sin(angles)))
    init_pts = pts_unit + np.array([1.0, 0.0])
    init_pts_swapped = init_pts.copy()
    init_pts_swapped[0], init_pts_swapped[4] = init_pts_swapped[4], init_pts_swapped[0]
    x0 = np.array([init_pts_swapped[i] for i in free_idx]).flatten()
    
    best_x = x0.copy()
    best_obj = objective_general(x0)

    for method in ['BFGS', 'L-BFGS-B']:
        try:
            res = minimize(objective_general, x0, method=method,
                           options={'maxiter': 2000, 'gtol': 1e-10, 'disp': False})
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Compare: which is better, two-squares or general?
    if best_pts is not None:
        # Compute score for best_pts
        diffs = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
        dists_sq = np.sum(diffs**2, axis=-1)
        iu = np.triu_indices(n, k=1)
        d = np.sqrt(dists_sq[iu])
        d = np.maximum(d, 1e-15)
        log_prod = np.sum(np.log(d))
        
        # Compute score for best_x
        p_general = unpack(best_x)
        diffs_g = p_general[:, np.newaxis, :] - p_general[np.newaxis, :, :]
        dists_sq_g = np.sum(diffs_g**2, axis=-1)
        iu_g = np.triu_indices(n, k=1)
        d_g = np.sqrt(dists_sq_g[iu_g])
        d_g = np.maximum(d_g, 1e-15)
        log_prod_g = np.sum(np.log(d_g))
        
        # Scale both to diameter 2 and compare
        max_d_sq = np.max(d)
        max_d_gen = np.max(d_g)
        
        # Product scales as (scale_factor)^num_pairs
        # log_prod_scaled = log_prod + num_pairs * log(2/max_d)
        log_prod_sq_scaled = log_prod + num_pairs * np.log(2.0 / max_d_sq)
        log_prod_gen_scaled = log_prod_g + num_pairs * np.log(2.0 / max_d_gen)
        
        if log_prod_sq_scaled > log_prod_gen_scaled:
            p = best_pts
        else:
            p = p_general
    else:
        p = unpack(best_x)

    # Ensure distinct points
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(p[i] - p[j]) < 1e-8:
                p[j] += np.array([1e-6, 0])

    # Scale to ensure diameter is exactly 2
    max_d = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(p[i] - p[j])
            if d > max_d:
                max_d = d

    if max_d > 0 and abs(max_d - 2.0) > 1e-10:
        scale_factor = 2.0 / max_d
        p = p * scale_factor

    # Return as list of lists of floats
    return [[float(p[i][0]), float(p[i][1])] for i in range(n)]

# EVOLVE-BLOCK-END
