# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution
import warnings

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Strategy: Symmetry-Aware Spherical Optimization followed by Cartesian Refinement.
    
    1. Assume the optimal configuration lies close to the unit sphere.
    2. Reduce dimensionality from 42 (Cartesian) to 28 (Spherical: theta, phi for 14 points).
    3. Use multi-start Nelder-Mead to find the best spherical arrangement maximizing (d_min/d_max)^2.
    4. Use the best spherical solution as the initial guess for a final Cartesian optimization (Nelder-Mead) 
       to allow points to move radially off the sphere if that improves the ratio.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 580  # 20s margin

    def get_score(pts):
        """Calculate (d_min / d_max)^2"""
        if pts.shape != (n, d):
            return 0.0
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    def objective_spherical(params):
        """
        Objective for spherical coordinates.
        params: shape (2n, ) -> [theta_1, phi_1, theta_2, phi_2, ...]
        Returns negative score for minimization.
        """
        if time.time() > deadline:
            return 1e10
            
        thetas = params[0::2]
        phis = params[1::2]
        
        # Convert to Cartesian (on unit sphere)
        pts = np.zeros((n, 3))
        pts[:, 0] = np.sin(thetas) * np.cos(phis)
        pts[:, 1] = np.sin(thetas) * np.sin(phis)
        pts[:, 2] = np.cos(thetas)
        
        # Scale is invariant, but we work on unit sphere here.
        # The ratio d_min/d_max is scale invariant.
        score = get_score(pts)
        return -score

    def objective_cartesian(params):
        """
        Objective for Cartesian coordinates.
        params: shape (3n, )
        Returns negative score for minimization.
        """
        if time.time() > deadline:
            return 1e10
            
        pts = params.reshape(n, 3)
        
        # Center of mass constraint to break translation invariance
        pts = pts - np.mean(pts, axis=0)
        
        score = get_score(pts)
        return -score

    best_score_global = -np.inf
    best_params_cart = None

    # --- Phase 1: Spherical Optimization (28D) ---
    # Initialize with Icosahedron + 2 Poles (mapped to spherical)
    # Icosahedron vertices
    phi_golden = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi_golden], [0, -1, phi_golden],
        [0, 1, -phi_golden], [0, -1, -phi_golden],
        [1, phi_golden, 0], [-1, phi_golden, 0],
        [1, -phi_golden, 0], [-1, -phi_golden, 0],
        [phi_golden, 0, 1], [-phi_golden, 0, 1],
        [phi_golden, 0, -1], [-phi_golden, 0, -1]
    ])
    ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    # Add 2 poles (approximate, will be optimized)
    poles = np.array([[0, 0, 1], [0, 0, -1]])
    
    init_pts = np.vstack([ico_verts, poles])
    
    # Convert to spherical
    init_thetas = np.arccos(np.clip(init_pts[:, 2], -1, 1))
    init_phis = np.arctan2(init_pts[:, 1], init_pts[:, 0])
    init_spherical = np.zeros(2 * n)
    init_spherical[0::2] = init_thetas
    init_spherical[1::2] = init_phis

    # Multi-start Nelder-Mead on Spherical Space
    # We will run several starts with different initial configurations
    starts = []
    
    # Start 1: Icosahedron + Poles
    starts.append(init_spherical.copy())
    
    # Start 2: Random perturbation of Icosahedron + Poles
    np.random.seed(42)
    perturbed = init_spherical.copy()
    perturbed += np.random.normal(0, 0.1, size=perturbed.shape)
    starts.append(perturbed)
    
    # Start 3: Random points on sphere
    random_thetas = np.random.uniform(0, np.pi, n)
    random_phis = np.random.uniform(0, 2 * np.pi, n)
    random_spherical = np.zeros(2 * n)
    random_spherical[0::2] = random_thetas
    random_spherical[1::2] = random_phis
    starts.append(random_spherical)

    # Start 4: Symmetric "Rings" - 2 poles, 6 in top ring, 6 in bottom ring
    # This is a common high-symmetry guess
    ring_thetas = np.array([0, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4, 
                            np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2, 
                            3*np.pi/4, 3*np.pi/4, 3*np.pi/4, 3*np.pi/4, 3*np.pi/4, 3*np.pi/4,
                            np.pi])
    # Wait, N=14. 
    # Let's do: 1 North Pole, 1 South Pole, 6 in Top Ring, 6 in Bottom Ring? That's 14.
    # Or 2 Poles, 12 in a single ring? No.
    # Let's stick to the Icosahedron-based starts which are known to be good.
    
    for i, start in enumerate(starts):
        if time.time() > deadline - 60:
            break
            
        try:
            res = minimize(
                objective_spherical, 
                start, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if res.fun < -best_score_global:
                best_score_global = -res.fun
                # Convert best spherical result to Cartesian for Phase 2
                thetas = res.x[0::2]
                phis = res.x[1::2]
                pts_cart = np.zeros((n, 3))
                pts_cart[:, 0] = np.sin(thetas) * np.cos(phis)
                pts_cart[:, 1] = np.sin(thetas) * np.sin(phis)
                pts_cart[:, 2] = np.cos(thetas)
                best_params_cart = pts_cart.flatten()
                
        except Exception as e:
            pass

    # --- Phase 2: Cartesian Refinement (42D) ---
    # Allow points to move radially. The optimal 3D configuration might not be strictly spherical.
    if best_params_cart is not None and time.time() < deadline - 30:
        try:
            # Use Nelder-Mead on the full Cartesian space
            # Initialize with the best spherical solution
            # Center of mass
            pts_init = best_params_cart.reshape(n, 3)
            pts_init -= np.mean(pts_init, axis=0)
            
            res_cart = minimize(
                objective_cartesian, 
                pts_init.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-11}
            )
            
            if res_cart.fun < -best_score_global:
                best_score_global = -res_cart.fun
                best_params_cart = res_cart.x
                
        except Exception as e:
            pass

    # --- Fallback if optimization failed or score is bad ---
    if best_params_cart is None or best_score_global < 0.5:
        # Use the initial Icosahedron + Poles directly
        pts_final = init_pts
        best_params_cart = pts_final.flatten()
        best_score_global = get_score(pts_final)

    # Final Processing
    points = best_params_cart.reshape(n, 3)
    
    # Center of mass
    points -= np.mean(points, axis=0)
    
    # Normalize scale (optional, but good for numerical stability in return)
    dists = pdist(points)
    d_max = np.max(dists)
    if d_max > 1e-12:
        points = points / d_max

    # Ensure finite
    if not np.all(np.isfinite(points)):
        # Ultimate fallback: Icosahedron + Poles
        phi_golden = (1 + np.sqrt(5)) / 2
        ico_verts = np.array([
            [0, 1, phi_golden], [0, -1, phi_golden],
            [0, 1, -phi_golden], [0, -1, -phi_golden],
            [1, phi_golden, 0], [-1, phi_golden, 0],
            [1, -phi_golden, 0], [-1, -phi_golden, 0],
            [phi_golden, 0, 1], [-phi_golden, 0, 1],
            [phi_golden, 0, -1], [-phi_golden, 0, -1]
        ])
        ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
        poles = np.array([[0, 0, 1], [0, 0, -1]])
        points = np.vstack([ico_verts, poles])
        points -= np.mean(points, axis=0)
        dists = pdist(points)
        d_max = np.max(dists)
        if d_max > 0:
            points = points / d_max

    return points

# EVOLVE-BLOCK-END
