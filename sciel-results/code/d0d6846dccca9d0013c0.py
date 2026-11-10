# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill

    def exact_score_from_pts(pts):
        """Compute R = (d_min/d_max)^2 from a (16,2) array."""
        pts = np.asarray(pts, dtype=np.float64)
        if pts.shape != (n, d):
            pts = pts.reshape(n, d)
        dists = pdist(pts)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        d_min = np.min(dists)
        return (d_min / d_max) ** 2

    def exact_objective(x):
        pts = x.reshape(n, d)
        return -exact_score_from_pts(pts)

    # Hexagonal lattice parameterization:
    # Parameters: [s, aspect, dx0,dy0, ..., dx15,dy15] = 2 + 32 = 34 params
    # s: lattice spacing
    # aspect: aspect ratio (1.0 = perfect hexagonal)
    # dx_i, dy_i: small perturbations
    
    def hex_lattice_points(params):
        """
        params: [s, aspect, dx0, dy0, dx1, dy1, ..., dx15, dy15]
        s: lattice spacing
        aspect: aspect ratio of lattice (1.0 = perfect hexagonal)
        dx_i, dy_i: small perturbations for each point
        Returns: (16, 2) array of points
        """
        s = params[0]
        aspect = params[1]
        perturbations = params[2:].reshape(n, 2)
        
        # Define hexagonal lattice basis vectors
        a1 = np.array([s, 0.0])
        a2 = np.array([s * aspect * 0.5, s * aspect * np.sqrt(3) / 2])
        
        # Generate 4x4 grid of lattice points
        pts = np.zeros((n, 2))
        for i in range(4):
            for j in range(4):
                idx = i * 4 + j
                pts[idx] = i * a1 + j * a2
        
        # Center the configuration
        pts -= np.mean(pts, axis=0)
        
        # Add perturbations
        pts += perturbations
        
        return pts

    def hex_objective(params):
        pts = hex_lattice_points(params)
        return -exact_score_from_pts(pts)

    def hex_soft_objective(params, t=0.05):
        """Differentiable approximation for local optimization."""
        pts = hex_lattice_points(params)
        dists = pdist(pts)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        dists_norm = dists / d_max
        d_min_norm = -np.log(np.sum(np.exp(-dists_norm / t)) / len(dists_norm)) * t
        return -d_min_norm

    best_x = None
    best_score = 0.0

    def update_best(x, score):
        nonlocal best_x, best_score
        if score > best_score:
            best_score = score
            best_x = x.copy()

    # Phase 1: Global search with Differential Evolution on hex lattice parameters
    # Parameters: [s, aspect, 32 perturbations]
    # s in [0.5, 2.0], aspect in [0.8, 1.2], perturbations in [-0.1, 0.1]
    bounds = [(0.5, 2.0), (0.8, 1.2)] + [(-0.1, 0.1)] * (n * d)
    
    try:
        if time.time() < deadline - 90:
            res_de = differential_evolution(
                hex_objective,
                bounds,
                maxiter=200,
                popsize=30,
                seed=42,
                tol=1e-10,
                init='latinhypercube',
                polish=False,
                updating='immediate'
            )
            
            de_pts = hex_lattice_points(res_de.x)
            de_score = exact_score_from_pts(de_pts)
            update_best(de_pts.ravel(), de_score)
    except Exception:
        pass

    # Phase 2: Nelder-Mead local refinement on exact objective
    if best_x is not None and time.time() < deadline - 30:
        try:
            # Convert back to hex parameters for optimization
            # This is tricky - we need to estimate the hex parameters from the points
            # For now, just do free-coordinate Nelder-Mead
            res_nm = minimize(
                exact_objective,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            nm_score = exact_score_from_pts(res_nm.x.reshape(n, d))
            update_best(res_nm.x, nm_score)
        except Exception:
            pass

    # Phase 3: Multi-start with diverse hex lattice initializations + Nelder-Mead
    if time.time() < deadline - 20:
        np.random.seed(123)
        inits = []
        
        # Perfect hexagonal lattice (aspect=1.0, s=1.0, no perturbations)
        init_perfect = np.array([1.0, 1.0] + [0.0] * (n * d))
        inits.append(init_perfect)
        
        # Slightly varied aspects
        for aspect in [0.9, 0.95, 1.05, 1.1]:
            inits.append(np.array([1.0, aspect] + [0.0] * (n * d)))
        
        # Various spacings
        for s in [0.8, 0.9, 1.1, 1.2]:
            inits.append(np.array([s, 1.0] + [0.0] * (n * d)))
        
        # Random perturbations
        for _ in range(6):
            s = np.random.uniform(0.8, 1.2)
            aspect = np.random.uniform(0.9, 1.1)
            perturbations = np.random.uniform(-0.05, 0.05, n * d)
            inits.append(np.concatenate([[s, aspect], perturbations]))
        
        for init_params in inits:
            if time.time() > deadline - 15:
                break
            try:
                # First optimize in hex parameter space
                res_hex = minimize(
                    hex_objective,
                    init_params,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                hex_pts = hex_lattice_points(res_hex.x)
                s_hex = exact_score_from_pts(hex_pts)
                update_best(hex_pts.ravel(), s_hex)
                
                # Then refine in free coordinate space
                res_free = minimize(
                    exact_objective,
                    hex_pts.ravel(),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                s_free = exact_score_from_pts(res_free.x.reshape(n, d))
                update_best(res_free.x, s_free)
            except Exception:
                continue

    # Phase 4: Final polish with L-BFGS-B on smooth objective in free space
    if best_x is not None and time.time() < deadline - 5:
        try:
            def smooth_objective_free(x, t=0.03):
                pts = x.reshape(n, d)
                dists = pdist(pts)
                d_max = np.max(dists)
                if d_max < 1e-12:
                    return 0.0
                dists_norm = dists / d_max
                d_min_norm = -np.log(np.sum(np.exp(-dists_norm / t)) / len(dists_norm)) * t
                return -d_min_norm
            
            res_lbfgs = minimize(
                smooth_objective_free,
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            s = exact_score_from_pts(res_lbfgs.x.reshape(n, d))
            update_best(res_lbfgs.x, s)
        except Exception:
            pass

    # Fallback
    if best_x is None:
        # Use perfect hexagonal lattice
        best_x = hex_lattice_points(np.array([1.0, 1.0] + [0.0] * (n * d))).ravel()

    return best_x.reshape(n, d)

# EVOLVE-BLOCK-END
