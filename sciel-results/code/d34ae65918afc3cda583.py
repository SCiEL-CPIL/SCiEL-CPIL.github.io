import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50


def construct():
    """
    Construct centers for 50 equal spheres in the unit cube [-0.5, 0.5]^3.
    Uses a BCC lattice initialization followed by multi-start local optimization
    to maximize the minimum distance between centers and to the cube boundaries.
    """
    np.random.seed(42)

    def generate_bcc_lattice(n_per_side):
        """
        Generate BCC lattice points in [-0.5, 0.5]^3.
        """
        points = []
        a = 1.0 / n_per_side
        # Shift grid to center in [-0.5, 0.5]
        # Standard grid: (i + 0.5) * a - 0.5
        for i in range(n_per_side):
            for j in range(n_per_side):
                for k in range(n_per_side):
                    # Corner equivalent (shifted to center of cell for BCC? No, standard BCC is corners + body center)
                    # Let's use standard BCC: 
                    # Cell size a.
                    # Points at (i*a, j*a, k*a) and (i*a + a/2, j*a + a/2, k*a + a/2)
                    # We want them in [-0.5, 0.5].
                    # So we can generate points in [0, 1] and shift by -0.5.
                    
                    # Corner points of the sub-cubes
                    cx = i * a
                    cy = j * a
                    cz = k * a
                    points.append([cx - 0.5, cy - 0.5, cz - 0.5])
                    
                    # Body center
                    bx = i * a + a / 2.0
                    by = j * a + a / 2.0
                    bz = k * a + a / 2.0
                    points.append([bx - 0.5, by - 0.5, bz - 0.5])
        
        pts = np.array(points)
        # Filter points strictly inside or on boundary
        mask = np.all(np.abs(pts) <= 0.5, axis=1)
        pts = pts[mask]
        
        # If we have too many, we need to select N.
        # If we have too few, we need to add random points (though BCC with n=3 gives 3^3*2=54 points, which is good).
        if len(pts) > N:
            # Select N points using farthest-point sampling or random subset
            # Farthest point sampling is better for spreading
            selected = [0]
            # Start with a random point
            start_idx = np.random.choice(len(pts))
            selected = [start_idx]
            while len(selected) < N:
                unselected = [i for i in range(len(pts)) if i not in selected]
                best_idx = -1
                best_min_dist = -1
                for idx in unselected:
                    dists = np.linalg.norm(pts[idx] - pts[selected], axis=1)
                    min_d = np.min(dists)
                    if min_d > best_min_dist:
                        best_min_dist = min_d
                        best_idx = idx
                selected.append(best_idx)
            pts = pts[selected]
        elif len(pts) < N:
            # Pad with random points
            while len(pts) < N:
                pts = np.vstack([pts, np.random.uniform(-0.45, 0.45, 3)])
        
        return pts[:N]

    # Generate initial candidates from BCC lattices with different densities
    # n=3 gives 54 points. We select 50.
    # n=4 gives 128 points. We select 50.
    
    best_init = None
    best_init_score = -np.inf

    for n_per_side in [3, 4]:
        for trial in range(5): # Try 5 different random selections/subsets
            pts = generate_bcc_lattice(n_per_side)
            # Add small random noise to break symmetry
            pts = pts + np.random.normal(0, 0.01, pts.shape)
            pts = np.clip(pts, -0.49, 0.49)
            
            # Evaluate initial radius
            d = cdist(pts, pts)
            np.fill_diagonal(d, np.inf)
            min_pair = np.min(d)
            dist_to_bound = 0.5 - np.max(np.abs(pts), axis=1)
            min_bound = np.min(dist_to_bound)
            r = 0.5 * min(min_pair, min_bound)
            
            if r > best_init_score:
                best_init_score = r
                best_init = pts.copy()

    # Objective function for optimization
    # We use a soft repulsion potential to maximize the minimum distance.
    # Minimizing sum(1/d^p) for large p approximates maximizing min(d).
    # This is differentiable and helps Powell/Nelder-Mead.
    
    def soft_repulsion_obj(x):
        pts = x.reshape(N, 3)
        # Boundary penalty: push points inside if they go out
        # We use a smooth penalty for boundaries
        dist_to_bound = 0.5 - np.abs(pts)
        # If any coordinate is outside, penalty is huge
        # max(0, |x| - 0.5)
        boundary_violation = np.maximum(0, np.abs(pts) - 0.5)
        boundary_penalty = np.sum(boundary_violation**4) * 1000
        
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        
        # Avoid division by zero
        d_safe = np.maximum(d, 1e-6)
        
        # Sum of inverse powers
        # p=4 is a good balance
        repulsion = np.sum(1.0 / (d_safe**4))
        
        return repulsion + boundary_penalty

    def hard_min_obj(x):
        pts = x.reshape(N, 3)
        if np.any(np.abs(pts) > 0.5 + 1e-9):
            return 1e10
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        dist_to_bound = 0.5 - np.max(np.abs(pts), axis=1)
        min_bound_dist = np.min(dist_to_bound)
        r = 0.5 * min(min_pair_dist, min_bound_dist)
        return -r

    x0 = best_init.flatten()
    
    best_x = x0.copy()
    best_score = -hard_min_obj(x0)

    # Multi-start optimization
    n_restarts = 10
    for restart in range(n_restarts):
        if restart == 0:
            x_init = x0.copy()
        else:
            # Perturb the initial solution
            scale = 0.05 * (1 + restart/10.0)
            x_init = x0 + np.random.normal(0, scale, N * 3)
            x_init = np.clip(x_init.reshape(N, 3), -0.495, 0.495).flatten()

        # Phase 1: Optimize soft repulsion to find a good local minimum structure
        try:
            res_soft = minimize(
                soft_repulsion_obj,
                x_init,
                method='Powell',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            x_mid = res_soft.x
        except Exception:
            x_mid = x_init

        # Phase 2: Polish with hard min objective
        try:
            res_hard = minimize(
                hard_min_obj,
                x_mid,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res_hard.fun < -best_score:
                best_score = -res_hard.fun
                best_x = res_hard.x.copy()
        except Exception:
            pass

    # Final cleanup and validation
    result_centers = best_x.reshape(N, 3)
    result_centers = np.clip(result_centers, -0.499999, 0.499999)
    
    return result_centers
