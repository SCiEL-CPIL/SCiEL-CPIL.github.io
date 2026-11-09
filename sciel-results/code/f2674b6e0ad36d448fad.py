# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the points.

    """
    start_time = time.time()
    deadline = start_time + 590  # safety margin

    n = 16

    def score(pts):
        """Compute R = (d_min / d_max)^2"""
        d = pdist(pts, 'euclidean')
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def neg_ratio_flat(flat_pts):
        """Nonsmooth objective for L-BFGS-B"""
        pts = flat_pts.reshape(n, 2)
        d = pdist(pts, 'euclidean')
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        return -(d_min / d_max) ** 2

    def soft_neg_ratio_flat(flat_pts, beta=10.0):
        """
        Smooth surrogate using log-sum-exp for soft-min and soft-max.
        Smaller beta is sharper (closer to true min/max) but less smooth.
        """
        pts = flat_pts.reshape(n, 2)
        d = pdist(pts, 'euclidean')
        
        # To avoid overflow, shift by max
        d_max_val = np.max(d)
        d_min_val = np.min(d)
        
        # Soft min: -beta * log(sum(exp(-(d - d_min)/beta))) - d_min
        # Actually: soft_min = d_min - beta * log(sum(exp(-(d-d_min)/beta)))
        # This is an approximation that is <= d_min.
        # Let's use a more stable formulation.
        # soft_min = -beta * log(sum(exp(-d/beta)))
        # soft_max = beta * log(sum(exp(d/beta)))
        
        # Shift for stability
        d_shifted = d - d_max_val
        exp_pos = np.exp(d_shifted / beta)
        soft_max = d_max_val + beta * np.log(np.sum(exp_pos))
        
        d_shifted_min = d - d_min_val
        exp_neg = np.exp(-d_shifted_min / beta)
        soft_min = d_min_val - beta * np.log(np.sum(exp_neg))
        
        if soft_max <= 0:
            return 0.0
        return -(soft_min / soft_max) ** 2

    def make_hexagonal_subset():
        """
        Generate a 16-point configuration based on a 19-point hexagonal lattice
        by removing 3 points from the outer ring to reduce diameter.
        """
        pts = []
        # Center
        pts.append([0.0, 0.0])
        
        # Inner ring (6 points, radius 1)
        for k in range(6):
            angle = k * 2 * np.pi / 6
            pts.append([np.cos(angle), np.sin(angle)])
            
        # Outer ring (12 points, radius sqrt(3))
        # We want to select 9 of these 12.
        # To maximize the min distance between outer points, we should space them evenly.
        # 9 points on a circle: angle step = 2*pi/9.
        # However, they are constrained to be near the hexagonal lattice points?
        # Actually, we can just generate 9 points on a circle of radius R_outer.
        # The optimizer will adjust R_outer and angles.
        # A good initial R_outer is sqrt(3) from the lattice.
        r_outer = np.sqrt(3)
        # Start angle offset to align with inner ring?
        # Inner ring points are at 0, 60, 120...
        # Outer ring points in lattice are at 0, 30, 60...
        # If we take 9 points, we can't align perfectly with lattice.
        # Let's just distribute 9 points evenly on the circle.
        for k in range(9):
            angle = k * 2 * np.pi / 9
            pts.append([r_outer * np.cos(angle), r_outer * np.sin(angle)])
            
        pts = np.array(pts[:n])
        pts -= np.mean(pts, axis=0)
        return pts

    def make_grid_initial():
        """
        Generate a 4x4 grid configuration, slightly distorted to break symmetry.
        """
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        xx, yy = np.meshgrid(x, y)
        pts = np.column_stack((xx.ravel(), yy.ravel()))
        # Add small noise to break symmetry
        rng = np.random.RandomState(42)
        pts += rng.randn(16, 2) * 0.01
        pts -= np.mean(pts, axis=0)
        return pts

    def improve_smooth(pts, beta_start=10.0, max_iter=2000):
        """Local optimization using L-BFGS-B on smooth surrogate."""
        x0 = pts.flatten()
        best_pts = pts.copy()
        best_val = score(best_pts)
        
        # Anneal beta: start large (smooth), end small (sharp)
        betas = [beta_start, beta_start/2, beta_start/4, beta_start/8]
        
        for beta in betas:
            if time.time() > deadline - 30:
                break
            res = minimize(
                lambda x: soft_neg_ratio_flat(x, beta=beta),
                x0,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            new_pts = res.x.reshape(n, 2)
            new_val = score(new_pts)
            if new_val > best_val:
                best_val = new_val
                best_pts = new_pts
                x0 = new_pts.flatten()
        
        return best_pts

    def improve_nonsmooth(pts, max_iter=5000, tol=1e-14):
        """Local optimization using L-BFGS-B on nonsmooth objective."""
        x0 = pts.flatten()
        res = minimize(
            neg_ratio_flat,
            x0,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-12, 'maxls': 50}
        )
        best_pts = res.x.reshape(n, 2)
        return best_pts

    # Generate initial configurations
    initial_configs = [
        make_hexagonal_subset(),
        make_grid_initial()
    ]
    
    # Add a few random perturbations of the hexagonal subset
    rng = np.random.RandomState(123)
    base_hex = make_hexagonal_subset()
    for noise_scale in [0.05, 0.1, 0.2]:
        perturbed = base_hex + rng.randn(n, 2) * noise_scale
        perturbed -= np.mean(perturbed, axis=0)
        initial_configs.append(perturbed)

    best_pts = None
    best_val = -np.inf

    # Optimize each configuration
    for i, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 60:
            break
        
        # 1. Smooth optimization
        opt_pts = improve_smooth(init_pts, beta_start=10.0, max_iter=1000)
        val = score(opt_pts)
        
        if val > best_val:
            best_val = val
            best_pts = opt_pts
        
        # 2. Nonsmooth refinement
        if time.time() < deadline - 30:
            opt_pts2 = improve_nonsmooth(opt_pts, max_iter=2000, tol=1e-14)
            val2 = score(opt_pts2)
            if val2 > best_val:
                best_val = val2
                best_pts = opt_pts2

    # Final high-precision polish on the best candidate
    if best_pts is not None and time.time() < deadline - 10:
        final_pts = improve_nonsmooth(best_pts, max_iter=5000, tol=1e-15)
        final_val = score(final_pts)
        if final_val > best_val:
            best_pts = final_pts
            best_val = final_val
    
    # Ensure we have a valid result
    if best_pts is None:
        best_pts = make_hexagonal_subset()
    
    return best_pts

# EVOLVE-BLOCK-END
