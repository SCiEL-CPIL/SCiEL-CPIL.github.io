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
        """Nonsmooth objective for Powell"""
        pts = flat_pts.reshape(n, 2)
        d = pdist(pts, 'euclidean')
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        return -(d_min / d_max) ** 2

    def make_hexagonal_initial():
        """
        Generate a 16-point configuration based on a hexagonal lattice.
        We use a 19-point hexagon (center + 6 + 12) and remove 3 points
        to get 16, or just take the first 16 from a larger hex grid.
        A better approach: Center + 6 (r=1) + 9 (r=sqrt(3) approx).
        """
        pts = []
        # Center
        pts.append([0.0, 0.0])
        
        # Inner ring (6 points, radius 1)
        for k in range(6):
            angle = k * 2 * np.pi / 6
            pts.append([np.cos(angle), np.sin(angle)])
            
        # Outer ring (9 points, radius sqrt(3))
        # To maximize min distance, space them evenly.
        r_outer = np.sqrt(3)
        for k in range(9):
            angle = k * 2 * np.pi / 9
            pts.append([r_outer * np.cos(angle), r_outer * np.sin(angle)])
            
        pts = np.array(pts[:n])
        pts -= np.mean(pts, axis=0)
        return pts

    def make_grid_initial():
        """
        Generate a 4x4 grid configuration.
        """
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        xx, yy = np.meshgrid(x, y)
        pts = np.column_stack((xx.ravel(), yy.ravel()))
        pts -= np.mean(pts, axis=0)
        return pts

    def make_circle_initial():
        """
        Generate 16 points evenly spaced on a circle.
        """
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        pts = np.column_stack((np.cos(angles), np.sin(angles)))
        pts -= np.mean(pts, axis=0)
        return pts

    def make_random_initial(seed):
        """
        Generate random points in a unit square.
        """
        rng = np.random.RandomState(seed)
        pts = rng.rand(n, 2) * 2 - 1
        pts -= np.mean(pts, axis=0)
        return pts

    def improve_powell(pts, max_iter=1000):
        """
        Local optimization using Powell's method.
        """
        x0 = pts.flatten()
        res = minimize(
            neg_ratio_flat,
            x0,
            method='Powell',
            options={'maxiter': max_iter, 'xtol': 1e-10, 'ftol': 1e-10}
        )
        best_pts = res.x.reshape(n, 2)
        return best_pts

    # Generate initial configurations
    initial_configs = [
        make_hexagonal_initial(),
        make_grid_initial(),
        make_circle_initial(),
        make_random_initial(1),
        make_random_initial(42),
        make_random_initial(123),
        make_random_initial(999),
    ]
    
    # Add perturbed versions of the best structured ones
    rng = np.random.RandomState(0)
    base_hex = make_hexagonal_initial()
    for noise_scale in [0.05, 0.1, 0.2]:
        perturbed = base_hex + rng.randn(n, 2) * noise_scale
        perturbed -= np.mean(perturbed, axis=0)
        initial_configs.append(perturbed)
        
    base_grid = make_grid_initial()
    for noise_scale in [0.05, 0.1]:
        perturbed = base_grid + rng.randn(n, 2) * noise_scale
        perturbed -= np.mean(perturbed, axis=0)
        initial_configs.append(perturbed)

    best_pts = None
    best_val = -np.inf

    # Optimize each configuration
    for i, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 10:
            break
        
        # 1. Powell optimization
        opt_pts = improve_powell(init_pts, max_iter=500)
        val = score(opt_pts)
        
        if val > best_val:
            best_val = val
            best_pts = opt_pts
            
        # 2. If we have time, do a second pass with tighter tolerance on the best found so far
        if time.time() < deadline - 30:
             # Only polish if this candidate is close to the best or is the best
             if val > best_val - 0.01:
                 polished_pts = improve_powell(opt_pts, max_iter=1000)
                 polished_val = score(polished_pts)
                 if polished_val > best_val:
                     best_val = polished_val
                     best_pts = polished_pts

    # Final high-precision polish on the best candidate
    if best_pts is not None and time.time() < deadline - 5:
        final_pts = improve_powell(best_pts, max_iter=2000)
        final_val = score(final_pts)
        if final_val > best_val:
            best_pts = final_pts
            best_val = final_val
    
    # Ensure we have a valid result
    if best_pts is None:
        best_pts = make_hexagonal_initial()
    
    return best_pts

# EVOLVE-BLOCK-END
