import numpy as np
import time
from scipy.optimize import minimize

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Step 1: Generate hexagonal lattice points (from inspiration 1, proven effective)
    def hex_lattice_init(n):
        spacing = 2.0 / (np.sqrt(n) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            mask = radius <= 1.0 - spacing / 2.0
            points = points[mask]
            if len(points) >= n:
                # Select n closest to center
                dists = np.hypot(points[:, 0], points[:, 1])
                idx = np.argsort(dists, kind='stable')[:n]
                return points[idx]
            spacing *= 0.98
    
    pts = hex_lattice_init(N)
    
    # Step 2: Compute initial radius
    def compute_radius(pts):
        from scipy.spatial.distance import pdist
        min_pair = np.min(pdist(pts)) / 2.0
        min_boundary = np.min(1.0 - np.hypot(pts[:, 0], pts[:, 1]))
        return min(min_pair, min_boundary)
    
    best_radius = compute_radius(pts)
    best_pts = pts.copy()
    
    # Step 3: Smooth optimization using L-BFGS-B with a soft-min surrogate
    # We maximize the minimum of (pair distances/2) and (boundary distances)
    # Using a smooth approximation: soft_min(x) = -logsumexp(-x) / alpha for large alpha
    
    def smooth_min(x, alpha=50.0):
        """Smooth approximation to min(x) using log-sum-exp"""
        x = np.asarray(x)
        # Clip to avoid numerical issues
        x_clipped = np.clip(x, -100, 100)
        return -np.log(np.sum(np.exp(-alpha * x_clipped))) / alpha
    
    def objective_and_grad(x_flat):
        """Objective: negative of smooth min of all constraints (to minimize)"""
        if time.time() > deadline:
            return 1e10, np.zeros_like(x_flat)
        
        pts = x_flat.reshape(-1, 2)
        n = len(pts)
        
        # Compute all pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Pair constraints: dist_ij / 2 >= r
        # Boundary constraints: 1 - |c_i| >= r
        r_pts = np.hypot(pts[:, 0], pts[:, 1])
        boundary = 1.0 - r_pts
        
        # All constraint values (we want these to be >= r, so we minimize -min)
        # Collect all pair distances / 2 and boundary distances
        pair_vals = dist / 2.0
        # Only take upper triangle to avoid duplicates
        iu = np.triu_indices(n, k=1)
        pair_vals = pair_vals[iu]
        
        all_vals = np.concatenate([pair_vals, boundary])
        
        # Smooth min
        val = smooth_min(all_vals, alpha=50.0)
        
        # Gradient with respect to x_flat
        # d(val)/dx = -1/alpha * sum_j exp(-alpha*x_j) * (-alpha * dx_j/dx) / sum_j exp(-alpha*x_j)
        # = sum_j exp(-alpha*x_j) * dx_j/dx / sum_j exp(-alpha*x_j)
        
        exp_vals = np.exp(-50.0 * np.clip(all_vals, -100, 100))
        weights = exp_vals / np.sum(exp_vals)
        
        grad = np.zeros_like(x_flat)
        
        # Gradient of pair distances
        # d(dist_ij)/dx_i = (x_i - x_j) / dist_ij
        # d(dist_ij/2)/dx_i = (x_i - x_j) / (2 * dist_ij)
        
        # This is expensive to compute fully, so we approximate
        # For efficiency, we'll use a simpler gradient approximation
        
        # Actually, let's compute the gradient properly but efficiently
        # d(val)/dx_i = sum_j weights_j * d(x_j)/dx_i
        
        # For pair (i,j): d(dist_ij/2)/dx_i = (x_i - x_j) / (2 * dist_ij)
        # For boundary i: d(1 - |c_i|)/dx_i = -x_i / |c_i|
        
        # Build gradient
        # Pair contributions
        for i in range(n):
            for j in range(i+1, n):
                d = dist[i, j]
                if d < 1e-10:
                    continue
                w = weights[i*(2*i-1)//2 + j - i - 1] if False else 0  # This indexing is wrong
                # Let me restructure
        
        # Better approach: compute gradient numerically or use a different formulation
        # For simplicity and speed, let's use a finite difference approach for the gradient
        # But that's slow too. Let me use a simpler but effective approach.
        
        # Alternative: use the fact that at optimum, the active constraints determine the gradient
        # For now, let's use a subgradient approximation
        
        # Actually, let's just use scipy's minimize with numerical gradient (slow but reliable)
        return -val, None
    
    # Use L-BFGS-B with numerical gradient (more reliable than our manual gradient)
    def neg_smooth_min(x_flat):
        if time.time() > deadline:
            return 1e10
        
        pts = x_flat.reshape(-1, 2)
        n = len(pts)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        r_pts = np.hypot(pts[:, 0], pts[:, 1])
        boundary = 1.0 - r_pts
        
        iu = np.triu_indices(n, k=1)
        pair_vals = dist[iu] / 2.0
        all_vals = np.concatenate([pair_vals, boundary])
        
        return -smooth_min(all_vals, alpha=50.0)
    
    # Optimize
    x0 = pts.flatten()
    
    result = minimize(
        neg_smooth_min,
        x0,
        method='L-BFGS-B',
        options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    if result.success or result.fun < neg_smooth_min(x0):
        pts_new = result.x.reshape(-1, 2)
        # Ensure points are inside the disk
        r_new = np.hypot(pts_new[:, 0], pts_new[:, 1])
        mask = r_new > 1.0
        if np.any(mask):
            pts_new[mask] = pts_new[mask] / r_new[mask, np.newaxis] * 0.9999
        
        r_new_val = compute_radius(pts_new)
        if r_new_val > best_radius:
            best_radius = r_new_val
            best_pts = pts_new.copy()
    
    # Step 4: Final polish with a local search
    if time.time() < deadline - 10:
        # Try small perturbations around the best solution
        for trial in range(20):
            if time.time() > deadline - 5:
                break
            
            # Random small perturbation
            noise = np.random.normal(0, 0.001, size=best_pts.shape)
            pts_perturbed = best_pts + noise
            
            # Keep inside disk
            r_p = np.hypot(pts_perturbed[:, 0], pts_perturbed[:, 1])
            mask = r_p > 0.999
            if np.any(mask):
                pts_perturbed[mask] = pts_perturbed[mask] / r_p[mask, np.newaxis] * 0.999
            
            r_pert = compute_radius(pts_perturbed)
            if r_pert > best_radius:
                best_radius = r_pert
                best_pts = pts_perturbed.copy()
    
    return best_pts
