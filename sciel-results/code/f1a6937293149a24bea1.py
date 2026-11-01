"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
_DEADLINE = time.time() + 140.0  # 10s safety margin before 150s kill


def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    dists = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(dists > 1.0):
        return 0.0
    r_cont = 1.0 - np.max(dists)
    if r_cont < 0:
        return 0.0
    if N < 2:
        return r_cont
    # Efficient pairwise distance computation
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dist_mat = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist_mat, np.inf)
    r_pair = 0.5 * np.min(dist_mat)
    return min(r_cont, r_pair)


def _hex_lattice_init():
    """Generate a hexagonal lattice packing as initial guess."""
    best_pts = None
    best_score = 0.0
    for spacing in np.linspace(0.15, 0.35, 80):
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            r = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(r, kind="stable")[:N]
            pts = points[idx]
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts
    if best_pts is not None:
        return best_pts
    # Fallback: random points in unit disk
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.81, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _vectorized_repulsion_optimize(pts0, max_time=100.0):
    """Optimize using vectorized repulsion forces between circles."""
    pts = pts0.copy().astype(np.float64)
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Adaptive step size
    step_size = 0.01
    n_iter = 0
    stuck_count = 0
    start_time = time.time()
    
    while time.time() - start_time < max_time:
        n_iter += 1
        
        # Compute pairwise distances
        dx = pts[:, 0:1] - pts[0:1, 0]
        dy = pts[:, 1:2] - pts[0:1, 1]
        dist_mat = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dist_mat, 1e-10) # Avoid division by zero
        
        # Compute repulsion forces: F = sum (r_i - r_j) / d_ij^3
        # We want to maximize min distance, so we repel strongly when close.
        # Using 1/d^2 potential -> Force is 1/d^3 * direction.
        # To avoid huge forces, we can cap the force or use a softer potential.
        # Let's use a force proportional to 1/d^2.
        
        # Vectorized force calculation
        # inv_dist_sq = 1 / dist_mat**2
        # force_mat = inv_dist_sq[:, :, None] * (dx[:, :, None] - dx[None, :, None]) ... this is getting complex with broadcasting.
        
        # Simpler:
        # F_i = sum_j (P_i - P_j) / d_ij^3
        # Let's compute d_ij^3
        d_cubed = dist_mat ** 3
        # Mask diagonal
        d_cubed[np.diag_indices(N)] = 1e10
        
        # Force components
        # dx_ij = x_i - x_j. We have dx = x_i - x_j (column i, row j? No, dx[:, 0:1] - pts[0:1, 0] is x_i - x_j where i is row, j is col?
        # dx = pts[:, 0:1] (Nx1) - pts[0:1, 0] (1xN) -> shape (N, N). Element [i,j] is x_i - x_j.
        # dy = pts[:, 1:2] (Nx1) - pts[0:1, 1] (1xN) -> shape (N, N). Element [i,j] is y_i - y_j.
        
        # Force on i is sum over j of (dx[i,j] / d_ij^3, dy[i,j] / d_ij^3)
        # Note: dx[i,j] = x_i - x_j. The vector from j to i is (x_i-x_j, y_i-y_j).
        # So force on i is sum_j (dx[i,j], dy[i,j]) / d_ij^3.
        
        fx = np.sum(dx / d_cubed, axis=1)
        fy = np.sum(dy / d_cubed, axis=1)
        forces = np.column_stack((fx, fy))
        
        # Normalize forces to prevent instability
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms[force_norms == 0] = 1.0
        forces = forces / force_norms
        
        # Update positions
        new_pts = pts + step_size * forces
        
        # Keep points inside the container
        new_radii = np.hypot(new_pts[:, 0], new_pts[:, 1])
        mask = new_radii > 1.0
        if np.any(mask):
            new_pts[mask] = new_pts[mask] / new_radii[mask, np.newaxis]
        
        # Check if valid
        new_score = _score(new_pts)
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            stuck_count = 0
        else:
            stuck_count += 1
            # Reduce step size if stuck
            if stuck_count > 10:
                step_size *= 0.95
                stuck_count = 0
            # If step size becomes too small, break or restart
            if step_size < 1e-6:
                break

        # Occasionally add random perturbation to escape local minima
        if n_iter % 50 == 0:
            perturb = np.random.uniform(-0.01, 0.01, (N, 2))
            new_pts = pts + perturb
            new_radii = np.hypot(new_pts[:, 0], new_pts[:, 1])
            mask = new_radii > 1.0
            if np.any(mask):
                new_pts[mask] = new_pts[mask] / new_radii[mask, np.newaxis]
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
        
        pts = new_pts
    
    return best_pts

def _polish(pts0):
    """Polish the solution using scipy.optimize.minimize."""
    def objective(flat_pts):
        pts = flat_pts.reshape(N, 2)
        # Penalty for going outside
        dists = np.hypot(pts[:, 0], pts[:, 1])
        penalty = np.sum(np.maximum(0, dists - 1.0)**2)
        
        # We want to maximize min distance.
        # Minimize -min_distance
        # But min_distance is non-smooth.
        # Use a sum of inverse distances as a proxy?
        # Or use the actual min distance with a penalty for overlaps?
        
        # Let's try minimizing the sum of squared overlaps + boundary violations
        # If we just minimize sum(1/d_ij), it might cluster.
        # Let's stick to the standard approach: minimize a function that is low when packing is good.
        # f = -min_dist is good but non-smooth.
        # Let's use Nelder-Mead which handles non-smooth.
        
        # Calculate min pair distance
        if N < 2:
            return -min(1.0 - np.max(dists), 0.0)
        
        dx = pts[:, 0:1] - pts[0:1, 0]
        dy = pts[:, 1:2] - pts[0:1, 1]
        dist_mat = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dist_mat, np.inf)
        min_pair_dist = np.min(dist_mat)
        
        min_cont_dist = 1.0 - np.max(dists)
        
        current_r = min(min_pair_dist / 2.0, min_cont_dist)
        
        # Add penalty for invalid points (outside or non-finite)
        if not np.all(np.isfinite(pts)):
            return 1e10
        if np.any(dists > 1.0 + 1e-5):
            return 1e10
            
        return -current_r

    x0 = pts0.flatten()
    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
    
    if res.success or res.fun < objective(x0):
        new_pts = res.x.reshape(N, 2)
        # Ensure validity
        new_score = _score(new_pts)
        old_score = _score(pts0)
        if new_score > old_score:
            return new_pts
    return pts0

def construct():
    start_time = time.time()
    
    # Get initial hexagonal lattice configuration
    pts0 = _hex_lattice_init()
    best_pts = pts0
    best_score = _score(pts0)
    
    # Run vectorized repulsion-based optimization
    try:
        if time.time() < _DEADLINE:
            opt_pts = _vectorized_repulsion_optimize(pts0, max_time=80.0)
            opt_score = _score(opt_pts)
            if opt_score > best_score:
                best_pts = opt_pts
                best_score = opt_score
    except Exception:
        pass
    
    # Polish with local optimizer
    try:
        if time.time() < _DEADLINE - 10:
            polished_pts = _polish(best_pts)
            polished_score = _score(polished_pts)
            if polished_score > best_score:
                best_pts = polished_pts
                best_score = polished_score
    except Exception:
        pass

    # Final safety check: ensure validity
    final_score = _score(best_pts)
    if final_score <= 0:
        # Return the initial hex lattice as fallback
        return pts0
    
    return best_pts

# EVOLVE-BLOCK-END
