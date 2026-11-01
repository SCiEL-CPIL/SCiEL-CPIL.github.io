"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

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
    """Generate a hexagonal lattice packing as initial guess, optimized for N=50."""
    # Based on Inspiration 1: This specific spacing and selection yields a strong baseline
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    
    # We might need to adjust spacing slightly to fit exactly 50 or get a good subset
    # The inspiration 1 code uses a while loop to adjust spacing. We replicate that logic.
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        
        # Filter points that are safely inside the unit circle (with margin for radius)
        # We don't know the final radius exactly, but spacing/2 is a good estimate
        # To be safe, we keep points with r < 1.0 - spacing/2
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        radius = radius[mask]
        
        if len(points) >= N:
            # Select N points. 
            # Strategy: Sort by distance from origin and take the first N? 
            # This creates a compact core. 
            # Alternatively, take all valid points and if > N, remove the ones 
            # that are "least useful" (e.g., closest to boundary if we want to maximize interior spacing, 
            # or closest to each other).
            # For now, taking the N closest to the center is a standard robust heuristic for lattice starts.
            idx = np.argsort(radius, kind="stable")[:N]
            return points[idx]
        
        # If not enough points, reduce spacing (increase density)
        spacing *= 0.98
        if spacing < 0.05:
            # Fallback to random if lattice fails
            angles = np.random.uniform(0, 2 * np.pi, N)
            radii = np.sqrt(np.random.uniform(0, 0.81, N))
            return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _optimize(pts0, max_time=120.0):
    """
    Vectorized repulsion optimization.
    Uses force-based update: F_repulsion + F_boundary.
    """
    pts = pts0.copy().astype(np.float64)
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Parameters
    step_size = 0.02
    momentum = 0.5
    v = np.zeros_like(pts)
    stuck_count = 0
    last_improve_iter = 0
    
    start_time = time.time()
    
    # Pre-compute indices for pairwise differences to avoid memory overhead of full matrix if needed,
    # but for N=50, full matrix is fine (50x50x2 floats is tiny).
    
    while time.time() < min(start_time + max_time, _DEADLINE):
        # 1. Compute Forces
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion: F_ij = (p_i - p_j) / ||p_i - p_j||^3
        # This is a strong short-range repulsion.
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 2)
        dists = np.linalg.norm(diff, axis=-1)  # (N, N)
        
        # Avoid division by zero
        np.fill_diagonal(dists, 1.0)
        
        # We only care about pairs that are close, or we can just compute all.
        # Using 1/d^3 creates a very strong force when close.
        # To prevent numerical explosion, we can cap the force or use a softer potential.
        # Let's use 1/d^2 for stability, scaled.
        weights = 1.0 / (dists * dists)
        np.fill_diagonal(weights, 0.0)
        
        # Force on i is sum_j (p_i - p_j) * w_ij
        # diff[i,j] = p_i - p_j
        # forces[i] = sum_j diff[i,j] * weights[i,j]
        forces = np.sum(diff * weights[..., np.newaxis], axis=1)
        
        # Boundary force: If a point is close to the boundary, push it in.
        # If it's far, we can leave it or gently push out to fill space?
        # Actually, for packing, we want to maximize the minimum distance.
        # Points near the boundary are "wasted" space if they can't move out.
        # A common trick: Attract points to the boundary to utilize the curved edge?
        # No, usually repulsion pushes them to the boundary.
        # We just need to ensure they don't go out.
        dists_origin = np.linalg.norm(pts, axis=1)
        # If dist > 1 - current_est_radius, push in.
        # We don't know current_est_radius exactly, but we can use a soft constraint.
        # Let's just project back if they go out in the update step.
        
        # 2. Update Velocity and Position
        v = momentum * v + (1 - momentum) * forces
        # Normalize velocity to prevent huge jumps
        v_norms = np.linalg.norm(v, axis=1, keepdims=True)
        v_norms[v_norms == 0] = 1.0
        v = v / v_norms
        
        new_pts = pts + step_size * v
        
        # 3. Project to Unit Disk
        new_radii = np.linalg.norm(new_pts, axis=1)
        mask = new_radii > 1.0
        if np.any(mask):
            new_pts[mask] = new_pts[mask] / new_radii[mask, np.newaxis]
            
        # 4. Evaluate
        new_score = _score(new_pts)
        
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            pts = new_pts.copy()
            stuck_count = 0
            last_improve_iter = time.time()
            # Slightly increase step size if improving
            step_size = min(0.05, step_size * 1.05)
        else:
            stuck_count += 1
            # Decrease step size
            step_size *= 0.98
            
            # If stuck for a while, try a random kick to escape local minimum
            if stuck_count > 50:
                # Perturb current best
                kick_strength = 0.05
                pts = best_pts + np.random.normal(0, kick_strength, (N, 2))
                # Project back
                radii = np.linalg.norm(pts, axis=1)
                mask = radii > 1.0
                if np.any(mask):
                    pts[mask] = pts[mask] / radii[mask, np.newaxis]
                v = np.zeros_like(pts)
                stuck_count = 0
                step_size = 0.02
        
        # If no improvement for a long time, reduce kick strength or restart
        if time.time() - last_improve_iter > 5.0:
             # Aggressive restart
             pts = best_pts + np.random.normal(0, 0.1, (N, 2))
             radii = np.linalg.norm(pts, axis=1)
             mask = radii > 1.0
             if np.any(mask):
                 pts[mask] = pts[mask] / radii[mask, np.newaxis]
             v = np.zeros_like(pts)
             step_size = 0.01
             last_improve_iter = time.time()

    return best_pts


def construct():
    start_time = time.time()
    
    # 1. Get initial hexagonal lattice configuration
    pts0 = _hex_lattice_init()
    best_pts = pts0
    best_score = _score(pts0)
    
    # 2. Run optimization
    try:
        if time.time() < _DEADLINE - 5.0: # Leave 5s for final checks
            opt_pts = _optimize(pts0, max_time=130.0)
            opt_score = _score(opt_pts)
            if opt_score > best_score:
                best_pts = opt_pts
                best_score = opt_score
    except Exception:
        pass
    
    # 3. Final safety check
    final_score = _score(best_pts)
    if final_score <= 0:
        return pts0
        
    return best_pts

# EVOLVE-BLOCK-END
