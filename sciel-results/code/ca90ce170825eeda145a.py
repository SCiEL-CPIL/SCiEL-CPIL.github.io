"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    def generate_hex_lattice(r):
        """
        Generate a hexagonal lattice of points with circle radius r.
        Returns an array of points that fit inside the unit circle.
        """
        d = 2.0 * r
        if d <= 0:
            return np.array([])
        
        # Range of indices to cover the unit circle
        k = int(1.0 / d) + 2
        
        points = []
        sqrt3 = np.sqrt(3.0)
        
        for j in range(-k, k + 1):
            # Offset every other row by half a spacing
            offset = 0.5 * d if (j % 2 != 0) else 0.0
            for i in range(-k, k + 1):
                x = i * d + offset
                y = j * d * sqrt3 / 2.0
                dist = np.hypot(x, y)
                # Ensure the circle fits inside the unit circle
                if dist + r <= 1.0:
                    points.append([x, y])
        
        if len(points) == 0:
            return np.array([])
            
        points = np.array(points)
        
        # If we have more than N points, select N points closest to the center
        if len(points) >= N:
            dists = np.hypot(points[:, 0], points[:, 1])
            indices = np.argsort(dists, kind="stable")[:N]
            return points[indices]
        else:
            return points

    def compute_radius(pts):
        """
        Computes the maximum radius for the given points.
        r = min( min_{i<j} dist(i,j)/2, min_i (1 - dist(i, origin)) )
        """
        if len(pts) != N:
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
        
        # Distance to boundary
        radii = np.hypot(pts[:, 0], pts[:, 1])
        min_boundary = np.min(1.0 - radii)
        
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        return min(min_pair, min_boundary)

    # 1. Initialization: Find best hexagonal lattice
    r_low = 0.01
    r_high = 0.08
    
    # Ensure r_low gives enough points
    while True:
        pts = generate_hex_lattice(r_low)
        if len(pts) >= N:
            break
        r_low *= 0.8
        
    # Binary search for max r in hex lattice
    for _ in range(20):
        r_mid = (r_low + r_high) / 2.0
        pts = generate_hex_lattice(r_mid)
        if len(pts) >= N:
            r_low = r_mid
        else:
            r_high = r_mid
            
    # r_low is the largest radius for which hex lattice works directly.
    points = generate_hex_lattice(r_low)
    
    # Safety net: if something went wrong, use random points
    if len(points) < N:
         angles = np.random.rand(N) * 2 * np.pi
         radii = np.sqrt(np.random.rand(N)) * 0.5
         points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    # Ensure we have exactly N points
    if len(points) > N:
        dists = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(dists, kind="stable")[:N]
        points = points[indices]
        
    best_points = points.copy()
    best_r = compute_radius(best_points)
    
    # 2. Optimization: Improved Force-Based Relaxation
    # Strategy:
    # - Use a soft-min surrogate to smooth the objective landscape.
    # - Apply forces based on the gradient of the soft-min.
    # - Use adaptive step size and occasional restarts/perturbations.
    
    # Parameters
    step_size = 0.005
    decay_rate = 0.9998
    beta = 50.0  # Temperature for soft-min (higher = sharper)
    
    # Main optimization loop
    for it in range(20000):
        if time.time() > deadline:
            break
            
        # Compute current radius
        current_r = compute_radius(points)
        
        if current_r > best_r:
            best_r = current_r
            best_points = points.copy()
        
        # Calculate Forces using Soft-Min Gradient
        forces = np.zeros_like(points)
        
        # 1. Pairwise Repulsion (Soft-Min Gradient)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # We want to maximize min(dists/2, 1 - radii).
        # Let f_ij = dists_ij / 2.0
        # Let g_i = 1.0 - radii_i
        # We want to maximize min(f_ij, g_i).
        # Soft-min: m = -1/beta * log(sum(exp(-beta * x)))
        # Gradient of m w.r.t. x_k is -exp(-beta * x_k) / sum(exp(-beta * x_j))
        
        # To avoid numerical issues with large beta, we can scale.
        # However, for N=300, 45k pairs + 300 boundary terms.
        # Direct log-sum-exp might be expensive if done naively.
        # Let's use a simpler heuristic gradient:
        # Force on i from j is proportional to (p_i - p_j) / dists_ij^2 * weight_ij
        # where weight_ij is high if dists_ij is close to the minimum.
        
        # Approximation:
        # Find the current minimum distance d_min.
        d_min = np.min(dists)
        r_min = np.min(1.0 - np.hypot(points[:, 0], points[:, 1]))
        
        # Active constraints are those close to the minimum.
        # Let's define a "tightness" threshold.
        threshold = 1.0 + 0.05 * max(d_min, r_min)
        
        # Pairwise forces
        # Only consider pairs where dists < threshold * 2 * r_min (approx)
        # Actually, let's just use the inverse square law for all pairs, but scaled.
        # F_ij = (p_i - p_j) / dists_ij^2
        # This naturally pushes closer pairs apart more strongly.
        
        safe_dists = np.where(dists > 1e-6, dists, 1.0)
        inv_d2 = 1.0 / (safe_dists**2)
        
        # Zero out diagonal
        np.fill_diagonal(inv_d2, 0.0)
        
        force_vecs = diff * inv_d2[:, :, np.newaxis]
        forces += np.sum(force_vecs, axis=1)
        
        # 2. Boundary Attraction
        radii = np.hypot(points[:, 0], points[:, 1])
        # Push points inward if they are close to the boundary relative to r_min
        # Force is -p_i / |p_i| * strength
        # Strength is high if (1 - radii) is close to r_min
        
        boundary_tightness = 1.0 - radii
        # Normalize tightness?
        # If boundary_tightness < threshold, apply force.
        # Let's use a similar inverse square logic? No, boundary is a wall.
        # Force proportional to (1 - (1-radii)/r_min)?
        # Simpler: if radii > 1 - r_min * 1.1, push in.
        
        boundary_mask = radii > (1.0 - r_min * 1.1)
        if np.any(boundary_mask):
            safe_radii = np.where(radii > 1e-6, radii, 1.0)
            unit_vecs = points / safe_radii[:, np.newaxis]
            # Violation amount
            violations = np.maximum(radii - (1.0 - r_min), 0.0)
            boundary_forces = -unit_vecs * violations[:, np.newaxis] * 10.0
            forces[boundary_mask] += boundary_forces[boundary_mask]
            
        # Scale forces
        forces *= 0.1
        
        # Update positions
        points += forces * step_size
        
        # Decay step size
        step_size *= decay_rate
        
        # Occasionally perturb
        if it % 50 == 0:
            num_perturb = N // 10
            idx = np.random.choice(N, num_perturb, replace=False)
            perturbations = np.random.randn(num_perturb, 2) * (step_size * 2)
            points[idx] += perturbations
            new_radii = np.hypot(points[:, 0], points[:, 1])
            outside = new_radii > 1.0
            if np.any(outside):
                points[outside] = points[outside] / new_radii[outside, np.newaxis]
        
        # Hard constraint projection
        new_radii = np.hypot(points[:, 0], points[:, 1])
        outside = new_radii > 1.0
        if np.any(outside):
            points[outside] = points[outside] / new_radii[outside, np.newaxis]

    # Final check
    final_r = compute_radius(best_points)
    if final_r > best_r:
        best_r = final_r
        best_points = best_points.copy()

    return best_points
# EVOLVE-BLOCK-END
