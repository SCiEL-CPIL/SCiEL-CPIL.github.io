import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Initialization: Hexagonal Lattice
    def generate_hex_lattice(r):
        d = 2.0 * r
        if d <= 0:
            return np.array([])
        
        k = int(1.0 / d) + 2
        points = []
        sqrt3 = np.sqrt(3.0)
        
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * d
                y = j * d * sqrt3 / 2.0
                dist = np.hypot(x, y)
                if dist + r <= 1.0:
                    points.append([x, y])
        
        if len(points) == 0:
            return np.array([])
            
        points = np.array(points)
        if len(points) >= N:
            dists = np.hypot(points[:, 0], points[:, 1])
            indices = np.argsort(dists, kind="stable")[:N]
            return points[indices]
        else:
            return points

    # Find a good initial radius
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
            
    points = generate_hex_lattice(r_low)
    
    if len(points) < N:
        # Fallback random
        angles = np.random.rand(N) * 2 * np.pi
        radii = np.sqrt(np.random.rand(N))
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        points = points * 0.5 # Scale down

    if len(points) > N:
        dists = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(dists, kind="stable")[:N]
        points = points[indices]
        
    # Add small random jitter to break symmetry and avoid degenerate states
    points += np.random.normal(0, 0.001, (N, 2))
    
    best_points = points.copy()
    best_r = 0.0
    
    def compute_radius(pts):
        if not np.all(np.isfinite(pts)):
            return 0.0
        radii = np.hypot(pts[:, 0], pts[:, 1])
        min_boundary = np.min(1.0 - radii)
        
        # Efficient pairwise distance
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        min_pair_sq = np.min(dists_sq)
        min_pair = np.sqrt(min_pair_sq) / 2.0
        
        return min(min_pair, min_boundary)

    best_r = compute_radius(best_points)
    
    # Optimization Loop: Vectorized Gradient Descent on Potential Energy
    # Objective: Minimize E = sum_{i<j} 1/d_ij^2 + lambda * sum_i 1/(1-|c_i|)^2
    # This pushes points apart and away from the boundary.
    
    step_size = 0.001
    no_improve_count = 0
    max_no_improve = 100
    lambda_bound = 10.0
    
    for it in range(200000):
        if time.time() > deadline:
            break
            
        # Compute current radius
        current_r = compute_radius(points)
        
        if current_r > best_r:
            best_r = current_r
            best_points = points.copy()
            no_improve_count = 0
            # Slightly increase step size if improving? Or keep it?
            # Let's keep it stable.
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Perturb to escape local minimum
                perturb_scale = 0.01
                points += np.random.normal(0, perturb_scale, (N, 2))
                # Ensure validity
                radii = np.hypot(points[:, 0], points[:, 1])
                outside = radii > 1.0
                if np.any(outside):
                    points[outside] = points[outside] / radii[outside, np.newaxis]
                
                no_improve_count = 0
                step_size *= 0.9 # Reduce step size after perturbation

        # Compute Forces (Gradients)
        # dE/dp_i = sum_{j != i} -2 * (p_i - p_j) / |p_i - p_j|^4 
        #         + lambda * -2 * p_i / |p_i|^2 * 1 / (1-|p_i|)^3 * (1/|p_i|) ?
        # Let's use a simpler potential for boundary: U_bound = 1 / (1 - |p|)^2
        # dU_bound/dp = -2 / (1-|p|)^3 * d(1-|p|)/dp = -2 / (1-|p|)^3 * (-p/|p|) = 2*p / (|p| * (1-|p|)^3)
        # So force is -dE/dp.
        # F_pair = sum_{j != i} 2 * (p_i - p_j) / |p_i - p_j|^4
        # F_bound = -lambda * 2 * p_i / (|p_i| * (1-|p_i|)^3)
        
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, 1.0) # Avoid div by zero, will be masked or inf
        
        # Inverse squared distance for force magnitude
        # F_ij vector = (p_i - p_j) / dist_ij^3 * 2 ? No, gradient of 1/d^2 is -2/d^4 * (p_i-p_j).
        # Force is -grad = 2/d^4 * (p_i - p_j).
        
        inv_dists_sq = 1.0 / dists_sq
        # Mask diagonal
        mask = np.ones((N, N), dtype=bool)
        np.fill_diagonal(mask, False)
        inv_dists_sq[~mask] = inv_dists_sq[~mask]
        inv_dists_sq[mask] = 0.0
        
        # Force from pairs: sum_j 2 * (p_i - p_j) * inv_dists_sq_ij^2 ?
        # Wait. U = 1/d^2. Grad U = -2 (p_i-p_j) / d^4.
        # Force = -Grad U = 2 (p_i-p_j) / d^4.
        # (p_i - p_j) is diff[i,j].
        # So F_i = sum_j 2 * diff[i,j] * (inv_dists_sq[i,j])^2
        
        # Vectorized sum:
        # factor[i,j] = 2 * (inv_dists_sq[i,j])^2
        factors = 2.0 * (inv_dists_sq ** 2)
        factors[~mask] = factors[~mask]
        factors[mask] = 0.0
        
        # F_i = sum_j factors[i,j] * diff[i,j]
        # This is a matrix multiplication: factors @ diff? No, diff is (N, N, 2).
        # We can do: np.einsum('ij,ijk->ik', factors, diff)
        pair_forces = np.einsum('ij,ijk->ik', factors, diff)
        
        # Boundary Force
        radii = np.hypot(points[:, 0], points[:, 1])
        # Avoid division by zero for center point
        radii_safe = np.where(radii > 1e-6, radii, 1.0)
        
        # Term: 2 * lambda * p_i / (|p_i| * (1-|p_i|)^3)
        # Only apply if close to boundary? Or always?
        # If always, it pushes everything to center strongly.
        # We only want to push in if 1 - |p_i| is small.
        # Let's use a soft constraint: only apply if 1 - |p_i| < r_target_est
        # Or just use the potential everywhere but with small lambda.
        
        # Let's use the potential everywhere but scaled.
        # If |p| is small, 1-|p| is ~1, force is ~2*lambda*p/|p| which is direction to center?
        # No, p/|p| is direction OUT. Force is -grad U.
        # U = lambda / (1-|p|)^2.
        # dU/dp = lambda * 2 * (1-|p|)^-3 * (p/|p|).
        # Force = -dU/dp = -2 * lambda * p / (|p| * (1-|p|)^3).
        # This force points INWARD (opposite to p).
        
        denom = radii_safe * (1.0 - radii_safe)**3
        # If radii_safe is 1, denom is 0. Clip radii_safe to 1 - epsilon.
        radii_clipped = np.minimum(radii_safe, 1.0 - 1e-6)
        denom = radii_clipped * (1.0 - radii_clipped)**3
        denom = np.where(denom > 1e-9, denom, 1.0)
        
        bound_forces = -2.0 * lambda_bound * points / denom[:, np.newaxis]
        
        total_forces = pair_forces + bound_forces
        
        # Update
        points += total_forces * step_size
        
        # Enforce hard constraint: inside unit circle
        new_radii = np.hypot(points[:, 0], points[:, 1])
        outside = new_radii > 1.0
        if np.any(outside):
            points[outside] = points[outside] / new_radii[outside, np.newaxis]
            
        # Decay step size slowly
        step_size *= 0.9999
        
        # Optional: If best_r is very high, reduce search scope
        if best_r > 0.0535:
             break

    return best_points
