import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Initialization: Hexagonal Lattice
    # Approximate optimal spacing for N circles in unit circle
    # Area of circle = pi. Area of N circles = N * pi * r^2.
    # Max density ~ 0.9069.
    # N * r^2 ~ 0.9069 * 1^2 => r ~ sqrt(0.9069/N)
    # For N=300, r ~ sqrt(0.00302) ~ 0.055.
    # Lattice spacing s ~ 2r.
    
    def generate_lattice():
        # Try a few different lattice orientations or densities to get a good start
        best_points = None
        best_r = 0
        
        for scale_factor in [0.9, 0.95, 1.0, 1.05, 1.1]:
            # Estimated radius
            r_est = np.sqrt(0.9 / N) * scale_factor
            spacing = 2.0 * r_est
            
            # Generate grid
            k = int(1.0 / spacing) + 2
            rows = []
            for j in range(-k, k + 1):
                for i in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    if x*x + y*y <= 1.0:
                        rows.append((x, y))
            
            points = np.array(rows)
            if len(points) < N:
                continue
                
            # Select N points closest to center to ensure symmetry and density
            radii = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(radii, kind='stable')
            points = points[idx[:N]]
            
            # Calculate current radius
            d_min = np.inf
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(points[i] - points[j])
                    if d < d_min:
                        d_min = d
                d_bound = 1.0 - np.linalg.norm(points[i])
                if d_bound < d_min:
                    d_min = d_bound
            
            r_cur = d_min / 2.0
            
            if r_cur > best_r:
                best_r = r_cur
                best_points = points.copy()
                
        return best_points, best_r

    points, r = generate_lattice()
    
    if points is None:
        # Fallback: Random points
        points = np.random.rand(N, 2) * 2 - 1
        # Filter to inside unit circle
        mask = np.hypot(points[:, 0], points[:, 1]) < 1.0
        while np.sum(mask) < N:
            new_pts = np.random.rand(N, 2) * 2 - 1
            new_mask = np.hypot(new_pts[:, 0], new_pts[:, 1]) < 1.0
            points = np.vstack([points, new_pts[new_mask]])
            mask = new_mask
        points = points[:N]
        # Calculate r
        d_min = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(points[i] - points[j])
                if d < d_min:
                    d_min = d
            d_bound = 1.0 - np.linalg.norm(points[i])
            if d_bound < d_min:
                d_min = d_bound
        r = d_min / 2.0

    best_r = r
    best_points = points.copy()
    
    # 2. Optimization Loop: Force-based relaxation to increase r
    # We try to increase r slightly and then relax the configuration to satisfy constraints.
    
    # Precompute indices for pairwise distances
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    def compute_r(pts):
        # Vectorized distance calculation
        # pts is (N, 2)
        # dists is (N, N)
        # This is memory intensive for N=300 (90k floats), but fast.
        dists = np.sqrt(np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :])**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        radii = np.hypot(pts[:, 0], pts[:, 1])
        min_bound_dist = np.min(1.0 - radii)
        
        return min(min_pair_dist, min_bound_dist) / 2.0

    def relax(pts, target_r, steps=50, lr=0.01):
        # Gradient descent on violation energy for fixed target_r
        # E = sum(max(0, 2*target_r - d_ij)^2) + sum(max(0, target_r - (1-|c_i|))^2)
        # This is non-smooth, so we use a soft approximation or just simple force push.
        # Force push is simpler and robust.
        
        pts = pts.copy()
        for step in range(steps):
            forces = np.zeros_like(pts)
            
            # Pairwise repulsion
            # Calculate all pairwise distances
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=-1))
            np.fill_diagonal(dists, np.inf)
            
            # Only consider pairs that are too close
            # threshold = 2 * target_r
            # mask = dists < threshold
            # To avoid division by zero or large forces, we use a smooth function
            
            # Force magnitude: proportional to overlap
            # F_ij = (2*r - d_ij) * direction_ij
            # direction_ij = (p_i - p_j) / d_ij
            
            # We can vectorize this:
            # overlap = np.maximum(0, 2*target_r - dists)
            # But we need direction.
            
            # Let's just sum forces for all pairs to be safe, it's O(N^2)
            # diff is (N, N, 2)
            # dists is (N, N)
            
            # Avoid division by zero
            dists_safe = np.where(dists == 0, 1e-6, dists)
            
            # Unit vectors
            unit_vecs = diff / dists_safe[:, :, np.newaxis]
            
            # Overlap amount
            overlap = np.maximum(0, 2.0 * target_r - dists)
            
            # Force on i from j is overlap * unit_vec (pointing away from j)
            # force_ij = overlap[:, :, np.newaxis] * unit_vecs
            # Sum over j for each i
            forces += np.sum(overlap[:, :, np.newaxis] * unit_vecs, axis=1)
            
            # Boundary constraint
            radii = np.hypot(pts[:, 0], pts[:, 1])
            bound_overlap = np.maximum(0, target_r - (1.0 - radii))
            # Direction is towards center: -pts / |pts|
            radii_safe = np.where(radii == 0, 1e-6, radii)
            unit_center = -pts / radii_safe[:, np.newaxis]
            forces += bound_overlap[:, np.newaxis] * unit_center
            
            # Update
            pts += lr * forces
            
            # Project back to inside unit circle if they go out?
            # The force should pull them back, but let's be safe.
            new_radii = np.hypot(pts[:, 0], pts[:, 1])
            mask_out = new_radii > 1.0
            if np.any(mask_out):
                pts[mask_out] *= (1.0 / new_radii[mask_out])[:, np.newaxis]
                
        return pts

    # Iterative increase of r
    current_r = r
    step_r = current_r * 0.001 # 0.1% increase per try
    max_iterations = 1000
    
    for _ in range(max_iterations):
        if time.time() > deadline - 5.0:
            break
            
        target_r = current_r + step_r
        
        # Try to relax to fit target_r
        # We need to check if it's feasible.
        # Run relaxation
        new_pts = relax(best_points, target_r, steps=20, lr=0.05)
        
        # Check if new_pts satisfies constraints for target_r
        new_r = compute_r(new_pts)
        
        if new_r >= target_r - 1e-6: # Feasible
            current_r = target_r
            best_points = new_pts
            best_r = current_r
            # Maybe increase step size slightly?
            # step_r = step_r * 1.05
        else:
            # Not feasible, decrease step size
            step_r *= 0.8
            if step_r < current_r * 1e-5:
                break
                
    # Final polish: Use scipy optimize if time permits?
    # The force method is usually good enough.
    # Let's do a few more steps of fine-tuning with smaller steps.
    
    if time.time() < deadline - 10.0:
        step_r = best_r * 0.0001
        for _ in range(200):
            if time.time() > deadline - 2.0:
                break
            target_r = best_r + step_r
            new_pts = relax(best_points, target_r, steps=10, lr=0.01)
            new_r = compute_r(new_pts)
            if new_r >= target_r - 1e-7:
                best_r = target_r
                best_points = new_pts
            else:
                break

    return best_points
