import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit

    # --- Initialization: Hexagonal lattice ---
    def hex_lattice_init(n):
        spacing = 2.0 / (np.sqrt(n) + 1.0)
        for _ in range(20):
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            mask = radius <= 1.0 - spacing / 2.0
            points = points[mask]
            if len(points) >= n:
                dists = np.hypot(points[:, 0], points[:, 1])
                idx = np.argsort(dists, kind='stable')[:n]
                return points[idx]
            spacing *= 0.98
        # Fallback: random points
        angles = np.random.uniform(0, 2 * np.pi, n)
        radii = np.random.uniform(0, 0.9, n)
        return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    pts = hex_lattice_init(N)

    # --- Helper: Compute exact packing radius ---
    def compute_radius(pts):
        # Use KDTree for efficiency if N is large, but for 300, pdist is fast enough
        # However, for repeated calls in SA, we might need a faster local check or just accept pdist cost
        # 300 points -> 44850 pairs. pdist is vectorized and fast.
        min_pair = np.min(pdist(pts)) / 2.0
        min_boundary = np.min(1.0 - np.hypot(pts[:, 0], pts[:, 1]))
        return min(min_pair, min_boundary)

    best_radius = compute_radius(pts)
    best_pts = pts.copy()

    # --- Phase 1: Adaptive Repulsive Force Relaxation ---
    # Strategy: For each pair closer than a target distance, apply a repulsive force.
    # The target distance is set slightly higher than the current best minimum distance
    # to push the system into a better configuration.
    
    def force_relax(pts, steps=5000, target_r=None):
        if target_r is None:
            # Estimate a good target: current best radius + small epsilon
            current_r = compute_radius(pts)
            target_r = current_r * 1.01 + 0.0001
        
        # We want min dist >= 2 * target_r
        min_dist = 2.0 * target_r
        
        # Adaptive step size
        step_size = 0.005
        # Decay step size over iterations
        decay = 0.995
        
        for step in range(steps):
            if time.time() > deadline - 10:
                break
                
            # Compute pairwise distances
            # Using cKDTree to find neighbors within min_dist * 1.5 to save time?
            # For N=300, full pairwise is ~45k distances. 
            # Let's use full pairwise for simplicity and vectorization, it's fast enough.
            
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dist, np.inf)
            
            # Find violating pairs: dist < min_dist
            # We only care about pairs that are too close
            # To be efficient, we can threshold.
            # But we need the gradient (direction) for all violating pairs.
            
            # Create a mask of violating pairs
            # Note: dist is symmetric. We only need to process upper triangle to avoid double counting forces?
            # No, force on i from j is equal and opposite to force on j from i.
            # So we can compute force on all points by summing contributions from all j != i.
            
            # Force magnitude: k * (1 - dist/min_dist) if dist < min_dist, else 0
            # Direction: (p_i - p_j) / dist
            
            # Vectorized force calculation
            # dist_safe = np.where(dist < min_dist, dist, 1.0) # Avoid div by 0
            # Actually, if dist == 0, we need a random kick.
            
            # Let's compute the "overlap" factor
            overlap = np.maximum(0, 1.0 - dist / min_dist)
            
            # Force vector: sum_j (overlap_ij * (p_i - p_j) / dist_ij)
            # We need to divide by dist. Where dist is 0, set direction to 0 (will handle separately or rely on noise)
            # To avoid NaN, replace 0 dist with 1 in denominator, but overlap will be 1.
            dist_safe = np.where(dist > 1e-12, dist, 1.0)
            
            # Normalize diff vector
            # diff[i,j] = p_i - p_j
            # unit_vec = diff / dist_safe
            unit_vec = diff / dist_safe[:, :, np.newaxis]
            
            # Weight by overlap
            # force[i] = sum_j overlap[i,j] * unit_vec[i,j]
            # Note: overlap is symmetric, unit_vec is antisymmetric (unit_vec[j,i] = -unit_vec[i,j])
            # So summing over j is correct.
            
            forces = np.sum(overlap[:, :, np.newaxis] * unit_vec, axis=1)
            
            # Boundary force: if point is too close to boundary, push inward
            r_pts = np.hypot(pts[:, 0], pts[:, 1])
            # Boundary constraint: r_pts <= 1 - target_r
            # If r_pts > 1 - target_r, push towards center
            boundary_violation = np.maximum(0, r_pts - (1.0 - target_r))
            
            # Direction is -p / r
            r_safe = np.where(r_pts > 1e-12, r_pts, 1.0)
            boundary_forces = -boundary_violation[:, np.newaxis] * pts / r_safe[:, np.newaxis]
            
            # Combine forces
            total_forces = forces + boundary_forces
            
            # Update positions
            pts = pts + step_size * total_forces
            
            # Decay step size
            step_size *= decay
            
            # Periodically check if we improved
            # This is expensive (pdist), so do it every 100 steps
            if step % 100 == 0:
                # Ensure points stay inside (hard constraint for validity)
                r_check = np.hypot(pts[:, 0], pts[:, 1])
                mask = r_check > 1.0
                if np.any(mask):
                    pts[mask] = pts[mask] / r_check[mask, np.newaxis] * 0.9999
                    
                # Compute radius
                current_r = compute_radius(pts)
                if current_r > best_radius:
                    best_radius = current_r
                    best_pts = pts.copy()
                    # If we improved, we might want to increase target_r slightly?
                    # For now, keep target_r fixed to push for a specific radius.
                    # Or adapt target_r to best_radius * 1.01
                    target_r = best_radius * 1.005 + 0.0001

        return pts, best_radius, best_pts

    # Run force relaxation for several rounds with increasing target radius
    # Start with a conservative target
    current_best = compute_radius(pts)
    
    # We will try to push for a radius slightly larger than current best
    # Number of relaxation rounds
    num_rounds = 10
    
    for round_idx in range(num_rounds):
        if time.time() > deadline - 20:
            break
            
        # Target radius: current best + small increment
        # The increment should be small to allow convergence
        # Heuristic: try to gain 0.1% of current radius per round
        target_r = current_best * 1.002 + 0.00005
        
        # Run relaxation
        pts, best_radius, best_pts = force_relax(pts, steps=3000, target_r=target_r)
        current_best = best_radius
        
        # If no improvement in last few rounds, break
        if round_idx > 2 and abs(current_best - best_radius) < 1e-7:
            # Check if we are stuck
            # We can break if progress is negligible
            if current_best < best_radius * 1.0001:
                break

    # --- Phase 2: Simulated Annealing Refinement ---
    # Use the exact objective function (min distance)
    # This allows escaping local minima found by the smooth force relaxation
    
    def simulated_annealing(pts, steps=50000, T_start=0.001, T_end=1e-6, cooling=0.9995):
        current_pts = pts.copy()
        current_score = compute_radius(current_pts)
        best_pts_sa = current_pts.copy()
        best_score_sa = current_score
        
        T = T_start
        scale = 0.005 # Initial perturbation scale
        
        for step in range(steps):
            if time.time() > deadline - 5:
                break
                
            # Perturb a random subset of points
            # Or all points with small noise?
            # For packing, perturbing all points slightly is good.
            # Or perturbing one point?
            # Let's perturb all points with Gaussian noise scaled by T
            
            noise = np.random.normal(0, scale, size=current_pts.shape)
            new_pts = current_pts + noise
            
            # Enforce hard boundary constraint (projection)
            r_new = np.hypot(new_pts[:, 0], new_pts[:, 1])
            mask = r_new > 1.0
            if np.any(mask):
                new_pts[mask] = new_pts[mask] / r_new[mask, np.newaxis] * 0.9999
            
            # Compute new score
            new_score = compute_radius(new_pts)
            
            # Acceptance criterion
            delta = new_score - current_score
            if delta > 0 or np.random.random() < np.exp(delta / T):
                current_pts = new_pts
                current_score = new_score
                
                if current_score > best_score_sa:
                    best_score_sa = current_score
                    best_pts_sa = current_pts.copy()
                    
                    # Update global best
                    if current_score > best_radius:
                        best_radius = current_score
                        best_pts = current_pts.copy()
            
            # Cool down
            T *= cooling
            # Reduce scale as we cool
            scale *= 0.9999

        return best_pts_sa, best_score_sa

    # Run SA
    if time.time() < deadline - 10:
        # Use the best points found so far
        pts, sa_best_score = simulated_annealing(best_pts, steps=10000, T_start=0.002, T_end=1e-6, cooling=0.999)
        
        if sa_best_score > best_radius:
            best_radius = sa_best_score
            best_pts = pts

    # --- Final Validation ---
    r_final = np.hypot(best_pts[:, 0], best_pts[:, 1])
    mask = r_final > 0.9999
    if np.any(mask):
        best_pts[mask] = best_pts[mask] / r_final[mask, np.newaxis] * 0.9999

    # Ensure no coincident points (though unlikely with SA and force)
    # If any, nudge them
    # This is a safety check
    dists = pdist(best_pts)
    if np.min(dists) < 1e-6:
        # Find close pairs and separate
        # For simplicity, just add tiny noise to all points
        best_pts += np.random.normal(0, 1e-5, size=best_pts.shape)
        r_final = np.hypot(best_pts[:, 0], best_pts[:, 1])
        mask = r_final > 0.9999
        if np.any(mask):
            best_pts[mask] = best_pts[mask] / r_final[mask, np.newaxis] * 0.9999

    return best_pts
