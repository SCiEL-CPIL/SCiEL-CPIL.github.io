import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # --- Initialization: Hexagonal Lattice ---
    # This provides a good starting point close to the optimal packing.
    def hex_lattice_init(n):
        # Target radius is approx 0.05315
        # Lattice spacing s = 2 * r
        # We want to fit n points in a circle of radius 1.
        # Area of hex lattice packing is roughly n * s^2 * sqrt(3)/2
        # Area of container is pi * 1^2
        # s^2 ~ 2*pi / (n*sqrt(3))
        # s ~ sqrt(2*pi / (300*1.732)) ~ sqrt(6.28/519) ~ sqrt(0.012) ~ 0.11
        best_pts = None
        best_r = -1
        
        # Try a range of spacings to ensure we get exactly N points
        # and maximize the initial radius
        for spacing in np.linspace(0.09, 0.13, 10):
            k = int(1.0 / spacing) + 2
            points = []
            for j in range(-k, k + 1):
                for i in range(-k, k + 1):
                    x = i * spacing
                    y = j * spacing * np.sqrt(3)
                    if (j % 2) != 0:
                        x += spacing / 2
                    points.append([x, y])
            points = np.array(points)
            r = np.hypot(points[:, 0], points[:, 1])
            mask = r <= 1.0 - spacing
            points = points[mask]
            
            if len(points) >= n:
                # Select points closest to center to keep them inside
                dists = np.hypot(points[:, 0], points[:, 1])
                idx = np.argsort(dists, kind='stable')[:n]
                pts = points[idx]
                
                # Calculate current radius
                tree_dists = []
                # For init, just estimate or use a quick check
                # Actually, let's just use the spacing as a proxy for r initially
                # But we need to check boundary.
                min_bound = 1.0 - np.max(np.linalg.norm(pts, axis=1))
                # Approx min pair dist is spacing
                r_val = 0.5 * min(spacing, min_bound)
                
                if r_val > best_r:
                    best_r = r_val
                    best_pts = pts
            elif len(points) < n:
                # If we don't have enough points, increase spacing? No, decrease spacing to fit more?
                # Wait, if len < n, we need smaller spacing to fit more points?
                # No, smaller spacing fits MORE points.
                # If len < n, we have too few points. We need smaller spacing.
                pass
        
        if best_pts is None:
            # Fallback: Random
            best_pts = np.random.rand(n, 2) * 2 - 1
            norms = np.linalg.norm(best_pts, axis=1)
            best_pts = best_pts * (0.9 / norms)[:, np.newaxis]
            
        return best_pts

    pts = hex_lattice_init(N)
    
    # Helper to compute the actual radius for the current configuration
    def compute_radius(pts):
        if len(pts) == 0:
            return 0.0
        # Min pair distance
        # Using cKDTree is efficient, but for SA we call this frequently.
        # N=300, O(N^2) is 90k ops. 90k * 10000 iters = 900M ops. Might be slow in Python.
        # Let's use a vectorized distance matrix calculation for speed in the loop?
        # 300x300 matrix is 90k floats. Creating it takes ~1ms.
        # 10000 iters * 1ms = 10s. This is acceptable.
        
        # Distance matrix
        # D_ij = ||p_i - p_j||
        # Using broadcasting: (300, 1, 2) - (1, 300, 2) -> (300, 300, 2)
        # This uses 300*300*2*8 bytes = 1.44 MB. Very fast.
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Boundary distance
        radii = np.linalg.norm(pts, axis=1)
        min_bound_dist = 1.0 - np.max(radii)
        
        return 0.5 * min(min_pair_dist, min_bound_dist)

    current_r = compute_radius(pts)
    best_r = current_r
    best_pts = pts.copy()
    
    # --- Simulated Annealing ---
    # Objective: Maximize r
    # Energy function E(r, pts) = -r + penalty
    # We want to find pts such that min_dist(pts) >= 2*r AND |pts| <= 1-r
    
    # Strategy:
    # 1. Start with a small r (from init).
    # 2. Try to increase r slightly.
    # 3. If constraints are violated, apply forces to fix them.
    # 4. Use SA to accept moves that might temporarily decrease "quality" (violations)
    #    to escape local minima, while pushing r up.
    
    # A simpler and often effective variant for packing:
    # "Repulsive Force" SA.
    # Each pair i,j has a repulsive force if dist < 2*r.
    # Each point i has a repulsive force from boundary if |p_i| > 1-r.
    # We move points along the net force.
    # We accept the move if it reduces the total "energy" (sum of squared violations)
    # OR if we are in the "annealing" phase where we accept worse states with probability exp(-dE/T).
    # Simultaneously, we try to increase r.
    
    # Let's implement a standard SA loop on the positions, with r as a parameter
    # that we slowly increase.
    
    # Actually, a better formulation for SA in packing:
    # State: (pts, r)
    # Move: Perturb pts randomly.
    # Energy: E = sum_{i<j} max(0, 2*r - dist_ij)^2 + sum_i max(0, 1-r - |p_i|)^2
    # We want to minimize E.
    # We slowly increase r. If E becomes large, we stay at this r or decrease it slightly.
    # If E is 0, we increase r.
    
    T = 0.01 # Initial temperature
    T_min = 1e-5
    cooling_rate = 0.999
    
    # We will iterate. In each step, we propose a small random perturbation to pts.
    # We calculate the change in energy.
    
    # To make it efficient, we don't recalculate the full energy matrix every step if we can avoid it.
    # But for N=300, recalculating is fast enough (~1-2ms).
    # 150s / 2ms = 75,000 iterations. This is plenty.
    
    # Let's use a more aggressive approach:
    # 1. Calculate current energy.
    # 2. Propose move.
    # 3. Calculate new energy.
    # 4. Accept/Reject.
    # 5. If accepted, update best if r increased and energy is low.
    
    # Actually, let's just maximize the minimum distance directly with a "soft" objective.
    # Objective: J = min_pair_dist + min_bound_dist
    # We want to maximize J.
    # Gradient of J is hard.
    
    # Let's stick to the Energy Minimization with increasing r.
    
    current_r = compute_radius(pts)
    # Start r slightly below the current valid radius to allow movement
    r = current_r * 0.99
    
    # Pre-compute some constants? No, r changes.
    
    def calc_energy(pts, r):
        # Returns sum of squared violations
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Pairwise violations: dist < 2*r
        # Violation amount = 2*r - dist (if positive)
        # We only care about pairs where dist < 2*r
        # Using numpy:
        pair_violations = np.maximum(0, 2*r - dists)
        # Sum of squares
        e_pairs = np.sum(pair_violations**2)
        
        # Boundary violations: |p| > 1-r
        radii = np.linalg.norm(pts, axis=1)
        bound_violations = np.maximum(0, radii - (1.0 - r))
        e_bound = np.sum(bound_violations**2)
        
        return e_pairs + e_bound
    
    current_energy = calc_energy(pts, r)
    
    # SA Loop
    # We want to increase r.
    # We will run the SA for a fixed number of steps or time.
    # We will try to increase r every K steps if energy is low.
    
    steps_since_r_increase = 0
    r_increase_step = 50 # Try to increase r every 50 accepted moves? Or just every 50 steps?
    # Better: Increase r when energy is close to 0.
    
    # Let's use a schedule:
    # 1. Run SA to minimize energy at current r.
    # 2. If energy < threshold, increase r.
    # 3. Repeat.
    
    # To save time, we'll interleave.
    
    max_iters = 50000 # We have ~150s. 50k * 1ms = 50s. Safe.
    
    for it in range(max_iters):
        if time.time() > deadline - 5:
            break
            
        # Propose a random perturbation
        # Gaussian noise
        noise_scale = 0.01 * (1 + 0.5 * np.sin(it / 100)) # Varying noise scale
        new_pts = pts + np.random.randn(N, 2) * noise_scale
        
        # Project back into container to ensure valid geometry?
        # No, let the energy handle it. But if they fly away, energy explodes.
        # Let's clip to a larger radius, say 1.5, to prevent overflow.
        new_radii = np.linalg.norm(new_pts, axis=1)
        mask_out = new_radii > 1.5
        if np.any(mask_out):
            new_pts[mask_out] = new_pts[mask_out] / new_radii[mask_out, np.newaxis] * 1.5
            
        # Calculate new energy
        new_energy = calc_energy(new_pts, r)
        
        # Acceptance criterion
        dE = new_energy - current_energy
        if dE < 0 or np.random.rand() < np.exp(-dE / T):
            pts = new_pts
            current_energy = new_energy
            
            # Check if we can increase r
            # If energy is very small, it means we are in a valid configuration for r.
            # We can try to increase r slightly.
            if current_energy < 1e-6:
                # Try to increase r
                # How much?
                # We want to find the max r.
                # Let's increase by a small delta.
                delta_r = 1e-5
                new_r = r + delta_r
                # Check if new_r is valid?
                # We don't need to check validity immediately, the energy will rise.
                # But if we jump too high, we might get stuck in high energy.
                # Let's just set r = new_r.
                r = new_r
                current_energy = calc_energy(pts, r)
                
                # Update best if this r is valid (energy should be 0 or close if we just increased slightly)
                # Actually, if we just increased r, energy might be non-zero.
                # We only update best if the configuration is actually valid for r.
                # A configuration is valid for r if min_pair_dist >= 2r and max_bound <= 1-r.
                # This is equivalent to energy == 0.
                if current_energy == 0.0:
                    if r > best_r:
                        best_r = r
                        best_pts = pts.copy()
                else:
                    # If energy > 0, we are invalid. We need to minimize energy.
                    # We keep r as is and let SA fix the positions.
                    pass
            
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    # Final Polish:
    # We have best_pts and best_r.
    # Let's try to push best_r a bit more with a tighter local search.
    # Use the gradient method from the previous program on the best solution found by SA.
    
    # Re-define a quick local optimizer using gradients for the final polish
    def local_optimize_gradient(pts, max_iter=200):
        pts = pts.copy()
        current_r = compute_radius(pts)
        
        for it in range(max_iter):
            if time.time() > deadline - 1:
                break
                
            # Calculate gradients for the minimum distance
            tree_dists = []
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            
            min_pair_dist = np.min(dists)
            radii = np.linalg.norm(pts, axis=1)
            min_bound_dist = 1.0 - np.max(radii)
            
            global_min = min(min_pair_dist, min_bound_dist)
            
            grads = np.zeros_like(pts)
            
            # Active pairs
            # Find pairs with dist < global_min * 1.05
            # This is O(N^2) mask.
            active_mask = dists < global_min * 1.05 + 1e-9
            # Exclude diagonal
            np.fill_diagonal(active_mask, False)
            
            # For each active pair, add gradient
            # Vectorized gradient update is tricky. Let's loop over active points?
            # Or just use the nearest neighbor approach.
            
            # Let's use cKDTree for speed in gradient calc
            from scipy.spatial import cKDTree
            tree = cKDTree(pts)
            d_nn, idx_nn = tree.query(pts, k=2)
            
            min_pair_dists = d_nn[:, 1]
            min_pair_idxs = idx_nn[:, 1]
            
            active_pairs_mask = min_pair_dists < global_min * 1.05 + 1e-9
            for i in np.where(active_pairs_mask)[0]:
                j = min_pair_idxs[i]
                d_vec = pts[i] - pts[j]
                d = np.linalg.norm(d_vec)
                if d > 1e-9:
                    grads[i] += d_vec / d
                    grads[j] += (pts[j] - pts[i]) / d
            
            active_bound_mask = radii > 1.0 - (global_min * 1.05 + 1e-9)
            for i in np.where(active_bound_mask)[0]:
                r_i = np.linalg.norm(pts[i])
                if r_i > 1e-9:
                    grads[i] += -pts[i] / r_i
            
            grad_norms = np.linalg.norm(grads, axis=1)
            mask_nonzero = grad_norms > 1e-9
            if np.any(mask_nonzero):
                max_norm = np.max(grad_norms[mask_nonzero])
                if max_norm > 0:
                    grads[mask_nonzero] /= max_norm
                
                step_size = 0.01
                new_pts = pts + step_size * grads
                
                new_radii = np.linalg.norm(new_pts, axis=1)
                mask_out = new_radii > 1.0
                if np.any(mask_out):
                    new_pts[mask_out] = new_pts[mask_out] / new_radii[mask_out, np.newaxis] * 0.9999
                
                new_r = compute_radius(new_pts)
                if new_r > current_r:
                    pts = new_pts
                    current_r = new_r
                else:
                    # Try smaller step
                    step_size *= 0.5
                    if step_size < 1e-6:
                        break
                        
        return pts, current_r

    final_pts, final_r = local_optimize_gradient(best_pts)
    
    if final_r > best_r:
        best_pts = final_pts
        best_r = final_r
        
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(f"Shape: {pts.shape}")
    # Need to define compute_radius outside or just print best_r?
    # The task says return array.
    # We just return.
    pass
