import numpy as np
import time
from scipy.spatial import cKDTree

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # --- Step 1: Initialization ---
    # Generate a good initial packing. 
    # Hexagonal lattice is a strong baseline.
    def hex_lattice_init(n):
        # Target radius is ~0.053. Spacing ~0.106.
        # In unit circle, area is pi. 300 circles of area pi*r^2.
        # r ~ 0.053 => area per circle ~ 0.0088. Total area ~ 2.64. Pi ~ 3.14. Density ~ 84%.
        # Hexagonal density is 90.7%. So it fits.
        
        # Start with a spacing that allows ~300 points in unit disk
        # Area of hexagon cell = s^2 * sqrt(3)/2 * 2 (for two points per cell? No, 1 point per cell in triangular lattice)
        # Triangular lattice: points at (i*s, j*s*sqrt(3)) and (i*s + s/2, j*s*sqrt(3) + s*sqrt(3)/2)
        # Distance between nearest neighbors is s.
        # We want s ~ 2*r ~ 0.106.
        
        spacing = 0.11
        while True:
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
            # Keep points that fit comfortably
            mask = r <= 1.0 - spacing
            points = points[mask]
            
            if len(points) >= n:
                # Select n closest to center
                dists = r[mask]
                idx = np.argsort(dists, kind='stable')[:n]
                return points[idx]
            else:
                spacing *= 0.95
                if spacing < 0.05:
                    # Fallback: random
                    return np.random.rand(n, 2) * 2 - 1
                    # Actually, just return whatever we have if we run out of time? No, must be N.
                    # If we get here, something is wrong.
                    # Force fit by scaling?
                    scale = 1.0 / (np.max(np.hypot(points[:,0], points[:,1])) + 1e-6)
                    return points * scale * 0.99

    pts = hex_lattice_init(N)
    
    # Ensure validity
    r_init = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(r_init > 1.0):
        pts = pts / r_init[:, np.newaxis] * 0.999

    # --- Step 2: Local Active Set Optimization ---
    # Strategy: 
    # 1. Build KDTree of current points.
    # 2. For each point i, find k nearest neighbors (k=20) and check boundary.
    # 3. If any constraint is violated (dist < 2r or |p| > 1-r), solve a local QP to push i away.
    #    Actually, simpler: move i in the direction that maximizes the minimum margin to its active neighbors/boundary.
    #    This is a "push" method.
    
    def compute_radius(pts):
        # Use KDTree for speed
        tree = cKDTree(pts)
        # Find all pairs with distance < some threshold? 
        # We need the MINIMUM distance.
        # cKDTree.query_pairs is good for finding close pairs.
        # But we need the global min.
        # Let's just compute min distance via query.
        # For each point, find 1 nearest neighbor.
        dists, _ = tree.query(pts, k=2) # k=2 to exclude self
        min_pair = np.min(dists[:, 1]) / 2.0
        min_boundary = np.min(1.0 - np.hypot(pts[:, 0], pts[:, 1]))
        return min(min_pair, min_boundary)

    best_radius = compute_radius(pts)
    best_pts = pts.copy()
    
    # Optimization Loop
    # We will iterate:
    # 1. Update KDTree
    # 2. For each point, check local constraints.
    # 3. If a point is "too close" to a neighbor or boundary, move it.
    
    # To be efficient, we only move points that are in "conflict".
    # Conflict: min_dist_to_neighbor < 2 * current_target_r OR boundary_violation.
    # But we want to MAXIMIZE r. 
    # So we can define a "local radius" for each point based on its neighbors.
    # r_i = min( min_j dist_ij/2, 1-|p_i| )
    # We want to maximize min_i r_i.
    
    # Let's use a simpler heuristic:
    # Iterate points. For each point, compute the "bottleneck" neighbor.
    # Move point slightly away from the bottleneck to increase the local min distance.
    # This is essentially gradient ascent on the local min.
    
    # Better: Use a "soft" repulsion force from all neighbors within a certain range.
    # Force F_i = sum_{j != i} w(d_ij) * (p_i - p_j) / d_ij + F_boundary
    # If d_ij < 2*r_target, push away.
    # But we don't know r_target.
    
    # Let's stick to the "Active Set" idea but simplified:
    # 1. Estimate current global r.
    # 2. For each point, find neighbors with dist < 2*r * 1.1 (slightly larger to catch near-misses).
    # 3. If a point has a neighbor closer than 2*r, it's "stuck".
    # 4. We try to "relax" the packing by moving points to increase the minimum gap.
    
    # Actually, a very effective method for circle packing is "Lloyd's algorithm" variant or "Force Directed Layout".
    # Let's implement a Force Directed Layout with a repulsive force that is very strong for close pairs.
    # F_rep(i, j) = 1 / d_ij^2 * (p_i - p_j)
    # F_attr(i) = -p_i (to keep them centered, or rather, boundary repulsion)
    # F_bound(i) = -1/|p_i|^2 * p_i (if close to boundary)
    
    # However, standard force directed layouts often converge to local minima.
    # The "Active Set" local relaxation is more targeted.
    
    # Let's try a hybrid:
    # 1. Global force-directed relaxation for a few iterations to "shake up" the lattice.
    # 2. Then, local active-set polishing.
    
    # Phase 1: Force Directed Relaxation
    # Use a repulsive force that penalizes closeness.
    # We want to maximize the minimum distance.
    # This is equivalent to minimizing the sum of exp(-d_ij / sigma) or similar.
    # Let's use a simple iterative push:
    
    current_pts = best_pts.copy()
    
    # Pre-compute some constants
    # We will run this for a fixed number of iterations or until time runs out.
    
    # To speed up, we only calculate forces for neighbors within a cutoff.
    # Cutoff should be related to the current radius.
    
    for iteration in range(200):
        if time.time() > deadline - 20:
            break
            
        # Update Tree
        tree = cKDTree(current_pts)
        
        # Find close pairs
        # We want to find all pairs with distance < 2 * r_current * 1.5
        # Estimate r_current
        # We can sample a few points to estimate min dist
        # Or just use a fixed cutoff that decreases over time?
        # Let's use a cutoff of 0.2 (which is ~4x the target radius 0.05).
        cutoff = 0.25
        
        # Get pairs
        pairs = tree.query_pairs(cutoff)
        
        if not pairs:
            # No close pairs, maybe we can expand? 
            # But we are maximizing radius, so if no pairs are close, we are good locally.
            # But we need to check boundary.
            pass
        
        # Calculate forces
        forces = np.zeros_like(current_pts)
        
        # Boundary force: push inwards if too close to edge
        # d_bound = 1 - |p|
        # If d_bound < r_target, push in.
        # Let's use a force proportional to (r_target - d_bound) if negative, or just always push in if |p| > 0.8
        # Actually, boundary constraint is |p| <= 1 - r.
        # So if |p| > 1 - r, we must move in.
        # Let's estimate r from the current min pair distance / 2.
        # This is expensive. Let's just use a heuristic boundary force.
        
        # Force from pairs
        for i, j in pairs:
            d_vec = current_pts[i] - current_pts[j]
            d = np.linalg.norm(d_vec)
            if d < 1e-6:
                # Random nudge
                forces[i] += np.random.randn(2) * 0.01
                forces[j] -= np.random.randn(2) * 0.01
                continue
            
            # Repulsive force: stronger when closer
            # We want to increase d.
            # Force magnitude: 1/d^2 is good.
            f_mag = 1.0 / (d * d)
            f_vec = d_vec / d * f_mag
            
            forces[i] += f_vec
            forces[j] -= f_vec
        
        # Boundary Force
        # For each point, if it's close to the boundary, push it towards the center.
        # "Close" means |p| > 0.95 (approx)
        # Or more precisely, if |p| > 1 - r_est.
        # Let's just use a constant boundary repulsion for |p| > 0.9.
        r_pts = np.hypot(current_pts[:, 0], current_pts[:, 1])
        mask_bound = r_pts > 0.9
        if np.any(mask_bound):
            # Force towards center: -p / |p|^2
            # Scale by how close it is to boundary
            # dist_to_bound = 1 - r_pts
            # If dist_to_bound is small, force is large.
            # F = -p * (1 / (1 - r_pts)^2) ?
            # Let's use F = -p * (r_pts - 0.9) / (1 - r_pts + 1e-6)
            # This ensures it stays inside.
            weights = (r_pts[mask_bound] - 0.9) / (1 - r_pts[mask_bound] + 1e-6)
            forces[mask_bound] -= current_pts[mask_bound] * weights[:, np.newaxis]
        
        # Update positions
        # Learning rate: small to avoid instability
        lr = 0.01
        # Normalize forces?
        # Forces can be large if d is small.
        # Let's cap the force magnitude.
        f_mags = np.linalg.norm(forces, axis=1)
        max_f = 0.1
        mask_f = f_mags > max_f
        if np.any(mask_f):
            forces[mask_f] = forces[mask_f] / f_mags[mask_f, np.newaxis] * max_f
        
        current_pts += lr * forces
        
        # Enforce hard boundary constraint
        r_new = np.hypot(current_pts[:, 0], current_pts[:, 1])
        mask_out = r_new > 1.0
        if np.any(mask_out):
            current_pts[mask_out] = current_pts[mask_out] / r_new[mask_out, np.newaxis] * 0.9999
            
        # Check improvement
        # Computing full radius is expensive (O(N^2) or O(N log N)).
        # Let's check every 10 iterations.
        if iteration % 10 == 0:
            r_curr = compute_radius(current_pts)
            if r_curr > best_radius:
                best_radius = r_curr
                best_pts = current_pts.copy()
        
        # Check for stagnation
        # If we haven't improved in 50 iterations, maybe restart or change strategy?
        # For now, just continue.
        
    # Phase 2: Local Active Set Polish
    # Now that the global structure is relaxed, we can do fine-tuning.
    # For each point, find its k-nearest neighbors (k=10).
    # If the point is the "worst" in a pair (i.e., moving it increases the min distance of that pair), move it.
    
    # We will iterate through points and try to optimize each one locally.
    # For point i, we want to maximize min( min_j dist_ij, 1-|p_i| ).
    # This is a 2D optimization problem.
    # We can use a small local search (e.g., 100 random perturbations) for each point.
    # But N=300, so 300 * 100 = 30,000 evaluations.
    # Each evaluation requires checking distances to all other points? No, just neighbors.
    # With KDTree, checking 10 neighbors is fast.
    
    # Let's do a coordinate descent:
    # For i in range(N):
    #    Try to move point i to a better position.
    #    Candidate positions: 
    #       1. Current position
    #       2. Position that maximizes distance to nearest neighbor (on a small circle around current)
    #       3. Position that maximizes distance to boundary
    
    # Simpler: Random Perturbation with Acceptance
    # For each point, try 20 random small moves. Accept if global radius improves.
    # This is slow if we check global radius each time.
    # Instead, check LOCAL radius improvement.
    # If local radius improves, it *might* improve global, but could hurt others.
    # So we must check global or at least check against all neighbors.
    
    # Let's do a "Local Relaxation" step:
    # 1. Build Tree.
    # 2. For each point i:
    #    a. Find 10 nearest neighbors.
    #    b. Compute local_min = min( dist to neighbors, dist to boundary )
    #    c. If local_min < global_best_radius + epsilon:
    #       - Try to move i to increase local_min.
    #       - Use a simple gradient step on the local min.
    
    # Gradient of local min:
    # If the bottleneck is neighbor j: grad = (p_i - p_j) / dist_ij
    # If the bottleneck is boundary: grad = -p_i / |p_i|
    
    # Let's implement this.
    
    for iteration in range(100):
        if time.time() > deadline - 5:
            break
            
        tree = cKDTree(current_pts)
        
        improved = False
        for i in range(N):
            if time.time() > deadline - 5:
                break
                
            # Find neighbors
            # We need the closest one.
            d, idx = tree.query(current_pts[i], k=2)
            nearest_idx = idx[1]
            nearest_d = d[1]
            
            # Boundary distance
            r_i = np.hypot(current_pts[i, 0], current_pts[i, 1])
            bound_d = 1.0 - r_i
            
            # Determine bottleneck
            if nearest_d / 2.0 < bound_d:
                # Bottleneck is neighbor
                bottleneck_val = nearest_d / 2.0
                # Direction to move: away from nearest
                direction = current_pts[i] - current_pts[nearest_idx]
                direction /= (np.linalg.norm(direction) + 1e-9)
            else:
                # Bottleneck is boundary
                bottleneck_val = bound_d
                # Direction to move: towards center
                direction = -current_pts[i]
                direction /= (np.linalg.norm(direction) + 1e-9)
            
            # Only move if this point is "holding back" the global radius
            # If bottleneck_val < best_radius, we should try to improve it.
            # But moving i might hurt other points.
            # Let's try a small step and check if it improves the LOCAL min.
            # Then verify if it doesn't break other constraints too badly.
            
            # Step size
            step = 0.005
            
            # Try moving
            new_p = current_pts[i] + direction * step
            
            # Check if new_p is valid (inside boundary)
            new_r = np.hypot(new_p[0], new_p[1])
            if new_r > 1.0:
                # Project back
                new_p = new_p / new_r * 0.9999
            
            # Check local improvement
            # New distance to nearest
            new_d = np.linalg.norm(new_p - current_pts[nearest_idx])
            new_bound = 1.0 - np.hypot(new_p[0], new_p[1])
            new_local_min = min(new_d / 2.0, new_bound)
            
            # Also check distance to OTHER neighbors to ensure we don't create a new conflict
            # Query 5 nearest neighbors for new_p
            # This requires a temporary tree or just checking against existing neighbors.
            # Let's just check against the 10 nearest neighbors of the ORIGINAL position.
            # This is an approximation.
            
            d_list, idx_list = tree.query(current_pts[i], k=11)
            # idx_list[0] is i itself
            conflict = False
            for j in idx_list[1:]:
                if j == i: continue
                d_j = np.linalg.norm(new_p - current_pts[j])
                if d_j < 2.0 * best_radius * 0.95: # Allow slight violation? No, must be non-overlapping.
                    # If we violate the current best radius, it's bad.
                    # But we are trying to IMPROVE best_radius.
                    # So we must ensure d_j >= 2 * new_local_min? No.
                    # We must ensure d_j >= 2 * (some target).
                    # If d_j < 2 * bottleneck_val, we might be making it worse?
                    # Actually, if we move away from the bottleneck, we might get closer to another neighbor.
                    # So we must check ALL neighbors.
                    if d_j / 2.0 < new_local_min:
                         # We created a new tighter constraint
                         # This move is bad
                         conflict = True
                         break
            
            if not conflict and new_local_min > bottleneck_val:
                # Accept move
                current_pts[i] = new_p
                improved = True
        
        # Update best
        if improved or iteration % 10 == 0:
             r_curr = compute_radius(current_pts)
             if r_curr > best_radius:
                 best_radius = r_curr
                 best_pts = current_pts.copy()
        
        if not improved:
             # Stagnation?
             # Try a random kick to escape local minimum
             if iteration % 20 == 0:
                 idx = np.random.randint(N)
                 current_pts[idx] += np.random.randn(2) * 0.01
                 # Project to boundary if needed
                 r_k = np.hypot(current_pts[idx, 0], current_pts[idx, 1])
                 if r_k > 1.0:
                     current_pts[idx] = current_pts[idx] / r_k * 0.9999

    # Final check
    r_final = compute_radius(current_pts)
    if r_final > best_radius:
        best_radius = r_final
        best_pts = current_pts.copy()

    return best_pts
