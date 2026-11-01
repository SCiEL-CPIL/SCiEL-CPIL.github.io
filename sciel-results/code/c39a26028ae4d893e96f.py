"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit

    def compute_exact_score(pts):
        """Compute the exact packing radius for given centers."""
        if len(pts) != N:
            return -1.0
        if not np.all(np.isfinite(pts)):
            return -1.0
        
        # Radial distances
        dists = np.linalg.norm(pts, axis=1)
        if np.any(dists > 1.0):
            return -1.0
        
        # Pairwise distances
        i_idx, j_idx = np.triu_indices(N, k=1)
        diffs = pts[i_idx] - pts[j_idx]
        pair_dists = np.linalg.norm(diffs, axis=1)
        
        min_pair_dist = np.min(pair_dists)
        min_radial_dist = 1.0 - np.max(dists)
        
        r = min(min_pair_dist / 2.0, min_radial_dist)
        return r

    def generate_hexagonal_initialization():
        """Generate a hexagonal lattice clipped to the unit circle."""
        best_pts = None
        best_score = -1.0
        
        for scale in np.linspace(0.85, 1.15, 13):
            spacing = 2.0 / (np.sqrt(N) * scale)
            
            nx = int(2.0 / spacing) + 5
            ny = int(2.0 / (spacing * np.sqrt(3.0) / 2.0)) + 5
            
            x_coords = (np.arange(-nx//2, nx//2 + 1) + 0.5) * spacing
            y_coords = (np.arange(-ny//2, ny//2 + 1)) * spacing * np.sqrt(3.0) / 2.0
            
            pts_list = []
            for i, y in enumerate(y_coords):
                offset = 0.0 if i % 2 == 0 else spacing / 2.0
                for x in x_coords:
                    px = x + offset
                    py = y
                    if np.hypot(px, py) <= 1.0 - spacing / 2.0:
                        pts_list.append([px, py])
            
            if len(pts_list) < N:
                continue
                
            pts = np.array(pts_list)
            norms = np.linalg.norm(pts, axis=1)
            idx = np.argsort(norms)
            pts = pts[idx[:N]]
            
            score = compute_exact_score(pts)
            if score > best_score:
                best_score = score
                best_pts = pts
        
        if best_pts is None:
            best_pts = np.random.rand(N, 2) * 2 - 1
            norms = np.linalg.norm(best_pts, axis=1)
            best_pts = best_pts * (0.9 / norms)[:, None]
            best_score = compute_exact_score(best_pts)
            
        return best_pts, best_score

    def build_neighbor_graph(pts, target_dist, max_neighbors=12):
        """Build a neighbor graph for the spring network."""
        # Find pairs that should be at target_dist
        i_idx, j_idx = np.triu_indices(N, k=1)
        diffs = pts[i_idx] - pts[j_idx]
        dists = np.linalg.norm(diffs, axis=1)
        
        # Keep only pairs within 1.5 * target_dist
        mask = dists < 1.5 * target_dist
        i_idx = i_idx[mask]
        j_idx = j_idx[mask]
        
        # Limit to max_neighbors per point
        # For each point, keep only the closest neighbors
        neighbor_counts = np.zeros(N, dtype=int)
        for i in range(N):
            neighbor_counts[i] = 0
        
        # Sort by distance and keep closest
        order = np.argsort(dists[mask])
        kept_i = []
        kept_j = []
        kept_dists = []
        
        count_per_point = np.zeros(N, dtype=int)
        for idx in order:
            i = i_idx[idx]
            j = j_idx[idx]
            if count_per_point[i] < max_neighbors and count_per_point[j] < max_neighbors:
                kept_i.append(i)
                kept_j.append(j)
                kept_dists.append(dists[mask][idx])
                count_per_point[i] += 1
                count_per_point[j] += 1
        
        kept_i = np.array(kept_i, dtype=int)
        kept_j = np.array(kept_j, dtype=int)
        kept_dists = np.array(kept_dists)
        
        return kept_i, kept_j, kept_dists

    def spring_objective(pts_init, target_dist, spring_k, wall_k, deadline):
        """
        Create objective using spring network to maintain inter-circle distances.
        Minimize: -min_clearance + spring_penalty + wall_penalty
        """
        # Build neighbor graph from initial configuration
        kept_i, kept_j, kept_dists = build_neighbor_graph(pts_init, target_dist)
        
        # Precompute for speed
        n_pairs = len(kept_i)
        
        def objective(params):
            if time.time() > deadline:
                return 1e10
            
            pts = params.reshape(-1, 2)
            
            if not np.all(np.isfinite(pts)):
                return 1e10
            
            # Spring energy: penalize deviation from target_dist
            if n_pairs > 0:
                diffs = pts[kept_i] - pts[kept_j]
                current_dists = np.linalg.norm(diffs, axis=1)
                # Quadratic spring penalty
                spring_penalty = spring_k * np.sum((current_dists - target_dist) ** 2)
            else:
                spring_penalty = 0.0
            
            # Wall constraint: penalize circles too close to or outside boundary
            radial_dists = np.linalg.norm(pts, axis=1)
            wall_violation = np.maximum(0.0, radial_dists - 1.0 + target_dist / 2.0)
            wall_penalty = wall_k * np.sum(wall_violation ** 2)
            
            # Compute actual clearance
            if n_pairs > 0:
                min_pair_dist = np.min(current_dists)
            else:
                min_pair_dist = 2.0
            min_radial_dist = 1.0 - np.max(radial_dists)
            clearance = min(min_pair_dist / 2.0, min_radial_dist)
            
            return -clearance + spring_penalty + wall_penalty
        
        return objective

    def spring_gradient_descent(pts_init, target_dist, spring_k, wall_k, lr, n_iters, deadline):
        """
        Perform gradient descent on the spring network objective.
        """
        obj_func = spring_objective(pts_init, target_dist, spring_k, wall_k, deadline)
        
        pts = pts_init.copy()
        best_pts = pts.copy()
        best_score = compute_exact_score(pts)
        
        # Simple gradient descent with numerical gradients for first few steps,
        # then use finite differences for the full gradient
        for it in range(n_iters):
            if time.time() > deadline - 5:
                break
            
            # Compute objective
            f0 = obj_func(pts.flatten())
            
            if not np.isfinite(f0):
                break
            
            # Numerical gradient (central differences)
            eps = 1e-6
            grad = np.zeros(2 * N)
            
            for i in range(2 * N):
                if time.time() > deadline - 3:
                    break
                
                old_val = pts.flatten()[i]
                pts.flatten()[i] = old_val + eps
                f_plus = obj_func(pts.flatten())
                pts.flatten()[i] = old_val - eps
                f_minus = obj_func(pts.flatten())
                pts.flatten()[i] = old_val
                
                if np.isfinite(f_plus) and np.isfinite(f_minus):
                    grad[i] = (f_plus - f_minus) / (2 * eps)
                else:
                    grad[i] = 0.0
            
            # Update
            pts = pts - lr * grad
            
            # Clip to valid range
            pts = np.clip(pts, -1.0, 1.0)
            
            # Track best
            current_score = compute_exact_score(pts)
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
        
        return best_pts, best_score

    # 1. Generate initial points
    pts, best_score = generate_hexagonal_initialization()
    
    # Estimate target distance from initial configuration
    i_idx, j_idx = np.triu_indices(N, k=1)
    diffs = pts[i_idx] - pts[j_idx]
    dists = np.linalg.norm(diffs, axis=1)
    target_dist = np.median(dists)
    
    current_pts = pts
    current_score = best_score
    
    # 2. Spring network optimization with annealing
    spring_ks = [10.0, 50.0, 200.0, 1000.0, 5000.0]
    wall_ks = [100.0, 500.0, 2000.0]
    lrs = [0.01, 0.005, 0.002, 0.001, 0.0005]
    n_iters_list = [200, 300, 400, 500, 600]
    
    for stage in range(min(5, len(spring_ks))):
        if time.time() > deadline - 20:
            break
        
        sk = spring_ks[stage]
        wk = wall_ks[min(stage, len(wall_ks) - 1)]
        lr = lrs[min(stage, len(lrs) - 1)]
        n_iters = n_iters_list[min(stage, len(n_iters_list) - 1)]
        
        new_pts, new_score = spring_gradient_descent(
            current_pts, target_dist, sk, wk, lr, n_iters, deadline
        )
        
        if new_score > current_score:
            current_pts = new_pts
            current_score = new_score
    
    # 3. Final high-precision polish using soft-min approach
    if time.time() < deadline - 10:
        # Use a simple soft-min objective for final polish
        beta = 10000.0
        
        def soft_min_objective(params):
            if time.time() > deadline:
                return 1e10
            
            pts = params.reshape(-1, 2)
            
            if not np.all(np.isfinite(pts)):
                return 1e10
            
            dists = np.linalg.norm(pts, axis=1)
            if np.any(dists > 1.0):
                return 1e10 + np.sum(np.maximum(dists - 1.0, 0.0))
            
            i_idx, j_idx = np.triu_indices(N, k=1)
            diffs = pts[i_idx] - pts[j_idx]
            pair_dists = np.linalg.norm(diffs, axis=1)
            
            min_pair_dist = np.min(pair_dists)
            min_radial_dist = 1.0 - np.max(dists)
            
            x = min_pair_dist / 2.0
            y = min_radial_dist
            
            # Soft min
            m = min(x, y)
            ex = np.exp(-beta * (x - m))
            ey = np.exp(-beta * (y - m))
            val = m - 1.0 / beta * (np.log(ex + ey) - np.log(2.0))
            
            return -val
        
        # Use scipy minimize for the final polish
        from scipy.optimize import minimize
        
        try:
            res = minimize(soft_min_objective, current_pts.flatten(), method='L-BFGS-B',
                           bounds=[(-1.0, 1.0)] * (2 * N),
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 20})
            
            if res.success or res.fun < 0:
                new_pts = res.x.reshape(-1, 2)
                new_score = compute_exact_score(new_pts)
                
                if new_score > current_score:
                    current_pts = new_pts
                    current_score = new_score
        except Exception:
            pass

    # Final validation
    current_pts = current_pts.reshape(-1, 2)
    norms = np.linalg.norm(current_pts, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        current_pts[mask] *= (1.0 / norms[mask])[:, np.newaxis]
        
    return current_pts
# EVOLVE-BLOCK-END
