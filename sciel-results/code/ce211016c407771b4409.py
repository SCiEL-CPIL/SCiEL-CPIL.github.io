"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Place 100 equal non-overlapping spheres in the unit cube [-0.5, 0.5]^3
    to maximize the common radius.
    
    Strategy:
    1. Jittered grid initialization.
    2. Short gradient ascent phase to break symmetry.
    3. Local search on critical points with Simulated Annealing.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # --- Initialization: Jittered 5x5x4 grid ---
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.4, 0.4, nz)
    
    base_points = np.array([[x, y, z] for z in zs for y in ys for x in xs])
    
    np.random.seed(42)
    current_pts = base_points + np.random.uniform(-0.02, 0.02, (N, 3))
    current_pts = np.clip(current_pts, -0.5, 0.5)
    
    def compute_radius_and_crit(pts):
        """
        Compute radius and identify critical points.
        Returns: radius, critical_indices
        """
        n = len(pts)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        
        boundary_dists = 0.5 - np.abs(pts)
        min_boundary_per_point = np.min(boundary_dists, axis=1)
        
        iu = np.triu_indices(n, k=1)
        min_pair_dist = np.min(dists[iu])
        
        radius = min(min_pair_dist / 2.0, np.min(min_boundary_per_point))
        
        crit_mask = np.zeros(n, dtype=bool)
        tol = 1e-9
        
        min_pair_val = min_pair_dist / 2.0
        if min_pair_val <= radius + tol:
            pair_dists = dists[iu]
            close_pairs = np.where(np.abs(pair_dists - 2.0 * radius) < 1e-8)[0]
            if len(close_pairs) > 0:
                i_idx, j_idx = iu
                for k in close_pairs:
                    crit_mask[i_idx[k]] = True
                    crit_mask[j_idx[k]] = True
        
        min_boundary_val = np.min(min_boundary_per_point)
        if min_boundary_val <= radius + tol:
            close_bounds = np.where(np.abs(min_boundary_per_point - radius) < 1e-8)[0]
            crit_mask[close_bounds] = True
            
        return radius, np.where(crit_mask)[0]

    current_radius, critical_idx = compute_radius_and_crit(current_pts)
    best_radius = current_radius
    best_pts = current_pts.copy()
    
    def soft_min_grad(pts, alpha):
        n = len(pts)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        iu = np.triu_indices(n, k=1)
        
        pair_dists = dists[iu] / 2.0
        pair_diffs = diff[iu]
        
        boundary_terms = 0.5 - np.abs(pts)
        boundary_flat = boundary_terms.flatten()
        
        all_terms = np.concatenate([pair_dists, boundary_flat])
        
        neg_alpha_x = -alpha * all_terms
        max_val = np.max(neg_alpha_x)
        exp_vals = np.exp(neg_alpha_x - max_val)
        weights = exp_vals / np.sum(exp_vals)
        
        grad = np.zeros_like(pts)
        
        pair_weights = weights[:len(pair_dists)]
        safe_dists = np.maximum(dists[iu], 1e-10)
        unit_vecs = pair_diffs / safe_dists[:, np.newaxis]
        
        i_idx, j_idx = iu
        np.add.at(grad, i_idx, pair_weights[:, np.newaxis] * unit_vecs)
        np.add.at(grad, j_idx, -pair_weights[:, np.newaxis] * unit_vecs)
        
        boundary_weights = weights[len(pair_dists):]
        bw = boundary_weights.reshape(n, 3)
        signs = np.sign(pts)
        grad -= bw * signs
        
        return grad

    # Phase 1: Gradient Ascent for 10s
    grad_end_time = start_time + 10.0
    T = 0.01
    lr = 0.05
    alpha = 50.0
    
    while time.time() < grad_end_time:
        grad = soft_min_grad(current_pts, alpha)
        new_pts = current_pts + lr * grad
        new_pts = np.clip(new_pts, -0.5 + 1e-9, 0.5 - 1e-9)
        
        new_radius, _ = compute_radius_and_crit(new_pts)
        if new_radius > current_radius:
            current_radius = new_radius
            current_pts = new_pts
            lr *= 0.99
        else:
            lr *= 0.95
            if lr < 0.001:
                lr = 0.01
        if current_radius > best_radius:
            best_radius = current_radius
            best_pts = current_pts.copy()
            
    # Phase 2: Local Search
    current_radius, critical_idx = compute_radius_and_crit(current_pts)
    T = 0.005
    
    while time.time() < deadline - 2.0:
        if len(critical_idx) == 0:
            break
            
        i = critical_idx[np.random.randint(len(critical_idx))]
        
        move_size = 0.01 * np.sqrt(T)
        delta = np.random.normal(0, move_size, 3)
        new_pt = current_pts[i] + delta
        new_pt = np.clip(new_pt, -0.5 + 1e-9, 0.5 - 1e-9)
        
        new_bound = 0.5 - np.abs(new_pt).min()
        
        dists_to_others = np.sqrt(np.sum((current_pts - new_pt)**2, axis=1))
        dists_to_others[i] = np.inf
        new_min_pair = np.min(dists_to_others) / 2.0
        
        new_local_radius = min(new_bound, new_min_pair)
        
        if new_local_radius >= current_radius:
            current_pts[i] = new_pt
            current_radius, critical_idx = compute_radius_and_crit(current_pts)
            
            if current_radius > best_radius:
                best_radius = current_radius
                best_pts = current_pts.copy()
                
            T *= 0.999
        else:
            delta_r = new_local_radius - current_radius
            if np.random.random() < np.exp(delta_r / T):
                current_pts[i] = new_pt
                current_radius, critical_idx = compute_radius_and_crit(current_pts)
                T *= 0.98
            else:
                T *= 0.99
                if np.random.random() < 0.01:
                    j = np.random.randint(N)
                    current_pts[j] = np.random.uniform(-0.4, 0.4, 3)
                    current_radius, critical_idx = compute_radius_and_crit(current_pts)
                    T = 0.01
                    if current_radius > best_radius:
                        best_radius = current_radius
                        best_pts = current_pts.copy()

    return best_pts
# EVOLVE-BLOCK-END
