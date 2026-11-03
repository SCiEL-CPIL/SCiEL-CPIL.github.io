"""Circle packing in a unit square - 200 equal circles.

Strategy: Contact-constraint refinement.
- Start from a hexagonal grid initialization.
- Iteratively identify the "critical" pairs (those at or near the current minimum
  distance) and the critical walls.
- Use L-BFGS-B to maximize r by minimizing a penalty on constraint violations,
  where only active (near-contact) constraints are penalized.
- This focuses computational effort on the few constraints that actually limit r,
  enabling much tighter convergence than a global soft-min over all O(N^2) pairs.
"""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 200


def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # safety margin before 150s kill

    def compute_radius(c):
        """Return the largest feasible radius for centers c (N,2)."""
        wall = 0.5 - np.abs(c)
        min_wall = float(np.min(wall))
        diff = c[:, None, :] - c[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_pair = float(np.min(dist))
        return max(0.0, min(min_wall, min_pair * 0.5))

    def hex_grid(n_rows, n_cols, d_scale=1.0):
        """Generate hexagonal grid points."""
        d = d_scale / (n_cols + 0.5)
        row_spacing = d * math.sqrt(3) / 2
        total_height = (n_rows - 1) * row_spacing + d
        if total_height > 1.0 + 1e-6:
            return None
        pts = []
        for row in range(n_rows):
            y = -0.5 + d / 2 + row * row_spacing
            if row % 2 == 0:
                n_in_row = n_cols
                x_start = -0.5 + d / 2
            else:
                n_in_row = n_cols - 1
                x_start = -0.5 + d
            for col in range(n_in_row):
                x = x_start + col * d
                pts.append([x, y])
        pts = np.array(pts)
        if len(pts) >= N:
            return pts[:N]
        return None

    def square_grid(m):
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    # Generate diverse initial configurations
    inits = []
    for n_rows in range(13, 20):
        n_cols = math.ceil(N / n_rows)
        for d_scale in [0.88, 0.92, 0.96, 1.0, 1.04, 1.08]:
            pts = hex_grid(n_rows, n_cols, d_scale)
            if pts is not None:
                inits.append(pts)

    m = int(math.ceil(math.sqrt(N)))
    inits.append(square_grid(m))

    # Add random perturbations for diversity
    rng = np.random.default_rng(42)
    for base in inits[:3]:
        perturbed = base + rng.normal(0, 0.008, (N, 2))
        perturbed = np.clip(perturbed, -0.499, 0.499)
        inits.append(perturbed)

    # Score and select top initializations
    scored = []
    for pts in inits:
        r = compute_radius(pts)
        scored.append((r, pts))
    scored.sort(key=lambda x: -x[0])
    top_inits = [pts for _, pts in scored[:min(6, len(scored))]]

    if not top_inits:
        top_inits = [square_grid(m)]

    best_centers = None
    best_radius = 0.0

    def contact_refine(c_init, max_iter=30, time_budget=30.0):
        """
        Refine a configuration by iteratively optimizing with active-set constraints.
        
        The key idea: at each iteration, identify which constraints are "active"
        (i.e., at or very near the current minimum), and only penalize those.
        This is analogous to active-set methods in QP.
        """
        c = c_init.copy().ravel()
        r_current = compute_radius(c_init)
        
        # We'll use a penalty-based approach:
        # Minimize: -r + penalty * sum(max(0, 2r - d_ij)^2 for active pairs)
        #           + penalty * sum(max(0, r - wall_i)^2 for active walls)
        # But r is determined by the constraints, so we optimize centers only
        # and let r be implicitly the min distance.
        
        # Alternative: directly optimize a "max-min" objective using only active constraints.
        # At each step, find the critical set, then do a small L-BFGS-B run
        # that only considers those constraints.
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            elapsed_in_func = time.time() - (t0 + time_budget * 0)  # just check deadline
            
            c_2d = c.reshape(N, 2)
            r_now = compute_radius(c_2d)
            
            # Find active constraints: pairs with distance < r_now * 2.15
            # and walls with distance < r_now * 1.15
            threshold_pair = r_now * 2.15
            threshold_wall = r_now * 1.15
            
            diff = c_2d[:, None, :] - c_2d[None, :, :]
            dist = np.sqrt(np.sum(diff * diff, axis=2))
            np.fill_diagonal(dist, np.inf)
            
            wall = 0.5 - np.abs(c_2d)
            
            # Active pairs
            active_mask = dist < threshold_pair
            np.fill_diagonal(active_mask, False)
            rows, cols = np.where(active_mask)
            # Only upper triangle to avoid duplicates
            upper = rows < cols
            rows = rows[upper]
            cols = cols[upper]
            
            # Active walls: coordinate i of circle j is close to boundary
            # wall[j, k] = 0.5 - |c[j,k]|
            active_wall_mask = wall < threshold_wall
            wall_rows, wall_cols = np.where(active_wall_mask)
            
            n_active_pairs = len(rows)
            n_active_walls = len(wall_rows)
            
            # If no active constraints, we're at a local max of min-distance
            # (all distances are well above the minimum), so try a different approach
            if n_active_pairs == 0 and n_active_walls == 0:
                break
            
            # Define objective and gradient using only active constraints
            def active_obj(x):
                cx = x.reshape(N, 2)
                # Compute all distances
                d = cx[:, None, :] - cx[None, :, :]
                dist_m = np.sqrt(np.sum(d * d, axis=2))
                np.fill_diagonal(dist_m, np.inf)
                wall_m = 0.5 - np.abs(cx)
                
                # Soft min over active pairs and walls
                # Use a moderate beta for smoothness
                beta = 50.0
                
                pair_dists = dist_m[rows, cols]
                wall_dists = wall_m[wall_rows, wall_cols]
                
                all_d = np.concatenate([pair_dists, wall_dists])
                
                if len(all_d) == 0:
                    return 0.0
                
                d_max = np.max(all_d)
                exp_terms = np.exp(-beta * (all_d - d_max))
                log_sum = np.log(np.sum(exp_terms))
                soft_min_val = d_max - log_sum / beta
                return -soft_min_val
            
            def active_grad(x):
                cx = x.reshape(N, 2)
                d = cx[:, None, :] - cx[None, :, :]
                dist_m = np.sqrt(np.sum(d * d, axis=2))
                np.fill_diagonal(dist_m, np.inf)
                wall_m = 0.5 - np.abs(cx)
                
                beta = 50.0
                
                pair_dists = dist_m[rows, cols]
                wall_dists = wall_m[wall_rows, wall_cols]
                
                all_d = np.concatenate([pair_dists, wall_dists])
                
                if len(all_d) == 0:
                    return np.zeros(2 * N)
                
                d_max = np.max(all_d)
                exp_terms = np.exp(-beta * (all_d - d_max))
                sum_exp = np.sum(exp_terms)
                weights = exp_terms / sum_exp
                
                grad_c = np.zeros((N, 2))
                
                # Pair contributions
                w_pair = weights[:n_active_pairs]
                diff_ab = cx[rows] - cx[cols]
                safe_dist = np.where(pair_dists < 1e-12, 1.0, pair_dists)
                unit = diff_ab / safe_dist[:, None]
                contrib = beta * w_pair[:, None] * unit
                np.add.at(grad_c, rows, contrib)
                np.add.at(grad_c, cols, -contrib)
                
                # Wall contributions
                w_wall = weights[n_active_pairs:]
                # wall_dists[k] = wall_m[wall_rows[k], wall_cols[k]]
                # d(wall)/d(c[j, k]) = -sign(c[j, k])
                wall_grad_vals = -np.sign(cx[wall_rows, wall_cols])
                # Accumulate: for each active wall (j, k), grad_c[j, k] += beta * w * (-sign)
                # Use np.add.at for repeated indices
                grad_wall = np.zeros((N, 2))
                grad_wall[wall_rows, wall_cols] += beta * w_wall * wall_grad_vals
                grad_c += grad_wall
                
                return grad_c.ravel()
            
            # Run L-BFGS-B with active constraints
            try:
                result = minimize(
                    fun=active_obj,
                    x0=c,
                    jac=active_grad,
                    method='L-BFGS-B',
                    bounds=[(-0.4999, 0.4999)] * (2 * N),
                    options={
                        'maxiter': 80,
                        'ftol': 1e-13,
                        'gtol': 1e-9,
                        'maxls': 20
                    }
                )
                c = result.x
            except Exception:
                pass
            
            # Check if we made progress
            c_2d_new = c.reshape(N, 2)
            r_new = compute_radius(c_2d_new)
            if r_new <= r_now + 1e-12:
                # No progress, break
                break
            r_now = r_new
        
        return c.reshape(N, 2)

    # For each top initialization, run contact refinement
    for init_idx, init in enumerate(top_inits):
        if time.time() > deadline - 20:
            break
        
        r_init = compute_radius(init)
        if r_init > best_radius:
            best_radius = r_init
            best_centers = init.copy()
        
        # Phase 1: Contact refinement
        refined = contact_refine(init, max_iter=25, time_budget=25.0)
        r_refined = compute_radius(refined)
        if r_refined > best_radius:
            best_radius = r_refined
            best_centers = refined.copy()
        
        # Phase 2: A final global polish with a higher-beta soft-min
        # to escape any local minima from the active-set method
        if time.time() < deadline - 10:
            def global_soft_min(x, beta):
                c = x.reshape(N, 2)
                wall = 0.5 - np.abs(c)
                diff = c[:, None, :] - c[None, :, :]
                dist = np.sqrt(np.sum(diff * diff, axis=2))
                np.fill_diagonal(dist, np.inf)
                all_d = np.concatenate([wall.ravel(), dist[dist < np.inf].ravel()])
                d_max = np.max(all_d)
                exp_terms = np.exp(-beta * (all_d - d_max))
                log_sum = np.log(np.sum(exp_terms))
                soft_min_val = d_max - log_sum / beta
                return -soft_min_val
            
            def global_soft_min_grad(x, beta):
                c = x.reshape(N, 2)
                wall = 0.5 - np.abs(c)
                diff = c[:, None, :] - c[None, :, :]
                dist = np.sqrt(np.sum(diff * diff, axis=2))
                np.fill_diagonal(dist, np.inf)
                
                triu_idx = np.triu_indices(N, k=1)
                pair_dist_flat = dist[triu_idx]
                wall_flat = wall.ravel()
                all_d = np.concatenate([wall_flat, pair_dist_flat])
                
                d_max = np.max(all_d)
                exp_terms = np.exp(-beta * (all_d - d_max))
                sum_exp = np.sum(exp_terms)
                weights = exp_terms / sum_exp
                
                grad_c = np.zeros((N, 2))
                
                # Wall
                w_wall = weights[:2*N].reshape(N, 2)
                grad_c += beta * w_wall * (-np.sign(c))
                
                # Pairs
                rows, cols = triu_idx
                w_pair = weights[2*N:]
                diff_ab = c[rows] - c[cols]
                safe_dist = np.where(dist[rows, cols] < 1e-12, 1.0, dist[rows, cols])
                unit = diff_ab / safe_dist[:, None]
                contrib = beta * w_pair[:, None] * unit
                np.add.at(grad_c, rows, contrib)
                np.add.at(grad_c, cols, -contrib)
                
                return grad_c.ravel()
            
            try:
                result = minimize(
                    fun=global_soft_min,
                    x0=refined.ravel(),
                    jac=global_soft_min_grad,
                    args=(200.0,),
                    method='L-BFGS-B',
                    bounds=[(-0.4999, 0.4999)] * (2 * N),
                    options={
                        'maxiter': 150,
                        'ftol': 1e-14,
                        'gtol': 1e-10,
                        'maxls': 20
                    }
                )
                c_final = result.x.reshape(N, 2)
                r_final = compute_radius(c_final)
                if r_final > best_radius:
                    best_radius = r_final
                    best_centers = c_final.copy()
            except Exception:
                pass

    if best_centers is None:
        best_centers = top_inits[0]

    return best_centers


if __name__ == "__main__":
    result = construct()
    print(f"Shape: {result.shape}")
    print(f"Radius: {compute_radius(result)}")
