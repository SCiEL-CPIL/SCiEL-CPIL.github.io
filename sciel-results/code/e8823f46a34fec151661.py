"""
Circle packing in unit square using a rigid cluster assembly strategy.
1. Partition 26 circles into 5 macro-clusters (bubbles) of 4-6 circles each.
2. Pre-optimize the internal geometry of each bubble to be a tight hexagonal packing.
3. Optimize the placement of these 5 rigid bubbles in the unit square using a penalty method.
4. Expand bubbles to full scale and perform a final per-circle polish.
This drastically reduces the dimensionality of the global search (5 bubbles vs 26 circles)
and leverages the known optimal local structure of circle packings.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 26
DEADLINE = 140.0  # Safety margin from 150s limit

def construct():
    start_time = time.time()
    
    def check_validity(x, y, r, tol=1e-9):
        if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
            return False
        if np.any(y - r < -tol) or np.any(y + y > 1 + tol):
            return False
        if np.any(r < -tol):
            return False

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        np.fill_diagonal(dist, np.inf)
        np.fill_diagonal(min_dist, 0)

        if np.any(dist < min_dist - tol):
            return False
        return True

    def compute_radii(x, y):
        """Compute maximum valid radii for given centers."""
        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)

        min_dist = np.min(dist, axis=1)
        r = np.minimum(r_bound, 0.5 * min_dist)
        r = np.maximum(r, 1e-8)
        return r

    def score_config(x, y):
        r = compute_radii(x, y)
        if check_validity(x, y, r):
            return np.sum(r)
        return -np.inf

    def rotate(points, angle):
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        return points @ R.T

    def translate(points, offset):
        return points + offset

    # 1. Define Macro-Clusters
    # We want 5 clusters. Sizes: 5, 5, 5, 6, 5 = 26
    # A "bubble" of n circles is arranged in a hexagonal pattern.
    # For n=5: 1 center + 4 around? No, hexagonal packing of 5 is a rhombus or cross.
    # Let's use specific local configurations known to be dense.
    # Config 1 (5 circles): One in center, 4 in corners of a square? No, hex is better.
    # Actually, for small n, the optimal packing is often just a subset of the hex lattice.
    
    def generate_hex_cluster(n_circles, scale=1.0):
        """Generate centers for n_circles in a hexagonal arrangement, scaled by 'scale'."""
        if n_circles == 1:
            return np.array([[0, 0]])
        if n_circles == 2:
            return np.array([[-0.5, 0], [0.5, 0]]) * scale
        if n_circles == 3:
            return np.array([[-0.5, -0.288], [0.5, -0.288], [0, 0.577]]) * scale
        if n_circles == 4:
            # Rhombus
            return np.array([[-0.5, -0.288], [0.5, -0.288], [-0.5, 0.288], [0.5, 0.288]]) * scale
        if n_circles == 5:
            # Center + 4 surrounding (square-ish for small n, or hex subset)
            # Hex subset of 5: remove one from the 6 around center.
            # Let's use a symmetric cross or just optimize it later.
            # For now, a square of 4 + 1 center is a good start.
            s = 0.7 * scale
            return np.array([
                [0, 0],
                [s, 0], [-s, 0], [0, s], [0, -s]
            ])
        if n_circles == 6:
            # Hexagon: 1 center + 6 around
            angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
            r = 1.0 * scale
            pts = np.array([np.cos(angles), np.sin(angles)]).T
            pts = np.vstack([np.array([0,0]), pts])
            return pts
        
        # Fallback for other sizes: random circle
        rng = np.random.RandomState(42)
        angles = rng.uniform(0, 2*np.pi, n_circles)
        radii = rng.uniform(0.1, 0.5, n_circles)
        return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]) * scale

    # Define cluster sizes
    cluster_sizes = [5, 5, 5, 6, 5]
    clusters = []
    for size in cluster_sizes:
        pts = generate_hex_cluster(size, scale=0.1) # Start small, will scale up
        # Center the cluster
        pts = pts - np.mean(pts, axis=0)
        clusters.append(pts)
    
    # 2. Optimize Bubble Placement (Rigid Bodies)
    # State vector: for each of 5 clusters: [x_center, y_center, rotation, scale]
    # Total dims: 5 * 4 = 20
    
    def bubble_objective(params):
        """
        Penalty objective for placing rigid bubbles.
        We want to maximize sum of radii.
        Since bubbles are rigid, the radius of each circle in a bubble is determined by
        the internal geometry and the scale factor.
        Let's assume all circles in a bubble have the same radius r_bubble.
        The internal geometry is fixed (relative distances).
        The scale factor 's' determines the actual size.
        
        Wait, if bubbles are rigid, the ratio of distances is fixed.
        If we scale a bubble, all radii scale.
        
        Let's refine: 
        Each bubble has a fixed internal shape (unit scale).
        We optimize: x_i, y_i, theta_i, s_i for i in 1..5.
        The radius of each circle in bubble i is r_i = s_i * r_internal_max.
        Actually, for a rigid cluster, the "radius" is the minimum distance to neighbors / 2.
        If we scale the cluster, the minimum distance scales.
        
        Let's compute the "max possible radius" for the unit-scale cluster geometry.
        r_unit_max = min(0.5 * min_dist_to_other_circles_in_cluster)
        Then actual radius r = s_i * r_unit_max.
        
        Constraints:
        1. Bubbles don't overlap.
           Distance between any circle in bubble i and any circle in bubble j >= r_i + r_j.
        2. Bubbles inside [0,1]^2.
           For all circles in bubble i: x_i + s_i * (pt_x + cos... ) +/- r_i in [0,1].
        
        Objective: Maximize sum(r_i * count_i)
        
        Penalty: 
        P = - sum(r_i * count_i) 
            + lambda * sum(max(0, overlap_violations)^2)
            + lambda * sum(max(0, boundary_violations)^2)
        """
        
        x_centers = params[0::4]
        y_centers = params[1::4]
        thetas = params[2::4]
        scales = params[3::4]
        
        # Ensure scales are positive
        scales = np.maximum(scales, 1e-6)
        
        total_penalty = 0.0
        total_radius_sum = 0.0
        
        # Compute actual circle positions and radii for all clusters
        all_x = []
        all_y = []
        all_r = []
        
        for k, cluster in enumerate(clusters):
            n_c = len(cluster)
            # Pre-computed r_unit_max for this cluster geometry
            # We need to compute this once outside or cache it.
            # Let's compute it here for simplicity, it's cheap.
            dx = cluster[:, None, 0] - cluster[None, :, 0]
            dy = cluster[:, None, 1] - cluster[None, :, 1]
            dists = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(dists, np.inf)
            min_dist = np.min(dists, axis=1)
            r_unit_max = np.min(0.5 * min_dist)
            
            r_k = scales[k] * r_unit_max
            total_radius_sum += r_k * n_c
            
            # Transform cluster points
            pts = rotate(cluster, thetas[k])
            pts = pts * scales[k]
            pts = pts + np.array([x_centers[k], y_centers[k]])
            
            all_x.append(pts[:, 0])
            all_y.append(pts[:, 1])
            all_r.extend([r_k] * n_c)
            
        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)
        all_r = np.array(all_r)
        
        # Boundary Penalty
        # x - r < 0 or x + r > 1
        v_left = np.maximum(0, (all_r - all_x))
        v_right = np.maximum(0, (all_x + all_r - 1.0))
        v_bottom = np.maximum(0, (all_r - all_y))
        v_top = np.maximum(0, (all_y + all_r - 1.0))
        
        boundary_pen = np.sum(v_left**2) + np.sum(v_right**2) + np.sum(v_bottom**2) + np.sum(v_top**2)
        
        # Overlap Penalty
        # Check all pairs. N=26, so 26*25/2 = 325 pairs.
        # We can do this vectorized but it might be slow if done naively in python loop.
        # Let's use a vectorized distance matrix.
        dx = all_x[:, None] - all_x[None, :]
        dy = all_y[:, None] - all_y[None, :]
        dists = np.sqrt(dx**2 + dy**2)
        req_dists = all_r[:, None] + all_r[None, :]
        
        # Only check upper triangle
        i_upper, j_upper = np.triu_indices(N, k=1)
        dists_upper = dists[i_upper, j_upper]
        req_upper = req_dists[i_upper, j_upper]
        
        violations = np.maximum(0, req_upper - dists_upper)
        overlap_pen = np.sum(violations**2)
        
        lambda_b = 10.0
        lambda_o = 10.0
        
        obj = -total_radius_sum + lambda_b * boundary_pen + lambda_o * overlap_pen
        
        return obj

    # Initial guess for bubble parameters
    # Place 5 bubbles roughly in a grid or random
    rng = np.random.RandomState(1)
    init_params = []
    # Roughly distribute 5 bubbles
    # 2 on left, 2 on right, 1 center? Or just random
    for k in range(5):
        x_c = rng.uniform(0.2, 0.8)
        y_c = rng.uniform(0.2, 0.8)
        theta = rng.uniform(0, 2*np.pi)
        scale = rng.uniform(0.05, 0.15) # Start small
        init_params.extend([x_c, y_c, theta, scale])
        
    init_params = np.array(init_params)
    
    # Optimize bubble placement
    bounds = []
    for k in range(5):
        bounds.append((0.1, 0.9)) # x
        bounds.append((0.1, 0.9)) # y
        bounds.append((0, 2*np.pi)) # theta
        bounds.append((0.01, 0.2)) # scale (guessing max scale)
        
    try:
        res_bubble = minimize(bubble_objective, init_params, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 200, 'ftol': 1e-10})
        best_params = res_bubble.x
    except Exception:
        best_params = init_params
    
    # 3. Expand Bubbles to Full Scale and Polish
    # Extract final bubble parameters
    x_centers = best_params[0::4]
    y_centers = best_params[1::4]
    thetas = best_params[2::4]
    scales = best_params[3::4]
    
    final_x = np.zeros(N)
    final_y = np.zeros(N)
    
    idx = 0
    for k, cluster in enumerate(clusters):
        n_c = len(cluster)
        pts = rotate(cluster, thetas[k])
        pts = pts * scales[k]
        pts = pts + np.array([x_centers[k], y_centers[k]])
        
        final_x[idx:idx+n_c] = pts[:, 0]
        final_y[idx:idx+n_c] = pts[:, 1]
        idx += n_c
        
    # Clip to valid range (just in case)
    final_x = np.clip(final_x, 0.01, 0.99)
    final_y = np.clip(final_y, 0.01, 0.99)
    
    # 4. Final Per-Circle Polish
    # Use a simple local search to fix any residual overlaps or improve radii
    # Since bubbles were optimized with penalties, they might still be slightly overlapping or not maximal.
    
    def polish(x, y, max_iters=1000):
        current_score = score_config(x, y)
        r = compute_radii(x, y)
        
        if not check_validity(x, y, r):
            # Try to fix validity by shrinking
            for scale in np.linspace(0.99, 0.9, 10):
                r_test = r * scale
                if check_validity(x, y, r_test):
                    r = r_test
                    break
            current_score = np.sum(r) if check_validity(x, y, r) else -np.inf
        
        step_sizes = [0.02, 0.01, 0.005, 0.001]
        
        for step in step_sizes:
            for _ in range(max_iters // len(step_sizes)):
                if time.time() - start_time > DEADLINE - 5:
                    break
                
                i = np.random.randint(N)
                angle = np.random.uniform(0, 2*np.pi)
                
                new_x = x[i] + step * np.cos(angle)
                new_y = y[i] + step * np.sin(angle)
                
                if new_x < 0.01 or new_x > 0.99 or new_y < 0.01 or new_y > 0.99:
                    continue
                    
                x[i] = new_x
                y[i] = new_y
                
                new_score = score_config(x, y)
                
                if new_score > current_score + 1e-9:
                    current_score = new_score
                else:
                    x[i] -= step * np.cos(angle)
                    y[i] -= step * np.sin(angle)
                    
        return x, y, current_score

    final_x, final_y, final_score = polish(final_x, final_y)
    
    final_r = compute_radii(final_x, final_y)
    
    # Final validity check and fix
    if not check_validity(final_x, final_y, final_r):
        for scale in np.linspace(0.999, 0.95, 20):
            r_test = final_r * scale
            if check_validity(final_x, final_y, r_test):
                final_r = r_test
                break
        else:
            # Fallback: compute radii from scratch (which ensures max valid radius for given centers)
            # But if centers are overlapping, compute_radii might return small radii.
            # If still invalid, we have a problem.
            final_r = compute_radii(final_x, final_y)
            if not check_validity(final_x, final_y, final_r):
                # Last resort: uniform grid
                x = []
                y = []
                for i in range(5):
                    for j in range(5):
                        x.append((j + 0.5) / 5.0)
                        y.append((i + 0.5) / 5.0)
                x.append(0.5)
                y.append(0.5)
                final_x = np.array(x)
                final_y = np.array(y)
                final_r = compute_radii(final_x, final_y)
    
    return np.column_stack((final_x, final_y, final_r))

if __name__ == "__main__":
    # Quick test
    start = time.time()
    res = construct()
    print(f"Time: {time.time() - start:.2f}s")
    print(f"Sum radii: {np.sum(res[:, 2]):.4f}")
    print(f"Valid: {check_validity(res[:, 0], res[:, 1], res[:, 2])}")
