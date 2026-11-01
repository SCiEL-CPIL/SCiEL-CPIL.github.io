import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def compute_radius(c):
        if not np.all(np.isfinite(c)):
            return 0.0
        if np.any(c > 0.5) or np.any(c < -0.5):
            return 0.0
        wall_dists = 0.5 - np.abs(c)
        r_wall = np.min(wall_dists)
        try:
            tree = cKDTree(c)
            dists, _ = tree.query(c, k=2)
            r_pair = np.min(dists[:, 1]) / 2.0
        except Exception:
            r_pair = 0.0
        return min(r_wall, r_pair)

    def penalty_objective_and_grad(x_flat, r_target, mu):
        """
        Penalty method objective: minimize sum of violation^2 + mu * (r_target - r_actual)^2
        Actually, we minimize: sum_{i<j} max(0, 2*r_target - d_ij)^2 + sum_i max(0, r_target - wall_i)^2
        plus a term to encourage larger r.
        
        Better: minimize sum of (max(0, 2*r_target - d_ij))^2 + (max(0, r_target - wall_dist))^2
        This is the penalty for violating the constraint that all spheres of radius r_target fit.
        If the minimum is 0, then r_target is feasible.
        """
        c = x_flat.reshape(N, 3)
        
        # Wall violations
        wall_dists = 0.5 - np.abs(c)
        wall_viol = np.maximum(0, r_target - wall_dists)
        
        # Pairwise violations
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        dists = np.sqrt(dists_sq)
        pair_viol = np.maximum(0, 2 * r_target - dists)
        
        iu = np.triu_indices(N, k=1)
        pv = pair_viol[iu]
        
        obj = np.sum(wall_viol**2) + np.sum(pv**2)
        
        # Gradient
        grad = np.zeros_like(c)
        
        # Wall gradient
        wall_mask = wall_viol > 0
        for k in range(3):
            mask = wall_viol[:, k] > 0
            if np.any(mask):
                # d/dx max(0, r - (0.5 - |x|)) = sign(x) * 2 * (r - (0.5 - |x|)) if violated
                # d(0.5 - |x|)/dx = -sign(x)
                # d max(0, r - (0.5-|x|))/dx = -d(0.5-|x|)/dx = sign(x)
                # So d obj/dx = 2 * viol * sign(x)
                grad[mask, k] += 2 * wall_viol[mask, k] * np.sign(c[mask, k])
        
        # Pair gradient
        for idx in range(len(iu[0])):
            i, j = iu[0][idx], iu[1][idx]
            if pv[idx] > 0:
                d = dists[i, j]
                if d > 1e-12:
                    direction = (c[i] - c[j]) / d
                    # d/dc_i max(0, 2r - d_ij) = -direction * 2 * (2r - d_ij) ... wait
                    # f = max(0, 2r - d)^2, d f/d c_i = 2 * (2r - d) * d(2r-d)/dc_i
                    # d(2r - d)/dc_i = -d d/dc_i = -(c_i - c_j)/d
                    # So d f/d c_i = 2 * (2r - d) * (-(c_i - c_j)/d) = -2 * pv * direction
                    # And obj has sum of f, so grad[i] += -2 * pv * direction
                    # But wait, we want to minimize, and the gradient of the penalty should push spheres apart.
                    # If spheres are too close (viol > 0), we want to increase d.
                    # d/dc_i (2r - d) = -direction, so gradient of (2r-d)^2 wrt c_i is 2(2r-d)(-direction) = -2*viol*direction
                    # This means we should move c_i in the direction of +direction (away from j) to reduce the penalty?
                    # minimize: x_{k+1} = x_k - alpha * grad
                    # grad_i = -2 * viol * (c_i - c_j)/d
                    # -alpha * grad_i = +2 * alpha * viol * (c_i - c_j)/d
                    # So c_i moves away from c_j. Correct.
                    grad[i] += -2 * pv[idx] * direction
                    grad[j] += 2 * pv[idx] * direction
        
        return obj, grad.ravel()

    def penalty_refine(centers, r_target, time_limit, mu=1.0, max_iter=300):
        """Refine centers to make radius r_target feasible using penalty method."""
        centers = centers.copy()
        x0 = centers.ravel()
        
        def objective_and_grad(x):
            if time.time() > time_limit:
                return 1e10, np.zeros_like(x)
            return penalty_objective_and_grad(x, r_target, mu)
        
        try:
            result = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                bounds=[(-0.499999, 0.499999)] * (N * 3),
                options={'maxiter': max_iter, 'maxfun': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            new_centers = result.x.reshape(N, 3)
            new_centers = np.clip(new_centers, -0.499999, 0.499999)
            new_r = compute_radius(new_centers)
            if new_r >= r_target * 0.9999:
                return new_centers, new_r, True
        except Exception:
            pass
        
        return centers, compute_radius(centers), False

    def binary_search_radius(centers, r_low, r_high, time_limit, max_iters=20):
        """Binary search for the maximum feasible radius."""
        best_centers = centers.copy()
        best_r = compute_radius(centers)
        
        for _ in range(max_iters):
            if time.time() > time_limit - 5:
                break
            
            r_mid = (r_low + r_high) / 2.0
            if r_mid <= best_r:
                r_low = r_mid
                continue
            
            it_limit = time.time() + min(5.0, (time_limit - time.time()) / 3.0)
            new_centers, new_r, feasible = penalty_refine(centers, r_mid, it_limit)
            
            if feasible and new_r >= r_mid * 0.9999:
                best_r = new_r
                best_centers = new_centers
                r_low = r_mid
            else:
                r_high = r_mid
        
        return best_centers, best_r

    def force_directed_refine(centers, time_limit, step_scale=1.0):
        """Force-directed local search."""
        centers = centers.copy()
        r = compute_radius(centers)
        best_centers = centers.copy()
        best_r = r
        
        base_step = 0.05 * step_scale
        decay_rate = 0.999
        stagnation_count = 0
        it = 0
        
        while time.time() < time_limit:
            it += 1
            wall_dists = 0.5 - np.abs(centers)
            r_wall = np.min(wall_dists)
            
            try:
                tree = cKDTree(centers)
                dists, _ = tree.query(centers, k=2)
                min_pair_dist = np.min(dists[:, 1])
                r_pair = min_pair_dist / 2.0
            except Exception:
                break
                
            current_r = min(r_wall, r_pair)
            
            if current_r <= best_r * 1.000001:
                stagnation_count += 1
                if stagnation_count > 100:
                    break
            else:
                stagnation_count = 0
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
            
            force = np.zeros_like(centers)
            target_r = current_r + 0.002
            
            for k in range(3):
                d_wall = 0.5 - np.abs(centers[:, k])
                mask = d_wall < target_r
                if np.any(mask):
                    viol = target_r - d_wall[mask]
                    dirs = -np.sign(centers[mask, k])
                    dirs[dirs == 0] = 1.0
                    force[mask, k] += viol * dirs
            
            target_dist = 2 * target_r
            try:
                pairs = tree.query_pairs(target_dist)
            except Exception:
                pairs = []
            
            for i, j in pairs:
                diff = centers[i] - centers[j]
                d = np.linalg.norm(diff)
                if d > 1e-10 and d < target_dist:
                    f_mag = (target_dist - d) / d
                    force[i] += f_mag * diff
                    force[j] -= f_mag * diff
            
            force_norm = np.linalg.norm(force)
            if force_norm < 1e-12:
                break
            force = force / force_norm
            
            step = base_step * (decay_rate ** (stagnation_count / 10.0))
            new_centers = centers + step * force
            new_centers = np.clip(new_centers, -0.499999, 0.499999)
            new_r = compute_radius(new_centers)
            
            if new_r > r:
                r = new_r
                centers = new_centers
            else:
                step *= 0.5
                new_centers = centers + step * force
                new_centers = np.clip(new_centers, -0.499999, 0.499999)
                new_r = compute_radius(new_centers)
                if new_r > r:
                    r = new_r
                    centers = new_centers
                else:
                    centers += np.random.normal(0, 0.002, centers.shape)
                    centers = np.clip(centers, -0.499999, 0.499999)
                    r = compute_radius(centers)
                    if r > best_r:
                        best_r = r
                        best_centers = centers.copy()
        
        return best_centers, best_r

    def anneal_and_refine(centers, time_limit, initial_temp=0.04, cooling=0.998):
        """Simulated Annealing with periodic force-directed refinement."""
        centers = centers.copy()
        current_r = compute_radius(centers)
        best_centers = centers.copy()
        best_r = current_r
        temp = initial_temp
        it = 0
        
        while time.time() < time_limit and temp > 1e-5:
            it += 1
            n_move = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
            indices = np.random.choice(N, n_move, replace=False)
            step_size = temp * 0.4
            new_centers = centers.copy()
            new_centers[indices] += np.random.normal(0, step_size, (n_move, 3))
            new_centers = np.clip(new_centers, -0.499999, 0.499999)
            
            new_r = compute_radius(new_centers)
            
            delta = new_r - current_r
            if delta > 0 or np.random.random() < np.exp(delta / (temp + 1e-10)):
                current_r = new_r
                centers = new_centers
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
            
            temp *= cooling
            
            if it % 20 == 0 and time.time() < time_limit - 1:
                refine_limit = min(time.time() + 0.3, time_limit)
                refined_centers, refined_r = force_directed_refine(centers, refine_limit, step_scale=0.3)
                if refined_r > current_r:
                    current_r = refined_r
                    centers = refined_centers
                    if current_r > best_r:
                        best_r = current_r
                        best_centers = centers.copy()
        
        return best_centers, best_r

    # --- Main Execution ---
    np.random.seed(42)
    
    best_centers_global = None
    best_radius_global = 0.0
    
    # Generate initial configurations
    initializations = []
    
    # 5x5x4 grid
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.3, 0.3, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    grid_centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    for jitter in [0.0, 0.01, 0.03]:
        centers = grid_centers + np.random.normal(0, jitter, grid_centers.shape)
        centers = np.clip(centers, -0.49, 0.49)
        initializations.append(centers)
    
    # FCC-like lattice
    a = 0.33
    fcc_points = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                fcc_points.append([i*a, j*a, k*a])
                fcc_points.append([i*a + a/2, j*a + a/2, k*a])
                fcc_points.append([i*a + a/2, j*a, k*a + a/2])
                fcc_points.append([i*a, j*a + a/2, k*a + a/2])
    
    fcc_array = np.array(fcc_points)
    mask = np.all(np.abs(fcc_array) <= 0.49, axis=1)
    fcc_array = fcc_array[mask]
    
    if len(fcc_array) >= N:
        idx = np.random.choice(len(fcc_array), N, replace=False)
        fcc_centers = fcc_array[idx]
        for jitter in [0.0, 0.02]:
            centers = fcc_centers + np.random.normal(0, jitter, fcc_centers.shape)
            centers = np.clip(centers, -0.49, 0.49)
            initializations.append(centers)
    
    # Random
    for _ in range(3):
        centers = np.random.uniform(-0.45, 0.45, (N, 3))
        initializations.append(centers)

    # Optimization loop
    for idx, init_centers in enumerate(initializations):
        if time.time() > deadline - 25:
            break
        
        remaining = deadline - 25 - time.time()
        if remaining < 5:
            break
        
        r_init = compute_radius(init_centers)
        
        # Short SA phase
        anneal_time = min(remaining * 0.3, 10.0)
        anneal_limit = time.time() + anneal_time
        sa_centers, sa_r = anneal_and_refine(init_centers, anneal_limit, initial_temp=0.05, cooling=0.997)
        
        if sa_r > best_radius_global:
            best_radius_global = sa_r
            best_centers_global = sa_centers
        
        # Force-directed refinement
        if time.time() < deadline - 20:
            refine_limit = min(time.time() + 4.0, deadline - 20)
            fd_centers, fd_r = force_directed_refine(sa_centers, refine_limit, step_scale=0.8)
            if fd_r > best_radius_global:
                best_radius_global = fd_r
                best_centers_global = fd_centers
        
        # Penalty-based binary search for radius
        if time.time() < deadline - 15 and best_centers_global is not None:
            bs_limit = deadline - 15
            bs_centers, bs_r = binary_search_radius(best_centers_global, best_radius_global, 0.15, bs_limit, max_iters=10)
            if bs_r > best_radius_global:
                best_radius_global = bs_r
                best_centers_global = bs_centers

    # Final intensive polish
    if best_centers_global is not None and time.time() < deadline - 5:
        # Force-directed pass
        if time.time() < deadline - 3:
            fd_centers, fd_r = force_directed_refine(best_centers_global, deadline - 3, step_scale=0.3)
            if fd_r > best_radius_global:
                best_radius_global = fd_r
                best_centers_global = fd_centers
        
        # Final binary search
        if time.time() < deadline - 1:
            bs_centers, bs_r = binary_search_radius(best_centers_global, best_radius_global, 0.15, deadline - 1, max_iters=5)
            if bs_r > best_radius_global:
                best_radius_global = bs_r
                best_centers_global = bs_centers

    if best_centers_global is None:
        return grid_centers

    best_centers_global = np.clip(best_centers_global, -0.5, 0.5)
    return best_centers_global

if __name__ == "__main__":
    start = time.time()
    result = construct()
    end = time.time()
    r = compute_radius(result)
    print(f"Radius: {r}, Time: {end-start}s")
