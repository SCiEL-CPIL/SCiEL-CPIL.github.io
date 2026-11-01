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

    def penalty_objective_and_grad(x_flat, r_target):
        """
        Penalty objective for making radius r_target feasible.
        Minimizes sum of squared violations of constraints:
        1. ||c_i - c_j|| >= 2*r_target
        2. wall_dist_i >= r_target
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
        for k in range(3):
            mask = wall_viol[:, k] > 0
            if np.any(mask):
                # d/dc_i max(0, r - (0.5 - |c_ik|)) = sign(c_ik) * 2 * viol
                grad[mask, k] += 2 * wall_viol[mask, k] * np.sign(c[mask, k])
        
        # Pair gradient
        for idx in range(len(iu[0])):
            i, j = iu[0][idx], iu[1][idx]
            if pv[idx] > 0:
                d = dists[i, j]
                if d > 1e-12:
                    direction = (c[i] - c[j]) / d
                    # grad_i += -2 * pv * direction
                    # grad_j += +2 * pv * direction
                    grad[i] += -2 * pv[idx] * direction
                    grad[j] += 2 * pv[idx] * direction
        
        return obj, grad.ravel()

    def penalty_refine(centers, r_target, time_limit, max_iter=200):
        """
        Use L-BFGS-B to minimize penalty for target radius r_target.
        Returns (new_centers, new_r, success).
        """
        x0 = centers.ravel()
        
        def objective_and_grad(x):
            if time.time() > time_limit:
                return 1e10, np.zeros_like(x)
            return penalty_objective_and_grad(x, r_target)
        
        try:
            bounds = [(-0.499999, 0.499999)] * (N * 3)
            result = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': max_iter, 'maxfun': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            new_centers = result.x.reshape(N, 3)
            new_centers = np.clip(new_centers, -0.499999, 0.499999)
            new_r = compute_radius(new_centers)
            if new_r >= r_target * 0.9999:
                return new_centers, new_r, True
            else:
                return new_centers, new_r, False
        except Exception:
            return centers, compute_radius(centers), False

    def force_directed_refine(centers, time_limit, step_scale=1.0):
        """
        Fast force-directed local search.
        """
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
            if force_norm < 1e-10:
                break
            force = force / force_norm
            
            step = base_step * (decay_rate ** (it / 10.0))
            
            new_centers = centers + step * force
            new_centers = np.clip(new_centers, -0.4999, 0.4999)
            
            new_r = compute_radius(new_centers)
            
            if new_r > r:
                r = new_r
                centers = new_centers
                if r > best_r:
                    best_r = r
                    best_centers = centers.copy()
            else:
                step *= 0.5
                new_centers = centers + step * force
                new_centers = np.clip(new_centers, -0.4999, 0.4999)
                new_r = compute_radius(new_centers)
                if new_r > r:
                    r = new_r
                    centers = new_centers
                    if r > best_r:
                        best_r = r
                        best_centers = centers.copy()
                else:
                    centers += np.random.normal(0, 0.003, centers.shape)
                    centers = np.clip(centers, -0.4999, 0.4999)
                    r = compute_radius(centers)
                    if r > best_r:
                        best_r = r
                        best_centers = centers.copy()
        
        return best_centers, best_r

    def anneal_and_refine(centers, time_limit, initial_temp=0.05, cooling=0.999):
        """
        Hybrid SA + Force-Directed Refinement.
        """
        centers = centers.copy()
        current_r = compute_radius(centers)
        best_centers = centers.copy()
        best_r = current_r
        temp = initial_temp
        
        while time.time() < time_limit and temp > 1e-5:
            n_move = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
            indices = np.random.choice(N, n_move, replace=False)
            step_size = temp * 0.5
            new_centers = centers.copy()
            new_centers[indices] += np.random.normal(0, step_size, (n_move, 3))
            new_centers = np.clip(new_centers, -0.4999, 0.4999)
            
            new_r = compute_radius(new_centers)
            
            delta = new_r - current_r
            if delta > 0 or np.random.random() < np.exp(delta / (temp + 1e-10)):
                current_r = new_r
                centers = new_centers
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
            
            temp *= cooling
            
            if np.random.random() < 0.05:
                refine_limit = min(time.time() + 0.5, time_limit)
                refined_centers, refined_r = force_directed_refine(centers, refine_limit, step_scale=0.5)
                if refined_r > current_r:
                    current_r = refined_r
                    centers = refined_centers
                    if current_r > best_r:
                        best_r = current_r
                        best_centers = centers.copy()
        
        return best_centers, best_r

    # --- Main Execution Loop ---
    
    np.random.seed(42)
    
    best_centers_global = None
    best_radius_global = 0.0
    
    # 1. Generate Initial Configurations
    initializations = []
    
    # Config 1: 5x5x4 Grid
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.3, 0.3, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    grid_centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    for jitter in [0.0, 0.01, 0.02, 0.05]:
        centers = grid_centers + np.random.normal(0, jitter, grid_centers.shape)
        centers = np.clip(centers, -0.49, 0.49)
        initializations.append(centers)
        
    # Config 2: FCC-like Lattice
    a = 0.35
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
            
    # Config 3: Random Uniform
    for _ in range(4):
        centers = np.random.uniform(-0.45, 0.45, (N, 3))
        initializations.append(centers)

    # 2. Optimization Loop
    # Strategy: 
    # Phase 1: Quick SA/Force refinement to find a good local minimum.
    # Phase 2: Binary search on radius r, using penalty L-BFGS to make r feasible.
    #          This directly targets the bottleneck constraints.
    
    time_per_init = 15.0

    for idx, init_centers in enumerate(initializations):
        if time.time() > deadline - 5:
            break
        
        remaining = deadline - 5 - time.time()
        if remaining < 2:
            break
            
        r_init = compute_radius(init_centers)
        if r_init > 0.10:
            t_alloc = min(remaining * 0.5, 20.0)
        elif r_init > 0.08:
            t_alloc = min(remaining * 0.3, 15.0)
        else:
            t_alloc = min(remaining * 0.2, 10.0)
            
        t_limit = time.time() + t_alloc
        
        # Phase A: Annealing to explore
        anneal_time = t_alloc * 0.3
        anneal_limit = time.time() + anneal_time
        
        sa_centers, sa_r = anneal_and_refine(init_centers, anneal_limit, initial_temp=0.08, cooling=0.998)
        
        if sa_r > best_radius_global:
            best_radius_global = sa_r
            best_centers_global = sa_centers
            
        # Phase B: Intensive Force-Directed Refinement
        refine_time = t_alloc * 0.2
        refine_limit = time.time() + refine_time
        
        fd_centers, fd_r = force_directed_refine(sa_centers, refine_limit, step_scale=1.0)
        
        if fd_r > best_radius_global:
            best_radius_global = fd_r
            best_centers_global = fd_centers
            
        # Phase C: Binary Search with Penalty L-BFGS
        # Try to increase radius beyond current best
        if time.time() < deadline - 10:
            # Set search bounds
            r_low = best_radius_global
            r_high = best_radius_global * 1.05 # Try to improve by up to 5%
            
            # If we haven't found a good solution yet, try a higher range
            if best_radius_global < 0.11:
                r_high = 0.12
            
            c_start = best_centers_global.copy()
            
            # Binary search for a few iterations
            for _ in range(5):
                if time.time() > deadline - 5:
                    break
                    
                r_mid = (r_low + r_high) / 2.0
                if r_mid <= best_radius_global:
                    r_low = r_mid
                    continue
                
                # Time limit for this penalty solve
                it_limit = time.time() + min(3.0, (deadline - 5 - time.time()) / 2.0)
                
                new_centers, new_r, feasible = penalty_refine(c_start, r_mid, it_limit)
                
                if feasible and new_r >= r_mid * 0.9999:
                    best_radius_global = new_r
                    best_centers_global = new_centers
                    c_start = new_centers.copy()
                    r_low = r_mid
                else:
                    r_high = r_mid
                    
                # Update global best if we improved
                if new_r > best_radius_global:
                    best_radius_global = new_r
                    best_centers_global = new_centers

    # Final Safety Check
    if best_centers_global is None:
        return grid_centers
        
    # Ensure validity
    best_centers_global = np.clip(best_centers_global, -0.5, 0.5)
    
    return best_centers_global

if __name__ == "__main__":
    start = time.time()
    result = construct()
    end = time.time()
    r = compute_radius(result)
    print(f"Radius: {r}, Time: {end-start}s")
