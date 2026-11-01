import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    # Safety margin: The evaluator kills at 150s. We stop at 145s.
    deadline = start_time + 145.0 

    def compute_radius(c):
        """Compute the maximum common radius for given centers."""
        if not np.all(np.isfinite(c)):
            return 0.0
        if np.any(c > 0.5) or np.any(c < -0.5):
            return 0.0
        # Wall distance
        wall_dists = 0.5 - np.abs(c)
        r_wall = np.min(wall_dists)
        
        # Pairwise distance using cKDTree for efficiency
        try:
            tree = cKDTree(c)
            dists, _ = tree.query(c, k=2)
            r_pair = np.min(dists[:, 1]) / 2.0
        except Exception:
            r_pair = 0.0
            
        return min(r_wall, r_pair)

    def smooth_objective_and_grad(x_flat):
        """
        Smooth surrogate for the sphere packing objective.
        x_flat: shape (300,) -> centers shape (100, 3)
        Returns: -r (to minimize), gradient
        
        Uses soft-min via log-sum-exp for smoothness.
        """
        c = x_flat.reshape(N, 3)
        
        # Wall distances: 0.5 - |c_ik|
        wall_dists = 0.5 - np.abs(c)  # (N, 3)
        
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]  # (N, N, 3)
        dists = np.sqrt(np.sum(diff**2, axis=2))  # (N, N)
        np.fill_diagonal(dists, np.inf)
        pair_dists = dists / 2.0  # radius from pair constraint
        
        # Soft-min of all constraints (wall + pair)
        # We want max r such that r <= all constraints
        # So minimize -min(constraints) = -softmin
        # Use log-sum-exp for soft-min: softmin(a) = -logsumexp(-a)
        # Combine all wall dists and all pair dists (unique pairs)
        
        # For efficiency, use the minimum over all constraints
        # Wall constraints: 300 values
        # Pair constraints: N*(N-1)/2 = 4950 values
        # Total: ~5250 constraints
        
        # Use a temperature parameter for soft-min
        beta = 50.0  # higher = sharper (closer to true min)
        
        # Negative of constraints (for logsumexp)
        neg_wall = -wall_dists.ravel()  # (300,)
        # Unique pairs only
        iu = np.triu_indices(N, k=1)
        neg_pair = -pair_dists[iu]  # (4950,)
        
        all_neg = np.concatenate([neg_wall, neg_pair])
        
        # Soft-min = -logsumexp(all_neg / beta) * beta
        max_neg = np.max(all_neg)
        logsum = max_neg + beta * np.log(np.sum(np.exp((all_neg - max_neg) / beta)))
        soft_min_val = -logsum / beta
        
        # Gradient w.r.t. centers
        # d(soft_min)/d(c) = -sum_j (exp((neg_j - max_neg)/beta) / sum_k exp(...)) * d(neg_j)/d(c)
        # = sum_j (exp((neg_j - max_neg)/beta) / sum_k exp(...)) * d(constraint_j)/d(c)
        
        weights = np.exp((all_neg - max_neg) / beta)
        weights /= np.sum(weights)
        
        grad = np.zeros_like(c)
        
        # Wall gradient: d(0.5 - |c_ik|)/d(c_ik) = -sign(c_ik)
        w_wall = weights[:300].reshape(N, 3)
        grad -= w_wall * np.sign(c)
        
        # Pair gradient: d(|c_i - c_j|/2)/d(c_i) = (c_i - c_j) / (2*|c_i - c_j|)
        w_pair = weights[300:]
        # Accumulate pair gradients
        for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
            d = dists[i, j]
            if d > 1e-12:
                direction = (c[i] - c[j]) / d
                grad[i] += w_pair[idx] * direction / 2.0
                grad[j] -= w_pair[idx] * direction / 2.0
        
        # We want to maximize soft_min, so minimize -soft_min
        # Objective = -soft_min_val
        # Gradient of objective = -gradient of soft_min
        obj_val = -soft_min_val
        grad_obj = -grad
        
        return obj_val, grad_obj.ravel()

    def force_directed_refine(centers, time_limit, step_scale=1.0):
        """
        Force-directed local search to maximize the minimum radius.
        Uses vectorized pairwise computation for speed.
        """
        centers = centers.copy()
        r = compute_radius(centers)
        best_centers = centers.copy()
        best_r = r
        
        # Adaptive parameters
        base_step = 0.1 * step_scale
        decay_rate = 0.998
        stagnation_count = 0
        
        it = 0
        while time.time() < time_limit:
            it += 1
            
            # Compute current radius
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
            
            # Stagnation check
            if current_r <= best_r * 1.00001:
                stagnation_count += 1
                if stagnation_count > 30:
                    break
            else:
                stagnation_count = 0
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
            
            # Compute forces using vectorized operations
            force = np.zeros_like(centers)
            
            # 1. Wall constraints
            target_r = current_r + 0.005 
            for k in range(3):
                d_wall = 0.5 - np.abs(centers[:, k])
                mask = d_wall < target_r
                if np.any(mask):
                    viol = target_r - d_wall[mask]
                    dirs = -np.sign(centers[mask, k])
                    dirs[dirs == 0] = 1.0 
                    force[mask, k] += viol * dirs
            
            # 2. Pair constraints - use KDTree query_pairs
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
            
            # Normalize force
            force_norm = np.linalg.norm(force)
            if force_norm < 1e-12:
                break
            force = force / force_norm
            
            # Adaptive step size
            step = base_step * (decay_rate ** (it / 5.0))
            
            # Propose new position
            new_centers = centers + step * force
            new_centers = np.clip(new_centers, -0.4999, 0.4999)
            
            new_r = compute_radius(new_centers)
            
            if new_r > r:
                r = new_r
                centers = new_centers
            else:
                step *= 0.5
                new_centers = centers + step * force
                new_centers = np.clip(new_centers, -0.4999, 0.4999)
                new_r = compute_radius(new_centers)
                if new_r > r:
                    r = new_r
                    centers = new_centers
                else:
                    # Small random perturbation
                    centers += np.random.normal(0, 0.003, centers.shape)
                    centers = np.clip(centers, -0.4999, 0.4999)
                    r = compute_radius(centers)
                    if r > best_r:
                        best_r = r
                        best_centers = centers.copy()
        
        return best_centers, best_r

    def lbfgs_polish(centers, time_limit, max_iter=200):
        """
        L-BFGS-B polish using smooth surrogate objective.
        This provides high-precision local optimization.
        """
        centers = centers.copy()
        x0 = centers.ravel()
        
        def objective_and_grad(x):
            if time.time() > time_limit:
                # Return large value to signal stop
                return 1e10, np.zeros_like(x)
            val, grad = smooth_objective_and_grad(x)
            return val, grad
        
        try:
            result = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                bounds=[(-0.4999, 0.4999)] * (N * 3),
                options={'maxiter': max_iter, 'maxfun': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < 0:
                new_centers = result.x.reshape(N, 3)
                new_centers = np.clip(new_centers, -0.4999, 0.4999)
                new_r = compute_radius(new_centers)
                if new_r > compute_radius(centers):
                    return new_centers, new_r
        except Exception:
            pass
        
        return centers, compute_radius(centers)

    def anneal_and_refine(centers, time_limit, initial_temp=0.05, cooling=0.999):
        """
        Hybrid Simulated Annealing and Force-Directed Refinement.
        Uses SA to jump between basins and force-directed method to settle.
        """
        centers = centers.copy()
        current_r = compute_radius(centers)
        best_centers = centers.copy()
        best_r = current_r
        temp = initial_temp
        it = 0
        
        while time.time() < time_limit and temp > 1e-5:
            it += 1
            # 1. Random Perturbation (SA Move)
            n_move = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
            indices = np.random.choice(N, n_move, replace=False)
            step_size = temp * 0.5
            new_centers = centers.copy()
            new_centers[indices] += np.random.normal(0, step_size, (n_move, 3))
            new_centers = np.clip(new_centers, -0.4999, 0.4999)
            
            new_r = compute_radius(new_centers)
            
            # Metropolis Criterion
            delta = new_r - current_r
            if delta > 0 or np.random.random() < np.exp(delta / (temp + 1e-10)):
                current_r = new_r
                centers = new_centers
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
            
            temp *= cooling
            
            # 2. Local Refinement every 10 iterations
            if it % 10 == 0 and time.time() < time_limit - 1:
                refine_limit = min(time.time() + 0.3, time_limit)
                refined_centers, refined_r = force_directed_refine(centers, refine_limit, step_scale=0.3)
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
    
    # Config 1: 5x5x4 Grid (Standard dense packing start)
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.3, 0.3, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    grid_centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    # Add jittered versions of grid
    for jitter in [0.0, 0.01, 0.02, 0.05]:
        centers = grid_centers + np.random.normal(0, jitter, grid_centers.shape)
        centers = np.clip(centers, -0.49, 0.49)
        initializations.append(centers)
        
    # Config 2: FCC-like Lattice
    # FCC unit cell points
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
    # Strategy: Quick SA exploration per init, then intensive L-BFGS polish on best
    
    for idx, init_centers in enumerate(initializations):
        if time.time() > deadline - 15:
            break
        
        remaining = deadline - 15 - time.time()
        if remaining < 3:
            break
            
        # Dynamic time allocation
        r_init = compute_radius(init_centers)
        if r_init > 0.10:
            t_alloc = min(remaining * 0.4, 15.0)
        elif r_init > 0.08:
            t_alloc = min(remaining * 0.3, 12.0)
        else:
            t_alloc = min(remaining * 0.2, 8.0)
            
        t_limit = time.time() + t_alloc
        
        # Phase A: Short annealing to find a good basin
        anneal_time = t_alloc * 0.5
        anneal_limit = time.time() + anneal_time
        
        sa_centers, sa_r = anneal_and_refine(init_centers, anneal_limit, initial_temp=0.06, cooling=0.997)
        
        if sa_r > best_radius_global:
            best_radius_global = sa_r
            best_centers_global = sa_centers
            
        # Phase B: Force-directed refinement
        if time.time() < t_limit - 2:
            refine_limit = min(time.time() + (t_limit - time.time() - 2), t_limit)
            fd_centers, fd_r = force_directed_refine(sa_centers, refine_limit, step_scale=0.8)
            
            if fd_r > best_radius_global:
                best_radius_global = fd_r
                best_centers_global = fd_centers
        
        # Phase C: L-BFGS polish on the best found so far
        if time.time() < deadline - 10 and best_centers_global is not None:
            polish_limit = min(time.time() + 3.0, deadline - 10)
            pol_centers, pol_r = lbfgs_polish(best_centers_global, polish_limit, max_iter=100)
            
            if pol_r > best_radius_global:
                best_radius_global = pol_r
                best_centers_global = pol_centers

    # Final intensive polish with remaining time
    if best_centers_global is not None and time.time() < deadline - 3:
        # Run L-BFGS polish with more iterations using remaining time
        polish_limit = deadline - 3
        max_iter = 300
        pol_centers, pol_r = lbfgs_polish(best_centers_global, polish_limit, max_iter=max_iter)
        
        if pol_r > best_radius_global:
            best_radius_global = pol_r
            best_centers_global = pol_centers
        
        # One more force-directed pass if time permits
        if time.time() < deadline - 2:
            fd_centers, fd_r = force_directed_refine(best_centers_global, deadline - 2, step_scale=0.3)
            if fd_r > best_radius_global:
                best_radius_global = fd_r
                best_centers_global = fd_centers
    
    # Final Safety Check
    if best_centers_global is None:
        return grid_centers
        
    # Ensure validity
    best_centers_global = np.clip(best_centers_global, -0.5, 0.5)
    
    return best_centers_global

if __name__ == "__main__":
    # Quick test
    start = time.time()
    result = construct()
    end = time.time()
    r = compute_radius(result)
    print(f"Radius: {r}, Time: {end-start}s")
