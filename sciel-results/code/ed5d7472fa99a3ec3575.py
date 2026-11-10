import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    def calc_score(x):
        # Calculate the actual min radius
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        wall_dists = 0.5 - np.abs(x)
        min_wall = np.min(np.min(wall_dists, axis=1))
        
        return min(min_pair / 2.0, min_wall)
    
    def objective_and_grad(x_flat, beta):
        x = x_flat.reshape(N, 3)
        
        # Calculate pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=2)
        dists = np.sqrt(dists_sq + 1e-15)
        
        # Mask out diagonal
        mask = ~np.eye(N, dtype=bool)
        pair_dists = dists[mask]
        
        # Wall distances
        wall_dists = 0.5 - np.abs(x)
        min_wall_dists = np.min(wall_dists, axis=1)
        
        all_dists = np.concatenate([pair_dists, min_wall_dists])
        
        # Log-sum-exp trick
        vals = -beta * all_dists
        max_val = np.max(vals)
        log_sum_exp = max_val + np.log(np.sum(np.exp(vals - max_val)))
        
        obj = -log_sum_exp / beta
        
        # Gradient
        exp_vals = np.exp(vals - max_val)
        S = np.sum(exp_vals)
        
        grad = np.zeros_like(x)
        
        # Pair distances gradient
        pair_exp = exp_vals[:N*(N-1)]
        W = np.zeros((N, N))
        W[mask] = pair_exp / S
        
        unit_vecs = diff / (dists[:, :, np.newaxis] + 1e-15)
        grad_pair = np.sum(W[:, :, np.newaxis] * unit_vecs, axis=1)
        
        # Wall distances gradient
        wall_exp = exp_vals[N*(N-1):]
        weights_wall = wall_exp / S
        
        grad_wall = np.zeros_like(x)
        min_dims = np.argmin(wall_dists, axis=1)
        for i in range(N):
            k = min_dims[i]
            grad_wall[i, k] = weights_wall[i] * (-np.sign(x[i, k]))
        
        grad = grad_pair + grad_wall
        
        return obj, grad

    def run_optimization(init_x, beta, maxiter=500):
        x0 = init_x.flatten()
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        def obj_wrapper(x_flat):
            return objective_and_grad(x_flat, beta)
        
        res = minimize(
            obj_wrapper, 
            x0, 
            method='L-BFGS-B', 
            jac=True, 
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
        )
        return res.x.reshape(N, 3)

    def resolve_overlaps(x, r_target, steps=100):
        x = x.copy()
        for _ in range(steps):
            forces = np.zeros_like(x)
            # Pairwise
            for i in range(N):
                for j in range(i+1, N):
                    d = x[i] - x[j]
                    dist = np.linalg.norm(d)
                    if dist < 2 * r_target and dist > 1e-9:
                        f = (2 * r_target - dist) / dist * d
                        forces[i] += f
                        forces[j] -= f
            # Walls
            for i in range(N):
                for k in range(3):
                    d_left = x[i, k] + 0.5
                    if d_left < r_target:
                        forces[i, k] += (r_target - d_left) * (-1)
                    d_right = 0.5 - x[i, k]
                    if d_right < r_target:
                        forces[i, k] += (r_target - d_right) * (1)
            x += forces * 0.1
            x = np.clip(x, -0.5, 0.5)
        return x

    def generate_fcc_init(seed):
        np.random.seed(seed)
        points = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                for k in range(-2, 3):
                    points.append([i*0.5, j*0.5, k*0.5])
                    points.append([i*0.5+0.25, j*0.5+0.25, k*0.5])
                    points.append([i*0.5+0.25, j*0.5, k*0.5+0.25])
                    points.append([i*0.5, j*0.5+0.25, k*0.5+0.25])
        
        points = np.array(points)
        dists = np.linalg.norm(points, axis=1)
        sorted_indices = np.argsort(dists)
        x_init = points[sorted_indices[:N]].copy()
        
        max_coord = np.max(np.abs(x_init))
        if max_coord > 0.4:
            x_init *= (0.4 / max_coord)
            
        x_init += np.random.normal(0, 0.01, x_init.shape)
        return x_init

    def generate_bcc_init(seed):
        np.random.seed(seed)
        points = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                for k in range(-2, 3):
                    points.append([i*0.5, j*0.5, k*0.5])
                    points.append([i*0.5+0.25, j*0.5+0.25, k*0.5+0.25])
        
        points = np.array(points)
        dists = np.linalg.norm(points, axis=1)
        sorted_indices = np.argsort(dists)
        x_init = points[sorted_indices[:N]].copy()
        
        max_coord = np.max(np.abs(x_init))
        if max_coord > 0.4:
            x_init *= (0.4 / max_coord)
            
        x_init += np.random.normal(0, 0.01, x_init.shape)
        return x_init

    def polish(x, iterations=50):
        """Final high-precision polish using exact min-distance objective with subgradient"""
        x = x.copy()
        for it in range(iterations):
            # Compute exact min radius
            diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff * diff, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            wall_dists = 0.5 - np.abs(x)
            min_wall_dists = np.min(wall_dists, axis=1)
            
            # Find critical pairs
            min_pair_dist = np.min(dists)
            min_wall = np.min(min_wall_dists)
            
            r = min(min_pair_dist / 2.0, min_wall)
            
            # Subgradient step
            forces = np.zeros_like(x)
            
            # If pair distance is limiting
            if min_pair_dist / 2.0 <= min_wall:
                for i in range(N):
                    for j in range(i+1, N):
                        d = x[i] - x[j]
                        dist = np.linalg.norm(d)
                        if dist < min_pair_dist + 1e-10:
                            if dist > 1e-9:
                                f = d / dist * 0.01
                                forces[i] += f
                                forces[j] -= f
            
            # If wall distance is limiting
            if min_wall <= min_pair_dist / 2.0:
                for i in range(N):
                    min_dim = np.argmin(wall_dists[i])
                    k = min_dim
                    if wall_dists[i, k] < min_wall + 1e-10:
                        # Push away from wall
                        if x[i, k] > 0:
                            forces[i, k] -= 0.01
                        else:
                            forces[i, k] += 0.01
            
            x += forces
            x = np.clip(x, -0.5, 0.5)
            
            # Check for improvement
            new_r = calc_score(x)
            if new_r <= r - 1e-12:
                x -= forces * 2  # Backtrack
                x = np.clip(x, -0.5, 0.5)
        
        return x

    best_x = None
    best_score = -np.inf
    
    # Strategy 1: FCC Lattice + Multi-stage optimization
    for seed in range(8):
        if time.time() > deadline:
            break
        
        x_init = generate_fcc_init(seed)
        x_resolved = resolve_overlaps(x_init, 0.05, steps=50)
        
        # Multi-stage beta annealing
        x_opt = run_optimization(x_resolved, beta=30, maxiter=300)
        x_opt = run_optimization(x_opt, beta=80, maxiter=300)
        x_opt = run_optimization(x_opt, beta=200, maxiter=200)
        
        x_opt = polish(x_opt, iterations=30)
        
        score = calc_score(x_opt)
        if score > best_score:
            best_score = score
            best_x = x_opt

    # Strategy 2: BCC Lattice Init
    for seed in range(8, 16):
        if time.time() > deadline:
            break
        
        x_init = generate_bcc_init(seed)
        x_resolved = resolve_overlaps(x_init, 0.05, steps=50)
        
        x_opt = run_optimization(x_resolved, beta=30, maxiter=300)
        x_opt = run_optimization(x_opt, beta=80, maxiter=300)
        x_opt = run_optimization(x_opt, beta=200, maxiter=200)
        
        x_opt = polish(x_opt, iterations=30)
        
        score = calc_score(x_opt)
        if score > best_score:
            best_score = score
            best_x = x_opt

    # Strategy 3: Random + Overlap Resolution
    for seed in range(16, 24):
        if time.time() > deadline:
            break
        np.random.seed(seed)
        x_init = np.random.uniform(-0.4, 0.4, (N, 3))
        x_resolved = resolve_overlaps(x_init, 0.05, steps=50)
        
        x_opt = run_optimization(x_resolved, beta=30, maxiter=300)
        x_opt = run_optimization(x_opt, beta=80, maxiter=300)
        x_opt = run_optimization(x_opt, beta=200, maxiter=200)
        
        x_opt = polish(x_opt, iterations=30)
        
        score = calc_score(x_opt)
        if score > best_score:
            best_score = score
            best_x = x_opt

    # Strategy 4: Final polish on best solution
    if best_x is not None and time.time() < deadline - 10:
        best_x = polish(best_x, iterations=50)
        best_score = calc_score(best_x)

    if best_x is None:
        # Fallback to simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_x = grid[:N]
        
    return best_x
