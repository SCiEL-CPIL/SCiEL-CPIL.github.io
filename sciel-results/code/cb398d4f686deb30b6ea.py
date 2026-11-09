"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from scipy.spatial.distance import cdist
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s kill

    def time_ok():
        return time.time() < deadline

    # Fast distance computation
    def fast_dist_matrix(centers):
        dx = centers[:, 0][:, None] - centers[:, 0][None, :]
        dy = centers[:, 1][:, None] - centers[:, 1][None, :]
        dists = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dists, np.inf)
        return dists

    # Compute the score (radius) for a given set of centers
    def compute_radius(centers):
        dists = fast_dist_matrix(centers)
        min_pair = np.min(dists) / 2.0
        dist_border = np.min(0.5 - np.abs(centers))
        return min(min_pair, dist_border)

    # Generate diverse initial configurations
    def make_hex_config(seed=0):
        """Hexagonal lattice with offset."""
        rng = np.random.RandomState(seed)
        centers = []
        spacing = 0.143
        row_height = spacing * np.sqrt(3) / 2
        y = -0.45
        row_idx = 0
        while y < 0.5 and len(centers) < N:
            x_offset = (spacing * 0.5) if (row_idx % 2 == 1) else 0.0
            x = -0.45 + x_offset
            while x < 0.5 and len(centers) < N:
                centers.append([x, y])
                x += spacing
            y += row_height
            row_idx += 1
        centers = np.array(centers[:N])
        if len(centers) < N:
            extra = rng.uniform(-0.4, 0.4, size=(N - len(centers), 2))
            centers = np.vstack([centers, extra])
        centers += rng.normal(0, 0.002, size=centers.shape)
        return np.clip(centers, -0.49, 0.49)

    def make_grid_config(seed=0):
        """Grid-based configuration."""
        rng = np.random.RandomState(seed)
        n_per_row = 7
        rows = N // n_per_row + (1 if N % n_per_row else 0)
        centers = []
        x_spacing = 0.9 / (n_per_row - 1) if n_per_row > 1 else 0.9
        y_spacing = 0.9 / (rows - 1) if rows > 1 else 0.9
        for i in range(rows):
            for j in range(n_per_row):
                if len(centers) < N:
                    x = -0.45 + j * x_spacing
                    y = -0.45 + i * y_spacing
                    centers.append([x, y])
        centers = np.array(centers[:N])
        if len(centers) < N:
            extra = rng.uniform(-0.4, 0.4, size=(N - len(centers), 2))
            centers = np.vstack([centers, extra])
        centers += rng.normal(0, 0.002, size=centers.shape)
        return np.clip(centers, -0.49, 0.49)

    def make_random_config(seed=0):
        """Random configuration."""
        rng = np.random.RandomState(seed)
        centers = rng.uniform(-0.45, 0.45, size=(N, 2))
        return centers

    # Gradient-based local optimization using scipy
    def local_optimize_scipy(centers, max_time=10.0):
        """Use scipy's Powell method for local optimization."""
        t0 = time.time()
        
        def neg_radius(x):
            if time.time() - t0 > max_time:
                return 1e10
            c = x.reshape(-1, 2)
            # Penalize going out of bounds
            penalty = 0.0
            for i in range(N):
                for k in range(2):
                    if c[i, k] > 0.499:
                        penalty += (c[i, k] - 0.499) ** 2 * 1000
                    elif c[i, k] < -0.499:
                        penalty += (-0.499 - c[i, k]) ** 2 * 1000
            r = compute_radius(np.clip(c, -0.499, 0.499))
            return -r + penalty
        
        x0 = centers.flatten()
        result = minimize(neg_radius, x0, method='Powell', options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
        
        if result.success or result.fun < neg_radius(x0):
            best_c = np.clip(result.x.reshape(-1, 2), -0.499, 0.499)
            best_r = compute_radius(best_c)
            return best_c, best_r
        else:
            return centers, compute_radius(centers)

    # Simulated annealing with improved moves
    def simulated_annealing(centers, max_time=60.0):
        """Optimize using simulated annealing."""
        current = centers.copy()
        current_r = compute_radius(current)
        best = current.copy()
        best_r = current_r
        
        T = 0.008
        T_min = 0.000005
        cooling = 0.997
        
        t0 = time.time()
        rng = np.random.RandomState(12345)
        
        while time_ok() and T > T_min and time.time() - t0 < max_time:
            n_move = rng.randint(1, 3)
            indices = rng.choice(N, size=n_move, replace=False)
            
            new = current.copy()
            for idx in indices:
                dx = rng.uniform(-0.015, 0.015)
                dy = rng.uniform(-0.015, 0.015)
                new[idx] += [dx, dy]
            
            new = np.clip(new, -0.499, 0.499)
            new_r = compute_radius(new)
            delta = new_r - current_r
            
            if delta > 0 or rng.random() < np.exp(delta / T):
                current = new
                current_r = new_r
                if current_r > best_r:
                    best = current.copy()
                    best_r = current_r
            
            T *= cooling
            
            if rng.random() < 0.0005:
                T = max(T * 3, 0.003)
        
        return best, best_r

    # Main optimization
    best_centers = None
    best_r = -1.0
    
    # Phase 1: Try multiple initial configs with SA + local opt
    configs = [
        make_hex_config(0),
        make_hex_config(1),
        make_grid_config(0),
        make_random_config(0),
    ]
    
    for cfg in configs:
        if not time_ok():
            break
        
        # SA on this config
        sa_c, sa_r = simulated_annealing(cfg, max_time=30.0)
        
        # Local opt on SA result
        lo_c, lo_r = local_optimize_scipy(sa_c, max_time=8.0)
        
        # Local opt on original
        lo2_c, lo2_r = local_optimize_scipy(cfg, max_time=8.0)
        
        for c, r in [(sa_c, sa_r), (lo_c, lo_r), (lo2_c, lo2_r)]:
            if r > best_r:
                best_r = r
                best_centers = c.copy()
    
    # Phase 2: Refine best with more SA iterations
    if best_centers is not None and time_ok():
        # Perturb and re-optimize
        for attempt in range(3):
            if not time_ok():
                break
            perturbed = best_centers + np.random.normal(0, 0.008, size=(N, 2))
            perturbed = np.clip(perturbed, -0.499, 0.499)
            
            sa_c, sa_r = simulated_annealing(perturbed, max_time=20.0)
            lo_c, lo_r = local_optimize_scipy(sa_c, max_time=8.0)
            
            if lo_r > best_r:
                best_r = lo_r
                best_centers = lo_c.copy()
    
    # Phase 3: Final polish
    if best_centers is not None and time_ok():
        final_c, final_r = local_optimize_scipy(best_centers, max_time=15.0)
        if final_r > best_r:
            best_r = final_r
            best_centers = final_c.copy()
    
    if best_centers is None:
        best_centers = make_hex_config(0)
    
    result = best_centers.copy()
    if not np.all(np.isfinite(result)):
        result = make_hex_config(0)
    
    result = np.clip(result, -0.499, 0.499)
    
    return result

# EVOLVE-BLOCK-END
