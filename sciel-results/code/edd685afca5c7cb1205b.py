"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist, squareform

def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 120s limit
    
    def compute_radius(c):
        """Compute the maximum common radius for given centres."""
        if not np.all(np.isfinite(c)):
            return 0.0
        if c.shape != (N, 3):
            return 0.0
        if np.any(np.abs(c) > 0.5 + 1e-10):
            return 0.0
        wall_min = 0.5 - np.max(np.abs(c), axis=1)
        if np.any(wall_min < -1e-10):
            return 0.0
        r_wall = np.min(wall_min)
        d = pdist(c)
        if len(d) == 0:
            r_pair = np.inf
        else:
            r_pair = np.min(d) / 2.0
        return min(r_wall, r_pair)
    
    def _bcc_grid(n_per_axis, scale=1.0):
        """Generate BCC lattice points scaled to fit in [-0.5, 0.5]^3."""
        points = []
        for i in range(n_per_axis):
            for j in range(n_per_axis):
                for k in range(n_per_axis):
                    # Corner point
                    x = (i + 0.5) / n_per_axis - 0.5
                    y = (j + 0.5) / n_per_axis - 0.5
                    z = (k + 0.5) / n_per_axis - 0.5
                    points.append([x * scale, y * scale, z * scale])
                    # Body center
                    points.append([((i + 1) / n_per_axis - 0.5) * scale,
                                   ((j + 1) / n_per_axis - 0.5) * scale,
                                   ((k + 1) / n_per_axis - 0.5) * scale])
        return np.array(points)
    
    def _cubic_grid(n_per_axis):
        """Generate simple cubic grid points in [-0.5, 0.5]^3."""
        ticks = (np.arange(n_per_axis) + 0.5) / n_per_axis - 0.5
        return np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    def _fcc_grid(n_per_axis):
        """Generate FCC lattice points scaled to fit in [-0.5, 0.5]^3."""
        points = []
        for i in range(n_per_axis):
            for j in range(n_per_axis):
                for k in range(n_per_axis):
                    x = (i + 0.5) / n_per_axis - 0.5
                    y = (j + 0.5) / n_per_axis - 0.5
                    z = (k + 0.5) / n_per_axis - 0.5
                    points.append([x, y, z])
                    # Face centers
                    if i < n_per_axis - 1:
                        points.append([((i + 1) / n_per_axis - 0.5), y, z])
                    if j < n_per_axis - 1:
                        points.append([x, ((j + 1) / n_per_axis - 0.5), z])
                    if k < n_per_axis - 1:
                        points.append([x, y, ((k + 1) / n_per_axis - 0.5)])
        return np.array(points)
    
    def _soft_min_radius(c, beta=50.0):
        """Smooth surrogate for the minimum radius using soft-min."""
        if not np.all(np.isfinite(c)):
            return 0.0
        if c.shape != (N, 3):
            return 0.0
        if np.any(np.abs(c) > 0.5 + 1e-10):
            return 0.0
        
        # Wall distances
        wall_dists = 0.5 - np.max(np.abs(c), axis=1)
        if np.any(wall_dists < -1e-10):
            return 0.0
        
        # Pair distances
        d = pdist(c)
        if len(d) == 0:
            r_pair = np.inf
        else:
            r_pair = d / 2.0
        
        # Soft min of wall distances
        w_max = np.max(wall_dists)
        log_sum_exp_w = w_max + np.log(np.sum(np.exp(wall_dists - w_max)))
        soft_min_wall = -log_sum_exp_w / beta
        
        # Soft min of pair distances
        if np.isfinite(r_pair).any():
            p_max = np.max(r_pair)
            log_sum_exp_p = p_max + np.log(np.sum(np.exp(r_pair - p_max)))
            soft_min_pair = -log_sum_exp_p / beta
        else:
            soft_min_pair = np.inf
        
        return min(soft_min_wall, soft_min_pair)
    
    def _gradient_ascent_smooth(c, iters, step0, step1, beta=50.0, rng=None):
        """Gradient ascent using smooth surrogate with subgradients."""
        if rng is None:
            rng = np.random.RandomState(42)
        
        cur = c.copy()
        cur_r = compute_radius(cur)
        best = cur.copy()
        best_r = cur_r
        
        for i in range(iters):
            if time.time() > deadline - 5.0:
                break
            
            step = step0 * (step1 / step0) ** (i / max(iters, 1))
            
            # Compute pairwise distances
            d_mat = squareform(pdist(cur))
            np.fill_diagonal(d_mat, np.inf)
            min_d = np.min(d_mat)
            
            # Compute wall distances
            wall_dists = 0.5 - np.max(np.abs(cur), axis=1)
            min_wall = np.min(wall_dists)
            
            # Build subgradient
            g = np.zeros_like(cur)
            
            # Pair constraints: push apart closest pairs
            if min_d / 2.0 < min_wall:
                near_pairs = d_mat < (min_d + 1e-6)
                near_pairs.fill_diagonal(False)
                for a in range(N):
                    for b in range(a + 1, N):
                        if near_pairs[a, b]:
                            diff = cur[a] - cur[b]
                            n = np.linalg.norm(diff)
                            if n > 1e-12:
                                g[a] += diff / n
                                g[b] -= diff / n
            
            # Wall constraints: push inward closest walls
            if min_wall <= min_d / 2.0 + 1e-6:
                for j in range(N):
                    for k in range(3):
                        if wall_dists[j] < min_wall + 1e-6:
                            # Find which wall is closest for point j
                            coords = np.abs(cur[j])
                            max_coord = np.max(coords)
                            if np.any(coords == max_coord):
                                idx_max = np.where(coords == max_coord)[0]
                                for idx in idx_max:
                                    if cur[j, idx] > 0:
                                        g[j, idx] -= 1.0
                                    else:
                                        g[j, idx] += 1.0
            
            # Add small noise for exploration
            g += rng.normal(0.0, 0.003, size=cur.shape)
            
            # Update
            cand = cur + step * g
            cand = np.clip(cand, -0.499, 0.499)
            
            new_r = compute_radius(cand)
            if new_r >= cur_r - 1e-9:
                cur = cand
                cur_r = new_r
                if cur_r > best_r:
                    best_r = cur_r
                    best = cur.copy()
            else:
                # Reduce step if no improvement
                step *= 0.95
        
        return best, best_r
    
    def _anneal(pts, T0, T1, steps, rng):
        """Simulated annealing on the minimum radius."""
        cur = pts.copy()
        cur_r = compute_radius(cur)
        best = cur.copy()
        best_r = cur_r
        
        for i in range(steps):
            if time.time() > deadline - 10.0:
                break
            
            T = T0 * (T1 / T0) ** (i / max(steps, 1))
            kick = 0.02 * (T / T0 + 0.05)
            idx = rng.choice(N, size=max(1, N // 5), replace=False)
            cand = cur.copy()
            cand[idx] += rng.normal(0.0, kick, size=(len(idx), 3))
            cand = np.clip(cand, -0.499, 0.499)
            
            cand_r = compute_radius(cand)
            if cand_r > cur_r or rng.random() < np.exp((cand_r - cur_r) / max(T, 1e-9)):
                cur = cand
                cur_r = cand_r
                if cur_r > best_r:
                    best_r = cur_r
                    best = cur.copy()
        
        return best, best_r
    
    # Generate multiple initial configurations
    rng = np.random.RandomState(42)
    
    configs = []
    
    # Config 1: Simple cubic 4x4x4 grid
    grid4 = _cubic_grid(4)
    if len(grid4) >= N:
        configs.append(grid4[:N].copy())
    
    # Config 2: BCC lattice
    bcc = _bcc_grid(3)
    if len(bcc) >= N:
        configs.append(bcc[:N].copy())
    else:
        # Scale down to fit
        bcc_scaled = _bcc_grid(3, scale=0.9)
        configs.append(bcc_scaled[:N].copy())
    
    # Config 3: FCC lattice
    fcc = _fcc_grid(3)
    if len(fcc) >= N:
        configs.append(fcc[:N].copy())
    else:
        fcc_scaled = _fcc_grid(3) * 0.9
        configs.append(fcc_scaled[:N].copy())
    
    # Config 4: Jittered grid
    grid4_jitter = grid4[:N].copy() + rng.normal(0, 0.01, (N, 3))
    grid4_jitter = np.clip(grid4_jitter, -0.49, 0.49)
    configs.append(grid4_jitter)
    
    # Config 5: Random valid points
    rand_pts = rng.uniform(-0.45, 0.45, (N, 3))
    configs.append(rand_pts)
    
    best_centres = None
    best_r = 0.0
    
    for config in configs:
        if time.time() > deadline - 30.0:
            break
        
        config = np.clip(config, -0.499, 0.499)
        config_r = compute_radius(config)
        
        if best_centres is None or config_r > best_r:
            best_r = config_r
            best_centres = config.copy()
        
        # Run gradient ascent with smooth surrogate
        if time.time() > deadline - 20.0:
            break
        
        asc_result, asc_r = _gradient_ascent_smooth(
            config, iters=300, step0=0.01, step1=0.001, beta=50.0, rng=rng
        )
        
        if asc_r > best_r:
            best_r = asc_r
            best_centres = asc_result.copy()
        
        # Run simulated annealing on the best so far
        if time.time() > deadline - 15.0:
            break
        
        anneal_result, anneal_r = _anneal(
            best_centres, T0=0.01, T1=0.001, steps=200, rng=rng
        )
        
        if anneal_r > best_r:
            best_r = anneal_r
            best_centres = anneal_result.copy()
        
        # Final gradient ascent polish
        if time.time() > deadline - 5.0:
            break
        
        polish_result, polish_r = _gradient_ascent_smooth(
            best_centres, iters=200, step0=0.005, step1=0.0005, beta=100.0, rng=rng
        )
        
        if polish_r > best_r:
            best_r = polish_r
            best_centres = polish_result.copy()
    
    # Final validation
    if best_centres is None or not np.all(np.isfinite(best_centres)):
        # Fallback to simple grid
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centres = grid[:N].copy()
    
    if best_centres.shape != (N, 3):
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centres = grid[:N].copy()
    
    if np.any(np.abs(best_centres) > 0.5 + 1e-10):
        best_centres = np.clip(best_centres, -0.499, 0.499)
    
    return best_centres
# EVOLVE-BLOCK-END
