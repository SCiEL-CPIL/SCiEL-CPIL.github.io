import numpy as np
import time
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

_IU = np.triu_indices(N, k=1)

def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers.
    Centers are in [-0.5, 0.5]^3.
    """
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0

    # Distance to walls
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)

    # Pairwise distances (upper triangle only)
    i, j = _IU
    diff = centers[i] - centers[j]
    dists_sq = np.einsum('ij,ij->i', diff, diff)
    min_pair_dist = np.sqrt(np.min(dists_sq))

    return min(min_wall_dist, min_pair_dist / 2.0)

def _soft_min_objective(centers_flat, beta, iu):
    """
    Objective function for L-BFGS-B.
    Minimizes the soft-min approximation of the radius.
    """
    centers = centers_flat.reshape(-1, 3)
    
    # 1. Wall distances
    dists_to_walls = 0.5 - np.abs(centers)
    wall_vals = dists_to_walls.flatten()
    
    # 2. Pairwise distances / 2 (upper triangle only)
    i, j = iu
    diff = centers[i] - centers[j]
    dists_sq = np.einsum('ij,ij->i', diff, diff)
    pair_vals = 0.5 * np.sqrt(dists_sq)
    
    all_vals = np.concatenate([wall_vals, pair_vals])
    
    # Calculate LSE: 1/beta * ln(sum(exp(-beta * x)))
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    
    sum_exp = np.sum(np.exp(neg_beta_x - max_val))
    obj = (max_val + np.log(sum_exp)) / beta
    
    return obj

def _build_initial_grid(n=100, noise_scale=0.003, seed=42):
    """
    Build a well-distributed grid initialization.
    Uses a 5x5x4 grid which gives exactly 100 points.
    """
    nx, ny, nz = 5, 5, 4
    # Use full extent with small margin to allow optimization to push to boundaries
    margin = 0.01
    xs = np.linspace(-0.5 + margin, 0.5 - margin, nx)
    ys = np.linspace(-0.5 + margin, 0.5 - margin, ny)
    zs = np.linspace(-0.5 + margin, 0.5 - margin, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, noise_scale, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers[:n]

def _build_halton_init(n=100, seed=42):
    """
    Build a Halton sequence based initialization for good space-filling properties.
    """
    def halton(index, base):
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result
    
    rng = np.random.default_rng(seed)
    centers = np.zeros((n, 3))
    for i in range(1, n+1):
        centers[i-1, 0] = -0.5 + halton(i, 2)
        centers[i-1, 1] = -0.5 + halton(i, 3)
        centers[i-1, 2] = -0.5 + halton(i, 5)
    
    centers += rng.normal(0, 0.003, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _coordinate_descent_minmax(centers, iu, deadline, max_sweeps=200):
    """
    Coordinate-descent minmax refinement: directly pushes apart the binding pair
    and moves wall-violating points inward. Structurally different from soft-min L-BFGS.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    i, j = iu
    rng = np.random.default_rng(999)
    
    for sweep in range(max_sweeps):
        if time.time() > deadline - 1.0:
            break
        
        # Find the current bottleneck
        dists_to_walls = 0.5 - np.abs(best_centers)
        min_wall_idx = np.argmin(dists_to_walls)
        min_wall_val = dists_to_walls[min_wall_idx]
        
        diff = best_centers[i] - best_centers[j]
        dists_sq = np.einsum('ij,ij->i', diff, diff)
        min_pair_idx = np.argmin(dists_sq)
        min_pair_dist = np.sqrt(dists_sq[min_pair_idx]) / 2.0
        
        current_r = min(min_wall_val, min_pair_dist)
        
        if min_pair_dist < min_wall_val:
            # Pairwise constraint is binding
            a, b = i[min_pair_idx], j[min_pair_idx]
            d = best_centers[a] - best_centers[b]
            dist = np.linalg.norm(d)
            if dist < 1e-12:
                continue
            direction = d / dist
            # Push apart: move a in +direction, b in -direction
            # But respect walls
            step = 0.0005
            new_a = best_centers[a] + direction * step
            new_b = best_centers[b] - direction * step
            
            # Check wall constraints
            for idx, new_c in [(a, new_a), (b, new_b)]:
                for k in range(3):
                    if new_c[k] > 0.5:
                        new_c[k] = 0.5
                    elif new_c[k] < -0.5:
                        new_c[k] = -0.5
            
            best_centers[a] = new_a
            best_centers[b] = new_b
        else:
            # Wall constraint is binding
            idx = min_wall_idx
            # Move inward from the closest wall
            c = best_centers[idx]
            for k in range(3):
                if c[k] > 0:
                    # Closest to +0.5 wall
                    if 0.5 - c[k] == min_wall_val:
                        best_centers[idx, k] -= 0.0005
                        break
                else:
                    # Closest to -0.5 wall
                    if c[k] + 0.5 == min_wall_val:
                        best_centers[idx, k] += 0.0005
                        break
            
            # Also try moving in other directions to find a better spot
            # Small random perturbation to escape local trap
            if rng.random() < 0.3:
                pert = rng.normal(0, 0.001, 3)
                new_c = best_centers[idx] + pert
                new_c = np.clip(new_c, -0.5, 0.5)
                best_centers[idx] = new_c
        
        # Clip to ensure validity
        best_centers = np.clip(best_centers, -0.5, 0.5)
        
        new_r = _compute_radius(best_centers)
        if new_r > best_radius:
            best_radius = new_r
        elif new_r < best_radius - 1e-10:
            # Revert if we made things worse
            best_centers = centers.copy() if sweep == 0 else best_centers
            break
    
    return best_centers, best_radius

def _polish(centers, iu, deadline):
    """
    Final high-precision polish pass using very high beta soft-min
    with tight tolerances and many iterations, plus coordinate-descent minmax.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Try several high-beta values with increasing precision
    betas = [5000.0, 10000.0, 20000.0, 50000.0]
    
    for beta in betas:
        if time.time() > deadline - 5.0:
            break
        
        try:
            res = minimize(
                _soft_min_objective,
                best_centers.flatten(),
                args=(beta, iu),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 200}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass
        
        # Coordinate-descent minmax after each soft-min pass
        if time.time() > deadline - 3.0:
            break
        try:
            cd_centers, cd_radius = _coordinate_descent_minmax(
                best_centers, iu, deadline - 2.0, max_sweeps=100
            )
            if cd_radius > best_radius:
                best_radius = cd_radius
                best_centers = cd_centers
        except Exception:
            pass
    
    return best_centers, best_radius

def _anneal(centers, iu, deadline, initial_temp=0.003, cooling_rate=0.9999):
    """
    Targeted simulated annealing that focuses on improving the minimum constraint.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    current_radius = best_radius
    temp = initial_temp
    
    rng = np.random.default_rng(123)
    stuck_count = 0
    
    while time.time() < deadline and temp > 0.0001:
        # Identify the most constrained spheres (those with smallest clearance)
        dists_to_walls = 0.5 - np.abs(best_centers)
        min_wall = np.min(dists_to_walls)
        
        diff = best_centers[:, np.newaxis, :] - best_centers[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        min_pair = np.sqrt(np.min(dists_sq)) / 2.0
        
        # Perturb spheres that are near the bottleneck
        n_perturb = rng.integers(1, 4)
        
        # Focus on spheres close to walls or each other
        wall_close = np.where(dists_to_walls < min_wall + 0.02)[0]
        if len(wall_close) > 0:
            idx = rng.choice(wall_close, size=min(n_perturb, len(wall_close)), replace=False)
        else:
            idx = rng.choice(N, size=n_perturb, replace=False)
        
        new_centers = best_centers.copy()
        step_size = temp * 0.3
        new_centers[idx] += rng.normal(0, step_size, (len(idx), 3))
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        new_radius = _compute_radius(new_centers)
        
        delta = new_radius - current_radius
        if delta > 0:
            best_centers = new_centers
            best_radius = new_radius
            current_radius = new_radius
            stuck_count = 0
        else:
            prob = np.exp(delta / temp)
            if rng.random() < prob:
                best_centers = new_centers
                current_radius = new_radius
                stuck_count += 1
            else:
                stuck_count += 1
        
        temp *= cooling_rate
        
        if stuck_count > 500:
            break
    
    return best_centers, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Precompute upper triangle indices
    iu = np.triu_indices(N, k=1)
    
    # Initial guess: Multiple diverse initializations
    best_centers = None
    best_radius = 0.0
    
    inits = []
    
    # Primary grid init (5x5x4 = 100 points)
    inits.append(_build_initial_grid(seed=42))
    inits.append(_build_initial_grid(seed=123))
    inits.append(_build_initial_grid(seed=456))
    
    # Halton init
    inits.append(_build_halton_init(seed=42))
    inits.append(_build_halton_init(seed=789))
    
    for centers in inits:
        r = _compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
            
    if best_centers is None:
        best_centers = _build_initial_grid()
        best_radius = _compute_radius(best_centers)
        
    # Phase 1: Soft-min optimization with more aggressive beta schedule
    # 30 -> 30 -> 90 -> 270 -> 810 -> 2430 -> 3000 (6 steps, capped)
    beta_schedule = [30.0, 90.0, 270.0, 810.0, 2430.0, 3000.0]
    
    for beta in beta_schedule:
        if time.time() > deadline - 40.0:
            break
        
        x0 = best_centers.flatten()
        
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta, iu),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 500, 'ftol': 1e-13, 'gtol': 1e-10, 'maxls': 100}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass
        
        # Coordinate-descent minmax between soft-min passes
        if time.time() > deadline - 30.0:
            break
        try:
            cd_centers, cd_radius = _coordinate_descent_minmax(
                best_centers, iu, deadline - 25.0, max_sweeps=150
            )
            if cd_radius > best_radius:
                best_radius = cd_radius
                best_centers = cd_centers
        except Exception:
            pass
        
    # Phase 2: Annealing to escape local minima
    if time.time() < deadline - 20.0:
        try:
            sa_centers, sa_radius = _anneal(
                best_centers, 
                iu,
                deadline - 20.0,
                initial_temp=0.003,
                cooling_rate=0.9999
            )
            if sa_radius > best_radius:
                best_radius = sa_radius
                best_centers = sa_centers
        except Exception:
            pass
            
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 5.0:
        try:
            polished, p_radius = _polish(
                best_centers,
                iu,
                deadline - 5.0
            )
            if p_radius > best_radius:
                best_radius = p_radius
                best_centers = polished
        except Exception:
            pass
    
    # Final safety: ensure all centers are strictly inside
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers

# EVOLVE-BLOCK-END
