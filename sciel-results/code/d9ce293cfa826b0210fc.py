"""Improved packing: Multi-phase optimization with repulsion annealing, Voronoi-Lloyd,
smooth soft-min polish, and exact local search."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi

N = 50

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    centers = np.asarray(centers)
    wall_min = 0.5 - np.max(np.abs(centers), axis=1)
    r_wall = np.min(wall_min)
    n = len(centers)
    if n < 2:
        return r_wall
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    r_pair = min_pair_dist / 2.0
    return min(r_wall, r_pair)

def _pair_dist_matrix(centers):
    """Compute pairwise distance matrix."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))

def _softmin_radius(centers, beta=50.0):
    """Smooth approximation to the minimum radius constraint."""
    c = centers
    wall_constraints = 0.5 - np.abs(c).ravel()
    dists = _pair_dist_matrix(c)
    np.fill_diagonal(dists, np.inf)
    pair_constraints = dists / 2.0
    pair_constraints = pair_constraints[np.triu_indices_from(pair_constraints, k=1)]
    all_constraints = np.concatenate([wall_constraints, pair_constraints])
    m = np.min(all_constraints)
    exponent = -beta * (all_constraints - m)
    exponent = np.clip(exponent, -700, 700)
    log_sum = np.log(np.sum(np.exp(exponent)))
    return m - log_sum / beta

def _smooth_obj(x, beta=50.0):
    """Smooth surrogate objective to minimize."""
    c = x.reshape(-1, 3)
    s = _softmin_radius(c, beta)
    out_of_bounds = np.maximum(0, np.abs(c) - 0.4999)
    penalty = 10.0 * np.sum(out_of_bounds ** 2)
    return -s + penalty

def _smooth_polish(centers, deadline, betas=(50.0, 100.0, 200.0)):
    """Polish using smooth surrogate with L-BFGS-B at increasing beta."""
    cur = centers.copy()
    best_score = _compute_radius(cur)
    for beta in betas:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(
                _smooth_obj,
                cur.ravel(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * 3 * N,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 200}
            )
            new_centers = res.x.reshape(-1, 3)
            new_score = _compute_radius(new_centers)
            if new_score > best_score:
                cur = new_centers.copy()
                best_score = new_score
        except Exception:
            pass
    return cur

def _repulsion_anneal(centers, deadline, iters=5000, t0=0.05, t1=0.001, seed=42):
    """Repulsion-based annealing to push spheres apart."""
    rng = np.random.RandomState(seed)
    cur = centers.copy()
    best = cur.copy()
    best_score = _compute_radius(cur)
    cur_score = best_score
    
    for it in range(iters):
        if time.time() > deadline - 10:
            break
        t = t0 * (t1 / t0) ** (it / iters)
        
        # Compute repulsion forces
        dists = _pair_dist_matrix(cur)
        np.fill_diagonal(dists, np.inf)
        
        # Force: push apart when close
        forces = np.zeros_like(cur)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = dists[i, j]
                if d < 0.3:
                    direction = (cur[i] - cur[j]) / (d + 1e-10)
                    force_mag = 1.0 / (d + 1e-10) ** 2
                    forces[i] += direction * force_mag
        
        # Wall repulsion
        for i in range(N):
            for k in range(3):
                if cur[i, k] > 0.3:
                    forces[i, k] -= 10.0 * (cur[i, k] - 0.3)
                elif cur[i, k] < -0.3:
                    forces[i, k] += 10.0 * (-0.3 - cur[i, k])
        
        # Random perturbation
        noise = rng.normal(0, t, size=cur.shape)
        
        # Update
        new_cur = cur + 0.1 * forces + noise
        new_cur = np.clip(new_cur, -0.4999, 0.4999)
        
        new_score = _compute_radius(new_cur)
        
        # Metropolis acceptance
        if new_score >= cur_score or rng.random() < np.exp((new_score - cur_score) / (t * 0.01)):
            cur = new_cur
            cur_score = new_score
            if cur_score > best_score:
                best = cur.copy()
                best_score = cur_score
    
    return best

def _lloyd_step(centers, alpha=0.8):
    """One step of Lloyd's algorithm."""
    c = centers.copy()
    cube_min = -0.5
    cube_max = 0.5
    
    try:
        vor = Voronoi(c)
    except Exception:
        return c
    
    new_c = c.copy()
    
    for i in range(len(c)):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        
        if len(region) == 0:
            continue
        
        vertex_indices = [v for v in region if v >= 0]
        
        if len(vertex_indices) < 3:
            continue
        
        vertices = vor.vertices[vertex_indices]
        
        # Clip vertices to cube
        vertices = np.clip(vertices, cube_min, cube_max)
        
        # Compute centroid
        centroid = np.mean(vertices, axis=0)
        centroid = np.clip(centroid, cube_min + 1e-10, cube_max - 1e-10)
        
        new_c[i] = (1 - alpha) * c[i] + alpha * centroid
    
    new_c = np.clip(new_c, cube_min + 1e-6, cube_max - 1e-6)
    return new_c

def _lloyd_optimize(centers, deadline, max_iter=200, alpha=0.8):
    """Lloyd's algorithm optimization."""
    c = centers.copy()
    best = c.copy()
    best_score = _compute_radius(c)
    
    for it in range(max_iter):
        if time.time() > deadline - 15:
            break
        
        new_c = _lloyd_step(c, alpha)
        displacement = np.max(np.linalg.norm(new_c - c, axis=1))
        c = new_c
        
        score = _compute_radius(c)
        if score > best_score:
            best = c.copy()
            best_score = score
        
        if displacement < 1e-10:
            break
    
    return best

def _local_search(centers, deadline, iters=5000):
    """Exact local search with coordinate descent."""
    cur = centers.copy()
    best_score = _compute_radius(cur)
    step = 0.005
    rng = np.random.RandomState(123)
    
    for _ in range(iters):
        if time.time() > deadline - 2:
            break
        
        improved = False
        # Try moving each center
        idx = rng.randint(N)
        coord = rng.randint(3)
        
        for delta in [step, -step, step*0.5, -step*0.5, step*0.1, -step*0.1]:
            new_c = cur.copy()
            new_c[idx, coord] += delta
            if np.all(np.abs(new_c) <= 0.5):
                s = _compute_radius(new_c)
                if s > best_score + 1e-12:
                    cur = new_c
                    best_score = s
                    improved = True
                    break
        
        if not improved:
            # Try swapping two centers
            i, j = rng.choice(N, 2, replace=False)
            cur[i], cur[j] = cur[j].copy(), cur[i].copy()
            s = _compute_radius(cur)
            if s < best_score - 1e-9:
                cur[i], cur[j] = cur[j].copy(), cur[i].copy()
            else:
                best_score = max(best_score, s)
        
        # Occasionally reduce step
        if rng.random() < 0.01:
            step *= 0.8
            step = max(step, 1e-7)
    
    return cur

def _seed_fcc():
    """FCC-inspired initialization."""
    centers = []
    spacing = 1.0 / 3.0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                x = -0.5 + i * spacing
                y = -0.5 + j * spacing
                z = -0.5 + k * spacing
                centers.append([x, y, z])
    rng = np.random.RandomState(42)
    centers = np.array(centers[:N])
    jitter = rng.uniform(-0.02, 0.02, size=centers.shape)
    centers += jitter
    centers = np.clip(centers, -0.48, 0.48)
    return centers

def _seed_grid():
    """Jittered grid initialization."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(grid))
    centers = grid[perm[:N]].copy()
    jitter = rng.uniform(-0.01, 0.01, size=centers.shape)
    centers += jitter
    centers = np.clip(centers, -0.49, 0.49)
    return centers

def _seed_random():
    """Random initialization."""
    rng = np.random.RandomState(123)
    return rng.uniform(-0.4, 0.4, size=(N, 3))

def _seed_bcc():
    """BCC-inspired initialization."""
    centers = []
    spacing = 1.0 / 3.0
    # Corner points
    for i in range(4):
        for j in range(4):
            for k in range(4):
                centers.append([-0.5 + i * spacing, -0.5 + j * spacing, -0.5 + k * spacing])
    # Body center points
    for i in range(3):
        for j in range(3):
            for k in range(3):
                centers.append([-0.5 + (i + 0.5) * spacing, -0.5 + (j + 0.5) * spacing, -0.5 + (k + 0.5) * spacing])
    rng = np.random.RandomState(42)
    centers = np.array(centers[:N])
    jitter = rng.uniform(-0.02, 0.02, size=centers.shape)
    centers += jitter
    centers = np.clip(centers, -0.48, 0.48)
    return centers

def _seed_shell_core():
    """Shell-core initialization: points on a sphere shell plus core."""
    rng = np.random.RandomState(42)
    # Core points
    core = rng.uniform(-0.2, 0.2, size=(25, 3))
    # Shell points on a sphere of radius 0.35
    theta = rng.uniform(0, 2*np.pi, size=25)
    phi = np.arccos(rng.uniform(-1, 1, size=25))
    r = 0.35
    shell = np.column_stack([
        r * np.sin(phi) * np.cos(theta),
        r * np.sin(phi) * np.sin(theta),
        r * np.cos(phi)
    ])
    centers = np.vstack([core, shell])
    centers = np.clip(centers, -0.48, 0.48)
    return centers

def _metropolis_mc(centers, deadline, iters=10000, t0=0.03, t1=0.0005, seed=99):
    """Metropolis Monte Carlo with displacements and swaps."""
    rng = np.random.RandomState(seed)
    cur = centers.copy()
    best = cur.copy()
    best_score = _compute_radius(cur)
    cur_score = best_score
    
    for it in range(iters):
        if time.time() > deadline - 8:
            break
        t = t0 * (t1 / t0) ** (it / iters)
        
        # Choose move type
        if rng.random() < 0.7:
            # Displacement
            idx = rng.randint(N)
            disp = rng.normal(0, t, size=3)
            new_c = cur.copy()
            new_c[idx] += disp
            new_c = np.clip(new_c, -0.4999, 0.4999)
        else:
            # Swap
            i, j = rng.choice(N, 2, replace=False)
            new_c = cur.copy()
            new_c[i], new_c[j] = new_c[j].copy(), new_c[i].copy()
        
        new_score = _compute_radius(new_c)
        
        # Metropolis acceptance
        if new_score >= cur_score or rng.random() < np.exp((new_score - cur_score) / (t * 0.005)):
            cur = new_c
            cur_score = new_score
            if cur_score > best_score:
                best = cur.copy()
                best_score = cur_score
    
    return best

def construct():
    t_start = time.time()
    deadline = t_start + 140  # Safety margin before 150s limit
    
    best_centers = None
    best_score = -np.inf
    
    # Generate diverse initializations
    inits = [
        (_seed_fcc(), "fcc"),
        (_seed_grid(), "grid"),
        (_seed_bcc(), "bcc"),
        (_seed_shell_core(), "shell_core"),
        (_seed_random(), "random")
    ]
    
    for init_idx, (init, name) in enumerate(inits):
        if time.time() > deadline - 20:
            break
            
        cur = init.copy()
        cur_score = _compute_radius(cur)
        
        if best_centers is None or cur_score > best_score:
            best_centers = cur.copy()
            best_score = cur_score
        
        # Phase 1: Repulsion annealing
        if time.time() > deadline - 18:
            continue
        rep_result = _repulsion_anneal(cur, deadline, iters=3000, seed=42 + init_idx)
        rep_score = _compute_radius(rep_result)
        if rep_score > best_score:
            best_centers = rep_result.copy()
            best_score = rep_score
            cur = rep_result.copy()
        
        # Phase 2: Lloyd's algorithm
        if time.time() > deadline - 15:
            continue
        lloyd_result = _lloyd_optimize(cur, deadline, max_iter=100)
        lloyd_score = _compute_radius(lloyd_result)
        if lloyd_score > best_score:
            best_centers = lloyd_result.copy()
            best_score = lloyd_score
            cur = lloyd_result.copy()
        
        # Phase 3: Smooth polish
        if time.time() > deadline - 10:
            continue
        polished = _smooth_polish(cur, deadline, betas=(50.0, 100.0, 200.0))
        pol_score = _compute_radius(polished)
        if pol_score > best_score:
            best_centers = polished.copy()
            best_score = pol_score
            cur = polished.copy()
        
        # Phase 4: Metropolis MC
        if time.time() > deadline - 5:
            continue
        mc_result = _metropolis_mc(cur, deadline, iters=5000, seed=99 + init_idx)
        mc_score = _compute_radius(mc_result)
        if mc_score > best_score:
            best_centers = mc_result.copy()
            best_score = mc_score
            cur = mc_result.copy()
        
        # Phase 5: Local search
        if time.time() > deadline - 2:
            continue
        ls_result = _local_search(cur, deadline, iters=2000)
        ls_score = _compute_radius(ls_result)
        if ls_score > best_score:
            best_centers = ls_result.copy()
            best_score = ls_score
    
    # Final polish on best result
    if best_centers is not None and time.time() < deadline - 1:
        final = _smooth_polish(best_centers, deadline, betas=(100.0, 200.0))
        final_score = _compute_radius(final)
        if final_score > best_score:
            best_centers = final
            best_score = final_score
    
    if best_centers is None:
        best_centers = _seed_grid()
    
    return best_centers
