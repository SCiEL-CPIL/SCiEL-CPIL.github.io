"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

# Deadline guard
START_TIME = time.time()
DEADLINE = 145.0  # 5s safety margin before 150s kill

def _elapsed():
    return time.time() - START_TIME

def _time_left():
    return DEADLINE - _elapsed()

def _compute_radius(centres):
    """
    Compute the packing radius for a set of centres.
    Uses cKDTree for efficient nearest neighbor search.
    """
    if len(centres) < 2:
        return 0.5 - np.max(np.abs(centres))
    
    # Wall distance
    wall_dists = 0.5 - np.abs(centres)
    min_wall_dist = np.min(wall_dists)
    
    # Pairwise distance using KDTree
    try:
        tree = cKDTree(centres)
        dists, _ = tree.query(centres, k=2)
        min_pair_dist = np.min(dists[:, 1])
    except Exception:
        d = cdist(centres, centres)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = d.min()
        
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _soft_min_objective(x, beta=50.0):
    """
    Smooth surrogate for the negative packing radius using log-sum-exp.
    """
    centres = x.reshape(N, 3)
    
    # Pairwise distances
    d = cdist(centres, centres)
    np.fill_diagonal(d, np.inf)
    
    # Wall distances
    wall_dists = np.concatenate([0.5 - np.abs(centres[:, k]) for k in range(3)])
    
    # All radii: pairwise/2 and wall distances
    pair_radii = d[d < np.inf] / 2.0
    all_radii = np.concatenate([pair_radii, wall_dists])
    
    # Soft min via log-sum-exp
    neg_all = -all_radii
    max_neg = np.max(neg_all)
    if max_neg > 50:
        exp_vals = np.exp(beta * (neg_all - max_neg))
    else:
        exp_vals = np.exp(beta * neg_all)
    
    log_sum = np.log(np.sum(exp_vals))
    soft_max_val = (1.0 / beta) * (log_sum + beta * max_neg)
    soft_min_val = -soft_max_val
    
    return -soft_min_val

def _construct_bcc_init():
    """Body-centered cubic lattice initialization."""
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    np.random.seed(42)
    perturbation = np.random.randn(len(grid), 3) * 0.01
    grid = grid + perturbation
    grid = np.clip(grid, -0.49, 0.49)
    return grid[:N]

def _construct_fcc_init():
    """Face-centered cubic lattice initialization."""
    n_cells = 3
    a = 1.0 / n_cells
    positions = []
    for ix in range(n_cells):
        for iy in range(n_cells):
            for iz in range(n_cells):
                base = np.array([ix, iy, iz]) * a
                fcc_pos = [
                    np.array([0, 0, 0]),
                    np.array([0.5, 0.5, 0]) * a,
                    np.array([0.5, 0, 0.5]) * a,
                    np.array([0, 0.5, 0.5]) * a,
                ]
                for fp in fcc_pos:
                    positions.append(base + fp)
    positions = np.array(positions)
    positions = positions - 0.5
    np.random.seed(42)
    perturbation = np.random.randn(len(positions), 3) * 0.005
    positions = positions + perturbation
    positions = np.clip(positions, -0.49, 0.49)
    return positions[:N]

def _construct_grid_init():
    """Standard grid initialization."""
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.5, 0.5, nx)
    ys = np.linspace(-0.5, 0.5, ny)
    zs = np.linspace(-0.5, 0.5, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    if len(centers) != N:
        if len(centers) > N:
            centers = centers[:N]
        else:
            rng = np.random.default_rng(42)
            extra = rng.uniform(-0.45, 0.45, (N - len(centers), 3))
            centers = np.vstack([centers, extra])
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    return centers

def _construct_halton_init():
    """Halton sequence initialization - low-discrepancy sequence."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = []
    for i in range(1, N + 1):
        x = halton(i, 2)
        y = halton(i, 3)
        z = halton(i, 5)
        points.append([x - 0.5, y - 0.5, z - 0.5])
    
    points = np.array(points)
    # Add small perturbation to break exact symmetry
    np.random.seed(42)
    points += np.random.randn(N, 3) * 0.002
    points = np.clip(points, -0.49, 0.49)
    return points

def _repulsion_search(centers, time_budget=10.0):
    """
    Local search using repulsion forces with KD-tree acceleration.
    Fully vectorized force computation using np.add.at.
    """
    centers = centers.copy()
    r_current = _compute_radius(centers)
    best_r = r_current
    best_centers = centers.copy()
    
    step_size = 0.01
    decay = 0.985
    min_step = 1e-9
    margin = 1.03
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 300
    
    while _time_left() > 5.0 and (time.time() - start_time) < time_budget and step_size > min_step:
        r = _compute_radius(centers)
        if r > best_r + 1e-13:
            best_r = r
            best_centers = centers.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
        
        # Compute forces
        forces = np.zeros_like(centers)
        
        # 1. Wall forces
        for k in range(3):
            dists = 0.5 - np.abs(centers[:, k])
            limit = max(r_current * margin, 0.01)
            mask = dists < limit
            if np.any(mask):
                dirs = -np.sign(centers[mask, k])
                dirs[dirs == 0] = 1.0
                strengths = (limit - dists[mask]) / limit
                forces[mask, k] += dirs * strengths
                
        # 2. Pairwise forces using KD-tree - fully vectorized
        tree = cKDTree(centers)
        search_radius = 2 * max(r_current * margin, 0.01)
        
        # Get all neighbor pairs
        neighbor_pairs = tree.query_pairs(r=search_radius)
        
        if len(neighbor_pairs) > 0:
            i_idx = np.array([p[0] for p in neighbor_pairs], dtype=np.intp)
            j_idx = np.array([p[1] for p in neighbor_pairs], dtype=np.intp)
            
            diffs = centers[i_idx] - centers[j_idx]
            dists = np.linalg.norm(diffs, axis=1)
            
            valid = dists > 1e-10
            if np.any(valid):
                i_valid = i_idx[valid]
                j_valid = j_idx[valid]
                diffs_valid = diffs[valid]
                dists_valid = dists[valid]
                
                strengths = (search_radius - dists_valid) / search_radius
                unit_diffs = diffs_valid / dists_valid[:, np.newaxis]
                force_vecs = strengths[:, np.newaxis] * unit_diffs
                
                # Vectorized accumulation via np.add.at
                np.add.at(forces, i_valid, force_vecs)
                np.add.at(forces, j_valid, -force_vecs)
        
        # Normalize forces
        norms = np.linalg.norm(forces, axis=1, keepdims=True)
        norms[norms == 0] = 1
        forces = forces / norms
        
        centers += step_size * forces
        centers = np.clip(centers, -0.4999, 0.4999)
        
        step_size *= decay
        r_current = r
        
    return best_centers, best_r

def _fine_polish(centers, time_budget=5.0):
    """
    Fine polishing with very small steps to squeeze out last bits of improvement.
    Vectorized force accumulation.
    """
    centers = centers.copy()
    r_current = _compute_radius(centers)
    best_r = r_current
    best_centers = centers.copy()
    
    step_size = 0.001
    decay = 0.97
    min_step = 1e-10
    margin = 1.01
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 200
    
    while _time_left() > 3.0 and (time.time() - start_time) < time_budget and step_size > min_step:
        r = _compute_radius(centers)
        if r > best_r + 1e-14:
            best_r = r
            best_centers = centers.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
        
        forces = np.zeros_like(centers)
        
        # Wall forces
        for k in range(3):
            dists = 0.5 - np.abs(centers[:, k])
            limit = max(r_current * margin, 0.005)
            mask = dists < limit
            if np.any(mask):
                dirs = -np.sign(centers[mask, k])
                dirs[dirs == 0] = 1.0
                strengths = (limit - dists[mask]) / limit
                forces[mask, k] += dirs * strengths
                
        # Pairwise forces - vectorized
        tree = cKDTree(centers)
        search_radius = 2 * max(r_current * margin, 0.005)
        neighbor_pairs = tree.query_pairs(r=search_radius)
        
        if len(neighbor_pairs) > 0:
            i_idx = np.array([p[0] for p in neighbor_pairs], dtype=np.intp)
            j_idx = np.array([p[1] for p in neighbor_pairs], dtype=np.intp)
            diffs = centers[i_idx] - centers[j_idx]
            dists = np.linalg.norm(diffs, axis=1)
            valid = dists > 1e-10
            if np.any(valid):
                i_valid = i_idx[valid]
                j_valid = j_idx[valid]
                diffs_valid = diffs[valid]
                dists_valid = dists[valid]
                strengths = (search_radius - dists_valid) / search_radius
                unit_diffs = diffs_valid / dists_valid[:, np.newaxis]
                force_vecs = strengths[:, np.newaxis] * unit_diffs
                np.add.at(forces, i_valid, force_vecs)
                np.add.at(forces, j_valid, -force_vecs)
        
        norms = np.linalg.norm(forces, axis=1, keepdims=True)
        norms[norms == 0] = 1
        forces = forces / norms
        
        centers += step_size * forces
        centers = np.clip(centers, -0.49999, 0.49999)
        
        step_size *= decay
        r_current = r
        
    return best_centers, best_r

def _optimize_from_init(init_centres, time_budget=20.0):
    """
    Run optimization from given initialization.
    Phase 1: Coarse L-BFGS-B
    Phase 2: Repulsion Local Search
    Phase 3: Fine Polish
    Phase 4: High-beta L-BFGS polish
    """
    best_centres = init_centres.copy()
    best_r = _compute_radius(best_centres)
    
    # Phase 1: Brief L-BFGS to smooth out initial noise
    if _time_left() > 15:
        x0 = best_centres.flatten()
        try:
            result = minimize(
                _soft_min_objective,
                x0,
                args=(40.0,),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (N * 3),
                options={'maxiter': 80, 'ftol': 1e-10, 'gtol': 1e-7}
            )
            new_centres = result.x.reshape(N, 3)
            new_r = _compute_radius(new_centres)
            if new_r > best_r:
                best_r = new_r
                best_centres = new_centres.copy()
        except Exception:
            pass

    # Phase 2: Repulsion Search - main workhorse
    if _time_left() > 10:
        rep_budget = min(15.0, _time_left() - 8.0)
        if rep_budget > 2.0:
            rep_centres, rep_r = _repulsion_search(best_centres, time_budget=rep_budget)
            if rep_r > best_r:
                best_r = rep_r
                best_centres = rep_centres.copy()

    # Phase 3: Fine Polish
    if _time_left() > 5:
        polish_budget = min(5.0, _time_left() - 3.0)
        if polish_budget > 1.0:
            polished_centres, polished_r = _fine_polish(best_centres, time_budget=polish_budget)
            if polished_r > best_r:
                best_r = polished_r
                best_centres = polished_centres.copy()
        
        # Phase 4: High-beta L-BFGS polish for final precision
        if _time_left() > 3:
            x0 = best_centres.flatten()
            try:
                result = minimize(
                    _soft_min_objective,
                    x0,
                    args=(200.0,),
                    method='L-BFGS-B',
                    bounds=[(-0.49999, 0.49999)] * (N * 3),
                    options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                new_centres = result.x.reshape(N, 3)
                new_r = _compute_radius(new_centres)
                if new_r > best_r:
                    best_r = new_r
                    best_centres = new_centres.copy()
            except Exception:
                pass

    return best_centres, best_r

def construct():
    """Main entry point."""
    global START_TIME, DEADLINE
    START_TIME = time.time()
    DEADLINE = START_TIME + 145.0
    
    best_centres = None
    best_r = -np.inf
    
    # Try multiple initializations
    inits = [
        _construct_bcc_init(),
        _construct_fcc_init(),
        _construct_grid_init(),
        _construct_halton_init(),
    ]
    
    # Also try a random initialization
    np.random.seed(123)
    random_init = np.random.uniform(-0.45, 0.45, (N, 3))
    inits.append(random_init)
    
    # Allocate time: ~145s total
    # Give structured inits more time, random less
    for i, init in enumerate(inits):
        if _time_left() < 12:
            break
        
        # Budget allocation: first 3 inits get more time
        if i < 3:
            budget = 28.0
        elif i == 3:
            budget = 22.0
        else:
            budget = 12.0
        
        centres, r = _optimize_from_init(init, time_budget=budget)
        
        if r > best_r:
            best_r = r
            best_centres = centres.copy()
            
    # Final global polish on the best candidate found
    if best_centres is not None and _time_left() > 8:
        polish_budget = _time_left() - 4.0
        if polish_budget > 3.0:
            polished_centres, polished_r = _optimize_from_init(best_centres, time_budget=polish_budget)
            if polished_r > best_r:
                best_r = polished_r
                best_centres = polished_centres.copy()
            
    if best_centres is None:
        # Fallback to simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centres = grid[:N]
    
    # Ensure centres are strictly inside the cube
    best_centres = np.clip(best_centres, -0.49999, 0.49999)
    
    return best_centres
# EVOLVE-BLOCK-END
