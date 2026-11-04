import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 100
HALF = 0.5

# Jellyfish parameters
N_ANCHORS = 8  # 8 corner anchors
N_LATTICE = 125  # 5x5x5 base lattice sites

def _jellyfish_decode(anchors, lattice_offsets):
    """
    Decode jellyfish parameters into 100 sphere centers.
    anchors: (8, 3) array of corner anchor positions
    lattice_offsets: (100, 3) array of offsets from nearest lattice site
    
    Strategy: 100 centers = weighted blend of nearest 5x5x5 lattice site 
    and surrounding anchors, plus a local offset.
    """
    # 5x5x5 lattice in [-0.45, 0.45]
    lat_coords = np.linspace(-0.45, 0.45, 5)
    lattice_sites = np.array([[x, y, z] for x in lat_coords for y in lat_coords for z in lat_coords])
    
    # Assign each of 100 spheres to a lattice site (first 100 of 125)
    assigned_sites = lattice_sites[:100]
    
    # For each sphere, compute weight toward nearest anchor
    centers = np.zeros((100, 3))
    for i in range(100):
        site = assigned_sites[i]
        # Find nearest anchor
        dists_to_anchors = np.linalg.norm(anchors - site, axis=1)
        nearest_anchor_idx = np.argmin(dists_to_anchors)
        nearest_anchor = anchors[nearest_anchor_idx]
        
        # Blend: 70% lattice site + 30% toward anchor, plus offset
        base = 0.7 * site + 0.3 * nearest_anchor
        center = base + lattice_offsets[i]
        centers[i] = np.clip(center, -HALF, HALF)
    
    return centers

def _jellyfish_optimize(initial_centers, time_limit=25.0):
    """
    Optimize the jellyfish representation: 8 anchors + 100 offsets = 308 params.
    Uses L-BFGS-B to maximize minimum radius via soft-min surrogate.
    """
    start_time = time.time()
    
    # Initialize anchors at cube corners
    anchors = np.array([
        [-HALF, -HALF, -HALF], [HALF, -HALF, -HALF],
        [-HALF, HALF, -HALF], [HALF, HALF, -HALF],
        [-HALF, -HALF, HALF], [HALF, -HALF, HALF],
        [-HALF, HALF, HALF], [HALF, HALF, HALF]
    ], dtype=float)
    
    # Initialize offsets to match initial_centers approximately
    lat_coords = np.linspace(-0.45, 0.45, 5)
    lattice_sites = np.array([[x, y, z] for x in lat_coords for y in lat_coords for z in lat_coords])
    assigned_sites = lattice_sites[:100]
    
    # Compute initial offsets
    offsets = np.zeros((100, 3))
    for i in range(100):
        site = assigned_sites[i]
        dists_to_anchors = np.linalg.norm(anchors - site, axis=1)
        nearest_anchor = anchors[np.argmin(dists_to_anchors)]
        base = 0.7 * site + 0.3 * nearest_anchor
        offsets[i] = initial_centers[i] - base
        offsets[i] = np.clip(offsets[i], -0.15, 0.15)
    
    # Pack parameters: anchors (24) + offsets (300) = 324
    def pack_params(a, o):
        return np.concatenate([a.ravel(), o.ravel()])
    
    def unpack_params(x):
        a = x[:24].reshape(8, 3)
        o = x[24:].reshape(100, 3)
        return a, o
    
    # Soft-min objective with annealing
    def soft_min_obj(x, temp):
        a, o = unpack_params(x)
        centers = _jellyfish_decode(a, o)
        
        # Compute pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
        np.fill_diagonal(dists, np.inf)
        
        # Wall distances
        d_walls = HALF - np.abs(centers)
        
        # r values
        r_pairs = dists / 2.0
        r_walls = d_walls
        
        # Soft-min over all r values using log-sum-exp
        all_r_pairs = r_pairs.ravel()
        all_r_walls = r_walls.ravel()
        
        K = 200
        pair_flat_idx = np.argpartition(all_r_pairs, K)[:K]
        wall_flat_idx = np.argpartition(all_r_walls, K)[:K]
        
        r_pair_subset = all_r_pairs[pair_flat_idx]
        r_wall_subset = all_r_walls[wall_flat_idx]
        
        all_r = np.concatenate([r_pair_subset, r_wall_subset])
        neg_r_over_t = -all_r / temp
        max_neg = np.max(neg_r_over_t)
        log_sum = max_neg + np.log(np.sum(np.exp(neg_r_over_t - max_neg)))
        obj_val = temp * log_sum
        
        weights = np.exp(neg_r_over_t - max_neg)
        weights /= np.sum(weights)
        
        grad = np.zeros_like(centers)
        
        # Pair contributions
        n_pairs_subset = len(pair_flat_idx)
        pair_weights = weights[:n_pairs_subset]
        for pi in range(n_pairs_subset):
            flat_idx = pair_flat_idx[pi]
            i, j = np.unravel_index(flat_idx, (N, N))
            if i == j:
                continue
            d = dists[i, j]
            if d < 1e-12:
                continue
            vec = (centers[i] - centers[j]) / d
            w = pair_weights[pi]
            grad[i] += w * 0.5 * vec
            grad[j] += -w * 0.5 * vec
        
        # Wall contributions
        wall_weights = weights[n_pairs_subset:]
        for wi in range(len(wall_flat_idx)):
            flat_idx = wall_flat_idx[wi]
            i, k = np.unravel_index(flat_idx, (N, 3))
            s = np.sign(centers[i, k])
            if s == 0:
                s = 1.0
            w = wall_weights[wi]
            grad[i, k] += -w * s
        
        # Backpropagate to anchors and offsets
        # centers[i] = clip(0.7*site_i + 0.3*anchor_nearest_i + offset_i)
        # Approximate gradient: for each center, distribute grad to offset and nearest anchor
        grad_anchors = np.zeros_like(a)
        grad_offsets = np.zeros_like(o)
        
        for i in range(100):
            center = centers[i]
            # Check if clipped
            g = grad[i]
            # Only pass gradient through non-clipped dimensions
            mask = (center > -HALF + 1e-6) & (center < HALF - 1e-6)
            g_clipped = g * mask
            
            grad_offsets[i] = g_clipped
            
            # Distribute to nearest anchor (30% weight)
            site = assigned_sites[i]
            dists_to_anchors = np.linalg.norm(a - site, axis=1)
            nearest_anchor_idx = np.argmin(dists_to_anchors)
            grad_anchors[nearest_anchor_idx] += 0.3 * g_clipped
        
        # Also distribute to other anchors via soft assignment for smoother gradients
        # Use softmax over anchor distances for smoother weight distribution
        for i in range(100):
            center = centers[i]
            mask = (center > -HALF + 1e-6) & (center < HALF - 1e-6)
            g_clipped = grad[i] * mask
            
            site = assigned_sites[i]
            dists_to_anchors = np.linalg.norm(a - site, axis=1)
            # Softmax weights (inverse distance)
            inv_dists = 1.0 / (dists_to_anchors + 0.01)
            weights_soft = inv_dists / np.sum(inv_dists)
            
            for ai in range(8):
                grad_anchors[ai] += 0.3 * weights_soft[ai] * g_clipped
        
        full_grad = pack_params(grad_anchors, grad_offsets)
        return obj_val, -full_grad  # Minimize -soft_min
    
    # Annealing schedule
    temps = [0.05, 0.02, 0.008, 0.003, 0.001]
    current_params = pack_params(anchors, offsets)
    best_params = current_params.copy()
    best_score = _compute_radius(initial_centers)
    
    anchor_bounds = [(-0.48, 0.48)] * 24
    offset_bounds = [(-0.15, 0.15)] * 300
    bounds = anchor_bounds + offset_bounds
    
    for temp in temps:
        if time.time() - start_time > time_limit - 5.0:
            break
        
        try:
            res = minimize(
                lambda x: soft_min_obj(x, temp),
                current_params,
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'disp': False}
            )
            if np.all(np.isfinite(res.x)):
                current_params = res.x
                decoded = _jellyfish_decode(*unpack_params(current_params))
                score = _compute_radius(decoded)
                if score > best_score:
                    best_score = score
                    best_params = current_params.copy()
        except Exception:
            pass
    
    # Decode best
    best_anchors, best_offsets = unpack_params(best_params)
    best_centers = _jellyfish_decode(best_anchors, best_offsets)
    
    return best_centers, best_score

def _compute_radius(centers):
    """
    Computes the maximum common radius for a given set of centers.
    Centers: (N, 3) array.
    Cube: [-0.5, 0.5]^3.
    """
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
        
    # Wall constraints: distance from center to nearest wall
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    min_wall_dist = np.min(wall_dists)
    
    if min_wall_dist < 0:
        return 0.0

    # Pairwise constraints
    # Use pdist for efficiency
    min_pair_dist = np.min(pdist(centers))
    
    # The radius is half the minimum pairwise distance, limited by wall distance
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _energy_and_gradient(x, R):
    """
    Calculates the repulsion energy and gradient for a target radius R.
    Energy is the sum of squared violations of the distance constraints.
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # 1. Pairwise terms
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
    np.fill_diagonal(dists, np.inf)
    
    # Mask for pairs closer than 2R
    mask = dists < 2.0 * R
    if np.any(mask):
        i_idx, j_idx = np.triu_indices(n, k=1)
        
        d_ij = dists[i_idx, j_idx]
        mask_pairs = d_ij < 2.0 * R
        
        if np.any(mask_pairs):
            v_ij = 2.0 * R - d_ij[mask_pairs]
            d_valid = d_ij[mask_pairs]
            i_valid = i_idx[mask_pairs]
            j_valid = j_idx[mask_pairs]
            
            energy += np.sum(v_ij**2)
            
            factors = 2.0 * v_ij / d_valid
            vecs = centers[i_valid] - centers[j_valid]
            forces = factors[:, np.newaxis] * vecs
            
            np.add.at(grad, i_valid, -forces)
            np.add.at(grad, j_valid, forces)

    # 2. Wall terms
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_abs
    
    mask_wall = wall_dists < R
    if np.any(mask_wall):
        violations_wall = R - wall_dists[mask_wall]
        i_valid_wall = np.where(mask_wall)[0]
        
        energy += np.sum(violations_wall**2)
        
        for idx in i_valid_wall:
            v = violations_wall[np.where(i_valid_wall == idx)[0][0]]
            max_val = max_abs[idx]
            max_k = np.argmax(abs_centers[idx])
            
            grad[idx, max_k] += 2.0 * v * np.sign(centers[idx, max_k])
            
    return energy, grad.flatten()

def _minimize_energy(init_centers, target_R, max_iter=300, ftol=1e-12, gtol=1e-10):
    """
    Minimizes the energy for a given target radius.
    """
    def objective_and_grad(x):
        val, g = _energy_and_gradient(x, target_R)
        return val, g

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(
            objective_and_grad,
            init_centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'disp': False}
        )
        new_centers = res.x.reshape(N, 3)
        final_energy, _ = _energy_and_gradient(new_centers, target_R)
        return new_centers, final_energy, res.success
    except Exception:
        return init_centers, np.inf, False

def _soft_min_polish(pos, max_iter=500, ftol=1e-14, gtol=1e-12, temp_init=0.05, temp_min=1e-4):
    """
    Smooth polish using soft-min (log-sum-exp) surrogate for the nonsmooth radius objective.
    Minimizes -soft_min(r_pair, r_wall) where soft_min uses log-sum-exp for smoothness.
    """
    def soft_min_obj_clean(x):
        centers = x.reshape(N, 3)
        
        # Compute all pair distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(dists_sq + 1e-15)
        np.fill_diagonal(dists, np.inf)
        
        # Wall distances
        d_walls = HALF - np.abs(centers)
        
        # r values
        r_pairs = dists / 2.0  # (N, N)
        r_walls = d_walls  # (N, 3)
        
        # For soft-min, we need to handle this efficiently
        # Use only the smallest K values for the sum
        all_r_pairs = r_pairs.ravel()
        all_r_walls = r_walls.ravel()
        
        # Combine and find the K smallest
        K = 100
        # Pair indices
        pair_flat_idx = np.argpartition(all_r_pairs, K)[:K]
        wall_flat_idx = np.argpartition(all_r_walls, K)[:K]
        
        r_pair_subset = all_r_pairs[pair_flat_idx]
        r_wall_subset = all_r_walls[wall_flat_idx]
        
        # Log-sum-exp over all K*2 values
        all_r = np.concatenate([r_pair_subset, r_wall_subset])
        neg_r_over_t = -all_r / temp
        max_neg = np.max(neg_r_over_t)
        log_sum = max_neg + np.log(np.sum(np.exp(neg_r_over_t - max_neg)))
        obj_val = temp * log_sum
        
        weights = np.exp(neg_r_over_t - max_neg)
        weights /= np.sum(weights)
        
        grad = np.zeros_like(centers)
        
        # Pair contributions
        n_pairs_subset = len(pair_flat_idx)
        pair_weights = weights[:n_pairs_subset]
        for pi in range(n_pairs_subset):
            flat_idx = pair_flat_idx[pi]
            i, j = np.unravel_index(flat_idx, (N, N))
            if i == j:
                continue
            d = dists[i, j]
            if d < 1e-12:
                continue
            vec = (centers[i] - centers[j]) / d
            w = pair_weights[pi]
            grad[i] += w * 0.5 * vec
            grad[j] += -w * 0.5 * vec
        
        # Wall contributions
        wall_weights = weights[n_pairs_subset:]
        for wi in range(len(wall_flat_idx)):
            flat_idx = wall_flat_idx[wi]
            i, k = np.unravel_index(flat_idx, (N, 3))
            s = np.sign(centers[i, k])
            if s == 0:
                s = 1.0
            w = wall_weights[wi]
            grad[i, k] += -w * s
        
        return obj_val, -grad.flatten()  # We minimize -soft_min, so gradient is -grad
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Anneal temperature
    n_steps = min(5, max_iter // 100 + 1)
    temps = np.geomspace(temp_init, temp_min, n_steps)
    
    current_pos = pos.copy()
    for t_idx, temp in enumerate(temps):
        remaining_iter = max(50, max_iter // n_steps)
        try:
            res = minimize(
                soft_min_obj_clean,
                current_pos.flatten(),
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': remaining_iter, 'ftol': ftol, 'gtol': gtol, 'disp': False}
            )
            if np.all(np.isfinite(res.x)):
                current_pos = res.x.reshape(N, 3)
        except Exception:
            pass
    
    return current_pos

def _polish(pos, max_iter=1000, ftol=1e-14, gtol=1e-12):
    """
    High-precision polish: first soft-min, then hard-min.
    """
    # Phase 1: Soft-min polish with annealing
    pos = _soft_min_polish(pos, max_iter=min(500, max_iter // 2), 
                           ftol=ftol, gtol=gtol, temp_init=0.05, temp_min=1e-4)
    
    # Phase 2: Hard-min polish
    def hard_min_obj(x):
        centers = x.reshape(N, 3)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        min_pair_val = np.min(dists)
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_pair_idx
        
        d_walls = HALF - np.abs(centers)
        min_wall_val = np.min(d_walls)
        min_wall_idx = np.unravel_index(np.argmin(d_walls), d_walls.shape)
        wi, wk = min_wall_idx
        
        r_pair = min_pair_val / 2.0
        r_wall = min_wall_val
        
        grad = np.zeros_like(centers)
        
        if r_pair <= r_wall:
            d = min_pair_val
            if d > 1e-12:
                vec = (centers[i] - centers[j]) / d
                grad[i] += 0.5 * vec
                grad[j] += -0.5 * vec
            return -r_pair, -grad.flatten()
        else:
            s = np.sign(centers[wi, wk])
            if s == 0:
                s = 1.0
            grad[wi, wk] = -s
            return -r_wall, -grad.flatten()

    bounds = [(-0.5, 0.5)] * (N * 3)
    try:
        res = minimize(
            hard_min_obj,
            pos.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter // 2, 'ftol': ftol, 'gtol': gtol, 'disp': False}
        )
        if res.success or res.fun < 0:
            new_pos = res.x.reshape(N, 3)
            if np.all(np.isfinite(new_pos)):
                pos = new_pos
    except Exception:
        pass
    
    return pos

def _initial_hybrid():
    """
    Hybrid initialization: 64 spheres in a 4x4x4 simple cubic lattice 
    and 36 spheres in interstitial voids (BCC-like offsets).
    """
    # 4x4x4 grid
    n = 4
    spacing = 1.0 / n
    centers = []
    
    # Generate 4x4x4 grid points
    for i in range(n):
        for j in range(n):
            for k in range(n):
                x = -0.5 + spacing * (i + 0.5)
                y = -0.5 + spacing * (j + 0.5)
                z = -0.5 + spacing * (k + 0.5)
                centers.append([x, y, z])
    
    centers = np.array(centers)
    
    # Now add 36 more points in interstitial voids
    # BCC-like offsets: (0.5, 0.5, 0.5) relative to cell center, scaled by spacing
    # There are 8 body-centered positions per cell, 4^3 = 64 cells, but we only need 36 more
    # Let's pick the 36 body-centered positions that are most spread out
    
    rng = np.random.default_rng(42)
    
    # Generate all 64 body-centered positions
    body_centers = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                x = -0.5 + spacing * (i + 0.5)
                y = -0.5 + spacing * (j + 0.5)
                z = -0.5 + spacing * (k + 0.5)
                # Body center offset
                x += spacing * 0.5
                y += spacing * 0.5
                z += spacing * 0.5
                # Clamp to cube
                x = np.clip(x, -0.5, 0.5)
                y = np.clip(y, -0.5, 0.5)
                z = np.clip(z, -0.5, 0.5)
                body_centers.append([x, y, z])
    
    body_centers = np.array(body_centers)
    
    # Select 36 of them - use a random subset to break symmetry
    indices = rng.choice(len(body_centers), size=36, replace=False)
    extra_points = body_centers[indices]
    
    # Combine
    all_centers = np.vstack([centers, extra_points])
    
    # Add small jitter to break degeneracy
    all_centers += rng.normal(0, 0.002, size=(N, 3))
    all_centers = np.clip(all_centers, -0.5, 0.5)
    
    return all_centers

def _simulated_annealing(c, time_limit=30, T_start=0.015, T_end=1e-7):
    """
    Simulated annealing with adaptive step size.
    """
    start_time = time.time()
    c = c.copy()
    current_score = _compute_radius(c)
    best_c = c.copy()
    best_score = current_score
    
    T = T_start
    T_min = T_end
    base_step = 0.03
    iter_count = 0
    
    rng = np.random.default_rng(123)
    
    while time.time() - start_time < time_limit:
        iter_count += 1
        
        # Adaptive cooling
        tau = 1500
        T = max(T_start * np.exp(-iter_count / tau), T_min)
        
        # Step size scales with T
        step_scale = max(0.3, min(1.0, current_score / 0.1))
        step_size = base_step * step_scale * (T / T_start + 0.1)
        step_size = max(0.002, min(0.05, step_size))
        
        # Determine move type
        r = rng.random()
        c_new = c.copy()
        
        if r < 0.5:
            # Move one point
            idx = rng.integers(0, N)
            delta = rng.normal(0, step_size, size=3)
            c_new[idx] = c_new[idx] + delta
            c_new[idx] = np.clip(c_new[idx], -HALF, HALF)
            
        elif r < 0.8:
            # Move two points
            i1, i2 = rng.choice(N, size=2, replace=False)
            delta1 = rng.normal(0, step_size, size=3)
            delta2 = rng.normal(0, step_size, size=3)
            c_new[i1] = c_new[i1] + delta1
            c_new[i2] = c_new[i2] + delta2
            c_new[i1] = np.clip(c_new[i1], -HALF, HALF)
            c_new[i2] = np.clip(c_new[i2], -HALF, HALF)
            
        else:
            # Neighbor-aware move
            idx = rng.integers(0, N)
            other = rng.integers(0, N)
            if other == idx:
                other = (other + 1) % N
            dir_vec = c[idx] - c[other]
            norm = np.linalg.norm(dir_vec)
            if norm > 1e-10:
                dir_vec = dir_vec / norm
            else:
                dir_vec = rng.normal(size=3)
                dir_vec = dir_vec / np.linalg.norm(dir_vec)
            
            delta = dir_vec * step_size * 0.5 + rng.normal(0, step_size * 0.5, size=3)
            c_new[idx] = c_new[idx] + delta
            c_new[idx] = np.clip(c_new[idx], -HALF, HALF)
        
        new_score = _compute_radius(c_new)
        
        # Acceptance
        if new_score >= current_score:
            c = c_new
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_c = c.copy()
        else:
            delta_score = new_score - current_score
            T_safe = max(T, 1e-10)
            prob = np.exp(delta_score / T_safe)
            if rng.random() < prob:
                c = c_new
                current_score = new_score
    
    return best_c, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Hybrid Initialization: 4x4x4 grid + interstitial voids
    best_pos = _initial_hybrid()
    
    # Initial Polish to get a good baseline radius
    best_pos = _polish(best_pos, max_iter=200)
    best_r = _compute_radius(best_pos)
    
    if best_r <= 0:
        rng = np.random.default_rng(42)
        best_pos = rng.uniform(-0.45, 0.45, size=(N, 3))
        best_r = _compute_radius(best_pos)

    # 2. Jellyfish Optimization: low-dimensional anchor-based morphing
    if time.time() < deadline - 50.0:
        jf_time = min(25.0, deadline - time.time() - 50.0)
        jf_pos, jf_r = _jellyfish_optimize(best_pos, time_limit=jf_time)
        if jf_r > best_r:
            best_pos = jf_pos
            best_r = jf_r
    
    # 3. Simulated Annealing Phase
    if time.time() < deadline - 35.0:
        sa_time = min(15.0, deadline - time.time() - 35.0)
        sa_pos, sa_r = _simulated_annealing(best_pos, time_limit=sa_time)
        if sa_r > best_r:
            best_pos = sa_pos
            best_r = sa_r
    
    # 4. Iterative Radius Increase
    current_r = best_r
    step_size = 0.001
    
    consecutive_failures = 0
    max_failures = 5
    
    rng = np.random.default_rng(123)
    
    while time.time() < deadline - 15.0:
        target_r = current_r + step_size
        
        if consecutive_failures >= max_failures:
            step_size *= 0.5
            consecutive_failures = 0
            if step_size < 1e-6:
                break
        
        candidates = []
        candidates.append(best_pos)
        candidates.append(best_pos + rng.normal(0, 0.01, size=(N, 3)))
        candidates.append(best_pos + rng.normal(0, 0.05, size=(N, 3)))
        
        success = False
        for start_pos in candidates:
            start_pos = np.clip(start_pos, -0.5, 0.5)
            
            new_pos, energy, opt_success = _minimize_energy(start_pos, target_r, max_iter=200)
            
            if energy < 1e-8:
                actual_r = _compute_radius(new_pos)
                if actual_r >= target_r - 1e-5:
                    best_pos = new_pos
                    best_r = actual_r
                    current_r = actual_r
                    success = True
                    consecutive_failures = 0
                    break
            else:
                pass
        
        if not success:
            consecutive_failures += 1
            
            if time.time() < deadline - 30.0 and consecutive_failures % 2 == 0:
                polished, _, _ = _minimize_energy(best_pos, current_r, max_iter=100)
                if _compute_radius(polished) >= current_r - 1e-5:
                    best_pos = polished
                    best_r = _compute_radius(best_pos)

    # 5. Second Jellyfish pass from improved positions (if time permits)
    if time.time() < deadline - 20.0:
        jf_time2 = min(12.0, deadline - time.time() - 20.0)
        jf_pos2, jf_r2 = _jellyfish_optimize(best_pos, time_limit=jf_time2)
        if jf_r2 > best_r:
            best_pos = jf_pos2
            best_r = jf_r2
    
    # 6. Final High-Precision Polish
    if time.time() < deadline - 5.0:
        polished_pos = _polish(best_pos, max_iter=800)
        final_r = _compute_radius(polished_pos)
        if final_r > best_r:
            best_pos = polished_pos
            best_r = final_r
    
    # 7. Additional restart polish if time permits
    if time.time() < deadline - 5.0:
        for _ in range(3):
            if time.time() > deadline - 3.0:
                break
            # Try polishing from slightly perturbed positions
            perturbed = best_pos + rng.normal(0, 0.005, size=(N, 3))
            perturbed = np.clip(perturbed, -0.49, 0.49)
            polished = _polish(perturbed, max_iter=300)
            r_pert = _compute_radius(polished)
            if r_pert > best_r:
                best_pos = polished
                best_r = r_pert

    return best_pos
