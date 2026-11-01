"""
Pack 100 equal spheres in a unit cube [-0.5, 0.5]^3.
Strategy:
  1. Start from a good lattice (FCC/BCC/grid) initialization.
  2. Use a smooth "soft-min" objective based on the log-sum-exp of constraint violations
     to make the landscape differentiable everywhere.
  3. Optimize with L-BFGS-B (vectorized gradients via cKDTree for pair distances).
  4. Use Simulated Annealing to escape local minima: occasionally accept worse solutions.
  5. High-precision polish with very small step sizes near the end.
  6. Multiple restarts from different initializations.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 100
CUBE_HALF = 0.5

# EVOLVE-BLOCK-START

def _compute_radius(centres):
    """Compute the maximum valid radius for given centres in [-0.5, 0.5]^3."""
    if not np.all(np.isfinite(centres)):
        return 0.0
    if centres.shape != (N, 3):
        return 0.0
    # Wall distances: min over axes of (0.5 - |c_ik|)
    wall_dists = np.min(CUBE_HALF - np.abs(centres), axis=1)  # (N,)
    min_wall = np.min(wall_dists)
    if min_wall < 0:
        return 0.0
    # Pair distances
    tree = cKDTree(centres)
    pairs = tree.query_pairs(1.0, output_type='ndarray')
    if len(pairs) == 0:
        return min_wall
    diffs = centres[pairs[:, 0]] - centres[pairs[:, 1]]
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    min_pair = np.min(dists)
    return min(min_pair / 2.0, min_wall)


def _soft_min_objective_and_grad(x, beta):
    """
    Soft-min objective using log-sum-exp to smoothly approximate the minimum radius.
    
    We want to maximize r = min(min_wall, min_pair/2).
    Equivalently, minimize -r.
    
    Using log-sum-exp:
      -log-sum-exp(-a_i / beta) ≈ min(a_i) for large beta
    where a_i are the constraint values (wall distances and half pair distances).
    
    Actually, let's define:
      For walls: a_i = 0.5 - |c_i| (per component, but we take min over axes per sphere)
      For pairs: b_ij = ||c_i - c_j|| / 2
    
    We want min(all a_i, all b_ij).
    
    A cleaner approach: define violations.
    Let target = some reference radius. But that's what SCA does.
    
    Alternative: directly optimize the minimum distance.
    min_radius = min( min_i min_k (0.5 - |c_ik|), min_{i<j} ||c_i - c_j||/2 )
    
    Let's use:
      f(x) = -log( sum_i exp(-a_i / beta) + sum_{i<j} exp(-b_ij / beta) ) * beta
    This approximates -min(a_i, b_ij) * ... hmm, not quite.
    
    Better: Let's define the objective as:
      g(x) = log( sum_i exp(-a_i / beta) + sum_{i<j} exp(-b_ij / beta) ) * beta
    Then for large beta, g(x) ≈ -min(a_i, b_ij) + log(num_constraints) * beta
    So minimizing g(x) maximizes min(a_i, b_ij).
    
    Actually the standard soft-min is:
      softmin(a_1, ..., a_n) = -beta * log( sum_i exp(-a_i / beta) )
    For large beta, softmin → min(a_i).
    
    So we want to maximize softmin(all constraints), i.e., minimize -softmin.
    
    Let's define:
      a = [all wall distances (N*3 values), all half-pair distances (M values)]
      objective = -softmin(a) = beta * log(sum(exp(-a_i/beta)))
    
    We minimize this objective.
    """
    c = x.reshape(N, 3)
    
    # Wall distances: for each sphere i and axis k, a = 0.5 - |c_ik|
    wall_vals = CUBE_HALF - np.abs(c)  # (N, 3)
    
    # Pair distances / 2
    tree = cKDTree(c)
    pairs = tree.query_pairs(1.0, output_type='ndarray')
    
    if len(pairs) == 0:
        pair_vals = np.array([1.0])  # dummy
    else:
        diffs = c[pairs[:, 0]] - c[pairs[:, 1]]
        pair_dists = np.sqrt(np.sum(diffs**2, axis=1)) / 2.0
        pair_vals = pair_dists
    
    # Combine all constraint values
    all_vals = np.concatenate([wall_vals.flatten(), pair_vals])
    
    # For numerical stability, subtract min
    min_val = np.min(all_vals)
    shifted = all_vals - min_val
    
    # Clip to avoid overflow
    shifted = np.clip(shifted, 0, 100)
    
    # Soft-min: -beta * log(sum(exp(-shifted/beta)))
    # = -beta * log(sum(exp(-(all_vals - min_val)/beta)))
    # = -beta * log(exp(min_val/beta) * sum(exp(-all_vals/beta)))
    # = -beta * (min_val/beta + log(sum(exp(-all_vals/beta))))
    # = -min_val - beta * log(sum(exp(-all_vals/beta)))
    
    # For stability, use the shifted version:
    # sum(exp(-shifted/beta)) where shifted >= 0
    exp_terms = np.exp(-shifted / beta)
    log_sum = np.log(np.sum(exp_terms))
    
    # softmin = -beta * log_sum + min_val
    # We want to maximize softmin, so minimize -softmin = beta * log_sum - min_val
    obj = beta * log_sum - min_val
    
    # Gradient
    # d(obj)/d(c) = d(beta * log_sum - min_val)/d(c)
    # min_val is piecewise linear, its gradient is -sign(c) for the active wall or (c_i - c_j)/||c_i-c_j|| for active pair.
    # But in the soft version, the gradient is smooth.
    
    # d log_sum / d c = (1/sum_exp) * sum_i (-1/beta) * exp(-shifted_i/beta) * d(shifted_i)/dc
    # where shifted_i = all_vals_i - min_val
    # d(shifted_i)/dc = d(all_vals_i)/dc (since min_val doesn't depend on c in a differentiable way for the soft version)
    
    # Actually, let's be more careful.
    # all_vals_i are the constraint values. For walls: a_ik = 0.5 - |c_ik|, d(a_ik)/dc = -sign(c_ik)
    # For pairs: b_ij = ||c_i - c_j||/2, d(b_ij)/dc_i = (c_i - c_j)/(2||c_i-c_j||), d(b_ij)/dc_j = -(c_i-c_j)/(2||c_i-c_j||)
    
    # Let w_i = exp(-shifted_i/beta) / sum_exp (weights)
    weights = exp_terms / np.sum(exp_terms)
    
    grad = np.zeros_like(c)
    
    # Wall contributions
    # For each wall value a_ik, contribution to grad:
    # d obj / dc = beta * d(log_sum)/dc - d(min_val)/dc
    # In the soft version, we ignore the -min_val term's gradient (it's zero in the soft approximation)
    # d(log_sum)/dc = sum_i weights_i * (-1/beta) * d(all_vals_i)/dc
    # So beta * d(log_sum)/dc = -sum_i weights_i * d(all_vals_i)/dc
    
    # Wall part:
    wall_weights = weights[:N*3].reshape(N, 3)
    for axis in range(3):
        # d(a_ik)/dc_ik = -sign(c_ik)
        grad[:, axis] -= wall_weights[:, axis] * np.sign(c[:, axis])
    
    # Pair part:
    if len(pairs) > 0:
        pair_weights = weights[N*3:]
        # d(b_ij)/dc_i = (c_i - c_j) / (2 * ||c_i - c_j||)
        # d(b_ij)/dc_j = -(c_i - c_j) / (2 * ||c_i - c_j||)
        # Contribution: -pair_weights * d(b_ij)/dc
        for idx in range(len(pairs)):
            i, j = pairs[idx]
            w = pair_weights[idx]
            if w < 1e-15:
                continue
            d = c[i] - c[j]
            dist = np.sqrt(np.sum(d**2))
            if dist < 1e-12:
                continue
            # d(b_ij)/dc_i = d / (2*dist)
            # Contribution to grad[i]: -w * d/(2*dist)
            # Contribution to grad[j]: +w * d/(2*dist)
            factor = w / (2.0 * dist)
            grad[i] -= factor * d
            grad[j] += factor * d
    
    return obj, grad.ravel()


def _optimize_soft_min(centres, beta_start, beta_end, deadline, max_iters=200):
    """
    Optimize the soft-min objective with annealing (decreasing beta).
    """
    c = centres.copy().astype(np.float64)
    best_c = c.copy()
    best_obj = _soft_min_objective_and_grad(c.flatten(), beta_start)[0]
    best_r = _compute_radius(best_c)
    
    bounds = [(-CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)] * (N * 3)
    
    # Annealing schedule: beta decreases from beta_start to beta_end
    n_anneal = 5
    betas = np.geomspace(beta_start, beta_end, n_anneal)
    
    for beta_idx, beta in enumerate(betas):
        if time.time() > deadline:
            break
        
        def obj_grad(x):
            return _soft_min_objective_and_grad(x, beta)
        
        try:
            res = minimize(
                obj_grad,
                c.flatten(),
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iters // n_anneal, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            
            new_c = np.clip(res.x.reshape(N, 3), -CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)
            new_obj = _soft_min_objective_and_grad(new_c.flatten(), beta)[0]
            
            if new_obj < best_obj:
                best_obj = new_obj
                best_c = new_c.copy()
                best_r = _compute_radius(best_c)
            
            c = new_c
        except Exception:
            pass
    
    return best_c, best_r


def _force_directed_refinement(centres, max_iters=300, dt_init=0.005, deadline=None):
    """
    Force-directed repulsion to spread spheres apart.
    """
    centres = centres.copy().astype(np.float64)
    best_centres = centres.copy()
    best_r = _compute_radius(best_centres)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        tree = cKDTree(centres)
        pairs = tree.query_pairs(0.6, output_type='ndarray')
        
        forces = np.zeros_like(centres)
        
        if len(pairs) > 0:
            diffs = centres[pairs[:, 0]] - centres[pairs[:, 1]]
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            
            for idx in range(len(pairs)):
                i, j = pairs[idx]
                d = dists[idx]
                if d < 1e-12:
                    continue
                # Repulsive force ~ 1/d^2
                f = 1.0 / (d**2 + 1e-6)
                forces[i] += f * diffs[idx] / d
                forces[j] -= f * diffs[idx] / d
        
        # Wall forces: push away from walls
        for axis in range(3):
            # Positive wall
            over = centres[:, axis] - CUBE_HALF
            mask_pos = over > -0.05
            if np.any(mask_pos):
                forces[mask_pos, axis] -= 50.0 * np.clip(over[mask_pos], 0, None)
                forces[mask_pos, axis] -= 5.0 * np.clip(0.05 + over[mask_pos], 0, None)
            # Negative wall
            under = centres[:, axis] + CUBE_HALF
            mask_neg = under < 0.05
            if np.any(mask_neg):
                forces[mask_neg, axis] += 50.0 * np.clip(-under[mask_neg], 0, None)
                forces[mask_neg, axis] += 5.0 * np.clip(0.05 - under[mask_neg], 0, None)
        
        # Adaptive step size
        dt = dt_init * (0.995 ** it)
        centres += dt * forces
        centres = np.clip(centres, -CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)
        
        r = _compute_radius(centres)
        if r > best_r:
            best_r = r
            best_centres = centres.copy()
        
        # Early stopping if no improvement for a while
        if it > 50 and abs(r - best_r) < 1e-10 and it % 50 == 0:
            # Check if we've been stuck
            if it > 100:
                break
    
    return best_centres, best_r


def _simulated_annealing(centres, deadline, n_restarts=3):
    """
    Simulated annealing: occasionally accept worse solutions to escape local minima.
    """
    c = centres.copy().astype(np.float64)
    best_c = c.copy()
    best_r = _compute_radius(best_c)
    
    T_start = 0.01
    T_end = 0.0001
    n_steps = 200
    Ts = np.geomspace(T_start, T_end, n_steps)
    
    bounds = [(-CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)] * (N * 3)
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        T = Ts[step]
        
        # Propose a move: small random perturbation
        move_size = 0.005 * (T / T_start) ** 0.5
        c_new = c + np.random.randn(N, 3) * move_size
        c_new = np.clip(c_new, -CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)
        
        r_new = _compute_radius(c_new)
        r_old = _compute_radius(c)
        
        # Accept if better, or with probability exp(-(r_old - r_new)/T) if worse
        if r_new >= r_old:
            c = c_new
            if r_new > best_r:
                best_r = r_new
                best_c = c.copy()
        else:
            delta = r_old - r_new
            if delta < 1e-12:
                c = c_new
            else:
                accept_prob = np.exp(-delta / T)
                if np.random.random() < accept_prob:
                    c = c_new
                    if r_new > best_r:
                        best_r = r_new
                        best_c = c.copy()
        
        # Every 20 steps, do a local refinement
        if step % 20 == 19 and time.time() < deadline - 5:
            c_ref, r_ref = _force_directed_refinement(c, max_iters=30, dt_init=0.002, deadline=deadline - 2)
            if r_ref > best_r:
                best_r = r_ref
                best_c = c_ref.copy()
            c = c_ref
    
    return best_c, best_r


def _init_grid():
    """Simple cubic grid initialization."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def _init_fcc_like():
    """FCC lattice initialization."""
    a = 1.0 / 5.0
    points = []
    for i in range(6):
        for j in range(6):
            for k in range(6):
                x = i * a - 0.5
                y = j * a - 0.5
                z = k * a - 0.5
                # FCC: add face-center offsets
                points.append([x, y, z])
                points.append([x + a/2, y + a/2, z])
                points.append([x + a/2, y, z + a/2])
                points.append([x, y + a/2, z + a/2])
    
    points = np.array(points)
    mask = np.all(np.abs(points) <= CUBE_HALF - 0.01, axis=1)
    points = points[mask]
    
    if len(points) < N:
        grid = _init_grid()
        points = np.vstack([points, grid[len(points):]])
    
    return points[:N].copy()


def _init_bcc_like():
    """BCC lattice initialization."""
    a = 1.0 / 5.0
    points = []
    for i in range(6):
        for j in range(6):
            for k in range(6):
                x = i * a - 0.5
                y = j * a - 0.5
                z = k * a - 0.5
                points.append([x, y, z])
                points.append([x + a/2, y + a/2, z + a/2])
    
    points = np.array(points)
    mask = np.all(np.abs(points) <= CUBE_HALF - 0.01, axis=1)
    points = points[mask]
    
    if len(points) < N:
        grid = _init_grid()
        points = np.vstack([points, grid[len(points):]])
    
    return points[:N].copy()


def _init_halton():
    """Halton sequence initialization for good space-filling."""
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
    
    return np.array(points)


def _init_random():
    """Random initialization."""
    return np.random.uniform(-0.45, 0.45, (N, 3))


def _potential_energy_minimization(centres, deadline, max_iters=100):
    """
    Minimize a potential energy function that penalizes overlaps.
    E = sum_{i<j} max(0, 2r - d_ij)^2 + sum_i max(0, r - wall_i)^2
    """
    c = centres.copy().astype(np.float64)
    best_c = c.copy()
    best_r = _compute_radius(best_c)
    
    bounds = [(-CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)] * (N * 3)
    
    def energy_and_grad(x):
        c = x.reshape(N, 3)
        r = _compute_radius(c)
        r = max(r, 0.01)  # Minimum radius to avoid division issues
        
        # Wall violations
        wall_dists = CUBE_HALF - np.abs(c)
        wall_viol = np.maximum(0, r - wall_dists)
        
        # Pair violations
        tree = cKDTree(c)
        pairs = tree.query_pairs(2 * r + 0.05, output_type='ndarray')
        
        grad = np.zeros_like(c)
        energy = np.sum(wall_viol**2)
        
        # Wall gradient
        for axis in range(3):
            viol = wall_viol[:, axis]
            if np.any(viol > 0):
                grad[:, axis] -= 2 * viol * np.sign(c[:, axis])
        
        if len(pairs) > 0:
            diffs = c[pairs[:, 0]] - c[pairs[:, 1]]
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            viol = np.maximum(0, 2 * r - dists)
            
            for idx in range(len(pairs)):
                if viol[idx] <= 0:
                    continue
                i, j = pairs[idx]
                d = dists[idx]
                if d < 1e-12:
                    continue
                factor = 2 * viol[idx] / d
                grad[i] -= factor * diffs[idx]
                grad[j] += factor * diffs[idx]
                energy += viol[idx]**2
        
        return energy, grad.ravel()
    
    try:
        res = minimize(
            energy_and_grad,
            c.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iters, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        new_c = np.clip(res.x.reshape(N, 3), -CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)
        new_r = _compute_radius(new_c)
        if new_r > best_r:
            best_r = new_r
            best_c = new_c.copy()
    except Exception:
        pass
    
    return best_c, best_r


def _high_precision_polish(centres, deadline):
    """Final high-precision polish with very small steps."""
    c = centres.copy().astype(np.float64)
    best_c = c.copy()
    best_r = _compute_radius(best_c)
    
    for dt in [0.001, 0.0005, 0.0002, 0.0001]:
        if time.time() > deadline:
            break
        
        for it in range(50):
            if time.time() > deadline:
                break
            
            tree = cKDTree(c)
            pairs = tree.query_pairs(0.3, output_type='ndarray')
            
            forces = np.zeros_like(c)
            
            if len(pairs) > 0:
                diffs = c[pairs[:, 0]] - c[pairs[:, 1]]
                dists = np.sqrt(np.sum(diffs**2, axis=1))
                
                for idx in range(len(pairs)):
                    i, j = pairs[idx]
                    d = dists[idx]
                    if d < 1e-12:
                        continue
                    f = 1.0 / (d**2 + 1e-8)
                    forces[i] += f * diffs[idx] / d
                    forces[j] -= f * diffs[idx] / d
            
            # Wall forces
            for axis in range(3):
                over = c[:, axis] - CUBE_HALF
                mask_pos = over > -0.02
                if np.any(mask_pos):
                    forces[mask_pos, axis] -= 100.0 * np.clip(over[mask_pos], 0, None)
                under = c[:, axis] + CUBE_HALF
                mask_neg = under < 0.02
                if np.any(mask_neg):
                    forces[mask_neg, axis] += 100.0 * np.clip(-under[mask_neg], 0, None)
            
            c += dt * forces
            c = np.clip(c, -CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)
            
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    np.random.seed(42)
    
    initializations = [
        _init_fcc_like,
        _init_bcc_like,
        _init_grid,
        _init_halton,
        _init_random,
    ]
    
    best_centres = None
    best_r = 0.0
    
    for init_idx, init_fn in enumerate(initializations):
        if time.time() > deadline - 20.0:
            break
        
        init = init_fn()
        init = np.clip(init, -CUBE_HALF + 0.01, CUBE_HALF - 0.01)
        
        # Phase 1: Force-directed refinement
        c1, r1 = _force_directed_refinement(init, max_iters=150, dt_init=0.005, deadline=deadline - 15.0)
        
        if r1 > best_r:
            best_r = r1
            best_centres = c1.copy()
        
        if time.time() > deadline - 15.0:
            continue
        
        # Phase 2: Soft-min optimization with annealing
        c2, r2 = _optimize_soft_min(c1, beta_start=1.0, beta_end=0.01, deadline=deadline - 10.0, max_iters=100)
        
        if r2 > best_r:
            best_r = r2
            best_centres = c2.copy()
        
        if time.time() > deadline - 10.0:
            continue
        
        # Phase 3: Simulated annealing to escape local minima
        c3, r3 = _simulated_annealing(c2, deadline=deadline - 5.0)
        
        if r3 > best_r:
            best_r = r3
            best_centres = c3.copy()
        
        if time.time() > deadline - 5.0:
            continue
        
        # Phase 4: High-precision polish
        c4, r4 = _high_precision_polish(c3, deadline=deadline - 1.0)
        
        if r4 > best_r:
            best_r = r4
            best_centres = c4.copy()
    
    # Additional restarts from best solution
    if best_centres is not None and time.time() < deadline - 10.0:
        for pert_scale in [0.01, 0.02, 0.03]:
            if time.time() > deadline - 5.0:
                break
            
            perturbed = best_centres + np.random.randn(N, 3) * pert_scale
            perturbed = np.clip(perturbed, -CUBE_HALF + 0.01, CUBE_HALF - 0.01)
            
            c_ref, r_ref = _force_directed_refinement(perturbed, max_iters=80, dt_init=0.003, deadline=deadline - 3.0)
            
            if r_ref > best_r:
                best_r = r_ref
                best_centres = c_ref.copy()
            
            if time.time() > deadline - 3.0:
                break
            
            c_pol, r_pol = _high_precision_polish(c_ref, deadline=deadline - 1.0)
            if r_pol > best_r:
                best_r = r_pol
                best_centres = c_pol.copy()
    
    # Final validation
    if best_centres is not None:
        final_r = _compute_radius(best_centres)
        if final_r > best_r:
            best_r = final_r
        return best_centres
    else:
        return _init_grid()

# EVOLVE-BLOCK-END
