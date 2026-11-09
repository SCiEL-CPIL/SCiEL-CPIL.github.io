"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_radius(centres):
    """Compute the maximum common radius for given centres."""
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    
    wall_dist = 0.5 - np.abs(centres)
    min_wall_dist = np.min(wall_dist)
    
    return min(min_pair_dist / 2.0, min_wall_dist)


def _qp_objective(centres, r_target=0.13):
    """
    Quadratic programming objective: maximize sum of squared pairwise distances
    and boundary clearances, with a penalty for violations.
    
    This creates a smoother landscape than min-based objectives.
    """
    c = centres
    n = len(c)
    
    # Pairwise squared distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    sq_dist = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(sq_dist, 0)
    
    # Sum of all pairwise squared distances
    pair_sum = np.sum(sq_dist)
    
    # Boundary clearances: distance to each wall
    # For coordinate k of point i: clearance = 0.5 - |c[i,k]|
    wall_clearances = 0.5 - np.abs(c)
    wall_sum = np.sum(wall_clearances ** 2)
    
    # Penalty for pairs that are too close (quadratic penalty)
    # This helps push apart any pairs that violate the minimum distance
    min_target = 2.0 * r_target
    violation_mask = sq_dist < min_target ** 2
    violations = np.maximum(0, min_target ** 2 - sq_dist)
    penalty = 10.0 * np.sum(violations)
    
    # Penalty for points outside or too close to walls
    wall_violation = np.maximum(0, np.abs(c) - 0.5)
    wall_penalty = 50.0 * np.sum(wall_violation ** 2)
    
    # Objective: maximize pair_sum + wall_sum - penalties
    # We minimize the negative
    return -(pair_sum + 0.5 * wall_sum - penalty - wall_penalty)


def _qp_gradient(centres, r_target=0.13):
    """
    Gradient of the QP objective.
    """
    c = centres
    n = len(c)
    grad = np.zeros_like(c)
    
    # Pairwise contribution
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    sq_dist = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(sq_dist, 0)
    
    # Gradient of sum of squared distances:
    # d/dc_i [sum_j |c_i - c_j|^2] = 2 * sum_j (c_i - c_j)
    pair_grad = 2.0 * np.sum(diff, axis=1)
    grad += pair_grad
    
    # Wall clearance contribution
    # d/dc[i,k] [(0.5 - |c[i,k]|)^2] = -2 * (0.5 - |c[i,k]|) * sign(c[i,k])
    wall_clearances = 0.5 - np.abs(c)
    wall_grad = -2.0 * wall_clearances * np.sign(c)
    grad += 0.5 * wall_grad
    
    # Penalty gradient for close pairs
    min_target = 2.0 * r_target
    min_target_sq = min_target ** 2
    
    # violation_ij = max(0, min_target_sq - sq_dist_ij)
    # d/dc_i [violation_ij] = -2 * (c_i - c_j) * indicator(sq_dist < min_target_sq)
    violation_mask = sq_dist < min_target_sq
    viol_grad = -2.0 * diff * violation_mask[:, :, np.newaxis]
    grad += 10.0 * np.sum(viol_grad, axis=1)
    
    # Wall violation penalty gradient
    # d/dc[i,k] [max(0, |c[i,k]| - 0.5)^2] = 2 * max(0, |c[i,k]| - 0.5) * sign(c[i,k])
    wall_violation = np.maximum(0, np.abs(c) - 0.5)
    wall_viol_grad = 2.0 * wall_violation * np.sign(c)
    grad += 50.0 * wall_viol_grad
    
    return grad


def _qp_optimize(centres, time_limit=10.0, r_target=0.13):
    """
    Optimize using gradient descent with the QP objective.
    """
    start_time = time.time()
    c = centres.copy()
    
    lr = 0.001
    best_c = c.copy()
    best_obj = _qp_objective(c, r_target)
    
    while time.time() - start_time < time_limit:
        obj = _qp_objective(c, r_target)
        if obj < best_obj:
            best_obj = obj
            best_c = c.copy()
        
        grad = _qp_gradient(c, r_target)
        
        # Adaptive learning rate
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-8:
            step = lr * grad / grad_norm
        else:
            break
        
        c = c - step
        c = np.clip(c, -0.499, 0.499)
        
        # Decay learning rate
        lr *= 0.999
    
    return best_c


def _simulated_annealing(centres, time_limit=20.0, r_target=0.13):
    """
    Simulated annealing with QP-based moves to escape local optima.
    """
    start_time = time.time()
    c = centres.copy()
    best_c = c.copy()
    best_radius = _compute_radius(c)
    
    T = 0.05
    T_min = 0.001
    cooling = 0.999
    
    while time.time() - start_time < time_limit and T > T_min:
        # Generate a random perturbation
        n_perturb = np.random.randint(1, 4)
        indices = np.random.choice(len(c), n_perturb, replace=False)
        c_perturbed = c.copy()
        c_perturbed[indices] += np.random.normal(0, T, size=(n_perturb, 3))
        c_perturbed = np.clip(c_perturbed, -0.499, 0.499)
        
        # Compute radius improvement
        r_current = _compute_radius(c)
        r_perturbed = _compute_radius(c_perturbed)
        
        delta_r = r_perturbed - r_current
        
        # Accept if better, or with probability exp(delta_r / T) if worse
        if delta_r >= 0 or np.random.random() < np.exp(delta_r / T):
            c = c_perturbed
            if r_perturbed > best_radius:
                best_radius = r_perturbed
                best_c = c.copy()
        
        T *= cooling
    
    # Final QP polish
    polished = _qp_optimize(c, time_limit=min(3, time_limit - (time.time() - start_time)), r_target=best_radius)
    r_polished = _compute_radius(polished)
    if r_polished > best_radius:
        best_radius = r_polished
        best_c = polished
    
    return best_c


def _make_fcc_packing():
    """Create an FCC-inspired packing for 50 spheres."""
    basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5]
    ])
    
    nx, ny, nz = 3, 3, 4
    points = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                for b in basis:
                    p = np.array([ix, iy, iz]) + b
                    p = (p - 1.25) / 2.5
                    points.append(p)
    
    points = np.array(points)
    
    selected = [0]
    remaining = list(range(1, len(points)))
    
    while len(selected) < N and remaining:
        best_idx = -1
        best_min_dist = -1
        for idx in remaining:
            min_d = np.min(np.linalg.norm(points[selected] - points[idx], axis=1))
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_idx = idx
        
        if best_idx == -1:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
    
    centres = points[selected]
    rng = np.random.RandomState(42)
    centres += rng.normal(0, 0.003, size=centres.shape)
    centres = np.clip(centres, -0.49, 0.49)
    
    return centres


def _make_grid_packing():
    """Create a grid-based packing."""
    # 4x4x4 grid gives 64 points, take 50
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    points = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return points[:N]


def _make_random_packing(seed=42):
    """Create a random initial packing."""
    rng = np.random.RandomState(seed)
    centres = rng.uniform(-0.45, 0.45, size=(N, 3))
    return centres


def _make_fibonacci_packing():
    """Generate points on a Fibonacci lattice scaled to fit in the cube."""
    phi = np.pi * (3.0 - np.sqrt(5.0))
    z = np.linspace(-1, 1, N)
    radius = np.sqrt(1 - z**2)
    theta = phi * np.arange(N)
    
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    
    points = np.column_stack([x, y, z])
    points *= 0.45
    
    return points


def construct():
    """Construct centres of N equal spheres in unit cube."""
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_centres = None
    best_radius = 0.0
    
    def try_solution(centres, label=""):
        nonlocal best_centres, best_radius
        if centres is None:
            return
        r = _compute_radius(centres)
        if r > best_radius:
            best_radius = r
            best_centres = centres.copy()
    
    # Strategy 1: FCC initialization + QP optimization + Simulated Annealing
    if time.time() < deadline - 40:
        try:
            init = _make_fcc_packing()
            time_left = deadline - time.time() - 10
            
            # QP optimization
            qp_centres = _qp_optimize(init, time_limit=min(15, time_left * 0.4), r_target=0.13)
            try_solution(qp_centres, "FCC+QP")
            
            # Simulated annealing from QP result
            sa_centres = _simulated_annealing(qp_centres, time_limit=min(20, time_left * 0.6), r_target=0.13)
            try_solution(sa_centres, "FCC+QP+SA")
        except Exception:
            pass
    
    # Strategy 2: Grid initialization + QP + SA
    if time.time() < deadline - 30:
        try:
            init = _make_grid_packing()
            time_left = deadline - time.time() - 10
            
            qp_centres = _qp_optimize(init, time_limit=min(10, time_left * 0.4), r_target=0.13)
            try_solution(qp_centres, "Grid+QP")
            
            sa_centres = _simulated_annealing(qp_centres, time_limit=min(15, time_left * 0.6), r_target=0.13)
            try_solution(sa_centres, "Grid+QP+SA")
        except Exception:
            pass
    
    # Strategy 3: Fibonacci initialization + QP + SA
    if time.time() < deadline - 25:
        try:
            init = _make_fibonacci_packing()
            time_left = deadline - time.time() - 10
            
            qp_centres = _qp_optimize(init, time_limit=min(10, time_left * 0.4), r_target=0.13)
            try_solution(qp_centres, "Fib+QP")
            
            sa_centres = _simulated_annealing(qp_centres, time_limit=min(15, time_left * 0.6), r_target=0.13)
            try_solution(sa_centres, "Fib+QP+SA")
        except Exception:
            pass
    
    # Strategy 4: Random restarts with SA
    if time.time() < deadline - 15:
        seeds = [1, 7, 13, 42, 99, 123, 456, 789, 1000, 2024]
        for seed in seeds:
            if time.time() > deadline - 5:
                break
            try:
                init = _make_random_packing(seed)
                time_left = deadline - time.time() - 3
                if time_left < 5:
                    break
                
                # Quick QP then SA
                qp_centres = _qp_optimize(init, time_limit=min(3, time_left * 0.3), r_target=0.13)
                sa_centres = _simulated_annealing(qp_centres, time_limit=min(8, time_left * 0.7), r_target=0.13)
                try_solution(sa_centres, f"Random{seed}+QP+SA")
            except Exception:
                continue
    
    # Strategy 5: Multiple SA restarts from best solution
    if best_centres is not None and time.time() < deadline - 10:
        for restart in range(3):
            if time.time() > deadline - 5:
                break
            try:
                time_left = deadline - time.time() - 2
                if time_left < 5:
                    break
                
                # Perturb the best solution
                c_perturbed = best_centres.copy()
                n_perturb = np.random.randint(5, 10)
                indices = np.random.choice(N, n_perturb, replace=False)
                c_perturbed[indices] += np.random.normal(0, 0.02, size=(n_perturb, 3))
                c_perturbed = np.clip(c_perturbed, -0.499, 0.499)
                
                sa_centres = _simulated_annealing(c_perturbed, time_limit=min(10, time_left * 0.8), r_target=best_radius)
                try_solution(sa_centres, f"BestPerturb{restart}")
            except Exception:
                continue
    
    # Final QP polish on best solution
    if best_centres is not None and time.time() < deadline - 2:
        try:
            polished = _qp_optimize(best_centres, time_limit=min(5, deadline - time.time() - 1), r_target=best_radius)
            try_solution(polished, "FinalQP")
        except Exception:
            pass
    
    # Fallback
    if best_centres is None:
        best_centres = _make_grid_packing()
    
    # Validation
    if not np.all(np.isfinite(best_centres)) or np.any(np.abs(best_centres) > 0.5):
        best_centres = _make_grid_packing()
    
    return best_centres
# EVOLVE-BLOCK-END
