"""
Packing 50 equal circles in a unit circle using constrained variational formulation
with Newton-Raphson on active contact graph and gradient-based relaxation.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
_DEADLINE = 145.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START

def _compute_score(centers):
    """Compute the packing score for given centers."""
    n = len(centers)
    if n < 2:
        return min(1.0 - np.linalg.norm(centers[0]), 0.5)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    norms = np.linalg.norm(centers, axis=1)
    min_bound = np.min(1.0 - norms)
    
    return min(min_pair / 2.0, min_bound)

def _find_active_contacts(centers, r, tol=1e-6):
    """
    Find active contacts (pairs at distance ~2r) and boundary contacts (|c_i| ~ 1-r).
    Returns (pair_indices, boundary_indices).
    """
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    # Active pair contacts: distance close to 2r
    pair_tol = 2 * r * tol
    active_pairs = np.where(np.abs(pair_dists - 2 * r) < pair_tol)[0]
    active_pair_indices = list(zip(i_idx[active_pairs], j_idx[active_pairs]))
    
    # Active boundary contacts: |c_i| close to 1-r
    norms = np.linalg.norm(centers, axis=1)
    boundary_tol = r * tol
    active_boundary = np.where(np.abs(norms - (1 - r)) < boundary_tol)[0]
    
    return active_pair_indices, active_boundary

def _get_initial_hex_lattice(n):
    """Generate a hexagonal lattice and pick n points inside unit circle."""
    r_est = 0.125
    s = 2.0 * r_est
    sqrt3 = np.sqrt(3.0)
    
    points = []
    max_range = int(2.0 / s) + 5
    for j in range(-max_range, max_range + 1):
        for i in range(-max_range, max_range + 1):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * sqrt3 / 2.0
            if np.hypot(x, y) <= 0.98:
                points.append([x, y])
                
    if len(points) == 0:
        rng = np.random.default_rng(42)
        points = []
        for _ in range(n * 10):
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(0, 0.9)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y])
    
    points = np.array(points)
    
    if len(points) < n:
        rng = np.random.default_rng(42)
        while len(points) < n:
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(0, 0.9)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points = np.vstack([points, [[x, y]]])
    
    # Select points closest to the center to form a dense core
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    return points[indices[:n]]

def _newton_solve_active_set(centers, r, max_iter=50, tol=1e-10):
    """
    Solve the constrained system using Newton-Raphson with active set.
    We maximize r subject to:
      |c_i - c_j| >= 2r for all i < j
      |c_i| <= 1 - r for all i
    
    At optimality, the active constraints have Lagrange multipliers >= 0.
    We solve the KKT system for the active constraints.
    
    Returns improved centers and r, or original if Newton fails.
    """
    n = len(centers)
    current = centers.copy()
    current_r = r
    
    for iteration in range(max_iter):
        # Find active contacts
        active_pairs, active_boundary = _find_active_contacts(current, current_r, tol=1e-5)
        
        n_pairs = len(active_pairs)
        n_boundary = len(active_boundary)
        n_constraints = n_pairs + n_boundary
        
        # Build the KKT system:
        # Variables: dx (2n), dr (1), multipliers lambda (n_constraints)
        # Total unknowns: 2n + 1 + n_constraints
        # 
        # Equations:
        # 1. For each active pair (i,j): |c_i - c_j| = 2r  =>  (c_i - c_j)·(c_i - c_j) = (2r)^2
        # 2. For each active boundary i: |c_i| = 1 - r
        # 3. Gradient of objective (maximize r) projected onto null space of constraints = 0
        #    This gives: sum_i lambda_i * grad_constraint_i = 0 (for center variables)
        #    and sum_i lambda_i * (grad_constraint_i wrt r) = 1 (for r variable)
        
        # Simpler approach: use the fact that at optimum, the gradient of r
        # is orthogonal to the feasible directions. We solve for multipliers such that
        # the net force on each circle is zero (for active constraints).
        
        # For a pair (i,j) at distance 2r: constraint g = |c_i - c_j| - 2r = 0
        # grad_{c_i} g = (c_i - c_j) / |c_i - c_j| = unit_vec
        # grad_{c_j} g = -(c_i - c_j) / |c_i - c_j| = -unit_vec
        # grad_r g = -2
        
        # For boundary i: constraint h = |c_i| + r - 1 = 0
        # grad_{c_i} h = c_i / |c_i|
        # grad_r h = 1
        
        # The stationarity condition for maximizing r:
        # For each circle k: sum of lambda * grad_{c_k} over active constraints involving k = 0
        # For r: sum over all constraints of lambda * grad_r = 1
        
        # Build linear system for multipliers
        # Unknowns: lambda for each active constraint
        # Equations:
        #   For each circle k (2n equations): sum_{constraints involving k} lambda * unit_grad = 0
        #   For r (1 equation): sum lambda * grad_r = 1
        
        total_unknowns = n_constraints
        if total_unknowns == 0:
            break
            
        # Build matrix A (2n+1) x n_constraints and vector b (2n+1)
        A = np.zeros((2 * n + 1, n_constraints))
        b = np.zeros(2 * n + 1)
        b[2 * n] = 1.0  # r equation
        
        # Fill in pair constraints
        for idx, (i, j) in enumerate(active_pairs):
            d = current[i] - current[j]
            d_mag = np.linalg.norm(d)
            if d_mag < 1e-12:
                continue
            unit = d / d_mag
            
            # grad_{c_i} = unit, grad_{c_j} = -unit, grad_r = -2
            A[2 * i, idx] = unit[0]
            A[2 * i + 1, idx] = unit[1]
            A[2 * j, idx] = -unit[0]
            A[2 * j + 1, idx] = -unit[1]
            A[2 * n, idx] = -2.0
        
        # Fill in boundary constraints
        for idx, i in enumerate(active_boundary):
            offset = n_pairs + idx
            c_norm = np.linalg.norm(current[i])
            if c_norm < 1e-12:
                continue
            unit = current[i] / c_norm
            
            # grad_{c_i} = unit, grad_r = 1
            A[2 * i, offset] = unit[0]
            A[2 * i + 1, offset] = unit[1]
            A[2 * n, offset] = 1.0
        
        # Solve for multipliers using least squares (may be overdetermined)
        try:
            lambdas, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=1e-10)
        except np.linalg.LinAlgError:
            break
        
        # Check if multipliers are non-negative (KKT condition)
        if np.any(lambdas < -1e-8):
            # Some multipliers are negative; remove those constraints from active set
            # and re-solve. For simplicity, just perturb and retry.
            # This is a simplification; a proper active-set method would iterate.
            pass
        
        # Update centers and r using the multipliers
        # The correction direction for centers is determined by the constraint violations
        # We use a damped Newton step
        
        # Compute constraint violations
        violations = np.zeros(n_constraints)
        for idx, (i, j) in enumerate(active_pairs):
            d = current[i] - current[j]
            d_mag = np.linalg.norm(d)
            violations[idx] = d_mag - 2 * current_r
        
        for idx, i in enumerate(active_boundary):
            offset = n_pairs + idx
            c_norm = np.linalg.norm(current[i])
            violations[offset] = c_norm + current_r - 1.0
        
        # Damped update
        damping = 0.5
        new_centers = current.copy()
        for idx, (i, j) in enumerate(active_pairs):
            if idx >= len(lambdas):
                break
            lam = lambdas[idx]
            d = current[i] - current[j]
            d_mag = np.linalg.norm(d)
            if d_mag < 1e-12:
                continue
            unit = d / d_mag
            # Move i and j apart if violation > 0, together if < 0
            step = damping * lam * violations[idx]
            new_centers[i] += step * unit
            new_centers[j] -= step * unit
        
        for idx, i in enumerate(active_boundary):
            offset = n_pairs + idx
            if offset >= len(lambdas):
                break
            lam = lambdas[offset]
            c_norm = np.linalg.norm(current[i])
            if c_norm < 1e-12:
                continue
            unit = current[i] / c_norm
            # Move i inward if violation > 0 (too far out), outward if < 0
            step = damping * lam * violations[offset]
            new_centers[i] -= step * unit
        
        # Update r: increase if constraints allow
        # dr = sum lambda * grad_r should be related to the r update
        dr = damping * np.sum(lambdas * (np.concatenate([
            -2.0 * np.ones(n_pairs),
            np.ones(n_boundary)
        ])))
        new_r = current_r + dr
        
        # Validate
        if not np.all(np.isfinite(new_centers)):
            break
        if new_r < 0:
            break
        
        new_score = _compute_score(new_centers)
        if new_score > current_r - 1e-12:
            current = new_centers
            current_r = max(current_r, min(new_r, new_score))
        else:
            # Step was bad; reduce damping
            damping *= 0.5
            if damping < 1e-6:
                break
        
        # Check convergence
        if np.max(np.abs(violations)) < tol:
            break
    
    return current, current_r

def _get_ring_initialization(n, num_rings=5):
    """Generate ring-based initialization with better spacing."""
    centers = []
    centers.append([0.0, 0.0])
    
    remaining = n - 1
    # Distribute points across rings with increasing radius
    ring_radii = np.linspace(0.25, 0.9, num_rings)
    points_per_ring = [remaining // num_rings] * num_rings
    for i in range(remaining - sum(points_per_ring)):
        points_per_ring[i % num_rings] += 1
    
    for ring_idx, count in enumerate(points_per_ring):
        if count <= 0:
            continue
        radius = ring_radii[ring_idx]
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + ring_idx * 0.3
        for angle in angles:
            centers.append([radius * np.cos(angle), radius * np.sin(angle)])
    
    return np.array(centers[:n])

def _gradient_ascent_on_radius(centers, r, max_iter=100, tol=1e-10):
    """
    Directly optimize r by moving centers to maintain feasibility.
    At each step, try to increase r slightly and adjust centers.
    """
    current = centers.copy()
    current_r = r
    
    for iteration in range(max_iter):
        if time.time() > _DEADLINE - 10:
            break
        
        # Try to increase r
        dr = 1e-6
        new_r = current_r + dr
        
        # Check if we can maintain feasibility with small center adjustments
        # For each pair at distance ~2*current_r, we need to increase their separation
        # For each circle at boundary, we need to move it inward
        
        # Compute required adjustments
        n = len(current)
        diff = current[:, np.newaxis, :] - current[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        i_idx, j_idx = np.triu_indices(n, k=1)
        pair_dists = dists[i_idx, j_idx]
        
        norms = np.linalg.norm(current, axis=1)
        
        # Find critical pairs (distance close to 2*new_r)
        critical_tol = 1e-4
        critical_pairs = np.where(np.abs(pair_dists - 2 * new_r) < critical_tol)[0]
        critical_boundary = np.where(np.abs(norms - (1 - new_r)) < critical_tol)[0]
        
        if len(critical_pairs) == 0 and len(critical_boundary) == 0:
            # No critical constraints; just increase r
            current_r = new_r
            continue
        
        # Apply small adjustments to critical pairs
        adjusted = current.copy()
        for idx in critical_pairs:
            i, j = i_idx[idx], j_idx[idx]
            d = adjusted[i] - adjusted[j]
            d_mag = np.linalg.norm(d)
            if d_mag < 1e-12:
                continue
            unit = d / d_mag
            # Need to increase separation by dr
            step = dr * 0.5
            adjusted[i] += step * unit
            adjusted[j] -= step * unit
        
        for idx in critical_boundary:
            i = critical_boundary[idx]
            c_norm = np.linalg.norm(adjusted[i])
            if c_norm < 1e-12:
                continue
            unit = adjusted[i] / c_norm
            # Move inward by dr
            adjusted[i] -= dr * unit
        
        # Project to disk
        adjusted = _project_to_disk(adjusted, 0.999)
        
        new_score = _compute_score(adjusted)
        if new_score >= new_r - 1e-8:
            current = adjusted
            current_r = new_r
        else:
            # Can't increase r; break
            break
        
        if current_r - r > 1e-4:
            break
    
    return current, current_r

def _get_random_initialization(n, seed=42):
    """Generate random initialization."""
    rng = np.random.default_rng(seed)
    centers = []
    for _ in range(n):
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0, 0.88)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        centers.append([x, y])
    return np.array(centers)

def _newton_polish(centers, start_time, max_iterations=20):
    """
    Apply Newton-Raphson active-set polishing iteratively.
    """
    current = centers.copy()
    current_r = _compute_score(current)
    best = current.copy()
    best_r = current_r
    
    for it in range(max_iterations):
        if time.time() - start_time > _DEADLINE - 15:
            break
        
        new_centers, new_r = _newton_solve_active_set(current, current_r, max_iter=30, tol=1e-10)
        
        if not np.all(np.isfinite(new_centers)):
            break
        
        new_centers = _project_to_disk(new_centers)
        new_score = _compute_score(new_centers)
        
        if new_score > best_r:
            best_r = new_score
            best = new_centers.copy()
            current = new_centers.copy()
            current_r = new_score
        else:
            # Try gradient ascent on radius
            new_centers2, new_r2 = _gradient_ascent_on_radius(current, current_r, max_iter=50)
            new_centers2 = _project_to_disk(new_centers2)
            new_score2 = _compute_score(new_centers2)
            
            if new_score2 > best_r:
                best_r = new_score2
                best = new_centers2.copy()
                current = new_centers2.copy()
                current_r = new_score2
            else:
                break
    
    return best, best_r

def _project_to_disk(centers, max_radius=0.99):
    """Project centers to be strictly inside the unit disk."""
    centers = centers.copy()
    norms = np.linalg.norm(centers, axis=1)
    mask = norms > max_radius
    if np.any(mask):
        # Fixed broadcasting: (K, 1) * (K, 2)
        centers[mask] *= (max_radius / norms[mask])[:, np.newaxis]
    return centers

def _objective_and_gradient(params, n, beta):
    """
    Objective to minimize: -soft_min(constraints) using numerically stable log-sum-exp
    params: (n, 2) array of centers
    Returns (objective, gradient) for L-BFGS-B.
    """
    centers = params.reshape(n, 2)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    norms = np.linalg.norm(centers, axis=1)
    vals = np.concatenate([pair_dists / 2.0, 1.0 - norms])
    
    # Numerically stable soft-min: -log(sum(exp(-beta * vals))) / beta
    # = -(min(vals) + log(sum(exp(-beta * (vals - min(vals))))) / beta
    m = np.min(vals)
    shifted = vals - m
    # Clamp shifted values to avoid underflow/overflow in exp
    shifted_clamped = np.clip(shifted, 0, 500) 
    log_sum = np.log(np.sum(np.exp(-beta * shifted_clamped)))
    
    obj = -(m - log_sum / beta)
    
    # Weights: w_i = exp(-beta * (vals_i - m)) / sum(exp(-beta * (vals_j - m)))
    exp_shifted = np.exp(-beta * shifted_clamped)
    sum_exp = np.sum(exp_shifted)
    if sum_exp == 0:
        sum_exp = 1e-300
    weights = exp_shifted / sum_exp
    
    grad = np.zeros_like(centers)
    
    # Pair contributions (vectorized)
    n_pairs = n*(n-1)//2
    pair_weights = weights[:n_pairs]
    d_vecs = centers[i_idx] - centers[j_idx]
    d_mags = pair_dists
    safe_mags = np.where(d_mags > 1e-10, d_mags, 1.0)
    unit_vecs = d_vecs / safe_mags[:, np.newaxis]
    
    # For each pair k: grad[i] += w_k * unit_vec[k] / 2, grad[j] -= w_k * unit_vec[k] / 2
    # Vectorize using np.add.at
    half_w = (pair_weights / 2.0)[:, np.newaxis]
    np.add.at(grad, i_idx, half_w * unit_vecs)
    np.add.at(grad, j_idx, -half_w * unit_vecs)
    
    # Boundary contributions
    bound_weights = weights[n_pairs:]
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    # grad -= (bound_weights / safe_norms)[:, np.newaxis] * centers
    bound_grad = (bound_weights / safe_norms)[:, np.newaxis] * centers
    grad -= bound_grad
    
    return obj, -grad

def _optimize_from_start(centers, beta, maxiter=100, gtol=1e-8):
    """Optimize from a given starting point using L-BFGS-B."""
    try:
        res = minimize(
            _objective_and_gradient, 
            centers.flatten(), 
            args=(N, beta),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 50}
        )
        if res.success or res.status in [0, 1, 2, 3, 4, 5, 6]:
            return res.x.reshape(N, 2)
        return centers
    except Exception:
        return centers

def _nelder_mead_polish(centers, maxiter=300):
    """Polish using Nelder-Mead on the raw (nonsmooth) objective."""
    def raw_obj(params):
        c = params.reshape(N, 2)
        # Add a small penalty for points outside to help NM stay in bounds if it drifts
        c = _project_to_disk(c)
        return -_compute_score(c)
    
    try:
        res = minimize(
            raw_obj,
            centers.flatten(),
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
        )
        return res.x.reshape(N, 2)
    except Exception:
        return centers

def _polish(centers, start_time):
    """Progressively polish the solution with increasing beta, Newton, and NM."""
    current = centers.copy()
    best_score = _compute_score(current)
    
    # Stage 1: L-BFGS with moderate to high beta
    betas = [500.0, 2000.0, 5000.0]
    for beta in betas:
        if time.time() - start_time > _DEADLINE - 30:
            break
        
        polished = _optimize_from_start(current, beta, maxiter=100, gtol=1e-10)
        if np.all(np.isfinite(polished)):
            polished = _project_to_disk(polished)
            r = _compute_score(polished)
            if r > best_score:
                best_score = r
                current = polished
    
    # Stage 2: Newton-Raphson active-set polishing
    if time.time() - start_time < _DEADLINE - 25:
        newton_result, newton_score = _newton_polish(current, start_time, max_iterations=15)
        if newton_score > best_score:
            best_score = newton_score
            current = newton_result
    
    # Stage 3: Nelder-Mead polish on the nonsmooth objective
    if time.time() - start_time < _DEADLINE - 15:
        nm_result = _nelder_mead_polish(current, maxiter=300)
        if np.all(np.isfinite(nm_result)):
            nm_result = _project_to_disk(nm_result)
            r = _compute_score(nm_result)
            if r > best_score:
                best_score = r
                current = nm_result
    
    return current, best_score

def _simulated_annealing(centers, start_time, max_iter=50):
    """
    Simulated annealing with L-BFGS-B and Newton steps to escape local minima.
    """
    current = centers.copy()
    current_score = _compute_score(current)
    best = current.copy()
    best_score = current_score
    
    T = 0.01
    T_min = 1e-6
    cooling_rate = 0.95
    rng = np.random.default_rng(12345)
    
    for it in range(max_iter):
        if time.time() - start_time > _DEADLINE - 30:
            break
        
        # Adaptive perturbation
        progress = it / max(max_iter, 1)
        noise_scale = 0.01 * (1 - progress) + 0.002
        
        noisy = current + rng.normal(0, noise_scale, current.shape)
        noisy = _project_to_disk(noisy, 0.97)
        
        # Gradient step
        optimized = _optimize_from_start(noisy, beta=200.0, maxiter=50, gtol=1e-8)
        if not np.all(np.isfinite(optimized)):
            continue
        optimized = _project_to_disk(optimized)
        
        # Newton polish after gradient step
        if time.time() - start_time < _DEADLINE - 25:
            newton_opt, newton_score = _newton_polish(optimized, start_time, max_iterations=5)
            if newton_score > _compute_score(optimized):
                optimized = newton_opt
        
        new_score = _compute_score(optimized)
        delta = new_score - current_score
        
        # Accept if better or with probability exp(delta/T)
        if delta > 0 or rng.random() < np.exp(delta / max(T, T_min)):
            current = optimized
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best = current.copy()
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    return best, best_score

def construct():
    start_time = time.time()
    best_score = 0.0
    best_centers = None
    
    # Multiple diverse initializations
    initializations = [
        _get_initial_hex_lattice(N),
        _get_ring_initialization(N),
        _get_random_initialization(N, seed=42),
        _get_random_initialization(N, seed=123),
    ]
    
    # Ensure all initializations are valid
    for i in range(len(initializations)):
        initializations[i] = _project_to_disk(initializations[i], 0.95)
        
    # First pass: optimize each initialization with L-BFGS + Newton
    for init in initializations:
        if time.time() - start_time > _DEADLINE - 70:
            break
        
        # Quick initial optimization with lower beta to find a basin
        optimized = _optimize_from_start(init, beta=100.0, maxiter=50, gtol=1e-7)
        if np.all(np.isfinite(optimized)):
            optimized = _project_to_disk(optimized)
            
            # Apply Newton polish
            if time.time() - start_time < _DEADLINE - 40:
                newton_opt, newton_score = _newton_polish(optimized, start_time, max_iterations=10)
                if newton_score > _compute_score(optimized):
                    optimized = newton_opt
            
            r = _compute_score(optimized)
            if r > best_score:
                best_score = r
                best_centers = optimized.copy()
                
    if best_centers is None:
        return initializations[0]
        
    # Second pass: Simulated Annealing to escape local minima
    if time.time() - start_time < _DEADLINE - 50:
        sa_centers, sa_score = _simulated_annealing(best_centers, start_time, max_iter=25)
        if sa_score > best_score:
            best_score = sa_score
            best_centers = sa_centers
            
    # Third pass: Final Polish with Newton and NM
    if time.time() - start_time < _DEADLINE - 15:
        best_centers, best_score = _polish(best_centers, start_time)
        
    # Final check and projection
    best_centers = _project_to_disk(best_centers)
    final_score = _compute_score(best_centers)
    
    return best_centers

# EVOLVE-BLOCK-END
