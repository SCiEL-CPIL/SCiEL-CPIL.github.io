"""
Minimize logarithmic energy of 150 points on the unit sphere using
multiple restarts with Riemannian L-BFGS-B style optimization,
momentum-based gradient descent, and simulated annealing polish.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.special import logsumexp

N = 150
TIME_LIMIT = 115.0  # Safety margin before 120s kill


def _energy_and_grad(pos):
    """
    Compute logarithmic energy and gradient for points on unit sphere.
    
    pos: (N, 3) array of points on unit sphere.
    Returns (energy, grad) where grad is the unconstrained gradient w.r.t. pos.
    Energy is sum_{i<j} log(1 / |x_i - x_j|).
    """
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid self-pairs (diagonal)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    # Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    iu = np.triu_indices(N, k=1)
    d = dist[iu]
    energy = -np.sum(np.log(d))
    
    # Gradient computation:
    # d/dp_i [ -log(|p_i - p_j|) ] = -(p_i - p_j) / |p_i - p_j|^2
    # grad_i = sum_{j != i} (p_j - p_i) / |p_i - p_j|^2
    #         = sum_{j != i} p_j / d_ij^2 - p_i * sum_{j != i} 1/d_ij^2
    
    inv_d2 = 1.0 / dist_sq
    np.fill_diagonal(inv_d2, 0.0)  # Exclude self-interaction for gradient
    
    # sum_j p_j / d_ij^2 for each i
    sum_pj_over_d2 = inv_d2 @ pos
    # sum_j 1/d_ij^2 for each i
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    
    grad = sum_pj_over_d2 - pos * sum_inv_d2[:, np.newaxis]
    
    return energy, grad


def _project_to_sphere(pos):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pos, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pos / norms


def _tangent_project(pos, grad):
    """Project gradient onto tangent space at each point."""
    grad_dot_x = np.sum(grad * pos, axis=1, keepdims=True)
    return grad - grad_dot_x * pos


def _retract(pos, tangent_vec, step_size):
    """
    Retraction on the sphere: move along tangent direction then normalize.
    Uses exponential map approximation: x_new = (x + step*tangent) / ||x + step*tangent||
    """
    new_pos = pos + step_size * tangent_vec
    return _project_to_sphere(new_pos)


def _fibonacci_sphere():
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pos = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project_to_sphere(pos)


def _random_sphere_points(rng, n=N):
    """Generate random uniform points on unit sphere."""
    pos = rng.normal(0, 1, size=(n, 3))
    return _project_to_sphere(pos)


def _optimize_riemannian(pos_init, time_budget, lr=0.01):
    """
    Optimize using Riemannian gradient descent with backtracking line search
    and momentum.
    """
    start_time = time.time()
    pos = _project_to_sphere(pos_init.copy())
    energy, grad = _energy_and_grad(pos)
    grad_tangent = _tangent_project(pos, grad)
    
    best_pos = pos.copy()
    best_energy = energy
    
    # Momentum
    beta1 = 0.9
    beta2 = 0.999
    m = np.zeros_like(pos)
    v = np.zeros_like(pos)
    t = 0
    
    # Initial learning rate
    alpha = lr
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        t += 1
        grad_tangent = _tangent_project(pos, grad)
        grad_norm = np.linalg.norm(grad_tangent)
        
        if grad_norm < 1e-12:
            break
        
        # Adam-style update on tangent space
        m = beta1 * m + (1 - beta1) * grad_tangent
        v = beta2 * v + (1 - beta2) * (grad_tangent ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        # Use raw direction (no normalization) for better convergence
        direction = -m_hat
        
        # Adaptive learning rate based on gradient magnitude
        eff_alpha = alpha / (np.sqrt(v_hat) + 1e-8)
        eff_alpha = np.minimum(eff_alpha, 0.05)  # Cap step size
        
        # Backtracking line search
        current_energy = energy
        accepted = False
        for _ in range(20):
            new_pos = _retract(pos, direction, eff_alpha)
            new_energy, new_grad = _energy_and_grad(new_pos)
            
            # Armijo condition
            if new_energy <= current_energy - 1e-4 * np.sum(eff_alpha * grad_tangent * direction):
                pos = new_pos
                energy = new_energy
                grad = new_grad
                accepted = True
                break
            else:
                eff_alpha *= 0.5
        
        if not accepted:
            # If no step accepted, reduce alpha significantly
            alpha *= 0.5
            if alpha < 1e-14:
                break
            continue
        
        # Update best
        if energy < best_energy:
            best_energy = energy
            best_pos = pos.copy()
        
        # Gradually increase alpha
        alpha = min(alpha * 1.03, 0.05)
    
    return _project_to_sphere(best_pos), best_energy


def _lbfgs_polish(pos_init, time_budget, tight=False):
    """
    Use scipy L-BFGS-B on flattened coordinates with sphere constraint handled
    via projection after each iteration.
    """
    start_time = time.time()
    
    # Flatten to 1D
    x0 = pos_init.flatten().copy()
    
    def objective(x):
        pos = x.reshape(N, 3)
        pos = _project_to_sphere(pos)
        energy, _ = _energy_and_grad(pos)
        return energy
    
    def gradient(x):
        pos = x.reshape(N, 3)
        pos = _project_to_sphere(pos)
        _, grad = _energy_and_grad(pos)
        # Project gradient to tangent space
        grad_t = _tangent_project(pos, grad)
        return grad_t.flatten()
    
    # Use L-BFGS-B with max time
    best_pos = _project_to_sphere(pos_init.copy())
    best_energy = objective(x0)
    
    # Set tolerances based on tight flag
    if tight:
        ftol = 1e-15
        gtol = 1e-12
        batch_iter = 100
    else:
        ftol = 1e-12
        gtol = 1e-10
        batch_iter = 50
    
    # Run in chunks to respect time budget
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        # Run a batch of iterations
        res = minimize(
            objective,
            x0,
            jac=gradient,
            method='L-BFGS-B',
            options={
                'maxiter': batch_iter,
                'ftol': ftol,
                'gtol': gtol,
                'maxls': 50,
            }
        )
        
        new_pos = _project_to_sphere(res.x.reshape(N, 3))
        new_energy = objective(res.x)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pos = new_pos.copy()
            x0 = res.x.copy()
        else:
            # No improvement, try with perturbed start
            rng = np.random.default_rng(int(time.time() * 1000) % 100000)
            x0 = (res.x + rng.normal(0, 0.01, size=x0.shape)).copy()
        
        if res.status == 1:  # Success
            break
    
    return _project_to_sphere(best_pos), best_energy


def _simulated_annealing(pos_init, time_budget, T0=0.1, T_end=1e-6, cooling=0.999):
    """
    Simulated annealing on the sphere using retraction moves.
    """
    start_time = time.time()
    pos = _project_to_sphere(pos_init.copy())
    energy, _ = _energy_and_grad(pos)
    
    best_pos = pos.copy()
    best_energy = energy
    T = T0
    
    rng = np.random.default_rng(42)
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_budget or T < T_end:
            break
        
        # Generate random move
        move = rng.normal(0, 1, size=(N, 3))
        move = _tangent_project(pos, move)
        move_norm = np.linalg.norm(move, axis=1, keepdims=True)
        move_norm[move_norm < 1e-15] = 1.0
        move = move / move_norm
        
        step_size = np.sqrt(2 * T / N)
        new_pos = _retract(pos, move, step_size)
        new_energy, _ = _energy_and_grad(new_pos)
        
        delta = new_energy - energy
        
        # Acceptance criterion
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
            pos = new_pos
            energy = new_energy
            
            if energy < best_energy:
                best_energy = energy
                best_pos = pos.copy()
        
        # Cool down
        T *= cooling
    
    return _project_to_sphere(best_pos), best_energy


def construct():
    start_time = time.time()
    
    # Generate starting configurations
    starts = []
    
    # 1. Fibonacci spiral
    starts.append(_fibonacci_sphere())
    
    # 2. Fibonacci with perturbation
    rng = np.random.default_rng(42)
    pos_fib_pert = _fibonacci_sphere() + rng.normal(0, 0.01, size=(N, 3))
    starts.append(_project_to_sphere(pos_fib_pert))
    
    # 3-7. Random starts
    for _ in range(5):
        starts.append(_random_sphere_points(rng))
    
    best_pos = starts[0]
    best_energy = float('inf')
    
    # Time allocation
    total_opt_time = TIME_LIMIT - 15.0  # 15s buffer
    
    # Strategy: 
    # - Run Riemannian GD on all starts with moderate time
    # - Then L-BFGS polish on best few
    # - Then simulated annealing on best
    # - Final L-BFGS polish
    
    # Phase 1: Riemannian GD on all starts
    time_per_gd = total_opt_time * 0.4 / len(starts)
    
    gd_results = []
    for idx, start_pos in enumerate(starts):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 10:
            break
        
        alloc = min(time_per_gd, TIME_LIMIT - 10 - elapsed)
        if alloc < 0.5:
            continue
        
        pos, energy = _optimize_riemannian(start_pos, alloc, lr=0.01)
        gd_results.append((energy, pos))
        
        if energy < best_energy:
            best_energy = energy
            best_pos = pos.copy()
    
    # Sort by energy
    gd_results.sort(key=lambda x: x[0])
    
    # Phase 2: L-BFGS polish on top 3 results
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 20:
        time_per_lbfgs = min((TIME_LIMIT - 20 - elapsed) * 0.3, total_opt_time * 0.2)
        
        for energy, pos in gd_results[:3]:
            elapsed = time.time() - start_time
            if elapsed > TIME_LIMIT - 20:
                break
            
            alloc = min(time_per_lbfgs, TIME_LIMIT - 20 - elapsed)
            if alloc < 0.5:
                continue
            
            pos_polished, energy_polished = _lbfgs_polish(pos, alloc, tight=False)
            
            if energy_polished < best_energy:
                best_energy = energy_polished
                best_pos = pos_polished.copy()
    
    # Phase 3: Simulated annealing on best (reduced time, favoring polish)
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 15:
        alloc_sa = min(10.0, TIME_LIMIT - 15 - elapsed)
        if alloc_sa > 1.0:
            pos_sa, energy_sa = _simulated_annealing(best_pos, alloc_sa, T0=0.05, T_end=1e-8)
            
            if energy_sa < best_energy:
                best_energy = energy_sa
                best_pos = pos_sa.copy()
    
    # Phase 4: Final L-BFGS polish (standard)
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 5:
        alloc_final = min(8.0, TIME_LIMIT - 5 - elapsed)
        if alloc_final > 0.5:
            pos_final, energy_final = _lbfgs_polish(best_pos, alloc_final, tight=False)
            
            if energy_final < best_energy:
                best_energy = energy_final
                best_pos = pos_final.copy()
    
    # Phase 5: Ultra-tight final polish
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 2:
        alloc_tight = min(5.0, TIME_LIMIT - 2 - elapsed)
        if alloc_tight > 0.5:
            pos_tight, energy_tight = _lbfgs_polish(best_pos, alloc_tight, tight=True)
            
            if energy_tight < best_energy:
                best_energy = energy_tight
                best_pos = pos_tight.copy()
    
    # Final validation and projection
    best_pos = _project_to_sphere(best_pos)
    
    if not np.all(np.isfinite(best_pos)):
        best_pos = _fibonacci_sphere()
    
    # Check minimum distance
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 1.0)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9:
        # Perturb and retry quickly
        rng2 = np.random.default_rng(123)
        perturbed = best_pos + rng2.normal(0, 0.01, size=(N, 3))
        perturbed = _project_to_sphere(perturbed)
        
        # Quick optimization
        elapsed = time.time() - start_time
        if elapsed < TIME_LIMIT - 1:
            pos, energy = _optimize_riemannian(perturbed, 5.0, lr=0.01)
            if energy < best_energy:
                best_pos = pos
                best_energy = energy
            else:
                best_pos = perturbed
    
    return best_pos
