"""Optimized logarithmic energy minimization for 100 points on the unit sphere."""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
import time
import math

N = 100
DEADLINE = 148.0  # Safety margin before 150s kill


def _energy_and_grad(points):
    """Compute logarithmic energy and Riemannian gradient for points on unit sphere.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Using chordal distance |x_i - x_j| = sqrt(2 - 2*x_i·x_j)
    
    Gradient of -log(d_ij) w.r.t. x_i:
      d/dx_i [-log(|x_i - x_j|)] = -(x_i - x_j)/|x_i - x_j|^2 = (x_j - x_i)/d2_ij
    
    So Riemannian grad_i = sum_{j!=i} (x_j - x_i)/d2_ij projected to tangent space.
    """
    n = points.shape[0]
    # Compute pairwise dot products
    dots = points @ points.T
    # Clip to valid range for numerical stability
    dots = np.clip(dots, -1.0, 1.0)
    # Chordal distances squared: |xi - xj|^2 = 2 - 2*dot
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)  # avoid log(0)
    
    # log(1/d) = -log(d) = -0.5*log(d2)
    log_d2 = np.log(d2)
    # Energy = sum_{i<j} -0.5 * log(d2_ij)
    # Sum over upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -0.5 * np.sum(log_d2[iu])
    
    # Gradient computation:
    # Riemannian grad_i = sum_{j!=i} (x_j - x_i) / d2_ij
    # = sum_{j!=i} x_j / d2_ij - x_i * sum_{j!=i} 1/d2_ij
    # Let W[i,j] = 1/d2_ij if i!=j else 0
    # grad_i = (W @ points)_i - x_i * row_sum_i
    
    d2_safe = d2.copy()
    np.fill_diagonal(d2_safe, 1.0)
    W = 1.0 / d2_safe
    np.fill_diagonal(W, 0.0)
    
    row_sums = W.sum(axis=1, keepdims=True)  # shape (n, 1)
    grad = W @ points - points * row_sums
    
    # Project gradient onto tangent space: grad_i -= (grad_i · x_i) * x_i
    radial = np.sum(grad * points, axis=1, keepdims=True)
    grad = grad - radial * points
    
    return energy, grad


def _energy_only(points):
    """Just the energy value (no gradient)."""
    n = points.shape[0]
    dots = points @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    log_d2 = np.log(d2)
    iu = np.triu_indices(n, k=1)
    energy = -0.5 * np.sum(log_d2[iu])
    return energy


def _riemannian_gradient_flow(points, n_iter=2000, lr_init=0.01, lr_decay=0.999,
                               momentum=0.9, deadline=None):
    """Riemannian gradient flow with momentum and adaptive step size.
    
    Uses exponential map (spherical) for updates, which is more natural
    than Euclidean step + renormalization.
    """
    n = points.shape[0]
    pts = points.copy()
    best_pts = pts.copy()
    best_energy = _energy_only(pts)
    v = np.zeros_like(pts)  # momentum velocity
    lr = lr_init
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        energy, grad = _energy_and_grad(pts)
        
        # Adaptive learning rate: reduce if energy increased
        v = momentum * v + (1 - momentum) * (-grad)
        
        # Exponential map: move along tangent direction
        # For small steps, exp_x(v) ≈ x + v (then normalize)
        # For larger steps, use the exact spherical exponential map
        step_norms = np.linalg.norm(v, axis=1, keepdims=True)
        # Avoid division by zero
        step_norms = np.maximum(step_norms, 1e-15)
        
        # Spherical exponential map:
        # exp_x(v) = cos(||v||) * x + sin(||v||)/||v|| * v
        cos_factor = np.cos(step_norms)
        sin_factor = np.sin(step_norms) / step_norms
        
        new_pts = cos_factor * pts + sin_factor * v
        
        # Normalize to ensure we're on the sphere
        new_pts = _normalize(new_pts)
        
        new_energy = _energy_only(new_pts)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pts = new_pts.copy()
        
        if new_energy < energy + 1e-12:
            pts = new_pts
        else:
            # Line search: try smaller step
            lr *= 0.5
            v = v * 0.5
            continue
        
        lr = max(lr_init * (lr_decay ** it), 1e-6)
    
    return best_pts, best_energy


def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _precise_energy(points):
    """Compute energy using math.fsum for maximum precision (matches evaluator)."""
    n = points.shape[0]
    total = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-15:
                return float('inf')
            total.append(math.log(1.0 / d))
    return math.fsum(total)


def _fibonacci_spiral(n, offset=0.5, scale=1.0):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i * scale
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)


def _random_points(n, rng):
    """Generate random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    return _normalize(pts)


def _shifted_spiral(n, shift):
    """Generate a shifted spiral pattern."""
    i = np.arange(n) + 0.5 + shift
    z = 1.0 - 2.0 * np.clip(i / n, 0, 1)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + shift * n)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)


def _sobol_points(n, seed=42):
    """Generate well-distributed points using Sobol sequence."""
    sampler = qmc.Sobol(d=3, seed=seed, scramble=True)
    samples = sampler.random(n)
    # Map [0,1]^3 to sphere using inverse CDF for uniform distribution
    # Use the fact that for uniform on sphere: z = 2*u - 1, theta = 2*pi*v
    z = 2.0 * samples[:, 0] - 1.0
    theta = 2.0 * np.pi * samples[:, 1]
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta), z])
    return _normalize(pts)


def _structured_lattice(n, rng):
    """Generate a structured lattice-like distribution with perturbation."""
    # Use a grid in spherical coordinates with jitter
    n_theta = int(np.sqrt(n))
    n_phi = n // n_theta
    if n_phi * n_theta < n:
        n_phi += 1
    
    thetas = np.linspace(0, np.pi, n_theta + 1)[1:-1]
    phis = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    
    theta_grid, phi_grid = np.meshgrid(thetas, phis, indexing='ij')
    theta_flat = theta_grid.ravel()
    phi_flat = phi_grid.ravel()
    
    # Take first n points
    if len(theta_flat) > n:
        theta_flat = theta_flat[:n]
        phi_flat = phi_flat[:n]
    elif len(theta_flat) < n:
        # Pad with random
        extra_n = n - len(theta_flat)
        theta_flat = np.concatenate([theta_flat, rng.uniform(0, np.pi, extra_n)])
        phi_flat = np.concatenate([phi_flat, rng.uniform(0, 2*np.pi, extra_n)])
    
    # Add small perturbation
    theta_flat += rng.normal(0, 0.05, n)
    phi_flat += rng.normal(0, 0.05, n)
    
    x = np.sin(theta_flat) * np.cos(phi_flat)
    y = np.sin(theta_flat) * np.sin(phi_flat)
    z = np.cos(theta_flat)
    pts = np.column_stack([x, y, z])
    return _normalize(pts)


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Standard Fibonacci spiral
    initializations.append(_fibonacci_spiral(N))
    
    # 2. Perturbed Fibonacci spiral
    base = _fibonacci_spiral(N)
    perturbed = base + 0.05 * rng.standard_normal((N, 3))
    initializations.append(_normalize(perturbed))
    
    # 3. Shifted spirals
    for shift in [0.1, 0.3, 0.7]:
        initializations.append(_shifted_spiral(N, shift))
    
    # 4. Sobol-based points (better space-filling)
    for seed in [42, 123, 456]:
        initializations.append(_sobol_points(N, seed=seed))
    
    # 5. Structured lattice with perturbation
    initializations.append(_structured_lattice(N, rng))
    
    # 6. Random points
    for _ in range(3):
        initializations.append(_random_points(N, rng))
    
    # 7. Anti-symmetric pairs (better distribution)
    half = N // 2
    pts_half = _random_points(half, rng)
    pts_full = np.vstack([pts_half, -pts_half])
    if N % 2 == 1:
        pts_full = np.vstack([pts_full, np.array([[0, 0, 1.0]])])
    initializations.append(_normalize(pts_full))
    
    # 8. More perturbed spirals
    for scale in [0.8, 1.2]:
        initializations.append(_fibonacci_spiral(N, scale=scale))
    
    # 9. Additional Sobol with different seeds
    for seed in [789, 101, 202]:
        initializations.append(_sobol_points(N, seed=seed))
    
    # Pre-compute initial energies to sort by quality
    init_energies = []
    for init_pts in initializations:
        init_energies.append(_energy_only(init_pts))
    
    # Sort initializations by energy (best first)
    sorted_indices = np.argsort(init_energies)
    sorted_initializations = [initializations[i] for i in sorted_indices]
    
    # Track best
    best_points = sorted_initializations[0].copy()
    best_energy = init_energies[sorted_indices[0]]
    
    # Phase 1: For each initialization, do Riemannian gradient flow + L-BFGS-B
    n_inits = len(sorted_initializations)
    time_per_init = (DEADLINE - 20.0) / n_inits  # Reserve 20s for final polish
    
    for init_idx, init_pts in enumerate(sorted_initializations):
        if time.time() > deadline - 20.0:
            break
        
        current = init_pts.copy()
        
        # Phase 1a: Riemannian gradient flow (good for escaping shallow local minima)
        time_left_for_flow = max(0, time_per_init * 0.4)
        flow_deadline = time.time() + time_left_for_flow
        flow_iters = int(time_left_for_flow * 500)  # ~500 iters per second estimate
        flow_iters = max(100, min(flow_iters, 5000))
        
        try:
            flow_pts, flow_energy = _riemannian_gradient_flow(
                current, n_iter=flow_iters, lr_init=0.005, lr_decay=0.9995,
                momentum=0.9, deadline=flow_deadline
            )
            if flow_energy < best_energy:
                best_energy = flow_energy
                best_points = flow_pts.copy()
            current = flow_pts
        except Exception:
            pass
        
        if time.time() > deadline - 20.0:
            break
        
        # Phase 1b: L-BFGS-B from the flow result
        time_left_for_lbfgs = max(0, time_per_init * 0.6)
        lbfgs_deadline = time.time() + time_left_for_lbfgs
        max_iter = int(time_left_for_lbfgs * 200)  # ~200 iters per second
        max_iter = max(100, min(max_iter, 2000))
        
        try:
            result = minimize(
                fun=lambda x: _energy_only(_normalize(x)),
                x0=current.flatten(),
                jac=lambda x: _energy_and_grad(_normalize(x))[1].flatten(),
                method='L-BFGS-B',
                options={
                    'maxiter': max_iter,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxls': 40
                }
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_points = _normalize(result.x.reshape(N, 3))
        except Exception:
            pass
    
    # Phase 2: Final high-precision polish with multiple strategies
    if time.time() < deadline - 5.0:
        # Strategy A: L-BFGS-B with very tight tolerances
        try:
            result_final = minimize(
                fun=lambda x: _energy_only(_normalize(x)),
                x0=best_points.flatten(),
                jac=lambda x: _energy_and_grad(_normalize(x))[1].flatten(),
                method='L-BFGS-B',
                options={
                    'maxiter': 5000,
                    'ftol': 1e-16,
                    'gtol': 1e-14,
                    'maxls': 60
                }
            )
            if result_final.fun < best_energy:
                best_energy = result_final.fun
                best_points = _normalize(result_final.x.reshape(N, 3))
        except Exception:
            pass
    
    if time.time() < deadline - 3.0:
        # Strategy B: Riemannian gradient flow with very small steps for final polish
        try:
            polish_deadline = deadline - 2.0
            polish_pts, polish_energy = _riemannian_gradient_flow(
                best_points, n_iter=10000, lr_init=0.0005, lr_decay=0.9999,
                momentum=0.95, deadline=polish_deadline
            )
            if polish_energy < best_energy:
                best_energy = polish_energy
                best_points = polish_pts
        except Exception:
            pass
    
    if time.time() < deadline - 2.0:
        # Strategy C: One more L-BFGS-B pass
        try:
            result_final2 = minimize(
                fun=lambda x: _energy_only(_normalize(x)),
                x0=best_points.flatten(),
                jac=lambda x: _energy_and_grad(_normalize(x))[1].flatten(),
                method='L-BFGS-B',
                options={
                    'maxiter': 3000,
                    'ftol': 1e-16,
                    'gtol': 1e-14,
                    'maxls': 50
                }
            )
            if result_final2.fun < best_energy:
                best_energy = result_final2.fun
                best_points = _normalize(result_final2.x.reshape(N, 3))
        except Exception:
            pass
    
    # Ensure valid output
    best_points = _normalize(best_points)
    
    # Check for any points too close together and nudge if needed
    for _nudge in range(10):
        min_d = np.inf
        min_pair = (0, 1)
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(best_points[i] - best_points[j])
                if d < min_d:
                    min_d = d
                    min_pair = (i, j)
        if min_d >= 1e-9:
            break
        # Nudge the closer pair apart
        i, j = min_pair
        direction = best_points[i] - best_points[j]
        norm_dir = np.linalg.norm(direction)
        if norm_dir > 1e-15:
            direction = direction / norm_dir
        else:
            direction = rng.standard_normal(3)
            direction = direction / np.linalg.norm(direction)
        best_points[i] = best_points[i] + 1e-8 * direction
        best_points[j] = best_points[j] - 1e-8 * direction
        best_points = _normalize(best_points)
    
    return best_points
